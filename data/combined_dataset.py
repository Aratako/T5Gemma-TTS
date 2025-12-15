

# ==== audio backend helper (auto-added) ====
def _torch_ge_29():
    try:
        import torch
        v = torch.__version__.split("+")[0]
        major, minor = map(int, v.split(".")[:2])
        return (major, minor) >= (2, 9)
    except Exception:
        return False

def load_audio_segment(audio_path, offset=0, num_frames=None):
    if not _torch_ge_29():
        import torchaudio
        return torchaudio.load(
            audio_path,
            frame_offset=offset,
            num_frames=num_frames
        )

    import soundfile as sf
    import torch

    wav, sr = sf.read(audio_path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    start = offset or 0
    end = None if num_frames is None else start + num_frames
    wav = wav[start:end]

    wav = torch.from_numpy(wav).unsqueeze(0)
    return wav, sr

import copy
import csv
import glob
import logging
import os
import random
import shutil

import ffmpeg
import numpy as np
import torch
import torch.distributed as dist
import torchaudio

from data.tokenizer import AudioTokenizer

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

class dataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.args.target_time_stretch_prob = getattr(self.args, "target_time_stretch_prob", 0)
        self.args.target_time_stretch_bound = getattr(self.args, "target_time_stretch_bound", 0.1)
        self.split = split

        assert self.split in ['train', 'valid', 'test'], f"split should be one of ['train', 'valid', 'test'], but it's {split}"

        if "[" not in self.args.dataset_dir or "]" not in self.args.dataset_dir:
            self.dataset_dir = f"['{self.args.dataset_dir}']"
        else:
            self.dataset_dir = copy.deepcopy(self.args.dataset_dir)
        self.dataset_dir = eval(self.dataset_dir)
        data = []
        if "[" not in self.args.manifest_name or "]" not in self.args.manifest_name:
            self.args.manifest_name = f"['{self.args.manifest_name}']"
        else:
            self.args.manifest_name = copy.deepcopy(self.args.manifest_name)
        self.manifest_name = eval(self.args.manifest_name)
        if len(self.manifest_name) != len(self.dataset_dir):
            assert len(self.manifest_name) == 1, f"len(self.manifest_name) should be 1 or equal to len(self.dataset_dir), but it's {len(self.manifest_name)}"
            self.manifest_name = self.manifest_name * len(self.dataset_dir)
        for i_data, dataset_dir in enumerate(self.dataset_dir):
            n_datapoints = 0
            manifest_fn = os.path.join(dataset_dir, self.manifest_name[i_data], self.split+".txt")
            if not os.path.isfile(manifest_fn):
                all_manifest_fn = glob.glob(manifest_fn.replace(".txt", "_*=*.txt"))
                if len(all_manifest_fn) == 0:
                    logging.info(f"no manifest file found for {split} split in {dataset_dir}")
                    continue
                if self.args.debug:
                    logging.info(f"debugging mode, only using the frist found manifest file: {all_manifest_fn[0]}")
                    all_manifest_fn = all_manifest_fn[:1]
                else:
                    if dist.is_initialized() and dist.get_rank() == 0:
                        logging.info(f"Combining found manifest files for {split}: {all_manifest_fn}")
                for cur_manifest_fn in all_manifest_fn:
                    with open(cur_manifest_fn, "r") as rf:
                        tmp = [l.strip().split("\t") + [i_data] for l in rf.readlines()] # i_data is the index of the dataset
                        n_datapoints += len(tmp)
                        data += tmp
            else:
                with open(manifest_fn, "r") as rf:
                    tmp = [l.strip().split("\t") + [i_data] for l in rf.readlines()]
                    data += tmp
                    n_datapoints += len(tmp)
            if dist.is_initialized() and dist.get_rank() == 0:
                logging.info(f"number of data points for {split} split in {dataset_dir}: {n_datapoints}")
        # Optionally cap validation set size for quicker val runs
        cap = getattr(self.args, "validation_sample_cap", None)
        if self.split == "valid" and cap is not None and cap > 0 and len(data) > cap:
            orig_len = len(data)
            g = torch.Generator()
            g.manual_seed(getattr(self.args, "seed", 0))
            perm = torch.randperm(len(data), generator=g).tolist()
            keep = set(perm[:cap])
            data = [d for idx, d in enumerate(data) if idx in keep]
            if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                logging.info(f"validation_sample_cap={cap}: trimmed valid set from {orig_len} to {len(data)} examples")
        assert len(data) > 0, f"no data found for {split} split"
        lengths_list = [int(item[1]) for item in data]
        self.data = []
        self.lengths_list = []
        total_duration = 0
        for d, l in zip(data, lengths_list):
            if l >= self.args.encodec_sr*self.args.audio_min_length:
                if self.args.drop_long and l > self.args.encodec_sr*self.args.audio_max_length:
                    continue
                self.data.append(d)
                self.lengths_list.append(l)
                total_duration += l / self.args.encodec_sr / 3600
        if dist.is_initialized() and dist.get_rank() == 0:
            logging.info(f"TOTAL number of data points for {self.split} split: {len(self.lengths_list)}")
            logging.info(f"TOTAL duration for {self.split} split: {total_duration:.1f} hours")
        self.text_mode = True
        if AutoTokenizer is None:
            raise ImportError("transformers is required for text tokenization. Please install it.")
        tokenizer_name = getattr(self.args, "text_tokenizer_name", None) or getattr(self.args, "t5gemma_model_name", "google/t5gemma-b-b-ul2")
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.text_tokenizer.pad_token_id is None:
            self.text_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.args.text_pad_token = self.text_tokenizer.pad_token_id
        if getattr(self.args, "add_eos_to_text", 0) == 0 and self.text_tokenizer.eos_token_id is not None:
            self.args.add_eos_to_text = self.text_tokenizer.eos_token_id
        # kept for checkpoint compatibility; unused in text mode
        self.phn2num = None

        tokenizer_backend = getattr(self.args, "audio_tokenizer", "xcodec2")
        # Runtime tokenizer is only needed when we actually perform on-the-fly
        # time-stretching (target or neighbor). Using xcodec2 alone should not
        # force GPU0 to load the tokenizer.
        need_runtime_tokenizer = (
            (self.args.neighbor_prompt_prob > 0 and self.args.time_stretch_prob > 0)
            or self.args.target_time_stretch_prob > 0
        )
        if need_runtime_tokenizer:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.audio_tokenizer = AudioTokenizer(
                backend="xcodec2",
                model_name=getattr(self.args, "xcodec2_model_name", None),
                device=device,
            )
            assert (
                self.audio_tokenizer.sample_rate == self.args.codec_audio_sr
            ), f"audio_tokenizer.sample_rate: {self.audio_tokenizer.sample_rate}, self.args.codec_audio_sr: {self.args.codec_audio_sr}"
            if dist.is_initialized() and dist.get_rank() == 0:
                logging.info(f"rank: {dist.get_rank()}, audio_tokenizer backend: {tokenizer_backend}, device: {self.audio_tokenizer._device}")

    def __len__(self):
        return len(self.lengths_list)

    def _maybe_resample_for_tokenizer(self, waveform, current_sr):
        target_sr = getattr(self.audio_tokenizer, "encode_sample_rate", self.audio_tokenizer.sample_rate)
        if current_sr != target_sr:
            waveform_2d = waveform.squeeze(0)
            resampler = torchaudio.transforms.Resample(current_sr, target_sr)
            waveform_2d = resampler(waveform_2d)
            waveform = waveform_2d.unsqueeze(0)
        return waveform

    def _load_text_tokens(self, dataset_dir, filename):
        text_folder = getattr(self.args, "text_folder_name", "text")
        text_fn = os.path.join(dataset_dir, text_folder, filename)
        if not os.path.isfile(text_fn):
            raise FileNotFoundError(f"text transcript not found: {text_fn}")
        with open(text_fn, "r") as tf:
            text = tf.read().strip()
        if len(text) == 0:
            return []
        return self.text_tokenizer.encode(text, add_special_tokens=False)

    def _load_text_and_codes(self, index):
        item = self.data[index]
        dataset_dir = self.dataset_dir[item[-1]]
        base_name = item[0] + ".txt"
        ef = os.path.join(dataset_dir, self.args.encodec_folder_name, base_name)
        try:
            x = self._load_text_tokens(dataset_dir, base_name)
        except Exception as exc:
            logging.info(f"loading failed for {base_name} (text mode): {exc}")
            return [], [[]], dataset_dir, None

        audio_ext = None
        audio_fn = None
        time_stretch_enabled = getattr(self.args, "target_time_stretch_prob", 0) > 0
        if time_stretch_enabled:
            audio_dir = os.path.join(dataset_dir, self.args.audio_folder_name)
            base_audio_name = item[0].replace(".txt", "")
            for ext in [".wav", ".flac", ".mp3", ".ogg"]:
                candidate = os.path.join(audio_dir, base_audio_name + ext)
                if os.path.isfile(candidate):
                    audio_ext = ext
                    audio_fn = candidate
                    break
            if audio_fn is None:
                time_stretch_enabled = False

        speed_factor = random.uniform(-self.args.target_time_stretch_bound, self.args.target_time_stretch_bound) + 1
        length_ok = (float(item[1]) / self.args.encodec_sr) / speed_factor < self.args.audio_max_length
        if (
            time_stretch_enabled
            and random.random() < self.args.target_time_stretch_prob
            and audio_fn is not None
            and length_ok
        ):
            # time stretch
            try:
                target_sr = getattr(self.audio_tokenizer, "encode_sample_rate", self.audio_tokenizer.sample_rate)
                process = (
                    ffmpeg.input(audio_fn, ss=0, t=float(item[1]) / self.args.encodec_sr)
                    .output('pipe:1', format='f32le', ac=1, ar=target_sr, filter='atempo={}'.format(speed_factor))
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )
                # Read the processed audio from ffmpeg stdout
                output, _ = process.communicate()

                # Convert the output to a numpy array
                output_np = np.frombuffer(output, dtype=np.float32).copy()

                # Reshape the numpy array back to the expected shape (1, samples for mono)
                waveform = torch.from_numpy(output_np).unsqueeze(0).unsqueeze(0)
                assert waveform.ndim == 3 and waveform.shape[0] == 1 and waveform.shape[1] == 1, waveform.shape
                with torch.no_grad():
                    encos = self.audio_tokenizer.encode(waveform.to(self.audio_tokenizer._device))
                assert encos.shape[1] == self.args.n_codebooks, f"encos.shape: {encos.shape}"
                encos = encos.cpu().squeeze(0).numpy().tolist() # [K, T]
                if self.args.special_first:
                    raise NotImplementedError
                    # y = [[int(n)+self.args.n_special for n in l] for l in encos]
                else:
                    y = [[int(n) for n in l] for l in encos]
                return x, y, dataset_dir, audio_ext
            except Exception as e:
                logging.info(f"failed with time stretch and codec encode for {audio_fn}")
                logging.info(f"error: {e}")
                pass

        try:
            with open(ef, "r") as e:
                encos = [l.strip().split() for k, l in enumerate(e.readlines()) if k < self.args.n_codebooks]
            assert len(encos) == self.args.n_codebooks, ef
            if self.args.special_first:
                raise NotImplementedError
            y = [[int(n) for n in l] for l in encos]
        except Exception:
            logging.info(f"loading failed for encoded audio {ef}, maybe files don't exist or are corrupted")
            return [], [[]], dataset_dir, audio_ext

        return x, y, dataset_dir, audio_ext

    # Neighbor selection for text-only pipeline (no IPA alignment)
    def find_neighbor(self, neighbors, y_len, dataset_dir, audio_ext):
        neighbor = random.choice(neighbors)
        neighbor_enc_fn = os.path.join(dataset_dir, self.args.encodec_folder_name, neighbor[0])
        if not os.path.isfile(neighbor_enc_fn):
            return None, None
        duration_factor = 1

        # Text-only path
        text_folder = getattr(self.args, "text_folder_name", "text")
        text_fn = os.path.join(dataset_dir, text_folder, neighbor[0])
        if not os.path.isfile(text_fn):
            return None, None
        with open(text_fn, "r", encoding="utf-8") as tf:
            text = tf.read().strip()
        if len(text) == 0:
            return None, None
        text_tokens = self.text_tokenizer.encode(text, add_special_tokens=False)
        if len(text_tokens) == 0:
            return None, None
        if not os.path.isfile(neighbor_enc_fn):
            return None, None

        time_stretch_prob = getattr(self.args, "time_stretch_prob", 0)
        neighbor_audio_path = None
        if audio_ext is not None:
            candidate_audio = os.path.join(
                dataset_dir,
                self.args.audio_folder_name,
                neighbor[0].replace(".txt", audio_ext),
            )
            if os.path.isfile(candidate_audio):
                neighbor_audio_path = candidate_audio

        time_stretch_flag = False
        speed_factor = 1.0
        try:
            target_sr = getattr(self.audio_tokenizer, "encode_sample_rate", self.audio_tokenizer.sample_rate)
        except AttributeError:
            target_sr = 16000
        if (
            neighbor_audio_path is not None
            and time_stretch_prob > 0
            and random.random() < time_stretch_prob
        ):
            time_stretch_flag = True
            speed_factor = (
                random.uniform(-self.args.time_stretch_bound, self.args.time_stretch_bound)
                + 1
            )
            duration_factor = 1 / speed_factor

        neighbor_enc = None
        if time_stretch_flag:
            try:
                process = (
                    ffmpeg.input(neighbor_audio_path)
                    .output(
                        'pipe:1',
                        format='f32le',
                        ac=1,
                        ar=target_sr,
                        filter='atempo={}'.format(speed_factor),
                    )
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )
                output, _ = process.communicate()
                output_np = np.frombuffer(output, dtype=np.float32).copy()
                waveform = torch.from_numpy(output_np).unsqueeze(0).unsqueeze(0)
                assert waveform.ndim == 3 and waveform.shape[0] == 1 and waveform.shape[1] == 1, waveform.shape
                with torch.no_grad():
                    encos = self.audio_tokenizer.encode(waveform.to(self.audio_tokenizer._device))
                assert encos.shape[1] == self.args.n_codebooks, f"encos.shape: {encos.shape}"
                neighbor_enc = encos.cpu().squeeze(0).numpy().tolist()
            except Exception as exc:
                logging.info(
                    f"failed to time-stretch neighbor {neighbor[0]}: {exc}"
                )
                neighbor_enc = None

        if neighbor_enc is None:
            with open(neighbor_enc_fn, "r") as f:
                neighbor_enc = [l.strip().split() for l in f.readlines()]
            if len(neighbor_enc) != self.args.n_codebooks:
                return None, None
            try:
                neighbor_enc = [[int(n) for n in l] for l in neighbor_enc]
            except ValueError as exc:
                # Corrupted neighbor encodec codes; fall back to no neighbor instead of crashing
                logging.warning(f"failed to parse encodec codes in {neighbor_enc_fn}: {exc}")
                return None, None
        if time_stretch_flag and neighbor_enc is not None:
            neighbor_dur = len(neighbor_enc[0]) / self.args.encodec_sr
            duration_factor = 1
        else:
            try:
                neighbor_dur = float(neighbor[2])
            except (IndexError, ValueError):
                neighbor_dur = len(neighbor_enc[0]) / self.args.encodec_sr
        if (
            neighbor_dur * duration_factor + y_len / self.args.encodec_sr > self.args.audio_max_length
            or neighbor_dur * duration_factor < self.args.min_prompt_len
        ):
            return None, None
        return text_tokens, neighbor_enc

    def __getitem__(self, index):
        x, y, dataset_dir, audio_ext = self._load_text_and_codes(index)
        x_len, y_len = len(x), len(y[0])
        extra_ret = {'x_sep_token_position': 0, 'y_sep_token_position': 0}
        if x_len == 0 or y_len == 0:
            ret = {
                "x": None,
                "x_len": None,
                "y": None,
                "y_len": None,
                }
            ret.update(extra_ret)
            return ret
        while y_len < self.args.encodec_sr*self.args.audio_min_length:
            assert not self.args.dynamic_batching
            index = random.choice(range(len(self))) # regenerate an index
            x, y, dataset_dir, audio_ext = self._load_text_and_codes(index)
            x_len, y_len = len(x), len(y[0])

        # if use neighbor prompt
        x_neighbor, y_neighbor = None, None
        use_neighbor_prob = random.random()
        neighbor_fn = os.path.join(dataset_dir, self.args.neighbor_folder_name, self.data[index][0]+".txt")
        if self.args.neighbor_prompt_prob > 0 and use_neighbor_prob < self.args.neighbor_prompt_prob and os.path.isfile(neighbor_fn): # it might not exist, just because we didn't find neighbor for this file (other than itself, which is common for emilia)
            with open(neighbor_fn, "r") as f:
                neighbors = [l.strip().split("\t") for l in f.readlines()]
            # select neighbors
            if "maxdist" in self.args.neighbor_selection_method:
                maxdist = int(self.args.neighbor_selection_method.split("_")[-1])
                # only keep neighbors with distance within maxdist
                neighbors = [n for n in neighbors if float(n[1]) <= maxdist]
            else:
                raise NotImplementedError
            x_neighbor, y_neighbor = None, None
            if len(neighbors) > 0:
                x_neighbor, y_neighbor = self.find_neighbor(neighbors, y_len, dataset_dir, audio_ext)
                i_trial = 0
                while x_neighbor is None and i_trial < self.args.num_trial and i_trial < len(neighbors):
                    x_neighbor, y_neighbor = self.find_neighbor(neighbors, y_len, dataset_dir, audio_ext)
                    i_trial += 1

        if x_neighbor != None:
            if self.args.x_sep_token != None:
                x = x_neighbor + [self.args.x_sep_token] + x
            else:
                x = x_neighbor + x
            if self.args.y_sep_token != None:
                y = [y_neighbor[i] + [self.args.y_sep_token] + y[i] for i in range(len(y))]
            else:
                y = [y_neighbor[i] + y[i] for i in range(len(y))]
            extra_ret['y_sep_token_position'] = len(y_neighbor[0]) + 1 # if using y_sep_token, this is actually the position of the token right before the y_sep_token, but since y_sep_token is ignored in loss computation, it's fine that we use the position of the token right before it
            extra_ret['x_sep_token_position'] = len(x_neighbor) + 1
            x_len, y_len = len(x), len(y[0])


        # consider adding eos to the end of the text
        if self.args.add_eos_to_text != 0:
            x.append(self.args.add_eos_to_text)
            x_len += 1
        if getattr(self.args, "add_bos_to_text", 0) != 0:
            x = [self.args.add_bos_to_text] + x
            x_len += 1
        ### padding and cropping ###
        # adjust the length of codes, pad to max_len or randomly crop
        orig_y_len = copy.copy(y_len)
        max_len = int(self.args.audio_max_length * self.args.encodec_sr)
        if y_len > max_len + 10: # give it some margin for rounding error
            raise RuntimeError(f"audio is too long, {y_len=}, {max_len=}")
        else:
            audio_start = 0
            if not self.args.dynamic_batching:
                pad = [0] * (max_len - y_len) if self.args.sep_special_token else [self.args.audio_pad_token] * (max_len - y_len)
                for i in range(len(y)):
                    y[i] = y[i] + pad

        if self.args.pad_x and x_len <= self.args.text_max_length:
            pad = [0] * (self.args.text_max_length - x_len) if self.args.sep_special_token else [self.args.text_pad_token] * (self.args.text_max_length - x_len)
            x = x + pad

        ret = {
            "x": torch.LongTensor(x),
            "x_len": x_len,
            "y": torch.LongTensor(y),
            "y_len": y_len,
            }
        ret.update(extra_ret)

        return ret


    def collate(self, batch):
        # make sure keys in every batch is the same
        for batch1, batch2 in zip(batch[:-1], batch[1:]):
            assert set(batch1.keys()) == set(batch2.keys()), f"keys in batch1: {batch1.keys()} and keys in batch2: {batch2.keys()} are different"
        out = {key:[] for key in batch[0]}
        for item in batch:
            if item['x'] == None: # deal with load failure
                continue
            for key, val in item.items():
                out[key].append(val)
        res = {}
        if self.args.pad_x:
            res["x"] = torch.stack(out["x"], dim=0)
        else:
            res["x"] = torch.nn.utils.rnn.pad_sequence(out["x"], batch_first=True, padding_value=self.args.text_pad_token)
        res["x_lens"] = torch.LongTensor(out["x_len"])
        if self.args.dynamic_batching:
            res['y'] = torch.nn.utils.rnn.pad_sequence([item.transpose(1,0) for item in out['y']],padding_value=self.args.audio_pad_token)
            res['y'] = res['y'].permute(1,2,0) # T B K -> B K T
        else:
            res['y'] = torch.stack(out['y'], dim=0)
        res["y_lens"] = torch.LongTensor(out["y_len"])
        res["text_padding_mask"] = torch.arange(res['x'][0].shape[-1]).unsqueeze(0) >= res['x_lens'].unsqueeze(1)
        res["audio_padding_mask"] = torch.arange(res['y'][0].shape[-1]).unsqueeze(0) >= res['y_lens'].unsqueeze(1)
        if "y_sep_token_position" in out:
            res["y_sep_token_position"] = torch.LongTensor(out["y_sep_token_position"])
        if "x_sep_token_position" in out:
            res["x_sep_token_position"] = torch.LongTensor(out["x_sep_token_position"])
        return res
    