

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
import json
import logging
import math
import os
import pickle
import random
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from data import librilight, gigaspeech, gigaspeech_waveform
from data import combined_dataset
from data.tokenizer import AudioTokenizer
from models.t5gemma import T5GemmaVoiceModel

from .optim import Eden, ScaledAdam
from .trainer_utils import (
    AverageMeter,
    DistributedDynamicBatchSampler,
    StatefulDistributedSampler,
    StatefulSampler,
    print_model_info,
)


class Trainer:

    def __init__(self, args, world_size, rank, local_rank):
        self.start_time = time.time()
        self.args = args
        if self.args.val_max_num_tokens == None:
            self.args.val_max_num_tokens = self.args.max_num_tokens
        self.world_size, self.rank, self.local_rank = world_size, rank, local_rank
        self.device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        if self.rank == 0:
            self.writer = SummaryWriter(args.exp_dir)
            self.wandb = wandb.init(
                project="t5gemma",
                name=args.exp_dir.split("/")[-1],
                config=args,
                dir=args.exp_dir,
                entity=self.args.wandb_entity,
            )
        self.seed_everything(seed=self.args.seed)
        self.meters = self._setup_meters()

        self.progress, self.total_progress = self._setup_progress()

        (
            self.model,
            self.trainables,
            self.optim_states,
            self.scheduler_states,
        ) = self._setup_models()

        (
            self.train_dataset_length,
            self.train_sampler,
            self.train_loader,
            self.valid_loader,
        ) = (
            self._setup_dataloader()
        )  # both are use DistributedSampler, train sampler is stateful
        if self.args.num_steps != None:
            self.total_step = self.args.num_steps
            self.args.num_epochs = (
                math.ceil(
                    self.total_step
                    / math.floor(self.train_dataset_length / self.args.batch_size)
                )
                if not self.args.dynamic_batching
                else None
            )
        else:
            self.total_step = (
                int(math.floor(self.train_dataset_length / self.args.batch_size))
                * self.args.num_epochs
            )

        self.optimizer, self.scheduler = self._setup_optimizer()
        # only need when using float16 (bfloat16 not needed)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.args.precision == "float16")
        )
        ddp_find_unused = getattr(self.args, "ddp_find_unused_parameters", None)
        ddp_find_unused = bool(ddp_find_unused)

        if getattr(self.args, "compile", 0):
            logging.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            find_unused_parameters=ddp_find_unused,
        )
        if self.rank == 0:
            logging.info(f"DDP find_unused_parameters={ddp_find_unused}")
        self.early_stop_accu_steps = 0
        if self.rank == 0:
            if self.args.dynamic_batching:
                logging.info(
                    f"max number of tokens per GPU in a training batch: {self.args.max_num_tokens}, max number of tokens per GPU in a inference batch: {self.args.val_max_num_tokens}"
                )
            else:
                logging.info(f"batch size (per gpu): {self.args.batch_size}")

        self.args.inference_every_n_steps = getattr(
            self.args, "inference_every_n_steps", self.args.val_every_n_steps * 5
        )
        assert (
            self.args.inference_every_n_steps > self.args.val_every_n_steps
            and self.args.inference_every_n_steps % self.args.val_every_n_steps == 0
        ), "inference_every_n_steps should be divisible by val_every_n_steps, otherwise the code will not get a chance to run inference"

        # Diagnostics defaults (used for validation-time generation stats)
        if not hasattr(self.args, "diag_top_k"):
            self.args.diag_top_k = getattr(self.args, "top_k", 0)
        if not hasattr(self.args, "diag_top_p"):
            self.args.diag_top_p = getattr(self.args, "top_p", 0.9)

        self._diag_audio_tokenizer = None
        self._diag_sample_index = 0

    def train(self):
        flag = True
        skip_flag = False
        data_start_time = time.time()
        if self.progress["step"] >= self.total_step:
            if self.rank == 0:
                self.writer.close()
                self.wandb.finish()
            return
        while flag:
            self.train_sampler.set_epoch(self.progress["epoch"])
            for i, batch in enumerate(self.train_loader):
                if len(batch["y_lens"]) < self.args.gradient_accumulation_steps:
                    continue
                data_end_time = time.time()
                self.model.train()
                if self.progress["step"] >= getattr(
                    self.args, "uniform_weight_start_step", 1e50
                ):
                    if (
                        self.progress["step"]
                        == getattr(self.args, "uniform_weight_start_step", 1e50)
                        and self.rank == 0
                    ):
                        logging.info(
                            "NOTE: start using uniform weight from step: {}".format(
                                self.progress["step"]
                            )
                        )
                    self.args.codebook_weight = [2.5, 2, 1.5, 0.6]
                    self.model.module.args.codebook_weight = [2.5, 2, 1.5, 0.6]

                if self.progress["step"] >= self.total_step:
                    dist.barrier()
                    flag = False
                    self.validate_and_save()
                    if self.rank == 0:
                        self.writer.close()
                        self.wandb.finish()
                    break
                if isinstance(self.scheduler, Eden):
                    self.scheduler.step_epoch(
                        self.progress["step"] // self.args.pseudo_epoch_size + 1
                    )
                if self.args.optimizer_name == "ScaledAdam":
                    cur_lr = self.scheduler.get_last_lr()[0]
                else:
                    lrs = [
                        param_group["lr"] for param_group in self.optimizer.param_groups
                    ]
                    assert lrs[0] == lrs[1]
                    cur_lr = lrs[0]

                if (
                    self.rank == 0
                    and self.progress["step"] % self.args.tb_write_every_n_steps == 0
                ):
                    self.writer.add_scalar("train/lr", cur_lr, self.progress["step"])
                    self.wandb.log({"train/lr": cur_lr}, step=self.progress["step"])

                all_inds = list(range(len(batch["y"])))
                sum_losses = 0
                sum_top10acc = 0
                sum_ntoken = 0
                sum_top10acc_cbi = [0 for _ in range(self.args.n_codebooks)]
                # extra losses
                sum_extra_losses = {}
                # when using prompt-based training, it's likely that due to prompt, the total length gets much longer, which make effective batch size in each accumulation step much bigger and then lead to OOM.
                # therefore we re-calculate graduent_accumulation_steps based on the effective batch size

                if self.args.neighbor_prompt_prob > 0:
                    effective_batch_size = (
                        self.args.max_num_tokens
                        // self.args.gradient_accumulation_steps
                    )
                    total_batch_size = sum(batch["y_lens"]).item()
                    cur_gradient_accumulation_steps = max(
                        self.args.gradient_accumulation_steps,
                        total_batch_size // effective_batch_size,
                    )
                    gas = torch.tensor(
                        cur_gradient_accumulation_steps,
                        dtype=torch.int,
                        device=self.local_rank,
                    )
                    dist.all_reduce(gas, op=dist.ReduceOp.MAX)
                    cur_gradient_accumulation_steps = gas.item()
                    len_batch = torch.tensor(
                        len(batch["y"]), dtype=torch.int, device=self.local_rank
                    )
                    dist.all_reduce(len_batch, op=dist.ReduceOp.MIN)
                    len_batch = len_batch.item()
                    cur_gradient_accumulation_steps = min(
                        cur_gradient_accumulation_steps, len_batch
                    )
                    # for those that cur_gradient_accumulation_steps * effective_batch_size < total_batch_size, we only use the first cur_gradient_accumulation_steps * effective_batch_size samples
                    cur_len = 0
                    final_all_inds = []
                    pointer = 0
                    while cur_len < self.args.max_num_tokens and pointer < len(
                        all_inds
                    ):
                        cur_len += batch["y_lens"][pointer]
                        final_all_inds.append(all_inds[pointer])
                        pointer += 1
                    all_inds = final_all_inds
                else:
                    cur_gradient_accumulation_steps = (
                        self.args.gradient_accumulation_steps
                    )

                sum_losses_local = 0.0
                sum_top10acc_local = 0.0
                sum_entropy_loss_local = 0.0
                sum_ctc_loss_local = 0.0
                sum_ntoken_local = 0.0
                sum_top10acc_cbi_local = [0.0 for _ in range(self.args.n_codebooks)]

                global_nan_flag = 0
                for j in range(cur_gradient_accumulation_steps):
                    cur_ind = all_inds[j::cur_gradient_accumulation_steps]
                    cur_batch = {key: batch[key][cur_ind] for key in batch}

                    # Automatic casting
                    if self.args.precision == "float16":
                        precision_used = torch.float16
                    elif self.args.precision in ["bf16", "bfloat16"]:
                        precision_used = torch.bfloat16
                    else:
                        precision_used = torch.float32

                    with torch.amp.autocast("cuda", dtype=precision_used):
                        out = self.model(cur_batch)
                    if out is None:
                        continue

                    if torch.isnan(out["loss"]).any():
                        local_nan_flag = torch.tensor(1, device=self.local_rank)
                    else:
                        local_nan_flag = torch.tensor(0, device=self.local_rank)

                    # All ranks check if *any* rank got a NaN
                    dist.all_reduce(local_nan_flag, op=dist.ReduceOp.SUM)
                    global_nan_flag = local_nan_flag.item()
                    if global_nan_flag > 0:
                        # Now *all* ranks break at the same j
                        logging.info(
                            f"rank: {self.rank}. Loss at micro-batch {j} in step {self.progress['step']} was NaN on at least one rank; skipping."
                        )
                        break

                    # Accumulate local values
                    record_loss = out["loss"].detach()
                    top10acc = out["top10acc"].detach()
                    effective_ntoken = out["effective_ntoken"].detach()

                    sum_losses_local += record_loss.item()
                    sum_top10acc_local += top10acc.item()
                    sum_ntoken_local += effective_ntoken.item()

                    # Optional losses
                    if "entropy_loss" in out:
                        sum_entropy_loss_local += out["entropy_loss"].detach().item()
                    if "ctc_loss" in out:
                        sum_ctc_loss_local += out["ctc_loss"].detach().item()

                    # Codebook accuracy
                    if "top10acc_by_codebook" in out:
                        for cb in range(self.args.n_codebooks):
                            sum_top10acc_cbi_local[cb] += (
                                out["top10acc_by_codebook"][cb].detach().item()
                            )

                    # Backprop on this micro-batch
                    if self.args.optimizer_name == "ScaledAdam":
                        self.scaler.scale(out["loss"]).backward()
                    else:
                        self.scaler.scale(
                            out["loss"] / out["effective_ntoken"]
                        ).backward()

                if global_nan_flag > 0:
                    # If *any* rank had NaN, skip this step
                    logging.info(
                        f"rank: {self.rank}. Loss at one micro-batch in step {self.progress['step']} was NaN on at least one rank; skipping."
                    )
                    self.progress["step"] += 1
                    self.progress["cur_step"] += 1
                    self.optimizer.zero_grad()
                    continue

                # Otherwise, do one big reduce for the summed metrics
                metrics_tensor = torch.tensor(
                    [
                        sum_losses_local,
                        sum_top10acc_local,
                        sum_entropy_loss_local,
                        sum_ctc_loss_local,
                        sum_ntoken_local,
                    ],
                    device=self.local_rank,
                    dtype=torch.float32,
                )

                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

                # Also reduce the codebook array in one shot if needed
                codebook_tensor = torch.tensor(
                    sum_top10acc_cbi_local, device=self.local_rank, dtype=torch.float32
                )
                dist.all_reduce(codebook_tensor, op=dist.ReduceOp.SUM)

                # Convert them back to Python scalars
                sum_losses = metrics_tensor[0].item()
                sum_top10acc = metrics_tensor[1].item()
                sum_entropy_loss = metrics_tensor[2].item()
                sum_ctc_loss = metrics_tensor[3].item()
                sum_ntoken = metrics_tensor[4].item()

                sum_top10acc_cbi = codebook_tensor.tolist()

                if self.args.optimizer_name != "ScaledAdam":
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.gradient_clip_val
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.args.optimizer_name == "ScaledAdam":
                    self.scheduler.step_batch(self.progress["step"])
                else:
                    self.scheduler.step()

                # logging
                if self.rank == 0:
                    average_loss = sum_losses / sum_ntoken
                    average_top10acc = sum_top10acc / sum_ntoken
                    average_top10acc_cbi = [
                        sum_top10acc_cbi[cb] / sum_ntoken * self.args.n_codebooks
                        for cb in range(self.args.n_codebooks)
                    ]
                    self.meters["train_loss"].update(
                        average_loss, batch["x"].shape[0] * self.world_size
                    )
                    self.meters["train_top10acc"].update(
                        average_top10acc, batch["x"].shape[0] * self.world_size
                    )
                    self.meters["train_top10acc"].update(
                        average_top10acc, batch["x"].shape[0] * self.world_size
                    )
                    for cb in range(self.args.n_codebooks):
                        self.meters[f"train_top10acc_cb{cb+1}"].update(
                            average_top10acc_cbi[cb],
                            batch["x"].shape[0] * self.world_size,
                        )
                    self.meters["data_time"].update(data_end_time - data_start_time)
                    self.meters["train_time"].update(time.time() - data_end_time)

                    # log extra losses
                    for key in sum_extra_losses:
                        if "train_" + key not in self.meters:
                            self.meters["train_" + key] = AverageMeter()
                        self.meters["train_" + key].update(
                            sum(sum_extra_losses[key]) / len(sum_extra_losses[key]),
                            batch["x"].shape[0] * self.world_size,
                        )

                    if self.progress["step"] % self.args.tb_write_every_n_steps == 0:
                        self.writer.add_scalar(
                            "train/loss", average_loss, self.progress["step"]
                        )
                        self.writer.add_scalar(
                            "train/top10acc", average_top10acc, self.progress["step"]
                        )
                        self.writer.add_scalar(
                            "train/ntokens", sum_ntoken, self.progress["step"]
                        )
                        self.wandb.log(
                            {
                                "train/loss": average_loss,
                                "train/top10acc": average_top10acc,
                                "train/ntokens": sum_ntoken,
                                "train/data_time": data_end_time - data_start_time,
                                "train/train_time": time.time() - data_end_time,
                            },
                            step=self.progress["step"],
                        )

                        for cb in range(self.args.n_codebooks):
                            self.writer.add_scalar(
                                f"train/top10acc_cb{cb+1}",
                                average_top10acc_cbi[cb],
                                self.progress["step"],
                            )
                            self.wandb.log(
                                {f"train/top10acc_cb{cb+1}": average_top10acc_cbi[cb]},
                                step=self.progress["step"],
                            )
                        self.writer.add_scalar(
                            "train/data_time",
                            data_end_time - data_start_time,
                            self.progress["step"],
                        )
                        self.writer.add_scalar(
                            "train/train_time",
                            time.time() - data_end_time,
                            self.progress["step"],
                        )
                        # write extra losses
                        for key in sum_extra_losses:
                            self.writer.add_scalar(
                                f"train/{key}",
                                sum(sum_extra_losses[key]) / len(sum_extra_losses[key]),
                                self.progress["step"],
                            )
                            self.wandb.log(
                                {
                                    f"train/{key}": sum(sum_extra_losses[key])
                                    / len(sum_extra_losses[key])
                                },
                                step=self.progress["step"],
                            )
                    # logging.info(f"ntoken: {sum_ntoken}")

                    # logging
                    if self.progress["step"] % self.args.print_every_n_steps == 0:
                        log_out = {}
                        log_out["cur_epoch"] = (
                            f"{self.progress['epoch']}/{self.args.num_epochs}"
                            if self.args.num_epochs is not None
                            else f"{self.progress['epoch']}"
                        )
                        log_out["cur_step"] = f"{int(self.progress['cur_step']+1)}"
                        log_out["total_step"] = (
                            f"{self.progress['step']}/{self.args.num_steps}"
                        )
                        log_out["lr"] = f"{cur_lr:.7f}"
                        log_out["ntokens"] = f"{sum_ntoken}"
                        for key in self.meters:
                            if self.meters[key].val != 0 or self.meters[key].avg != 0:
                                log_out[key] = (
                                    f"{self.meters[key].val:.4f} ({self.meters[key].avg:.4f})"
                                    if isinstance(self.meters[key].val, float)
                                    else f"{self.meters[key].val}"
                                )
                        logging.info(log_out)
                        if np.isnan(self.meters["train_loss"].avg):
                            logging.warning("training diverged...")
                            raise RuntimeError("training diverged...")

                # save the model only
                if self.progress["step"] % self.args.save_every_n_steps == 0:
                    dist.barrier()
                    if self.rank == 0:
                        save_path = os.path.join(
                            self.args.exp_dir, f"bundle_step{self.progress['step']}.pth"
                        )
                        self.save_progress(name=f"step{self.progress['step']}")
                        torch.save(
                            {
                                "model": self.model.module.state_dict(),
                                "args": self.args,
                                "optimizer": self.optimizer.state_dict(),
                                "scheduler": self.scheduler.state_dict(),
                            },
                            save_path,
                        )
                        logging.info(
                            f"save model, optimizer, scheduler and progress at {save_path} at global step {self.progress['step']}"
                        )
                    dist.barrier()

                # validation and save models
                if self.progress["step"] % self.args.val_every_n_steps == 0:
                    dist.barrier()
                    continue_training = self.validate_and_save()
                    # broadcast continue_training to all processes, so that all processes gets into generation stage
                    continue_training = torch.tensor(
                        int(continue_training), dtype=torch.int, device=self.local_rank
                    )
                    dist.broadcast(continue_training, src=0)
                    continue_training = bool(continue_training.item())
                    dist.barrier()  # need this to ensure all processes get to the next line?
                    logging.info(
                        f"rank: {self.rank}, continue_training: {continue_training}"
                    )
                    if not continue_training:
                        if self.rank == 0:
                            self.writer.close()
                            self.wandb.finish()
                        flag = False
                        break

                self.progress["step"] += 1
                self.progress["cur_step"] += 1

                data_start_time = time.time()
            self.progress["epoch"] += 1
            self.progress["cur_step"] = 0  # reset cur_step to be 0
        dist.destroy_process_group()

    def validate_and_save(self):
        self.model.eval()

        score = self.validate(self.valid_loader)

        if self.rank == 0:
            self._log_val_generation_stats(self.progress["step"])

        if self.args.early_stop_threshold > 0:
            if self.progress["best_score"] - score < self.args.early_stop_threshold:
                self.early_stop_accu_steps += self.args.val_every_n_steps
                if self.early_stop_accu_steps >= self.args.early_stop_step - 1:
                    logging.info(
                        f"early stop based on self.args.early_stop_threshold: {self.args.early_stop_threshold}, and self.args.early_stop_step: {self.args.early_stop_step}"
                    )
                    logging.info(
                        f"best validation score at step: {self.progress['best_step']}, and the score is {self.progress['best_score']:.4f}"
                    )
                    return False
            else:
                self.early_stop_accu_steps = 0

        if self.rank == 0:
            save_path = os.path.join(self.args.exp_dir, "bundle.pth")
            if os.path.isfile(save_path):
                os.system(f"mv {save_path} {save_path.replace('.pth', '_prev.pth')}")
            torch.save(
                {
                    "model": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "args": self.args,
                },
                save_path,
            )
            self.save_progress()
            logging.info(
                f"save models, indices, acc and other statistics at {save_path} and {self.args.exp_dir}/progress.pkl at global step {self.progress['step']}"
            )
            if score < self.progress["best_score"]:
                self.progress["best_step"] = self.progress["step"]
                self.progress["best_score"] = score
                save_path = os.path.join(self.args.exp_dir, "best_bundle.pth")
                if os.path.isfile(save_path):
                    os.system(
                        f"mv {save_path} {save_path.replace('.pth', '_prev.pth')}"
                    )
                torch.save(
                    {
                        "model": self.model.module.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "args": self.args,
                    },
                    save_path,
                )
                logging.info(
                    f"save *best* models at {save_path} at global step {self.progress['step']}"
                )

        # sync best score and best step, so that all processes early stop at the same time
        best_score_tensor = torch.tensor(
            self.progress["best_score"], device=self.local_rank
        )
        dist.broadcast(best_score_tensor, src=0)
        self.progress["best_score"] = float(best_score_tensor.item())
        best_step_tensor = torch.tensor(
            self.progress["best_step"], device=self.local_rank
        )
        dist.broadcast(best_step_tensor, src=0)
        self.progress["best_step"] = int(best_step_tensor.item())
        dist.barrier()
        return True

    def _get_diag_audio_tokenizer(self):
        if self._diag_audio_tokenizer is not None:
            return self._diag_audio_tokenizer
        backend = getattr(self.args, "audio_tokenizer", "xcodec2")
        device = torch.device("cpu")
        self._diag_audio_tokenizer = AudioTokenizer(
            backend="xcodec2",
            model_name=getattr(self.args, "xcodec2_model_name", None),
            device=device,
        )
        return self._diag_audio_tokenizer

    def _log_val_generation_stats(self, global_step: int) -> None:
        if self.valid_loader is None:
            return
        dataset = getattr(self.valid_loader, "dataset", None)
        if dataset is None:
            return
        dataset_len = len(dataset)
        if dataset_len == 0:
            return

        sample = None
        prompt_len = None

        # First try to pick a sample that has a neighbor/prompt (sep token present)
        for attempt in range(min(dataset_len, 16)):
            idx = (self._diag_sample_index + attempt) % dataset_len
            try:
                candidate = dataset[idx]
            except Exception as exc:
                logging.warning(
                    "diagnostic sample fetch failed at idx %d: %s", idx, exc
                )
                continue
            if candidate.get("x") is None or candidate.get("y") is None:
                continue
            if candidate.get("x_len") is None or candidate.get("y_len") is None:
                continue
            if candidate["x_len"] <= 0 or candidate["y_len"] <= 0:
                continue
            sep_pos_val = int(candidate.get("y_sep_token_position", 0) or 0)
            if 0 < sep_pos_val < candidate["y_len"]:
                sample = candidate
                prompt_len = sep_pos_val
                self._diag_sample_index = (idx + 1) % dataset_len
                break

        # Fallback: allow samples without neighbor/prompt (sep_pos == 0)
        if sample is None:
            for attempt in range(min(dataset_len, 16)):
                idx = (self._diag_sample_index + attempt) % dataset_len
                try:
                    candidate = dataset[idx]
                except Exception as exc:
                    logging.warning(
                        "diagnostic sample fetch failed at idx %d: %s", idx, exc
                    )
                    continue
                if candidate.get("x") is None or candidate.get("y") is None:
                    continue
                if candidate.get("x_len") is None or candidate.get("y_len") is None:
                    continue
                if candidate["x_len"] <= 0 or candidate["y_len"] <= 0:
                    continue
                sample = candidate
                prompt_len = 0  # no reference prompt
                self._diag_sample_index = (idx + 1) % dataset_len
                break

        if sample is None:
            logging.warning("diagnostic logging skipped: no usable validation sample")
            return

        module = self.model.module
        device = next(module.parameters()).device
        x_len = int(sample["x_len"])
        y_len = int(sample["y_len"])

        x = sample["x"][:x_len].unsqueeze(0).to(device)
        x_lens = torch.tensor([x_len], device=device)
        prompt_len = min(int(prompt_len), y_len)
        # Build prompt tokens: if no reference prompt, insert a single empty_token.
        if prompt_len > 0:
            prompt_only = sample["y"][:, :prompt_len]  # [1, prompt_len]
            y_tokens = prompt_only.unsqueeze(0).transpose(2, 1).contiguous().to(device)
            prompt_frames = prompt_len
        else:
            empty_tok = int(getattr(module.args, "empty_token", 0))
            y_tokens = (
                torch.full((1, module.args.n_codebooks, 1), empty_tok, device=device)
                if module.args.n_codebooks > 1
                else torch.tensor([[[empty_tok]]], device=device)
            )
            prompt_frames = y_tokens.shape[-1]

        target_token_count = max(1, y_len - prompt_len)

        codec_sr = int(getattr(module.args, "encodec_sr", 50))
        delay = 0 if module.args.n_codebooks == 1 else int(module.args.n_codebooks + 1)
        max_total = int(module.args.audio_max_length * codec_sr)
        desired_total = prompt_len + target_token_count + delay
        target_total = min(max_total, desired_total)
        tgt_y_lens = torch.tensor([target_total], device=device)

        # inference_tts sets backbone.config.use_cache to True; restore it afterward
        # or subsequent training steps will keep KV cache enabled and slow down.
        # Save the original value here and set it back after generation.
        backbone = getattr(module, "backbone", None)
        orig_use_cache = None
        if backbone is not None and hasattr(backbone, "config"):
            orig_use_cache = getattr(backbone.config, "use_cache", None)

        try:
            with torch.no_grad():
                _, gen_frames = module.inference_tts(
                    x,
                    x_lens,
                    y_tokens,
                    tgt_y_lens=tgt_y_lens,
                    top_k=self.args.diag_top_k,
                    top_p=self.args.diag_top_p,
                    min_p=getattr(self.args, "min_p", 0.0),
                    temperature=getattr(self.args, "temperature", 1.0),
                    stop_repetition=getattr(self.args, "stop_repetition", 3),
                    silence_tokens=getattr(self.args, "silence_tokens", []),
                    prompt_frames=prompt_frames,
                )
        except Exception as exc:
            logging.warning("diagnostic generation failed: %s", exc)
            return
        finally:
            if backbone is not None and hasattr(backbone, "config") and orig_use_cache is not None:
                backbone.config.use_cache = orig_use_cache

        gen_frames_cpu = gen_frames.squeeze(0).detach().cpu()
        if gen_frames_cpu.numel() == 0:
            logging.warning("diagnostic generation returned empty frames")
            return

        # Remove special/pad tokens that the codec cannot decode (e.g., empty_token == vocab size).
        audio_vocab_size = getattr(module.args, "audio_vocab_size", None)
        if isinstance(audio_vocab_size, list):
            audio_vocab_size = audio_vocab_size[0]
        if audio_vocab_size is not None:
            valid_mask = (gen_frames_cpu >= 0) & (
                gen_frames_cpu < int(audio_vocab_size)
            )
            cleaned = gen_frames_cpu[valid_mask]
            if cleaned.numel() == 0:
                logging.warning(
                    "diagnostic generation cleaned to empty after stripping OOV tokens"
                )
                return
            gen_frames_cpu = cleaned

        diag_metrics = {
            "val_diag/generated_frames": float(gen_frames_cpu.shape[-1]),
        }

        diag_audio = None
        diag_audio_np = None
        sample_rate = None
        try:
            audio_tokenizer = self._get_diag_audio_tokenizer()
            audio = audio_tokenizer.decode(gen_frames_cpu.unsqueeze(0))
            audio = audio.cpu()
            sample_rate = getattr(audio_tokenizer, "sample_rate", 44100)
            diag_audio = audio.squeeze(0)
            diag_audio_np = diag_audio.transpose(0, 1).contiguous().numpy()
            rms = torch.sqrt((audio**2).mean()).item()
            max_abs = audio.abs().max().item()
            diag_metrics["val_diag/generated_rms"] = rms
            diag_metrics["val_diag/generated_max_abs"] = max_abs
        except Exception as exc:
            logging.warning("diagnostic decode failed: %s", exc)

        for key, value in diag_metrics.items():
            self.writer.add_scalar(key, float(value), global_step)
        if diag_audio is not None and diag_audio_np is not None:
            audio_path = os.path.join(
                self.args.exp_dir,
                f"diag_audio_step{global_step}.wav",
            )
            try:
                saved = False
                try:
                    import soundfile as sf

                    sf.write(audio_path, diag_audio_np, sample_rate)
                    saved = True
                except Exception:
                    try:
                        import torchaudio

                        torchaudio.save(
                            audio_path,
                            diag_audio,
                            sample_rate,
                            format="wav",
                        )
                        saved = True
                    except Exception:
                        import wave

                        import numpy as np

                        pcm = np.clip(diag_audio_np, -1.0, 1.0)
                        pcm_i16 = (pcm * 32767.0).astype(np.int16)
                        with wave.open(audio_path, "wb") as wf:
                            num_channels = pcm_i16.shape[1] if pcm_i16.ndim > 1 else 1
                            wf.setnchannels(num_channels)
                            wf.setsampwidth(2)
                            wf.setframerate(sample_rate)
                            wf.writeframes(pcm_i16.tobytes())
                        saved = True
                if saved:
                    diag_metrics["val_diag/audio_path"] = audio_path
            except Exception as exc:
                logging.warning("saving diagnostic audio failed: %s", exc)
            try:
                import wandb

                diag_metrics["val_diag/audio"] = wandb.Audio(
                    (
                        diag_audio_np.squeeze(-1)
                        if diag_audio_np.shape[-1] == 1
                        else diag_audio_np
                    ),
                    sample_rate=sample_rate,
                    caption=f"val_diag_step{global_step}",
                )
            except Exception as exc:
                logging.warning("wandb audio log failed: %s", exc)
        self.wandb.log(diag_metrics, step=global_step)

    def validate(self, valid_loader=None, hide_progress=True):
        if valid_loader == None:
            valid_loader = self.valid_loader
        self.model.eval()

        start_val_time = time.time()
        sum_losses = 0
        sum_top10acc = 0
        sum_ntoken = 0
        sum_dur_loss = 0
        sum_dur_acc = 0
        sum_entropy_loss = 0
        sum_ctc_loss = 0

        sum_top10acc_cbi = [0 for _ in range(self.args.n_codebooks)]
        mean_perplexity_cbi = [0 for _ in range(self.args.n_codebooks)]

        with torch.no_grad():
            for i, batch in enumerate(tqdm(valid_loader, disable=hide_progress)):
                out = self.model(batch)  # no reduction is applied to loss
                sum_losses += out["loss"]
                sum_top10acc += out["top10acc"]
                sum_ntoken += out["effective_ntoken"]
                if "dur_loss" in out:
                    sum_dur_loss += out["dur_loss"]
                    sum_dur_acc += out["dur_acc"]
                if "entropy_loss" in out:
                    sum_entropy_loss += out["entropy_loss"]
                if "ctc_loss" in out:
                    sum_ctc_loss += out["ctc_loss"]
                # logging.info(f"iter {i}::: {sum_losses}, {sum_top10acc}, {sum_ntoken}")
                if "top10acc_by_codebook" in out:
                    for cb in range(self.args.n_codebooks):
                        sum_top10acc_cbi[cb] += out["top10acc_by_codebook"][cb]

                if "perplexity_by_codebook" in out:
                    for cb in range(self.args.n_codebooks):
                        mean_perplexity_cbi[cb] += out["perplexity_by_codebook"][cb]
                # if i > 10:
                #     break

        dist.all_reduce(sum_losses, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_top10acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_ntoken, op=dist.ReduceOp.SUM)
        if "dur_loss" in out:
            dist.all_reduce(sum_dur_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_dur_acc, op=dist.ReduceOp.SUM)
        if "entropy_loss" in out:
            dist.all_reduce(sum_entropy_loss, op=dist.ReduceOp.SUM)
        if "ctc_loss" in out:
            dist.all_reduce(sum_ctc_loss, op=dist.ReduceOp.SUM)

        if "top10acc_by_codebook" in out:
            for cb in range(self.args.n_codebooks):
                dist.all_reduce(sum_top10acc_cbi[cb], op=dist.ReduceOp.SUM)

        if "perplexity_by_codebook" in out:
            for cb in range(self.args.n_codebooks):
                dist.all_reduce(mean_perplexity_cbi[cb], op=dist.ReduceOp.SUM)

        val_loss = sum_losses / sum_ntoken
        val_top10acc = sum_top10acc / sum_ntoken

        if self.rank == 0:
            if "dur_loss" in out:
                val_dur_loss = sum_dur_loss / sum_ntoken
                val_dur_acc = sum_dur_acc / sum_ntoken
                self.meters["val_dur_loss"].update(val_dur_loss)
                logging.info(f"val dur_loss: {val_dur_loss:.5f}")
                self.meters["val_dur_acc"].update(val_dur_acc)
                logging.info(f"val dur_acc: {val_dur_acc:.5f}")
                self.writer.add_scalar(
                    "val/dur_loss", val_dur_loss, self.progress["step"]
                )
                self.writer.add_scalar(
                    "val/dur_acc", val_dur_acc, self.progress["step"]
                )
                self.wandb.log(
                    {"val/dur_loss": val_dur_loss, "val/dur_acc": val_dur_acc},
                    step=self.progress["step"],
                )
            # logging
            self.meters["val_loss"].update(val_loss)
            logging.info(f"val loss: {val_loss:.5f}")
            self.writer.add_scalar("val/loss", val_loss, self.progress["step"])
            self.wandb.log({"val/loss": val_loss}, step=self.progress["step"])

            self.meters["val_top10acc"].update(val_top10acc)
            logging.info(f"val top10acc: {val_top10acc:.5f}")
            self.writer.add_scalar("val/top10acc", val_top10acc, self.progress["step"])
            self.wandb.log({"val/top10acc": val_top10acc}, step=self.progress["step"])
            for cb in range(self.args.n_codebooks):
                average_top10acc_cbi = (
                    sum_top10acc_cbi[cb] / sum_ntoken * self.args.n_codebooks
                )
                self.meters[f"val_top10acc_cb{cb+1}"].update(average_top10acc_cbi)
                self.writer.add_scalar(
                    f"val/top10acc_cb{cb+1}",
                    average_top10acc_cbi,
                    self.progress["step"],
                )
                self.wandb.log(
                    {f"val/top10acc_cb{cb+1}": average_top10acc_cbi},
                    step=self.progress["step"],
                )

                temp = mean_perplexity_cbi[cb] / len(valid_loader)
                self.writer.add_scalar(
                    f"val/perplexity_cb{cb+1}", temp, self.progress["step"]
                )
                self.wandb.log(
                    {f"val/perplexity_cb{cb+1}": temp}, step=self.progress["step"]
                )

            average_perplexity = sum(mean_perplexity_cbi) / (
                self.args.n_codebooks * len(valid_loader)
            )
            self.wandb.log(
                {"val/average_perplexity": average_perplexity},
                step=self.progress["step"],
            )
            self.writer.add_scalar(
                "val/average_perplexity", average_perplexity, self.progress["step"]
            )

            # log entropy and ctc loss
            if "entropy_loss" in out:
                val_entropy_loss = sum_entropy_loss / ((i + 1) * self.world_size)
                self.meters["val_entropy_loss"].update(val_entropy_loss)
                logging.info(f"val entropy_loss: {val_entropy_loss:.5f}")
                self.writer.add_scalar(
                    "val/entropy_loss", val_entropy_loss, self.progress["step"]
                )
                self.wandb.log(
                    {"val/entropy_loss": val_entropy_loss}, step=self.progress["step"]
                )
            if "ctc_loss" in out:
                val_ctc_loss = sum_ctc_loss / ((i + 1) * self.world_size)
                self.meters["val_ctc_loss"].update(val_ctc_loss)
                logging.info(f"val ctc_loss: {val_ctc_loss:.5f}")
                self.writer.add_scalar(
                    "val/ctc_loss", val_ctc_loss, self.progress["step"]
                )
                self.wandb.log(
                    {"val/ctc_loss": val_ctc_loss}, step=self.progress["step"]
                )

            logging.info(f"validation takes: {time.time() - start_val_time:.2f}s")
            logging.info(
                f"Step [{self.progress['step']}/{self.total_step}]\t Time elapsed {(time.time() - self.start_time)/3600.:.2f}h, Val Loss: {val_loss:.4f}, Val Top10Acc: {val_top10acc:.4f}"
            )

        return val_loss.item()

    def _setup_meters(self):
        meters = {}
        meter_names = [
            "train_loss",
            "val_loss",
            "train_top10acc",
            "val_top10acc",
            "data_time",
            "train_time",
        ]
        meter_names += [
            "train_dur_loss",
            "train_dur_acc",
            "val_dur_loss",
            "val_dur_acc",
        ]
        meter_names += ["val_perplexity"]
        meter_names += [
            f"train_top10acc_cb{cb+1}" for cb in range(self.args.n_codebooks)
        ]
        meter_names += [f"val_top10acc_cb{cb+1}" for cb in range(self.args.n_codebooks)]
        meter_names += [
            f"val_perplexity_cb{cb+1}" for cb in range(self.args.n_codebooks)
        ]
        for name in meter_names:
            meters[name] = AverageMeter()
        return meters

    def _setup_progress(self):
        """
        Need to customize it
        """
        progress = {}
        progress["best_step"] = 1
        progress["best_score"] = np.inf  # this records loss value
        progress["step"] = 1
        progress["epoch"] = 1
        progress["cur_step"] = 0  # step in the current epoch, for resuming the sampler
        total_progress = []
        # if self.args.resume or self.args.validate:
        if self.args.resume:
            progress_pkl = "%s/progress.pkl" % self.args.exp_dir
            with open(progress_pkl, "rb") as f:
                total_progress = pickle.load(f)
                (
                    progress["best_step"],
                    progress["best_score"],
                    progress["step"],
                    progress["epoch"],
                    progress["cur_step"],
                    _,
                ) = total_progress[-1]
            if self.rank == 0:
                logging.info("\nResume training from:")
                logging.info("  epoch = %s" % progress["epoch"])
                logging.info("  cur_step = %s" % progress["cur_step"])
                logging.info("  step = %s" % progress["step"])
                logging.info("  best_step = %s" % progress["best_step"])
                logging.info("  best_score = %s" % progress["best_score"])
        return progress, total_progress

    def save_progress(self, name=None):
        self.total_progress.append(
            [
                self.progress["best_step"],
                self.progress["best_score"],
                int(self.progress["step"] + 1),
                self.progress["epoch"],
                int(self.progress["cur_step"] + 1),
                time.time() - self.start_time,
            ]
        )
        if name is not None:
            progress_fn = f"{self.args.exp_dir}/progress_{name}.pkl"
        else:
            progress_fn = f"{self.args.exp_dir}/progress.pkl"
        with open(progress_fn, "wb") as f:
            pickle.dump(self.total_progress, f)

    def _setup_dataloader(self):
        train_dataset, val_dataset = combined_dataset.dataset(
            self.args, "train"
        ), combined_dataset.dataset(
            self.args, "valid"
        )  # need to change 'train' to 'valid' in actual training

        if self.args.dynamic_batching:
            train_sampler = DistributedDynamicBatchSampler(
                train_dataset,
                self.args,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.args.seed,
                drop_last=True,
                lengths_list=train_dataset.lengths_list,
                verbose=True,
                epoch=0,
            )
            valid_sampler = DistributedDynamicBatchSampler(
                val_dataset,
                self.args,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.args.seed,
                drop_last=True,
                lengths_list=val_dataset.lengths_list,
                verbose=True,
                epoch=0,
            )
        else:
            train_sampler = StatefulDistributedSampler(
                train_dataset,
                self.args.batch_size // self.world_size,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.args.seed,
                drop_last=True,
            )
            valid_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                seed=self.args.seed,
                drop_last=False,
            )

        if self.progress["step"] > 1:
            train_sampler.set_epoch_resume(
                self.progress["epoch"], self.progress["cur_step"]
            )

        if self.args.dynamic_batching:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=self.args.num_workers,
                collate_fn=train_dataset.collate,
                persistent_workers=True,
            )
            valid_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_sampler=valid_sampler,
                num_workers=self.args.num_workers,
                collate_fn=val_dataset.collate,
                persistent_workers=True,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                sampler=train_sampler,
                num_workers=self.args.num_workers,
                collate_fn=train_dataset.collate,
                persistent_workers=True,
            )
            valid_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                sampler=valid_sampler,
                num_workers=self.args.num_workers,
                collate_fn=val_dataset.collate,
                persistent_workers=True,
            )
        return len(train_dataset), train_sampler, train_loader, valid_loader

    def _setup_models(self):
        model = T5GemmaVoiceModel(self.args)

        if self.rank == 0:
            logging.info(model)
            logging.info("model parameters")
            print_model_info(model)

        optim_states = None
        scheduler_states = None
        if self.progress["step"] > 1:
            bundle = torch.load(
                os.path.join(self.args.exp_dir, "bundle.pth"), map_location="cpu"
            )
            strict_load = not getattr(self.args, "use_lora", 0)
            model.load_state_dict(bundle["model"], strict=strict_load)
            optim_states = bundle["optimizer"]
            scheduler_states = bundle["scheduler"]
            if self.rank == 0:
                logging.info(
                    "loaded parameters and data indices from epoch %d, global step %d"
                    % (self.progress["epoch"], self.progress["step"])
                )
            del bundle["model"]

        if self.args.load_model_from != None and self.progress["step"] <= 1:
            logging.info(f"load weights from {self.args.load_model_from}")
            sd = torch.load(self.args.load_model_from, map_location="cpu", weights_only=False)
            if hasattr(model, "carefully_load_state_dict"):
                model.carefully_load_state_dict(sd["model"])
            else:
                strict_load = not getattr(self.args, "use_lora", 0)
                model.load_state_dict(sd["model"], strict=strict_load)
            del sd

        #### below operations is for getting params for optimizer, which is at wrapper level ###
        if self.args.optimizer_name == "ScaledAdam":
            trainables = [p for p in model.parameters() if p.requires_grad]
        else:
            no_decay = [
                ".bias",
                ".audio_embeddings.weight",
                ".text_embeddings.weight",
                ".norm.weight",
                ".norm1.weight",
                ".norm2.weight",
                "lora_",
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
            if len(optimizer_grouped_parameters[1]["params"]) == 0:
                logging.info(
                    "there is no embedding weights, bias, and layernorm parameters in the model, which should be True, check model parameter names"
                )
                trainables = optimizer_grouped_parameters[0]
            else:
                trainables = optimizer_grouped_parameters
        #### below operations is for getting params for optimizer, which is at wrapper level ###
        model.to(self.device)

        return model, trainables, optim_states, scheduler_states

    def _setup_optimizer(self):
        if self.args.optimizer_name == "ScaledAdam":
            parameters_names = []
            _model = (
                self.model.module
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                else self.model
            )
            parameters_names.append(
                [n for n, p in self.model.named_parameters() if p.requires_grad]
            )
            optimizer = ScaledAdam(
                self.trainables,
                lr=self.args.lr,
                betas=(0.9, 0.95),
                clipping_scale=2.0,
                parameters_names=parameters_names,
                show_dominant_parameters=False,
                clipping_update_period=self.args.clipping_update_period,
            )
            scheduler = Eden(
                optimizer,
                self.args.reduce_lr_start_step,
                self.args.reduce_lr_start_epoch,
                warmup_batches=self.total_step * self.args.warmup_fraction,
            )  # NOTE: if using ScaledAdam, we will use the Eden scheduler!

        else:
            optimizer = AdamW(self.trainables, lr=self.args.lr)
            warmup_steps = self.total_step * self.args.warmup_fraction

            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0,
                    float(self.total_step - current_step)
                    / float(max(1, self.total_step - warmup_steps)),
                )

            scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

        # if resume
        if self.progress["step"] > 1:
            optimizer.load_state_dict(self.optim_states)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            del self.optim_states

            scheduler.load_state_dict(self.scheduler_states)

        optimizer.zero_grad()
        return optimizer, scheduler

    def seed_everything(self, seed=1):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.deterministic = True