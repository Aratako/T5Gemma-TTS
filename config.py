

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

import argparse
import logging


def int_or_str(value):
    """Custom function to allow both int and str types."""
    try:
        return int(value)  # Try converting to integer
    except ValueError:
        return value  # If conversion fails, return as string

def MyParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general training
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--multinodes", type=int, default=0)
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--dist_backend", default="nccl", type=str)
    parser.add_argument("--precision", type=str, default="float16", help="we might need float32 for NAR model")
    parser.add_argument("--num_workers", type=int, default=8, help="per gpu")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--tb_write_every_n_steps", type=int, default=100)
    parser.add_argument("--print_every_n_steps", type=int, default=250)
    parser.add_argument("--val_every_n_steps", type=int, default=500)
    parser.add_argument("--inference_every_n_steps", type=int, default=3000, help="will only get to inference when model is saved, and therefore this needs to be multiple of val_every_n_steps")
    parser.add_argument("--save_every_n_steps", type=int, default=10000000, help="save the model every n steps, will save the model as bundle_step$step.pth")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=100, help="this is the effective batch size per gpu, no matter whether using gradient_accumulation_steps")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_fraction", type=float, default=0.1, help="use linear warmup, the proportion of the training steps that are used for warming up")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=None, help="if not None, will ignore n_epochs and use num_steps as the total number of amount of training, can try e.g. 400000 i.e. 400k steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="the value for torch.nn.utils.clip_grad_norm_()")
    parser.add_argument("--early_stop_step", type=int, default=3200, help="stop training after this many steps of non-improvement")
    parser.add_argument("--early_stop_threshold", type=float, default=-1.0, help="early stop after the improvement is below this threshold for certain number of steps")
    parser.add_argument("--ddp_find_unused_parameters", type=int, default=None, help="if set, overrides DistributedDataParallel(find_unused_parameters)")


    # path
    parser.add_argument("--exp_dir", type=str, default='/home/ubuntu/t5gemma-tts/working', help="will be combined with dataset name")
    parser.add_argument("--dataset", type=str, help="e.g. 'libritts', 'librilight', 'spotify', they are folder name in the data dir also")
    parser.add_argument("--dataset_dir", type=str, help="need to be compatible with corresponding dataset py file")
    parser.add_argument("--local_wandb", type=int, default=0, help="if 1, will use local wandb, otherwise use the global one")
    parser.add_argument("--wandb_entity", type=str, default="your-wandb-entity", help="the entity (usually your username) for wandb")
    parser.add_argument("--model_arch", type=str, default="t5gemma", choices=["t5gemma"], help="select architecture: T5Gemma-based model (VoiceStar removed)")
    parser.add_argument("--t5gemma_model_name", type=str, default="google/t5gemma-b-b-ul2", help="Hugging Face repo id or local path for the T5Gemma checkpoint")
    parser.add_argument("--t5_gradient_checkpointing", type=int, default=0, help="Set to 1 to enable gradient checkpointing for T5Gemma (reduces training memory). During inference use_cache is auto-enabled.")
    parser.add_argument("--freeze_t5gemma", type=int, default=0, help="if 1, freeze Transformer weights and only train new heads")
    parser.add_argument("--compile", type=int, default=0, help="if 1, use torch.compile for speedup")
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"], help="attention implementation to use")
    parser.add_argument(
        "--prune_text_modules",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0: keep all. 1: drop lm_head. 2: drop lm_head + decoder.embed_tokens (text decoding disabled).",
    )
    parser.add_argument("--audio_tokenizer", type=str, default="xcodec2", choices=["xcodec2"], help="audio tokenizer backend (xcodec2 only)")
    parser.add_argument("--xcodec2_model_name", type=str, default="NandemoGHS/Anime-XCodec2-44.1kHz-v2", help="Hugging Face repo id or local path for XCodec2 weights")
    parser.add_argument("--text_input_type", type=str, default="text", choices=["text"], help="text inputs use HF tokenizer ids (phoneme path removed)")
    # LoRA / PEFT
    parser.add_argument("--use_lora", type=int, default=0, help="if 1, wrap the T5Gemma backbone with LoRA adapters (PEFT)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names to apply LoRA to",
    )
    parser.add_argument(
        "--text_guard_frames_per_token",
        type=int,
        default=0,
        help="if >0, applies a per-token frame budget when running text-mode inference to avoid premature EOS; 0 disables the guard",
    )
    parser.add_argument("--text_tokenizer_name", type=str, default=None, help="Hugging Face tokenizer id to load when text_input_type=text")
    parser.add_argument("--text_folder_name", type=str, default="text", help="folder storing raw text transcripts when using text_input_type=text")
    parser.add_argument("--pseudo_epoch_size", type=int, default=37901, help="only use for Eden scheduler. 37901 is the epoch size in the default optim setting, this is probably too big")
    # data
    parser.add_argument("--encodec_folder_name", type=str, default="xcodec2_1cb", help="folder where codec codes are stored (xcodec2)")
    parser.add_argument("--manifest_name", type=str, default="manifest_final", help="manifest folder name")
    parser.add_argument("--pad_x", type=int, default=1, help="whether or not always pad x to have text_max_length. select 1 to get the maximal memory consumption, but the actual case should be smaller, better to have it being 0")
    parser.add_argument("--max_num_tokens", type=int, default=18750, help="max number of audio tokens per gpu, used when dynamic batching is enabled; ignores batch size")
    parser.add_argument("--val_max_num_tokens", type=int, default=6000, help="validation-only cap on audio tokens per gpu when using dynamic batching (helps memory for music-gen style data)")
    parser.add_argument("--num_buckets", type=int, default=10)
    parser.add_argument("--dynamic_batching", type=int, default=1)
    parser.add_argument("--audio_max_length", type=float, default=120, help="in second, crop the audio is length is longer than this")
    parser.add_argument("--audio_min_length", type=float, default=2, help="in second, drop the audio if length is shorter than this")
    parser.add_argument("--text_max_length", type=int, default=1000, help='if too long, we crop')
    parser.add_argument("--encodec_sr", type=float, default=50, help="codec frame rate (tokens per second); used for length calculations")

    # model
    parser.add_argument("--drop_long", type=int, default=1, help="if this is true, will drop example whose audio token sequence or phone sequence is too long, rather than cropping as we did before, to avoid hallucination")
    parser.add_argument("--eos", type=int, default=2051, help="this is to be used with reduced_eog, where we end the utterance with eos, and end the generated segment with eog, also when this is used, the n_special should be 4")

    # put special tokens first to handle different vocab_size
    parser.add_argument("--special_first", type=int, default=0, help="if 1, need to have special tokens to be the first few tokens, e.g. 0, 1, 2, which means we need to adjust the preprocessing and postprocessing of the codec codes. note that we hard coded to have 3 special tokens")
    parser.add_argument("--n_special", type=int, default=4, help="empty, eog, pad, eos")

    # weight codebook differently
    parser.add_argument("--codebook_weight", type=str, default=None, help="e.g. ['5','1','0.5','0.1']")

    parser.add_argument("--empty_token", default=2048, type=int, help="indicating the no token at the position for the codebook")
    # args for the optimizer and scheduler from Feiteng
    # original setup for the 3 params are 5000 4 and 1000
    # but that's because set_epoch is run on num_gradient_accumulation_step*step (with 4 being the accumulation step)
    # so I scaled down them a little bit
    # will try scaling them back if this doesn't work
    parser.add_argument("--optimizer_name", type=str, default="AdamW", help="can also use ScaledAdam, in which case we'll also use the Eden scheduler")
    parser.add_argument("--reduce_lr_start_step", type=int, default=3000, help='after which significantly reduce the lr. a param for the eden optimizer')
    parser.add_argument("--reduce_lr_start_epoch", type=int, default=4)
    parser.add_argument("--clipping_update_period", type=int, default=600)
    # add parallel_pattern
    parser.add_argument("--parallel_pattern", type=int, default=0, help="if 1, use parallel pattern, we also use LFSC codec")

    parser.add_argument('--sep_special_token', type=int, default=0, help="remove text/audio pad token, set audio_mask_token and start of continue to be separately learned embeddings. Therefore, for ve1 self.n_text_tokens == self.args.text_vocab_size, self.n_audio_tokens == self.args.audio_vocab_size + 2, for ve7, self.n_text_tokens == self.args.text_vocab_size, self.n_audio_tokens == self.args.audio_vocab_size")
    # XCodec2 always uses a single codebook; keep the arg for compatibility but fix it to 1.
    parser.add_argument('--n_codebooks', type=int, default=1, help='Number of audio codebooks (fixed to 1 for xcodec2)')
    parser.add_argument('--text_vocab_size', type=int, default=86, help='Size of text vocabulary')
    parser.add_argument('--text_pad_token', type=int, default=86, help='padding of the text tokens, not attended')
    # parser.add_argument('--audio_vocab_size', type=int, default=1024, help='Size of audio vocabulary')
    parser.add_argument('--audio_vocab_size', type=str, default='2048', help="Size of audio vocabulary, can be specified as '[128,512,1024,2048]'")
    parser.add_argument('--audio_mask_token', type=int, default=1024, help='Audio mask token, this the the extra mask used in the masked region for AR, for NAR, the entire masked region will be filled with it')
    parser.add_argument('--eog', type=int, default=2049, help='End of generation token')
    parser.add_argument('--audio_pad_token', type=int, default=2050, help='padding token id for audio codes (not attended)')
    parser.add_argument('--audio_embedding_dim', type=int, default=128, help='dimension for audio continuous embedding (before being quantized)')
    parser.add_argument('--text_embedding_dropout', type=float, default=0.1, help='Dropout for text embedding')
    parser.add_argument('--audio_embedding_dropout', type=float, default=0, help='Dropout for audio embedding')
    parser.add_argument('--eog_weight', type=float, default=1.0, help='Weight for End of generation token')
    parser.add_argument('--load_model_from', type=str, default=None, help='Path to load model from, this will be effective last, so will overwrite all previous load, including resume')


    ## below are args for the new long model
    parser.add_argument("--target_time_stretch_prob", type=float, default=0, help="the probability of time stretching the target audio")
    parser.add_argument("--target_time_stretch_bound", type=float, default=0.1, help="the bound of the time stretching target audio, e.g. 0.1 means the audio will be stretched by 0.9 to 1.1")
    parser.add_argument("--time_stretch_prob", type=float, default=0, help="the probability of time stretching the audio")
    parser.add_argument("--time_stretch_bound", type=float, default=0.3, help="the bound of the time stretching, e.g. 0.3 means the audio will be stretched by 0.7 to 1.3")
    parser.add_argument("--no_loss_on_prefix", type=int, default=0, help="if 1, will not calculate loss on the prefix acoustic tokens")
    parser.add_argument("--x_sep_token", type=int, default=None, help="if not None, will use this token in between prompt text and target generation text")
    parser.add_argument("--y_sep_token", type=int, default=None, help="if not None, will use this token in between prompt codec tokens and target codec tokens")
    parser.add_argument("--neighbor_prompt_prob", type=float, default=0, help="the probability of using the prompt from the neighbor")
    parser.add_argument("--neighbor_folder_name", type=str, default='neighbors',help="folder where the neighbors of the current audio files are stored, each row contains three tab separated entries: neighbor_fn, neighbor_temporal_distance, neighbor_duration")
    parser.add_argument("--min_prompt_len", type=float, default=0.5, help="in sec., minimal prompt length selected from some neighboring file")
    parser.add_argument("--neighbor_selection_method", type=str, default="maxdist_60", help="maxdist_60 means uniformly select a neighbor that's within 60 sec of the current audio file")
    parser.add_argument("--num_trial", type=int, default=5, help="number of tries to select a neighbor")
    parser.add_argument("--audio_folder_name", type=str, default='audio', help="folder where the audio files are stored")

    # rope parameters
    parser.add_argument("--add_eos_to_text", type=int, default=0, help="if not 0, use this number as eos and add to the end of text token, usually use the second to last token in the vocab size")
    parser.add_argument("--add_bos_to_text", type=int, default=0, help="if not 0, use this number as bos and add to the begining of text token, usually use the third to last token in the vocab size")
    parser.add_argument("--progress_scale", type=float, default=1.0, help="scale the progress, the smaller the value, the bigger the diagonal in attention score, see models/rope_playground.ipynb")
    parser.add_argument("--use_pm_rope", type=int, default=1, help="if 1, use PM-RoPE for cross-attention in T5Gemma")


    # inference parameters
    parser.add_argument("--codec_audio_sr", type=int, default=16000)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--silence_tokens", type=list, default=[])
    parser.add_argument("--stop_repetition", type=int, default=3)
    parser.add_argument("--extra_cutoff", type=float, default=5, help="in rare cases where the model doesn't follow specified target duration (only happened in extrapolation cases), we will terminate generation once the extra duration exceeds this value")

    # depth transformer parameters
    parser.add_argument("--validation_sample_cap", type=int, default=None, help="cap the validation data to this number")
    parser.add_argument("--uniform_weight_start_step", type=int, default=1e50, help="set all codebook weight to be uniform starting from this step")

    return parser


def apply_repo_defaults(args):
    """Apply normalization helpers and backend-specific default values to args."""
    if hasattr(args, "audio_vocab_size"):
        if isinstance(args.audio_vocab_size, str):
            args.audio_vocab_size = eval(args.audio_vocab_size)
        if isinstance(args.audio_vocab_size, list):
            if getattr(args, "audio_tokenizer", "xcodec2") == "xcodec2":
                if len(args.audio_vocab_size) != 1:
                    raise ValueError("audio_vocab_size must contain exactly one value when using xcodec2.")
                args.audio_vocab_size = args.audio_vocab_size[0]
    if getattr(args, "audio_tokenizer", "xcodec2") == "xcodec2":
        args.n_codebooks = 1
        if not hasattr(args, "audio_vocab_size") or isinstance(args.audio_vocab_size, list):
            raise ValueError("audio_vocab_size must be an int when using xcodec2.")
        args.empty_token = args.audio_vocab_size
        args.eog = args.audio_vocab_size + 1
        args.audio_pad_token = args.audio_vocab_size + 2
        args.eos = args.audio_vocab_size + 3
        args.y_sep_token = args.audio_vocab_size + 4
        args.codec_audio_sr = 44100
        if getattr(args, "encodec_sr", None) is None:
            args.encodec_sr = 50
        logging.info("Applied XCodec2 defaults: n_codebooks=1, codec_audio_sr=44100, tokens updated.")
    if getattr(args, "text_input_type", "text") == "text":
        if getattr(args, "text_tokenizer_name", None) is None:
            args.text_tokenizer_name = getattr(args, "t5gemma_model_name", "google/t5gemma-b-b-ul2")
        if getattr(args, "pad_x", None) is None:
            args.pad_x = 0
    if getattr(args, "model_arch", "t5gemma") == "t5gemma" and getattr(args, "ddp_find_unused_parameters", None) is None:
        args.ddp_find_unused_parameters = 1
    return args