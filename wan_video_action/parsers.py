import argparse
import json
import os
import sys
from omegaconf import OmegaConf


def merge_yaml_and_args(yaml_path, parser, args, cli_args=None):
    # priority: CLI args > YAML config > parser defaults
    if not yaml_path or not os.path.exists(yaml_path):
        return args

    if cli_args is None:
        cli_args = sys.argv[1:]

    explicit_cli_keys = set()
    for action in parser._actions:
        for option in action.option_strings:
            if option in cli_args:
                explicit_cli_keys.add(action.dest)

    yaml_dict = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True) or {}

    for section in yaml_dict.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if key not in explicit_cli_keys:
                    setattr(args, key, value)
    return args


def _expand_safetensors_index(path: str):
    if not str(path).endswith(".safetensors.index.json"):
        return path
    with open(path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    return [os.path.join(os.path.dirname(path), shard_name) for shard_name in shard_names]


def resolve_model_paths(args):
    cfg = OmegaConf.to_container(OmegaConf.load(args.model_config_path), resolve=True)
    module_files = cfg["modules"]
    paths = []
    for module in args.load_modules:
        rel_path = module_files[module]
        path = os.path.join(args.model_root_path, rel_path)
        paths.append(_expand_safetensors_index(path))
    args.model_paths_list = paths
    args.tokenizer_path = os.path.join(args.model_root_path, cfg["tokenizer_subdir"])
    return args

def resolve_data_keys(args):
    if args.stage == "infer" or args.vae_mode == "raw":
        keys = ["video"]
    else:
        keys = ["latents"]

    if args.action_mode != "none":
        keys.append("action")

    if args.text_mode == "emb":
        keys.extend(["prompt_emb", "negative_prompt_emb"])

    args.data_keys = keys
    return args


def add_dataset_base_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("dataset")
    group.add_argument("--dataset_name", type=str, default="robotwin", help="[KEY] Dataset builder name.")
    group.add_argument("--dataset_base_path", type=str, default="", help="[REQUIRED] Base path of the dataset.")
    group.add_argument("--dataset_metadata_path", type=str, default=None, help="[OPTIONAL] Path to the metadata file of the dataset.")
    group.add_argument("--dataset_repeat", type=int, default=1, help="[TUNABLE] Number of times to repeat the dataset per epoch.")
    group.add_argument("--dataset_num_workers", type=int, default=0, help="[OPTIONAL] Number of workers for data loading.")
    parser.set_defaults(data_keys=None)
    return parser


def add_video_size_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("video")
    group.add_argument("--height", type=int, default=None, help="[KEY] Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--width", type=int, default=None, help="[KEY] Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--max_pixels", type=int, default=1048576, help="[OPTIONAL] Maximum number of pixels per frame, used for dynamic resolution.")
    group.add_argument("--num_frames", type=int, default=81, help="[KEY] Number of frames per video. Frames are sampled from the video prefix.")
    group.add_argument("--resize_mode", type=str, default="fit", choices=["crop", "fit"], help="[OPTIONAL] Resize behavior: crop (center crop), fit (no crop).")
    group.add_argument("--num_history_frames", type=int, default=1, help="[KEY] Number of conditioning history frames. Must satisfy 1 <= num_history_frames < num_frames.")
    group.add_argument("--time_division_factor", type=int, default=4, help="[OPTIONAL] Temporal frame divisor used to align video/action frame counts.")
    group.add_argument("--time_division_remainder", type=int, default=1, help="[OPTIONAL] Temporal frame remainder used with time_division_factor.")
    group.add_argument("--spatial_division_factor", type=int, default=32, help="[OPTIONAL] Spatial size divisor used to align frame height and width.")
    group.add_argument("--enable_first_frame_anchor", action="store_true", default=False, help="[OPTIONAL] Include frame 0 as a fixed history anchor when num_history_frames > 1.")
    return parser


def add_model_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("model")
    group.add_argument("--model_root_path", type=str, default=None, help="[REQUIRED] Root directory of the pretrained model.")
    group.add_argument("--model_id_with_origin_paths", type=str, default=None, help="[OPTIONAL] Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    group.add_argument("--load_modules", nargs="+", default=["dit", "vae"], help="[KEY] Model modules to load.")
    group.add_argument("--extra_inputs", type=str, default=None, help="[OPTIONAL] Additional model inputs, comma-separated.")
    group.add_argument("--fp8_models", type=str, default=None, help="[OPTIONAL] Models with FP8 precision, comma-separated.")
    group.add_argument("--offload_models", type=str, default=None, help="[OPTIONAL] Models with offload, comma-separated. Only used in splited training.")
    group.add_argument("--model_config_path", type=str, default="configs/model/wan2_1_fun_1_3b_inp.yaml", help="[KEY] Path to model config YAML.")
    group.add_argument("--initialize_model_on_cpu", action="store_true", default=False, help="[OPTIONAL] Whether to initialize models on CPU.")
    group.add_argument("--stage", type=str, default="train", choices=["train", "infer"], help="[KEY] Runtime stage used to resolve dataset keys.")
    group.add_argument("--text_mode", type=str, default=None, choices=["t5", "emb", "none"], help="[KEY] Text conditioning mode.")
    group.add_argument("--vae_mode", type=str, default=None, choices=["raw", "emb"], help="[KEY] Video latent mode.")
    group.add_argument("--action_mode", type=str, default=None, choices=["noise", "adaln", "none"], help="[KEY] Action conditioning mode.")
    return parser


def add_action_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("action")
    group.add_argument("--action_type", type=str, choices=["joint_abs", "eef_abs", "joint_delta", "eef_delta"], default="eef_delta", help="[KEY] Action/state representation: joint/eef x abs/delta.")
    group.add_argument("--action_stat_path", type=str, default=None, help="[OPTIONAL] Path to robot normalization stats (stat.json).")
    group.add_argument("--action_dim", type=int, default=14, help="[OPTIONAL] Action dimension.")
    return parser


def add_training_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("training")
    group.add_argument("--learning_rate", type=float, default=1e-4, help="[TUNABLE] Learning rate.")
    group.add_argument("--num_epochs", type=int, default=1, help="[TUNABLE] Number of epochs.")
    group.add_argument("--find_unused_parameters", action="store_true", default=False, help="[OPTIONAL] Whether to find unused parameters in DDP.")
    group.add_argument("--weight_decay", type=float, default=0.01, help="[TUNABLE] Weight decay.")
    group.add_argument("--task", type=str, default="sft", help="[OPTIONAL] Task type.")
    group.add_argument("--seed", type=int, default=42, help="[OPTIONAL] Random seed for python/numpy/torch.")
    group.add_argument("--deterministic", action="store_true", default=False, help="[OPTIONAL] Enable deterministic PyTorch algorithms for strict resume debugging.")
    group.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="[OPTIONAL] Mixed precision mode.")
    group.add_argument("--max_timestep_boundary", type=float, default=1.0, help="[OPTIONAL] Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    group.add_argument("--min_timestep_boundary", type=float, default=0.0, help="[OPTIONAL] Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    group.add_argument("--max_train_steps", type=int, default=None, help="[OPTIONAL] Maximum optimizer steps. If None, train by epochs.")
    group.add_argument("--batch_size", type=int, default=1, help="[TUNABLE] Batch size per GPU.")
    return parser


def add_output_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("output")
    group.add_argument("--output_path", type=str, default="./models", help="[KEY] Output save path.")
    group.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help='[OPTIONAL] Remove prefix in ckpt. (default: "pipe.dit.")')
    group.add_argument("--save_steps", type=int, default=None, help="[OPTIONAL] Number of checkpoint saving intervals. If None, checkpoints will be saved every epoch.")
    group.add_argument("--ckpt_path", type=str, default=None, help="[OPTIONAL] Path to model checkpoint (.safetensors) used to initialize training weights (model-only resume).")
    group.add_argument("--resume_from", type=str, default=None, help="[OPTIONAL] Path to a full training state directory saved by accelerator (e.g., output_path/states/latest).")
    return parser


def add_lora_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("lora")
    group.add_argument("--lora_base_model", type=str, default=None, help="[OPTIONAL] Which model LoRA is added to.")
    group.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="[OPTIONAL] Which layers LoRA is added to.")
    group.add_argument("--lora_rank", type=int, default=32, help="[TUNABLE] Rank of LoRA.")
    group.add_argument("--lora_checkpoint", type=str, default=None, help="[OPTIONAL] Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    group.add_argument("--preset_lora_path", type=str, default=None, help="[OPTIONAL] Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.")
    group.add_argument("--preset_lora_model", type=str, default=None, help="[OPTIONAL] Which model the preset LoRA is fused to.")
    return parser


def add_gradient_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("gradient")
    group.add_argument("--use_gradient_checkpointing", action="store_true", default=False, help="[KEY] Whether to use gradient checkpointing.")
    group.add_argument("--use_gradient_checkpointing_offload", action="store_true", default=False, help="[KEY] Whether to offload gradient checkpointing to CPU memory.")
    group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="[TUNABLE] Gradient accumulation steps.")
    group.add_argument("--max_grad_norm", type=float, default=0.5, help="[OPTIONAL] Maximum gradient norm for clipping. (default: 0.5)")
    return parser


def add_tracking_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("tracking")
    group.add_argument("--use_wandb", action="store_true", default=False, help="[OPTIONAL] Enable Weights & Biases tracking.")
    group.add_argument("--use_swanlab", action="store_true", default=False, help="[OPTIONAL] Enable SwanLab tracking.")
    group.add_argument("--run_name", type=str, default=None, help="[OPTIONAL] Remote tracker run/project name. Defaults to output_path.")
    return parser


def add_infer_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("infer")
    group.add_argument("--cfg_scale", type=float, default=5.0, help="[OPTIONAL] CFG scale for generation.")
    group.add_argument("--num_inference_steps", type=int, default=50, help="[OPTIONAL] Number of inference steps.")
    group.add_argument("--negative_prompt", type=str, default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion", help="[OPTIONAL] Negative prompt for generation.")
    group.add_argument("--negative_prompt_emb", type=str, default=None, help="[OPTIONAL] Path to the pre-extracted negative prompt embedding.")
    group.add_argument("--quality", type=int, default=5, help="[OPTIONAL] Output video quality.")
    group.add_argument("--enable_chunk_infer", action="store_true", default=False, help="[OPTIONAL] Enable chunked inference with 81-frame segments.")
    group.add_argument("--fps", type=int, default=24, help="[OPTIONAL] Output video FPS.")
    group.add_argument("--enable_metrics", action="store_true", default=False, help="[OPTIONAL] Enable evaluation metrics.")
    return parser


def add_debug_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("debug")
    group.add_argument("--start_index", type=int, default=0, help="[OPTIONAL] First metadata row index to process.")
    group.add_argument("--max_samples", type=int, default=0, help="[OPTIONAL] Maximum number of metadata rows to process. 0 means all.")
    return parser


def add_config_support(parser: argparse.ArgumentParser):
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file. CLI args override YAML config.")
    return parser


def add_general_config(parser: argparse.ArgumentParser):
    parser = add_config_support(parser)
    parser = add_dataset_base_config(parser)
    parser = add_video_size_config(parser)
    parser = add_model_config(parser)
    parser = add_action_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_tracking_config(parser)
    parser = add_infer_config(parser)
    parser = add_debug_config(parser)
    return parser
