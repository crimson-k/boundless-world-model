"""用与训练一致的窗口分别编码完整视频和历史帧；不缓存噪声或扩散 timestep。"""
import json
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train import wan_parser
from wan_video_action.parsers import merge_yaml_and_args, prepare_model_config, resolve_data_keys
from wan_video_action.data import build_train_dataset
from wan_video_action.data.latent_cache import cache_settings, metadata_digest, source_fingerprint


def main():
    parser = wan_parser()
    args = parser.parse_args()
    if args.config:
        args = merge_yaml_and_args(args.config, parser, args)
    output = args.vae_latent_cache_dir
    if not output:
        parser.error("--vae_latent_cache_dir is required")
    args.vae_latent_cache_dir = None
    args.weights = ["vae"]
    config = prepare_model_config(args)
    args = resolve_data_keys(args, stage="train")
    accelerator = Accelerator(mixed_precision="bf16")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=accelerator.device,
        model_configs=[ModelConfig(path=p) for p in config["model_paths_list"]],
        tokenizer_config=None,
    )
    pipe.vae.requires_grad_(False).eval()
    for metadata in [args.dataset_metadata_path, args.val_dataset_metadata_path]:
        if not metadata:
            continue
        dataset = build_train_dataset(args, metadata_path=metadata, repeat=1)
        folder = Path(output) / Path(metadata).stem
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {"metadata_sha256": metadata_digest(metadata), "settings": cache_settings(args), "rows": len(dataset), "source_fingerprint": source_fingerprint(metadata, args.dataset_base_path)}
        manifest_path = folder / "manifest.json"
        pending_manifest = folder / "manifest.inprogress.json"
        for existing in [manifest_path, pending_manifest]:
            if existing.exists() and json.loads(existing.read_text()) != manifest:
                raise ValueError(f"Refusing incompatible cache: {folder}")
        if accelerator.is_main_process:
            pending_manifest.write_text(json.dumps(manifest, indent=2))
        accelerator.wait_for_everyone()
        for index in tqdm(range(accelerator.process_index, len(dataset), accelerator.num_processes), disable=not accelerator.is_main_process):
            path = folder / f"{index:06d}.pt"
            if path.exists():
                continue
            sample = dataset[index]
            video = sample["video"].to(accelerator.device, dtype=torch.bfloat16)
            with torch.no_grad(), accelerator.autocast():
                full = pipe.vae.encode(video, device=accelerator.device, tiled=False)
                history = pipe.vae.encode(video[:, :, :args.num_history_frames], device=accelerator.device, tiled=False)
            payload = {"input_latents": full.to(dtype=torch.bfloat16, device="cpu"), "history_latents": history.to(dtype=torch.bfloat16, device="cpu"), "frame_indices": sample.get("frame_indices")}
            pending = path.with_suffix(f".rank{accelerator.process_index}.pending")
            torch.save(payload, pending)
            os.replace(pending, path)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            assert len(list(folder.glob("*.pt"))) == len(dataset)
            os.replace(pending_manifest, manifest_path)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
