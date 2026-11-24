"""按原始 metadata 行缓存两个独立的 VAE 编码，噪声仍在训练时生成。"""
import hashlib
import json
from pathlib import Path


def metadata_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_settings(args):
    from wan_video_action.parsers import _resolve_model_paths
    weights = _resolve_model_paths(args.model_paths, args.model_config_path, ["vae"])
    weights = [p for item in weights for p in (item if isinstance(item, list) else [item])]
    return {
        "vae_weights": [file_identity(path) for path in weights],
        "format_version": 2,
        "dataset_base_path": str(Path(args.dataset_base_path).resolve()),
        "model_paths": str(Path(args.model_paths).resolve()),
        **{key: getattr(args, key) for key in (
            "height", "width", "max_pixels", "num_frames", "resize_mode",
            "num_history_frames", "history_template_sampling", "time_division_factor",
            "time_division_remainder", "spatial_division_factor",
        )},
        "vae_mode": args.modes["vae"],
        "dtype": "bfloat16",
    }


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_fingerprint(metadata_path, base_path):
    text = Path(metadata_path).read_text()
    rows = json.loads(text) if str(metadata_path).endswith(".json") else [json.loads(line) for line in text.splitlines() if line.strip()]
    videos = set()
    for row in rows:
        sources = row["video"] if isinstance(row["video"], list) else [row["video"]]
        for source in sources:
            videos.add(str(Path(base_path) / (source["data"] if isinstance(source, dict) else source)))
    identities = [file_identity(path) for path in sorted(videos)]
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
