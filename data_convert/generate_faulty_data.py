"""Generate two faulty episode variants for Stage 1 negative-data preparation.

Run: python scripts/generate_faulty_data.py
Outputs retain the expert window metadata and unchanged action files. The
metadata.jsonl alias contains training rows, as in the expert dataset.
These are whole-video corruptions, not history-preserving future edits.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import numpy as np


def write_video(source, destination, frame_count, fps, size, rng, noise_std, distortion):
    """Stream frames; uniformly resample donor clips to the action timeline."""
    with imageio.get_reader(str(source)) as reader:
        source_count = reader.count_frames()
        if tuple(reader.get_meta_data()["size"]) != size:
            raise ValueError(f"Video resolution differs from target: {source}")
        indices = np.rint(np.linspace(0, source_count - 1, frame_count)).astype(int)
        width, height = size
        x = np.arange(width)[None, :]
        y = np.arange(height)[:, None]
        source_index = -1
        with imageio.get_writer(
            str(destination), fps=fps, codec="libx264", quality=8,
            macro_block_size=1, ffmpeg_log_level="error",
        ) as writer:
            for t, index in enumerate(indices):
                while source_index < index:
                    frame = np.asarray(reader.get_next_data())
                    source_index += 1
                output = frame
                if rng is not None:
                    # Smooth horizontal wave distortion plus independent pixel noise.
                    shift = np.rint(distortion * np.sin(2 * np.pi * y / height + t * 0.15))
                    columns = np.clip(x + shift.astype(int), 0, width - 1)
                    output = frame[y, columns].astype(np.float32)
                    output += rng.normal(0, noise_std, output.shape).astype(np.float32)
                    output = np.clip(output, 0, 255).astype(np.uint8)
                writer.append_data(output)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-dir", type=Path, default=root / "converted_dataset_task1")
    parser.add_argument("--output-dir", type=Path, default=root / "converted_dataset_task1_faulty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=25.0, help="Pixel noise std on the 0–255 scale")
    parser.add_argument("--distortion", type=float, default=8.0, help="Horizontal wave amplitude in pixels")
    args = parser.parse_args()
    if not np.isfinite([args.noise_std, args.distortion]).all() or min(args.noise_std, args.distortion) < 0:
        parser.error("Noise and distortion must be finite and nonnegative")
    if args.noise_std == args.distortion == 0:
        parser.error("At least one degradation must be nonzero")

    expert = args.expert_dir.resolve()
    output = args.output_dir.resolve()
    if output == expert or expert in output.parents:
        parser.error("Output must be outside the expert dataset")
    if output.exists():
        parser.error(f"Output already exists; choose a new directory: {output}")

    manifests = {}
    episodes = {}
    groups = defaultdict(list)
    for split in ("train", "test"):
        path = expert / f"metadata_{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        manifests[split] = rows
        for row in rows:
            episode = int(row["source_episode_index"])
            expected_split = "train" if 0 <= episode < 40 else "test" if 40 <= episode < 50 else None
            if row["split"] != split or split != expected_split:
                raise ValueError(f"Episode violates the fixed 0–39/40–49 split: {row}")
            key = (row["task"], episode)
            if key not in episodes:
                episodes[key] = row
                groups[(row["task"], split)].append(key)
            if (row["action"], row["video"]) != (episodes[key]["action"], episodes[key]["video"]):
                raise ValueError(f"Inconsistent episode files: {key}")
            for field in ("action", "video"):
                relative = Path(row[field])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Expected a dataset-relative path: {relative}")
                if not (expert / relative).is_file():
                    raise FileNotFoundError(expert / relative)

    rng = np.random.default_rng(args.seed)
    donors = {}
    for group, keys in sorted(groups.items()):
        if len(keys) < 2:
            raise ValueError(f"Need at least two episodes for within-split mismatches: {group}")
        shuffled = [keys[i] for i in rng.permutation(len(keys))]
        donors.update(zip(shuffled, shuffled[1:] + shuffled[:1]))

    output.mkdir(parents=True)
    shutil.copy2(expert / "stat.json", output / "stat.json")
    provenance = []
    variants = {}
    for number, (key, row) in enumerate(sorted(episodes.items()), 1):
        source = expert / row["video"]
        with imageio.get_reader(str(source)) as reader:
            meta = reader.get_meta_data()
            frame_count = reader.count_frames()
        if frame_count != row["raw_length"]:
            raise ValueError(f"Metadata/video frame count mismatch: {source}")
        action = output / row["action"]
        action.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(expert / row["action"], action)
        donor = episodes[donors[key]]
        episode_seed = int(rng.integers(0, 2**32))
        for fault in ("mismatched_video", "degraded_video"):
            relative = Path("videos") / fault / Path(row["video"]).relative_to("videos")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            video_row = donor if fault == "mismatched_video" else row
            write_video(
                expert / video_row["video"], destination, frame_count,
                meta["fps"], tuple(meta["size"]),
                np.random.default_rng(episode_seed) if fault == "degraded_video" else None,
                args.noise_std, args.distortion,
            )
            variants[(key, fault)] = relative.as_posix()
            provenance.append(dict(
                task=key[0], source_episode_index=key[1], split=row["split"],
                fault_type=fault, video=relative.as_posix(), action=row["action"],
                source_video=video_row["video"],
                video_source_episode_index=video_row["source_episode_index"],
                frame_count=frame_count, fps=meta["fps"],
                degradation_seed=episode_seed if fault == "degraded_video" else None,
            ))
        print(f"[{number}/{len(episodes)}] {key[0]} episode {key[1]}: both variants written", flush=True)

    for split, rows in manifests.items():
        generated = []
        for row in rows:
            key = (row["task"], int(row["source_episode_index"]))
            for fault in ("mismatched_video", "degraded_video"):
                generated.append(dict(row, video=variants[(key, fault)], fault_type=fault))
        (output / f"metadata_{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in generated)
        )
    shutil.copy2(output / "metadata_train.jsonl", output / "metadata.jsonl")
    (output / "generation.json").write_text(json.dumps(dict(
        expert_dir=str(expert), seed=args.seed, noise_std=args.noise_std,
        distortion=args.distortion, temporal_resampling="uniform nearest frame",
        episodes=provenance,
    ), indent=2) + "\n")
    print(f"Wrote {2 * len(episodes)} faulty videos to {output}")


if __name__ == "__main__":
    main()
