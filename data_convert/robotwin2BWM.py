#!/usr/bin/env python3
"""Convert raw RoboTwin episodes into the flat BWM raw-video training format.

The output contains:

  metadata.jsonl                         BWM sample manifest
  stat.json                              14-D EEF normalization statistics
  data/<task>/<subset>/episode_*.parquet Parquets with observation.state
  videos/<task>/<subset>/episode_*.mp4   symlinks to source videos

Use the output with `model.modes.vae: raw` and `action_type: eef_abs`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_SOURCE = Path("/data1/common_data/RoboTwin2.0_640_480/dataset")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "converted_dataset_bwm"
FORMAT_VERSION = b"1"
STATE_DIM = 14


def episode_index(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("episode"))
    except ValueError as exc:
        raise ValueError(f"Cannot parse episode index from {path}") from exc


def quaternion_wxyz_to_euler_xyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized or unnormalized [w, x, y, z] quaternions to XYZ RPY."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    w, x, y, z = quaternion.T
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if np.any(norm == 0):
        raise ValueError("Encountered a zero-norm end-effector quaternion.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.where(
        np.abs(sin_pitch) >= 1.0,
        np.sign(sin_pitch) * (math.pi / 2.0),
        np.arcsin(sin_pitch),
    )
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return np.stack([roll, pitch, yaw], axis=-1)


def read_eef_state(hdf5_path: Path) -> np.ndarray:
    """Return [left xyz/rpy/gripper, right xyz/rpy/gripper] with shape (T, 14)."""
    arms = []
    with h5py.File(hdf5_path, "r") as handle:
        expected_length = int(handle["joint_action/vector"].shape[0])
        for arm in ("left", "right"):
            endpose = np.asarray(
                handle[f"endpose/{arm}_endpose"],
                dtype=np.float64,
            )
            gripper = np.asarray(
                handle[f"endpose/{arm}_gripper"],
                dtype=np.float64,
            ).reshape(-1, 1)
            if endpose.shape != (expected_length, 7):
                raise ValueError(
                    f"{hdf5_path}: expected endpose/{arm}_endpose shape "
                    f"({expected_length}, 7), got {endpose.shape}"
                )
            if gripper.shape != (expected_length, 1):
                raise ValueError(
                    f"{hdf5_path}: expected endpose/{arm}_gripper shape "
                    f"({expected_length}, 1), got {gripper.shape}"
                )
            euler = quaternion_wxyz_to_euler_xyz(endpose[:, 3:7])
            arms.append(np.concatenate([endpose[:, :3], euler, gripper], axis=1))

    state = np.concatenate(arms, axis=1).astype(np.float32)
    if state.shape != (expected_length, STATE_DIM) or not np.isfinite(state).all():
        raise ValueError(f"{hdf5_path}: invalid converted EEF state shape/content")
    return state


def read_prompt(instruction_path: Path) -> str:
    with instruction_path.open("r", encoding="utf-8") as handle:
        instructions = json.load(handle)
    seen = instructions.get("seen")
    if not isinstance(seen, list) or not seen or not isinstance(seen[0], str):
        raise ValueError(f"{instruction_path}: expected a non-empty string list at 'seen'")
    return seen[0]


def collect_episodes(
    source: Path,
    subset: str,
    selected_tasks: set[str] | None,
) -> list[tuple[str, int, Path, Path, Path]]:
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source}")

    episodes = []
    for task_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        if selected_tasks is not None and task_dir.name not in selected_tasks:
            continue
        run_dir = task_dir / subset
        data_dir = run_dir / "data"
        if not data_dir.is_dir():
            continue
        hdf5_paths = sorted(data_dir.glob("episode*.hdf5"), key=episode_index)
        for hdf5_path in hdf5_paths:
            index = episode_index(hdf5_path)
            video_path = run_dir / "video" / f"episode{index}.mp4"
            instruction_path = run_dir / "instructions" / f"episode{index}.json"
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing video paired with {hdf5_path}: {video_path}")
            if not instruction_path.is_file():
                raise FileNotFoundError(
                    f"Missing instruction paired with {hdf5_path}: {instruction_path}"
                )
            episodes.append(
                (task_dir.name, index, hdf5_path, video_path, instruction_path)
            )

    if selected_tasks is not None:
        found_tasks = {episode[0] for episode in episodes}
        missing_tasks = sorted(selected_tasks - found_tasks)
        if missing_tasks:
            raise ValueError(f"Requested tasks not found: {missing_tasks}")
    if not episodes:
        raise RuntimeError(f"No paired RoboTwin episodes found under {source}")
    return episodes


def output_paths(
    output: Path,
    task: str,
    subset: str,
    source_episode_index: int,
) -> tuple[Path, Path]:
    filename = f"episode_{source_episode_index:06d}"
    parquet_path = output / "data" / task / subset / f"{filename}.parquet"
    video_path = output / "videos" / task / subset / f"{filename}.mp4"
    return parquet_path, video_path


def write_parquet(
    path: Path,
    state: np.ndarray,
    global_episode_index: int,
    source_hdf5: Path,
    overwrite: bool,
) -> None:
    source_value = str(source_hdf5.resolve()).encode()
    if path.exists() and not overwrite:
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        reusable = (
            parquet.metadata.num_rows == len(state)
            and "observation.state" in parquet.schema_arrow.names
            and metadata.get(b"bwm_format_version") == FORMAT_VERSION
            and metadata.get(b"source_hdf5") == source_value
        )
        if reusable:
            return
        raise FileExistsError(
            f"Existing Parquet is not a matching resumable output: {path}. "
            "Use --overwrite to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    state_type = pa.list_(pa.float32(), STATE_DIM)
    table = pa.table(
        {
            "observation.state": pa.array(state.tolist(), type=state_type),
            "frame_index": np.arange(len(state), dtype=np.int64),
            "episode_index": np.full(
                len(state), global_episode_index, dtype=np.int64
            ),
        }
    )
    table = table.replace_schema_metadata(
        {
            b"bwm_format_version": FORMAT_VERSION,
            b"source_hdf5": source_value,
        }
    )
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        pq.write_table(table, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_video_symlink(source: Path, destination: Path, overwrite: bool) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        if not overwrite:
            raise FileExistsError(
                f"Symlink points to a different source: {destination}. "
                "Use --overwrite to replace it."
            )
        destination.unlink()
    elif destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Video destination already exists and is not a symlink: {destination}"
            )
        if not destination.is_file():
            raise FileExistsError(f"Refusing to replace non-file path: {destination}")
        destination.unlink()
    os.symlink(source, destination)


def action_stats(states: list[np.ndarray]) -> dict:
    state = np.concatenate(states, axis=0).astype(np.float64)
    stats = {
        "shape": [STATE_DIM],
        "min": np.min(state, axis=0).tolist(),
        "max": np.max(state, axis=0).tolist(),
        "p01": np.percentile(state, 1, axis=0).tolist(),
        "p99": np.percentile(state, 99, axis=0).tolist(),
        "mean": np.mean(state, axis=0).tolist(),
        "std": np.std(state, axis=0).tolist(),
    }
    # LoadCobotAction resolves eef_abs to its internal state_pose name first.
    return {"state_pose": stats}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def convert(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    selected_tasks = None if args.task is None else set(args.task)
    episodes = collect_episodes(source, args.subset, selected_tasks)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    output.mkdir(parents=True, exist_ok=True)

    records = []
    states = []
    for global_index, (
        task,
        source_index,
        hdf5_path,
        source_video,
        instruction_path,
    ) in enumerate(tqdm(episodes, desc="Converting episodes", unit="episode")):
        state = read_eef_state(hdf5_path)
        prompt = read_prompt(instruction_path)
        parquet_path, video_path = output_paths(
            output, task, args.subset, source_index
        )
        write_parquet(
            parquet_path,
            state,
            global_index,
            hdf5_path,
            overwrite=args.overwrite,
        )
        ensure_video_symlink(source_video, video_path, overwrite=args.overwrite)

        length = len(state)
        records.append(
            {
                "episode_index": global_index,
                "source_episode_index": source_index,
                "task": task,
                "prompt": prompt,
                "video": video_path.relative_to(output).as_posix(),
                "action": parquet_path.relative_to(output).as_posix(),
                "start_frame": 0,
                "end_frame": length - 1,
                "length": length,
                "raw_length": length,
            }
        )
        states.append(state)

    manifest_text = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    atomic_write_text(output / "metadata.jsonl", manifest_text)
    atomic_write_text(
        output / "stat.json",
        json.dumps(action_stats(states), indent=2) + "\n",
    )
    print(f"Converted {len(records)} episodes to {output}")
    print(f"  metadata: {output / 'metadata.jsonl'}")
    print(f"  stats:    {output / 'stat.json'}")
    print("  training: set model.modes.vae=raw and action_type=eef_abs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RoboTwin 2.0 data to the BWM raw-video input format."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subset", default="aloha-agilex_clean_50")
    parser.add_argument(
        "--task",
        action="append",
        help="Convert only this task; repeat the option for multiple tasks.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Limit the globally sorted episode list, primarily for validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace conflicting generated Parquets or video symlinks.",
    )
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    convert(args)


if __name__ == "__main__":
    main()
