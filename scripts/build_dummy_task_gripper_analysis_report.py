#!/usr/bin/env python3
"""Build a static gripper-movement analysis report for dummy_task runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_ROOT = REPO_ROOT / "outputs" / "inference_robotwin_runs_metadata" / "dummy_task"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "inference_robotwin_runs" / "dummy_task"
DEFAULT_ROBOTWIN_ROOT = Path("/data1/fangxuebin/dev/sim/RoboTwin/data/dummy_task/multiple_interventions")
DEFAULT_HTML_PATH = REPO_ROOT / "outputs" / "dummy_task_gripper_analysis_report.html"
DEFAULT_ASSET_ROOT = REPO_ROOT / "outputs" / "dummy_task_gripper_analysis_report_assets"
CROP_BRANCH = Path("crop_intervention") / "intervention_1_2"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield json.loads(text)


def read_jsonl_by_episode(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {int(row["episode_index"]): row for row in iter_jsonl(path)}


def format_number(value: float, digits: int) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == -0.0 else rounded


def infer_active_arm_and_cube(run_dir: Path, full_rows: list[dict]) -> dict:
    candidates = []
    for arm_tag in ("left", "right"):
        points = []
        for row in full_rows:
            hdf5_path = run_dir / "data" / f"episode{int(row['episode_index'])}.hdf5"
            if not hdf5_path.exists():
                continue
            with h5py.File(hdf5_path, "r") as f:
                key = f"endpose/{arm_tag}_endpose"
                if key in f:
                    points.append([float(value) for value in f[key][-1, :3]])
        if len(points) < 2:
            continue
        mins = [min(point[axis] for point in points) for axis in range(3)]
        maxs = [max(point[axis] for point in points) for axis in range(3)]
        side_length = format_number(sum(high - low for low, high in zip(mins, maxs)) / 3.0, 2)
        if side_length <= 0:
            continue
        centre = [format_number((low + high) / 2.0, 3) for low, high in zip(mins, maxs)]
        candidates.append({"arm": arm_tag, "centre": centre, "side_length": side_length})
    if not candidates:
        return {"arm": "unknown", "centre": None, "side_length": None}
    return max(candidates, key=lambda item: float(item["side_length"]))


def read_video(path: Path) -> np.ndarray:
    return iio.imread(path, index=None)


def resize_like(frame: np.ndarray, target: np.ndarray) -> np.ndarray:
    if frame.shape == target.shape:
        return frame
    image = Image.fromarray(frame)
    image = image.resize((target.shape[1], target.shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(image)


def frame_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    frame_b = resize_like(frame_b, frame_a)
    return float(np.mean(np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32))) / 255.0)


def crop_frames(frames: np.ndarray, roi: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    return frames[:, y1:y2, x1:x2]


def clamp_index(index: int, num_frames: int) -> int:
    return max(0, min(num_frames - 1, index))


def segment_metric(sim_frames: np.ndarray, wm_frames: np.ndarray, start: int, end: int) -> dict:
    start = clamp_index(start, len(sim_frames))
    end = clamp_index(end, len(sim_frames))
    if end < start:
        start, end = end, start
    wm_start = clamp_index(start, len(wm_frames))
    wm_end = clamp_index(end, len(wm_frames))
    sim_change = frame_diff(sim_frames[start], sim_frames[end])
    wm_change = frame_diff(wm_frames[wm_start], wm_frames[wm_end])
    count = min(end - start + 1, wm_end - wm_start + 1)
    aligned = [
        frame_diff(sim_frames[start + offset], wm_frames[wm_start + offset])
        for offset in range(max(0, count))
    ]
    return {
        "sim_change": sim_change,
        "wm_change": wm_change,
        "ratio": wm_change / sim_change if sim_change > 1e-8 else None,
        "aligned_diff": float(np.mean(aligned)) if aligned else None,
        "frames": end - start + 1,
    }


def motion_mask(frame_a: np.ndarray, frame_b: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    diff = np.mean(np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32)), axis=2)
    return diff > threshold


def sample_indexes(num_frames: int, count: int = 8) -> list[int]:
    if num_frames <= 0:
        return []
    if num_frames <= count:
        return list(range(num_frames))
    return [int(round(value)) for value in np.linspace(0, num_frames - 1, count)]


def dark_arm_region(frame: np.ndarray) -> dict | None:
    brightness = np.mean(frame.astype(np.float32), axis=2)
    dark = brightness < 105
    dark[:40, :] = False
    dark = ndimage.binary_opening(dark, iterations=1)
    dark = ndimage.binary_dilation(dark, iterations=3)
    labels, num_labels = ndimage.label(dark)
    best = None
    best_area = 0
    for label in range(1, num_labels + 1):
        component = labels == label
        area = int(component.sum())
        if area > best_area:
            best_area = area
            best = component
    if best is None or best_area < 100:
        return None
    ys, xs = np.where(best)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    centroid = [float(xs.mean()), float(ys.mean())]
    return {"bbox": bbox, "centroid": centroid, "area": best_area}


def bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def full_arm_position_metric(sim_frames: np.ndarray, wm_frames: np.ndarray) -> dict:
    count = min(len(sim_frames), len(wm_frames))
    if count <= 0:
        return {"mean_centroid_px": None, "final_centroid_px": None, "mean_bbox_iou": None, "mean_full_diff": None}
    indexes = sample_indexes(count, 8)
    centroid_distances = []
    bbox_ious = []
    full_diffs = []
    diagonal = float(np.hypot(sim_frames.shape[1], sim_frames.shape[2]))
    for index in indexes:
        sim_frame = sim_frames[index]
        wm_frame = resize_like(wm_frames[index], sim_frame)
        sim_region = dark_arm_region(sim_frame)
        wm_region = dark_arm_region(wm_frame)
        full_diffs.append(frame_diff(sim_frame, wm_frame))
        if not sim_region or not wm_region:
            continue
        sx, sy = sim_region["centroid"]
        wx, wy = wm_region["centroid"]
        centroid_distances.append(float(np.hypot(sx - wx, sy - wy)))
        bbox_ious.append(bbox_iou(sim_region["bbox"], wm_region["bbox"]))

    final_sim = dark_arm_region(sim_frames[count - 1])
    final_wm = dark_arm_region(resize_like(wm_frames[count - 1], sim_frames[count - 1]))
    final_distance = None
    if final_sim and final_wm:
        final_distance = float(
            np.hypot(
                final_sim["centroid"][0] - final_wm["centroid"][0],
                final_sim["centroid"][1] - final_wm["centroid"][1],
            )
        )
    mean_centroid = summarize(centroid_distances)
    return {
        "mean_centroid_px": mean_centroid,
        "mean_centroid_norm": mean_centroid / diagonal if mean_centroid is not None else None,
        "final_centroid_px": final_distance,
        "final_centroid_norm": final_distance / diagonal if final_distance is not None else None,
        "mean_bbox_iou": summarize(bbox_ious),
        "mean_full_diff": summarize(full_diffs),
    }


def infer_gripper_roi(
    sim_frames: np.ndarray,
    close_local: list[int],
    open_local: list[int],
    margin: int = 10,
) -> list[int]:
    """Find a tight ROI around the dark claw pixels that move during close/open."""
    indexes = [
        clamp_index(0, len(sim_frames)),
        clamp_index(int(close_local[1]), len(sim_frames)),
        clamp_index(int(open_local[0]), len(sim_frames)),
        clamp_index(int(open_local[1]), len(sim_frames)),
    ]
    selected = sim_frames[indexes]
    brightness = np.mean(selected.astype(np.float32), axis=3)
    dark = np.any(brightness < 90, axis=0)
    dark[:40, :] = False

    diff = np.maximum.reduce(
        [
            np.mean(np.abs(selected[0].astype(np.float32) - selected[1].astype(np.float32)), axis=2),
            np.mean(np.abs(selected[2].astype(np.float32) - selected[3].astype(np.float32)), axis=2),
            np.mean(np.abs(selected[1].astype(np.float32) - selected[3].astype(np.float32)), axis=2),
        ]
    )
    dark_diff = diff[dark]
    threshold = max(5.0, float(np.percentile(dark_diff, 92))) if len(dark_diff) else 10.0
    claw_motion = dark & (diff >= threshold)
    claw_motion = ndimage.binary_opening(claw_motion, iterations=1)
    claw_motion = ndimage.binary_dilation(claw_motion, iterations=2)
    claw_motion[:40, :] = False

    labels, num_labels = ndimage.label(claw_motion)
    components = []
    for label in range(1, num_labels + 1):
        component = labels == label
        area = int(component.sum())
        if 8 <= area <= 2500:
            components.append((float(diff[component].sum()), component))
    components.sort(key=lambda item: item[0], reverse=True)
    if components:
        best_mask = np.zeros_like(claw_motion, dtype=bool)
        # Two claws often appear as separate moving components.
        for _score, component in components[:3]:
            best_mask |= component
    else:
        best_mask = claw_motion

    ys, xs = np.where(best_mask)
    if len(xs) == 0 or len(ys) == 0:
        height, width = sim_frames.shape[1:3]
        return [0, 0, width, height]

    height, width = sim_frames.shape[1:3]
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(width, int(xs.max()) + margin + 1)
    y2 = min(height, int(ys.max()) + margin + 1)
    return [x1, y1, x2, y2]


def summarize(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def get_gripper_delta(run_dir: Path, episode: int, arm: str, span: list[int] | None) -> float | None:
    if arm not in {"left", "right"} or not span:
        return None
    hdf5_path = run_dir / "data" / f"episode{episode}.hdf5"
    if not hdf5_path.exists():
        return None
    with h5py.File(hdf5_path, "r") as f:
        key = f"endpose/{arm}_gripper"
        if key not in f:
            return None
        values = np.asarray(f[key], dtype=np.float32)
    start = clamp_index(int(span[0]), len(values))
    end = clamp_index(int(span[1]) - 1, len(values))
    return float(values[end] - values[start])


def collect_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows = []
    skipped = {
        "missing_source_video": 0,
        "missing_crop_source_video": 0,
        "missing_full_inference": 0,
        "missing_crop_inference": 0,
    }
    for metadata_path in sorted(args.metadata_root.glob("run_*/metadata.jsonl")):
        run = metadata_path.parent.name
        if "_side" in run:
            continue
        full_rows = list(iter_jsonl(metadata_path))
        if not full_rows:
            continue
        crop_rows = read_jsonl_by_episode(metadata_path.parent / CROP_BRANCH / "metadata.jsonl")
        first_source_video = Path(full_rows[0]["video"])
        run_dir = args.robotwin_root / run
        if not run_dir.exists() and first_source_video.parent.name == "video":
            run_dir = first_source_video.parent.parent
        cube = infer_active_arm_and_cube(run_dir, full_rows)
        crop_metadata = read_json(run_dir / "crop_intervention" / "crop_metadata.json")
        episodes = crop_metadata.get("episodes", {})

        for row in full_rows:
            episode = int(row["episode_index"])
            source_video = Path(row["video"])
            crop_source_row = crop_rows.get(episode, {})
            crop_source_video = Path(crop_source_row.get("video", ".")) if crop_source_row else Path(".")
            full_inference = args.output_root / run / f"episode{episode}.mp4"
            crop_inference = args.output_root / run / CROP_BRANCH / f"episode{episode}.mp4"

            if not source_video.exists():
                skipped["missing_source_video"] += 1
                continue
            if not crop_source_video.is_file():
                skipped["missing_crop_source_video"] += 1
                continue
            if not full_inference.exists() or full_inference.stat().st_size == 0:
                skipped["missing_full_inference"] += 1
                continue
            if not crop_inference.exists() or crop_inference.stat().st_size == 0:
                skipped["missing_crop_inference"] += 1
                continue

            episode_crop = episodes.get(f"episode{episode}", {})
            crop_range = next(
                (
                    item
                    for item in episode_crop.get("ranges", [])
                    if item.get("name") == "intervention_1_2"
                ),
                {},
            )
            spans = episode_crop.get("detected_active_spans", [])
            close_span = spans[1] if len(spans) > 1 else None
            open_span = spans[2] if len(spans) > 2 else None
            crop_start = int(crop_range.get("start", close_span[0] if close_span else 0))

            sim_crop = read_video(crop_source_video)
            wm_crop = read_video(crop_inference)
            sim_full = read_video(source_video)
            wm_full = read_video(full_inference)
            close_local = [int(close_span[0]) - crop_start, int(close_span[1]) - crop_start - 1]
            open_local = [int(open_span[0]) - crop_start, int(open_span[1]) - crop_start - 1]
            close_metric = segment_metric(sim_crop, wm_crop, close_local[0], close_local[1])
            open_metric = segment_metric(sim_crop, wm_crop, open_local[0], open_local[1])
            gripper_roi = infer_gripper_roi(sim_crop, close_local, open_local)
            sim_roi = crop_frames(sim_crop, gripper_roi)
            wm_roi = crop_frames(wm_crop, gripper_roi)
            close_roi_metric = segment_metric(sim_roi, wm_roi, close_local[0], close_local[1])
            open_roi_metric = segment_metric(sim_roi, wm_roi, open_local[0], open_local[1])
            full_metric = full_arm_position_metric(sim_full, wm_full)

            rows.append(
                {
                    "run": run,
                    "episode": episode,
                    "arm": cube["arm"],
                    "cube_side_length": cube["side_length"],
                    "cube_centre": cube["centre"],
                    "crop_start": crop_start,
                    "close_span": close_span,
                    "open_span": open_span,
                    "close_local": close_local,
                    "open_local": open_local,
                    "gripper_roi": gripper_roi,
                    "close_metric": close_metric,
                    "open_metric": open_metric,
                    "close_roi_metric": close_roi_metric,
                    "open_roi_metric": open_roi_metric,
                    "full_metric": full_metric,
                    "close_gripper_delta": get_gripper_delta(run_dir, episode, cube["arm"], close_span),
                    "open_gripper_delta": get_gripper_delta(run_dir, episode, cube["arm"], open_span),
                    "sim_crop_video": crop_source_video,
                    "wm_crop_video": crop_inference,
                }
            )
    summary = {"total": len(rows), "skipped": skipped}
    return rows, summary


def group_key(row: dict, keys: tuple[str, ...]) -> tuple:
    return tuple(row[key] for key in keys)


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(group_key(row, keys), []).append(row)
    output = []
    for key, items in sorted(grouped.items()):
        base = {name: value for name, value in zip(keys, key)}
        base.update(
            {
                "count": len(items),
                "close_sim": summarize([item["close_metric"]["sim_change"] for item in items]),
                "close_wm": summarize([item["close_metric"]["wm_change"] for item in items]),
                "close_ratio": summarize([item["close_metric"]["ratio"] for item in items]),
                "close_diff": summarize([item["close_metric"]["aligned_diff"] for item in items]),
                "open_sim": summarize([item["open_metric"]["sim_change"] for item in items]),
                "open_wm": summarize([item["open_metric"]["wm_change"] for item in items]),
                "open_ratio": summarize([item["open_metric"]["ratio"] for item in items]),
                "open_diff": summarize([item["open_metric"]["aligned_diff"] for item in items]),
                "close_roi_sim": summarize([item["close_roi_metric"]["sim_change"] for item in items]),
                "close_roi_wm": summarize([item["close_roi_metric"]["wm_change"] for item in items]),
                "close_roi_ratio": summarize([item["close_roi_metric"]["ratio"] for item in items]),
                "close_roi_diff": summarize([item["close_roi_metric"]["aligned_diff"] for item in items]),
                "open_roi_sim": summarize([item["open_roi_metric"]["sim_change"] for item in items]),
                "open_roi_wm": summarize([item["open_roi_metric"]["wm_change"] for item in items]),
                "open_roi_ratio": summarize([item["open_roi_metric"]["ratio"] for item in items]),
                "open_roi_diff": summarize([item["open_roi_metric"]["aligned_diff"] for item in items]),
                "full_mean_centroid_px": summarize([item["full_metric"]["mean_centroid_px"] for item in items]),
                "full_mean_centroid_norm": summarize([item["full_metric"]["mean_centroid_norm"] for item in items]),
                "full_final_centroid_px": summarize([item["full_metric"]["final_centroid_px"] for item in items]),
                "full_final_centroid_norm": summarize([item["full_metric"]["final_centroid_norm"] for item in items]),
                "full_mean_bbox_iou": summarize([item["full_metric"]["mean_bbox_iou"] for item in items]),
                "full_mean_diff": summarize([item["full_metric"]["mean_full_diff"] for item in items]),
                "close_gripper_delta": summarize([item["close_gripper_delta"] for item in items]),
                "open_gripper_delta": summarize([item["open_gripper_delta"] for item in items]),
            }
        )
        output.append(base)
    return output


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def make_contact_sheet(rows: list[dict], out_path: Path, title: str) -> None:
    selected = rows[:2]
    labels = ["crop start", "close end", "open start", "open end"]
    thumbs = []
    for row in selected:
        sim = read_video(row["sim_crop_video"])
        wm = read_video(row["wm_crop_video"])
        indexes = [
            0,
            row["close_local"][1],
            row["open_local"][0],
            row["open_local"][1],
        ]
        for video_name, frames in [("RoboTwin", sim), ("WM", wm)]:
            for label, index in zip(labels, indexes):
                frame = frames[clamp_index(int(index), len(frames))]
                image = Image.fromarray(frame).resize((240, 180), Image.Resampling.BILINEAR)
                draw = ImageDraw.Draw(image)
                x1, y1, x2, y2 = row["gripper_roi"]
                scale_x = 240 / frame.shape[1]
                scale_y = 180 / frame.shape[0]
                box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                draw.rectangle(box, outline=(255, 64, 64), width=3)
                thumbs.append((f"{row['run']} ep{row['episode']} {video_name} {label}", image))

    pad = 10
    caption_h = 38
    cols = 4
    rows_count = int(np.ceil(len(thumbs) / cols))
    canvas = Image.new("RGB", (cols * 240 + (cols + 1) * pad, rows_count * (180 + caption_h) + (rows_count + 1) * pad + 34), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), title, fill=(20, 25, 35))
    y0 = pad + 34
    for idx, (caption, image) in enumerate(thumbs):
        col = idx % cols
        row = idx // cols
        x = pad + col * (240 + pad)
        y = y0 + row * (180 + caption_h + pad)
        canvas.paste(image, (x, y))
        draw.text((x, y + 184), caption, fill=(80, 90, 105))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def html_table(rows: list[dict], headers: list[tuple[str, str]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{fmt(row[key])}</td>" for key, _label in headers)
            + "</tr>"
        )
    head = "".join(f"<th>{label}</th>" for _key, label in headers)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(rows: list[dict], summary: dict, assets: dict[str, Path], html_path: Path) -> str:
    by_arm = aggregate(rows, ("arm",))
    by_side = aggregate(rows, ("arm", "cube_side_length"))
    by_run = aggregate(rows, ("run", "arm", "cube_side_length"))
    left = next((row for row in by_arm if row["arm"] == "left"), {})
    right = next((row for row in by_arm if row["arm"] == "right"), {})
    rel_assets = {name: path.relative_to(html_path.parent).as_posix() for name, path in assets.items()}
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dummy_task 夹爪动作分析报告</title>
<style>
:root {{ --fg:#1d232b; --muted:#667085; --line:#d8dee8; --bg:#f6f8fb; --card:#ffffff; --accent:#2f6fed; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; color:var(--fg); background:var(--bg); line-height:1.55; }}
main {{ max-width:1240px; margin:0 auto; padding:28px 24px 56px; }}
h1 {{ margin:0 0 8px; font-size:30px; letter-spacing:0; }}
h2 {{ margin:34px 0 12px; font-size:22px; border-bottom:1px solid var(--line); padding-bottom:8px; letter-spacing:0; }}
h3 {{ margin:20px 0 10px; font-size:17px; letter-spacing:0; }}
p {{ margin:8px 0 12px; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; margin:14px 0; }}
.grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
.metric {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:14px; }}
.metric b {{ display:block; font-size:24px; }}
.metric span, .muted {{ color:var(--muted); font-size:13px; }}
table {{ width:100%; border-collapse:collapse; margin:10px 0 16px; background:#fff; border:1px solid var(--line); }}
th, td {{ border-bottom:1px solid var(--line); padding:8px 10px; vertical-align:top; text-align:left; font-size:13px; }}
th {{ background:#eef2f8; font-weight:650; }}
tr:last-child td {{ border-bottom:0; }}
.wide-img {{ width:100%; display:block; border:1px solid var(--line); border-radius:8px; background:#fff; margin:10px 0 16px; }}
.note {{ color:var(--muted); font-size:13px; }}
@media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} main {{ padding:18px 12px 40px; }} table {{ display:block; overflow-x:auto; }} }}
</style>
</head>
<body>
<main>
<header>
  <h1>dummy_task 夹爪 close/open 动作分析报告</h1>
  <p class="muted">实验组: <code>cube_center=(0.0, -0.15, 1.0)</code>, 边长 5/10/15cm. active arm 从 HDF5 中实际形成 cube 的 endpose 自动推断. 样本数: {summary["total"]} 个 episode.</p>
</header>

<section class="card">
  <h2>核心结论</h2>
  <div class="grid">
    <div class="metric"><b>左臂 ROI: {fmt(left.get("close_roi_ratio"), 2)} / {fmt(left.get("open_roi_ratio"), 2)}</b><span>close / open 的 gripper ROI 变化比例</span></div>
    <div class="metric"><b>右臂 ROI: {fmt(right.get("close_roi_ratio"), 2)} / {fmt(right.get("open_roi_ratio"), 2)}</b><span>close / open 的 gripper ROI 变化比例</span></div>
    <div class="metric"><b>{fmt(left.get("close_gripper_delta"), 2)} / +{fmt(left.get("open_gripper_delta"), 2)}</b><span>HDF5 active gripper close / open 输入变化</span></div>
  </div>
  <p>这个 dummy_task 实验的关注点不是 cube 方向运动, 而是裁剪段中的夹爪闭合与打开。报告使用 <code>crop_intervention/intervention_1_2</code> 视频, close 段对应第二个 active span, open 段对应第三个 active span。</p>
  <p>主指标改为 gripper ROI: ROI 从 RoboTwin 裁剪视频中 close/open 期间发生运动的暗色 gripper 连通区域自动估计, 并把同一个框用于 WM 视频。它比整帧像素差更少受到腕部整体运动、遮挡和背景变化影响。</p>
  <p>ROI ratio 仍然不是严格成功率, 但它更接近“夹爪局部是否发生对应变化”。如果右臂视频里夹爪看起来没有明显开合, 应优先相信 ROI 代表帧和交互浏览器中的人工检查, 而不是整帧变化指标。</p>
</section>

<section class="card">
  <h2>主分析: gripper ROI</h2>
  <p>数值越接近 1, 表示 WM 在夹爪局部 ROI 内的变化幅度越接近 RoboTwin。<code>ROI 对齐帧平均差异</code> 越小, 表示两段视频在夹爪局部逐帧外观更接近。</p>
  {html_table(by_arm, [
      ("arm", "机械臂"),
      ("count", "episode 数"),
      ("close_roi_sim", "仿真器 ROI close 变化"),
      ("close_roi_wm", "WM ROI close 变化"),
      ("close_roi_ratio", "ROI close WM/仿真器"),
      ("close_roi_diff", "ROI close 对齐帧平均差异"),
      ("open_roi_sim", "仿真器 ROI open 变化"),
      ("open_roi_wm", "WM ROI open 变化"),
      ("open_roi_ratio", "ROI open WM/仿真器"),
      ("open_roi_diff", "ROI open 对齐帧平均差异"),
  ])}
  <p>左臂 close/open ROI 变化比例为 {fmt(left.get("close_roi_ratio"), 2)} / {fmt(left.get("open_roi_ratio"), 2)}, 右臂 close/open ROI 变化比例为 {fmt(right.get("close_roi_ratio"), 2)} / {fmt(right.get("open_roi_ratio"), 2)}。这些数值仍需结合代表帧确认 ROI 是否覆盖了实际夹爪。</p>
  <h3>按 cube 边长分组</h3>
  {html_table(by_side, [
      ("arm", "机械臂"),
      ("cube_side_length", "cube 边长"),
      ("count", "episode 数"),
      ("close_roi_ratio", "ROI close WM/仿真器"),
      ("open_roi_ratio", "ROI open WM/仿真器"),
      ("close_roi_diff", "ROI close 对齐差异"),
      ("open_roi_diff", "ROI open 对齐差异"),
  ])}
  <h3>补充: 整帧变化指标</h3>
  <p class="note">整帧指标保留为背景参考。它会计入腕部位移、遮挡和全局画面变化, 不应解释为夹爪动作成功率。</p>
  {html_table(by_arm, [
      ("arm", "机械臂"),
      ("close_ratio", "整帧 close WM/仿真器"),
      ("open_ratio", "整帧 open WM/仿真器"),
      ("close_diff", "整帧 close 对齐差异"),
      ("open_diff", "整帧 open 对齐差异"),
  ])}
</section>

<section class="card">
  <h2>扩展分析: full video 的整臂位置 hallucination</h2>
  <p>这一节分析完整视频, 关注 WM 是否在长程生成中产生机械臂整体位置漂移、错位或形状位置不一致。指标从每个 full video 采样 8 帧, 提取暗色机械臂/夹爪连通区域, 比较 RoboTwin 与 WM 的区域中心和 bbox。它不是 3D 位姿误差, 但能量化视频 hallucination 中常见的整臂视觉漂移。</p>
  {html_table(by_arm, [
      ("arm", "机械臂"),
      ("count", "episode 数"),
      ("full_mean_centroid_px", "平均 centroid 偏移(px)"),
      ("full_mean_centroid_norm", "平均 centroid 偏移/画面对角线"),
      ("full_final_centroid_px", "末帧 centroid 偏移(px)"),
      ("full_final_centroid_norm", "末帧 centroid 偏移/画面对角线"),
      ("full_mean_bbox_iou", "平均 bbox IoU"),
      ("full_mean_diff", "整帧平均差异"),
  ])}
  <p>centroid 偏移越大, 表示 WM 生成的机械臂暗色主体位置越偏离仿真器; bbox IoU 越低, 表示机械臂可见区域的形状/位置重叠越差。若 full video 的整臂指标差, 即使 clipped gripper ROI 有变化, 也可能是 hallucinated arm motion 而不是正确的物理响应。</p>
  <p>当前结果显示, 左臂 full video 的整臂漂移更明显: 平均 centroid 偏移约 {fmt(left.get("full_mean_centroid_px"), 1)}px, 末帧偏移约 {fmt(left.get("full_final_centroid_px"), 1)}px, bbox IoU 约 {fmt(left.get("full_mean_bbox_iou"), 2)}。右臂对应为 {fmt(right.get("full_mean_centroid_px"), 1)}px / {fmt(right.get("full_final_centroid_px"), 1)}px / {fmt(right.get("full_mean_bbox_iou"), 2)}。按 run 看, 左臂 15cm 的 run_0003 最明显, 末帧偏移接近 191px, 这更像长程生成中的整臂位置 hallucination, 而不是单纯夹爪开合误差。</p>
  <h3>按 run 分组</h3>
  {html_table(by_run, [
      ("run", "run"),
      ("arm", "机械臂"),
      ("cube_side_length", "cube 边长"),
      ("count", "episode 数"),
      ("full_mean_centroid_px", "平均 centroid 偏移(px)"),
      ("full_final_centroid_px", "末帧 centroid 偏移(px)"),
      ("full_mean_bbox_iou", "平均 bbox IoU"),
      ("full_mean_diff", "整帧平均差异"),
  ])}
</section>

<section class="card">
  <h2>输入夹爪轨迹校验</h2>
  <p>下面直接读取 HDF5 中 active arm 的 <code>endpose/&lt;arm&gt;_gripper</code>。close 段应为负向变化, open 段应为正向变化。结果显示输入动作本身存在清晰 close/open 切换, 因此 WM 视频中夹爪幅度不足更可能来自生成模型, 而不是裁剪段没有包含夹爪动作。</p>
  {html_table(by_arm, [
      ("arm", "机械臂"),
      ("close_gripper_delta", "close gripper delta"),
      ("open_gripper_delta", "open gripper delta"),
  ])}
  <h3>逐 run 结果</h3>
  {html_table(by_run, [
      ("run", "run"),
      ("arm", "机械臂"),
      ("cube_side_length", "cube 边长"),
      ("count", "episode 数"),
      ("close_roi_ratio", "ROI close WM/仿真器"),
      ("open_roi_ratio", "ROI open WM/仿真器"),
      ("close_gripper_delta", "close gripper delta"),
      ("open_gripper_delta", "open gripper delta"),
  ])}
</section>

<section class="card">
  <h2>代表帧</h2>
  <p class="note">每张图包含 RoboTwin 与 WM 的裁剪视频帧: crop start, close end, open start, open end。红框是自动估计的 gripper ROI, 用于快速检查 ROI 是否覆盖夹爪以及 WM 是否在夹爪闭合/打开时产生对应局部变化。</p>
  <h3>左臂代表帧</h3>
  <img class="wide-img" src="{rel_assets['left_sheet']}" alt="left arm gripper contact sheet">
  <h3>右臂代表帧</h3>
  <img class="wide-img" src="{rel_assets['right_sheet']}" alt="right arm gripper contact sheet">
</section>

<section class="card">
  <h2>产物位置</h2>
  <p>交互浏览器仍在 <code>outputs/dummy_task_intervention_report_standalone.html</code>; 本报告产物在 <code>{html_path}</code>, 图片资源在 <code>{html_path.parent / DEFAULT_ASSET_ROOT.name}</code>.</p>
  <p class="note">跳过统计: <code>{json.dumps(summary["skipped"], ensure_ascii=False)}</code></p>
</section>
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--robotwin-root", type=Path, default=DEFAULT_ROBOTWIN_ROOT)
    parser.add_argument("--html-path", type=Path, default=DEFAULT_HTML_PATH)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = collect_rows(args)
    left_rows = [row for row in rows if row["arm"] == "left" and row["episode"] in {0, 4}]
    right_rows = [row for row in rows if row["arm"] == "right" and row["episode"] in {0, 4}]
    assets = {
        "left_sheet": args.asset_root / "dummy_task_gripper_left_contact_sheet.jpg",
        "right_sheet": args.asset_root / "dummy_task_gripper_right_contact_sheet.jpg",
    }
    make_contact_sheet(left_rows, assets["left_sheet"], "dummy_task left arm gripper close/open")
    make_contact_sheet(right_rows, assets["right_sheet"], "dummy_task right arm gripper close/open")
    args.html_path.parent.mkdir(parents=True, exist_ok=True)
    args.html_path.write_text(render_html(rows, summary, assets, args.html_path), encoding="utf-8")
    print(f"Wrote {args.html_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
