#!/usr/bin/env python3
"""Build a standalone HTML report for dummy_task intervention inference runs."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_ROOT = REPO_ROOT / "outputs" / "inference_robotwin_runs_metadata" / "dummy_task"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "inference_robotwin_runs" / "dummy_task"
DEFAULT_ROBOTWIN_ROOT = Path("/data/fangxuebin/RoboTwin/data/dummy_task/multiple_interventions")
DEFAULT_CONFIG = Path("/data/fangxuebin/RoboTwin/task_config/multiple_interventions.yml")
DEFAULT_HTML_PATH = REPO_ROOT / "outputs" / "dummy_task_intervention_report_standalone.html"
CROP_BRANCH = Path("crop_intervention") / "intervention_1_2"


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


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


def load_intervention_config(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    interventions = []
    for index in range(int(config.get("num_of_interventions", 0))):
        entry = config.get(f"intervention {index}", {})
        if entry:
            interventions.append({"index": index, **entry})
    return interventions


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ": "))


def build_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    skipped = {
        "missing_metadata": 0,
        "missing_source_video": 0,
        "missing_full_inference": 0,
        "missing_crop_inference": 0,
        "missing_crop_metadata": 0,
    }
    interventions = load_intervention_config(args.config)

    for metadata_path in sorted(args.metadata_root.glob("run_*/metadata.jsonl")):
        run = metadata_path.parent.name
        run_dir = args.robotwin_root / run
        crop_metadata = read_json(run_dir / "crop_intervention" / "crop_metadata.json")
        episodes = crop_metadata.get("episodes", {})
        if not episodes:
            skipped["missing_crop_metadata"] += 1

        for row in iter_jsonl(metadata_path):
            episode = int(row["episode_index"])
            episode_key = f"episode{episode}"
            source_video = Path(row["video"])
            full_inference = args.output_root / run / f"episode{episode}.mp4"
            crop_inference = args.output_root / run / CROP_BRANCH / f"episode{episode}.mp4"

            if not source_video.exists():
                skipped["missing_source_video"] += 1
                continue
            if not full_inference.exists() or full_inference.stat().st_size == 0:
                skipped["missing_full_inference"] += 1
                continue
            if not crop_inference.exists() or crop_inference.stat().st_size == 0:
                skipped["missing_crop_inference"] += 1
                continue

            episode_crop = episodes.get(episode_key, {})
            crop_range = next(
                (
                    item
                    for item in episode_crop.get("ranges", [])
                    if item.get("name") == "intervention_1_2"
                ),
                {},
            )
            active_spans = episode_crop.get("detected_active_spans", [])
            move_span = active_spans[0] if len(active_spans) > 0 else None
            close_span = active_spans[1] if len(active_spans) > 1 else None
            open_span = active_spans[2] if len(active_spans) > 2 else None
            seed_path = run_dir / "seed.txt"
            seed = seed_path.read_text(encoding="utf-8").strip() if seed_path.exists() else ""

            label = (
                f"move span {move_span}; gripper close span {close_span}; "
                f"gripper open span {open_span}; crop {crop_range.get('start')}-{crop_range.get('end')}"
            )
            rows.append(
                {
                    "task": "dummy_task",
                    "run": run,
                    "episode": episode,
                    "seed": seed,
                    "length": row.get("length", ""),
                    "num_frames": episode_crop.get("num_frames", row.get("length", "")),
                    "crop_name": "intervention_1_2",
                    "crop_start": crop_range.get("start"),
                    "crop_end": crop_range.get("end"),
                    "crop_frames": crop_range.get("frames"),
                    "active_spans": active_spans,
                    "move_span": move_span,
                    "gripper_close_span": close_span,
                    "gripper_open_span": open_span,
                    "configured_interventions": episode_crop.get("configured_interventions", []),
                    "intervention_config": interventions,
                    "label": label,
                    "source_video": data_uri(source_video),
                    "full_inference_video": data_uri(full_inference),
                    "crop_inference_video": data_uri(crop_inference),
                    "source_path": str(source_video),
                    "full_inference_path": str(full_inference),
                    "crop_inference_path": str(crop_inference),
                }
            )

    rows.sort(key=lambda item: (item["run"], item["episode"]))
    summary = {
        "total": len(rows),
        "runs": sorted({row["run"] for row in rows}),
        "interventions": interventions,
        "skipped": skipped,
        "metadata_root": str(args.metadata_root),
        "output_root": str(args.output_root),
        "robotwin_root": str(args.robotwin_root),
        "config": str(args.config),
    }
    return rows, summary


def render_html(rows: list[dict], summary: dict) -> str:
    data_json = json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False)
    data_json = data_json.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dummy Task Intervention Report</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #fff;
      --ink: #151923;
      --muted: #5c6675;
      --line: #dde2ea;
      --accent: #0f6f64;
      --accent-weak: #e0f3ef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(246, 247, 249, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }}
    .wrap {{ width: min(1760px, calc(100vw - 32px)); margin: 0 auto; }}
    .topbar {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: end;
      padding: 18px 0 14px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    .subline {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .stats {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .pill {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(150px, 220px) auto auto;
      gap: 10px;
      padding: 0 0 14px;
      align-items: center;
    }}
    input, select, button {{
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      color: var(--ink);
      padding: 0 10px;
      font: inherit;
      font-size: 14px;
    }}
    button {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      cursor: pointer;
      min-width: 92px;
    }}
    button.secondary {{ background: var(--panel); color: var(--accent); }}
    main {{ padding: 18px 0 40px; }}
    .result-line {{ margin-bottom: 12px; color: var(--muted); font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(21, 25, 35, 0.04);
    }}
    .card-head {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 14px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      align-items: start;
    }}
    .title {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-weight: 700; }}
    .tag {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--accent-weak);
      color: var(--accent);
      font-size: 12px;
      font-weight: 650;
    }}
    .meta {{ margin-top: 7px; color: var(--muted); font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }}
    .pair-actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .pair-actions button {{ min-width: 84px; height: 32px; padding: 0 9px; font-size: 12px; }}
    .setting {{
      max-width: 720px;
      text-align: right;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .videos {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .video-pane {{ min-width: 0; border-right: 1px solid var(--line); background: #07090c; }}
    .video-pane:last-child {{ border-right: 0; }}
    .video-label {{
      height: 36px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 0 10px;
      color: #eef2f6;
      background: #1a2029;
      font-size: 12px;
    }}
    video {{ display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #050607; }}
    details {{ padding: 10px 14px 12px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 650; }}
    pre {{
      margin: 8px 0 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      line-height: 1.45;
    }}
    .empty {{ padding: 30px; border: 1px dashed var(--line); border-radius: 8px; background: var(--panel); text-align: center; color: var(--muted); }}
    @media (max-width: 1120px) {{
      .topbar, .card-head, .videos {{ grid-template-columns: 1fr; }}
      .setting {{ max-width: none; text-align: left; }}
      .video-pane {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .video-pane:last-child {{ border-bottom: 0; }}
      .controls {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 620px) {{
      .wrap {{ width: min(100vw - 20px, 1760px); }}
      .controls {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 20px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <div class="topbar">
        <div>
          <h1>dummy_task 机械臂位置与夹爪开合推理对比</h1>
          <div class="subline">每组包含 RoboTwin 全程视频、Boundless World Model 全程推理、以及只裁剪夹爪闭合/打开片段后的推理。</div>
        </div>
        <div class="stats" id="stats"></div>
      </div>
      <div class="controls">
        <input id="query" type="search" placeholder="搜索 run / episode / frame span">
        <select id="run"></select>
        <button id="play-visible" type="button">播放当前</button>
        <button id="pause-visible" class="secondary" type="button">暂停当前</button>
      </div>
    </div>
  </header>
  <main class="wrap">
    <div class="result-line" id="result-line"></div>
    <section class="grid" id="grid"></section>
  </main>
  <script id="report-data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('report-data').textContent);
    const rows = data.rows;
    const summary = data.summary;
    const state = {{ query: '', run: 'all' }};
    const grid = document.getElementById('grid');
    const resultLine = document.getElementById('result-line');
    const queryInput = document.getElementById('query');
    const runSelect = document.getElementById('run');

    function esc(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[ch]));
    }}

    function renderStats() {{
      document.getElementById('stats').innerHTML = [
        `${{summary.total}} grouped samples`,
        `${{summary.runs.length}} runs`,
        `3 videos per group`,
        `standalone`
      ].map(text => `<span class="pill">${{esc(text)}}</span>`).join('');
    }}

    function fillSelects() {{
      runSelect.innerHTML = '<option value="all">全部 run</option>' +
        summary.runs.map(run => `<option value="${{esc(run)}}">${{esc(run)}}</option>`).join('');
    }}

    function rowMatches(row) {{
      if (state.run !== 'all' && row.run !== state.run) return false;
      if (!state.query) return true;
      const haystack = [
        row.run, row.episode, row.seed, row.label, row.crop_start, row.crop_end,
        JSON.stringify(row.active_spans), JSON.stringify(row.intervention_config)
      ].join(' ').toLowerCase();
      return haystack.includes(state.query.toLowerCase());
    }}

    function card(row) {{
      const title = `${{row.task}} / ${{row.run}} / episode${{row.episode}}`;
      const details = {{
        task: row.task,
        run: row.run,
        episode: row.episode,
        seed: row.seed,
        num_frames: row.num_frames,
        active_spans: row.active_spans,
        move_span: row.move_span,
        gripper_close_span: row.gripper_close_span,
        gripper_open_span: row.gripper_open_span,
        crop: {{ name: row.crop_name, start: row.crop_start, end: row.crop_end, frames: row.crop_frames }},
        configured_interventions: row.configured_interventions,
        intervention_config: row.intervention_config,
        robotwin_video: row.source_path,
        full_inference_video: row.full_inference_path,
        crop_inference_video: row.crop_inference_path
      }};
      return `<article class="card">`
        + `<div class="card-head">`
        + `<div><div class="title"><span>${{esc(title)}}</span><span class="tag">crop ${{esc(row.crop_start)}}-${{esc(row.crop_end)}}</span></div>`
        + `<div class="meta">move: ${{esc(JSON.stringify(row.move_span))}} · close: ${{esc(JSON.stringify(row.gripper_close_span))}} · open: ${{esc(JSON.stringify(row.gripper_open_span))}} · frames: ${{esc(row.num_frames)}} · seed: ${{esc(row.seed)}}</div>`
        + `<div class="pair-actions"><button type="button" data-group-action="play">播放这一组</button><button type="button" class="secondary" data-group-action="pause">暂停这一组</button><button type="button" class="secondary" data-group-action="restart">从头同步</button></div></div>`
        + `<div class="setting">${{esc(row.label)}}</div>`
        + `</div>`
        + `<div class="videos">`
        + `<div class="video-pane"><div class="video-label"><span>RoboTwin full trajectory</span><span>original</span></div><video controls muted loop preload="metadata" src="${{esc(row.source_video)}}"></video></div>`
        + `<div class="video-pane"><div class="video-label"><span>BWM full inference</span><span>full</span></div><video controls muted loop preload="metadata" src="${{esc(row.full_inference_video)}}"></video></div>`
        + `<div class="video-pane"><div class="video-label"><span>BWM gripper-crop inference</span><span>intervention_1_2</span></div><video controls muted loop preload="metadata" src="${{esc(row.crop_inference_video)}}"></video></div>`
        + `</div>`
        + `<details><summary>路径和完整参数</summary><pre>${{esc(JSON.stringify(details, null, 2))}}</pre></details>`
        + `</article>`;
    }}

    function render() {{
      const filtered = rows.filter(rowMatches);
      resultLine.textContent = `显示 ${{filtered.length}} / ${{rows.length}} 组样本`;
      grid.innerHTML = filtered.length ? filtered.map(card).join('') : '<div class="empty">没有匹配的 run、episode 或 span。</div>';
    }}

    function visibleVideos() {{
      return [...document.querySelectorAll('video')];
    }}

    function groupVideos(card) {{
      return [...card.querySelectorAll('video')];
    }}

    function playGroup(card, restart) {{
      const videos = groupVideos(card);
      if (!videos.length) return;
      const targetTime = restart ? 0 : Math.min(...videos.map(video => video.currentTime || 0));
      videos.forEach(video => {{
        try {{ video.currentTime = targetTime; }} catch (error) {{}}
      }});
      requestAnimationFrame(() => {{
        videos.forEach(video => video.play().catch(() => {{}}));
      }});
    }}

    function pauseGroup(card) {{
      groupVideos(card).forEach(video => video.pause());
    }}

    queryInput.addEventListener('input', event => {{
      state.query = event.target.value.trim();
      render();
    }});
    runSelect.addEventListener('change', event => {{
      state.run = event.target.value;
      render();
    }});
    document.getElementById('play-visible').addEventListener('click', () => {{
      visibleVideos().forEach(video => video.play().catch(() => {{}}));
    }});
    document.getElementById('pause-visible').addEventListener('click', () => {{
      visibleVideos().forEach(video => video.pause());
    }});
    grid.addEventListener('click', event => {{
      const button = event.target.closest('button[data-group-action]');
      if (!button) return;
      const card = button.closest('.card');
      if (!card) return;
      const action = button.dataset.groupAction;
      if (action === 'play') playGroup(card, false);
      if (action === 'pause') pauseGroup(card);
      if (action === 'restart') playGroup(card, true);
    }});

    fillSelects();
    renderStats();
    render();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--robotwin-root", type=Path, default=DEFAULT_ROBOTWIN_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--html-path", type=Path, default=DEFAULT_HTML_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_rows(args)
    args.html_path.parent.mkdir(parents=True, exist_ok=True)
    args.html_path.write_text(render_html(rows, summary), encoding="utf-8")
    print(f"Wrote {args.html_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
