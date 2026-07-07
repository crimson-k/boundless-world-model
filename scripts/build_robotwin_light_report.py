#!/usr/bin/env python3
"""Build a static HTML report for RoboTwin light-intervention inference runs."""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
from pathlib import Path
from urllib.parse import quote

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_ROOT = REPO_ROOT / "outputs" / "inference_robotwin_runs_metadata"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "inference_robotwin_runs"
DEFAULT_ROBOTWIN_DATA_ROOT = Path("/data/fangxuebin/RoboTwin/data")
DEFAULT_LIGHT_CONFIG = Path("/data/fangxuebin/RoboTwin/task_config/multiple_interventions.yml")
DEFAULT_HTML_PATH = REPO_ROOT / "outputs" / "robotwin_light_intervention_report.html"
DEFAULT_ASSET_ROOT = REPO_ROOT / "outputs" / "robotwin_light_intervention_report_assets"


def file_uri(path: Path) -> str:
    return "file://" + quote(str(path.resolve()))


def rel_or_file_uri(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return file_uri(path)


def materialize_video(source: Path, destination: Path, mode: str) -> Path:
    if mode == "none":
        return source
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return destination
    if destination.exists():
        destination.unlink()
    if mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return destination
        except OSError:
            shutil.copy2(source, destination)
            return destination
    if mode == "copy":
        shutil.copy2(source, destination)
        return destination
    raise ValueError(f"Unknown materialize mode: {mode}")


def video_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def load_light_settings(config_path: Path) -> list[dict]:
    if not config_path.exists():
        return []
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return (
        config.get("domain_randomization", {})
        .get("supervised_light_settings", [])
    )


def compact_json(value: object) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ": "))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield json.loads(text)


def load_scene_info(robotwin_data_root: Path, task: str, run: str) -> dict:
    scene_path = robotwin_data_root / task / "multiple_interventions" / run / "scene_info.json"
    if not scene_path.exists():
        return {}
    with scene_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    html_dir = args.html_path.parent
    light_settings = load_light_settings(args.light_config)
    rows: list[dict] = []
    skipped = {
        "dummy_task": 0,
        "missing_inference_video": 0,
        "missing_source_video": 0,
    }

    for metadata_path in sorted(args.metadata_root.glob("*/*/metadata.jsonl")):
        task = metadata_path.parts[-3]
        run = metadata_path.parts[-2]
        if task == "dummy_task":
            skipped["dummy_task"] += sum(1 for _ in iter_jsonl(metadata_path))
            continue

        scene_info = load_scene_info(args.robotwin_data_root, task, run)
        output_dir = args.output_root / task / run

        for row in iter_jsonl(metadata_path):
            episode = int(row["episode_index"])
            source_video = Path(row["video"])
            inference_video = output_dir / f"episode{episode}.mp4"
            if not source_video.exists():
                skipped["missing_source_video"] += 1
                continue
            if not inference_video.exists() or inference_video.stat().st_size == 0:
                skipped["missing_inference_video"] += 1
                continue

            if args.embed_videos:
                report_source_video = source_video
                source_video_src = video_data_uri(source_video)
                inference_video_src = video_data_uri(inference_video)
            else:
                report_source_video = materialize_video(
                    source_video,
                    args.asset_root / "robotwin" / task / run / f"episode{episode}.mp4",
                    args.materialize_source_videos,
                )
                source_video_src = rel_or_file_uri(report_source_video, html_dir)
                inference_video_src = rel_or_file_uri(inference_video, html_dir)

            scene = scene_info.get(f"episode_{episode}", {})
            setting_index = scene.get("supervised_light_setting_index")
            setting = {}
            if isinstance(setting_index, int) and 0 <= setting_index < len(light_settings):
                setting = light_settings[setting_index]

            rows.append(
                {
                    "task": task,
                    "run": run,
                    "episode": episode,
                    "source_episode": scene.get("source_episode_idx", row.get("source_episode_index", "")),
                    "source_seed": scene.get("source_seed", ""),
                    "length": row.get("length", ""),
                    "light_setting_index": setting_index,
                    "light_setting": setting,
                    "light_label": compact_json(setting) or "setting not found",
                    "scene_info": scene,
                    "source_video": source_video_src,
                    "inference_video": inference_video_src,
                    "source_path": str(source_video),
                    "inference_path": str(inference_video),
                }
            )

    rows.sort(key=lambda r: (r["task"], r["run"], r["episode"]))
    summary = {
        "total": len(rows),
        "tasks": sorted({row["task"] for row in rows}),
        "light_settings": light_settings,
        "skipped": skipped,
        "metadata_root": str(args.metadata_root),
        "output_root": str(args.output_root),
        "robotwin_data_root": str(args.robotwin_data_root),
        "light_config": str(args.light_config),
        "asset_root": str(args.asset_root),
        "materialize_source_videos": args.materialize_source_videos,
        "embed_videos": args.embed_videos,
    }
    return rows, summary


def render_html(rows: list[dict], summary: dict) -> str:
    data_json = json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False)
    data_json = (
        data_json.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    title = "RoboTwin Light Intervention Inference Report"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #14171f;
      --muted: #5d6676;
      --line: #dfe3ea;
      --accent: #126d7a;
      --accent-weak: #e3f4f6;
      --warn: #9a5b00;
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
      background: rgba(247, 248, 250, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }}
    .wrap {{ width: min(1680px, calc(100vw - 32px)); margin: 0 auto; }}
    .topbar {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: end;
      padding: 18px 0 14px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; font-weight: 720; letter-spacing: 0; }}
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
      grid-template-columns: minmax(180px, 1fr) minmax(150px, 220px) minmax(150px, 220px) auto auto;
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
    button.secondary {{
      background: var(--panel);
      color: var(--accent);
    }}
    main {{ padding: 18px 0 40px; }}
    .result-line {{ margin-bottom: 12px; font-size: 13px; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(20, 23, 31, 0.04);
    }}
    .card-head {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      align-items: start;
    }}
    .title {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      font-weight: 680;
      line-height: 1.35;
    }}
    .tag {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--accent-weak);
      color: var(--accent);
      font-size: 12px;
      font-weight: 620;
    }}
    .meta {{
      margin-top: 7px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .pair-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .pair-actions button {{
      min-width: 84px;
      height: 32px;
      padding: 0 9px;
      font-size: 12px;
    }}
    .setting {{
      max-width: 680px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      text-align: right;
      overflow-wrap: anywhere;
    }}
    .videos {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
    }}
    .video-pane {{ min-width: 0; border-right: 1px solid var(--line); background: #0b0d10; }}
    .video-pane:last-child {{ border-right: 0; }}
    .video-label {{
      height: 34px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 10px;
      color: #eef2f6;
      background: #1a1f28;
      font-size: 12px;
    }}
    video {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #050607;
    }}
    details {{
      padding: 10px 14px 12px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 620; }}
    pre {{
      margin: 8px 0 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      line-height: 1.45;
    }}
    .empty {{
      padding: 30px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--warn);
      text-align: center;
    }}
    @media (max-width: 980px) {{
      .topbar, .card-head, .videos {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr 1fr; }}
      .setting {{ text-align: left; max-width: none; }}
      .video-pane {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .video-pane:last-child {{ border-bottom: 0; }}
    }}
    @media (max-width: 620px) {{
      .wrap {{ width: min(100vw - 20px, 1680px); }}
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
          <h1>RoboTwin 光线扰动推理对比</h1>
          <div class="subline">左侧为 RoboTwin 光线扰动后视频，右侧为 Boundless World Model 推理视频；不包含 dummy_task。</div>
        </div>
        <div class="stats" id="stats"></div>
      </div>
      <div class="controls">
        <input id="query" type="search" placeholder="搜索 task / run / episode / 参数">
        <select id="task"></select>
        <select id="setting"></select>
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
    const state = {{ query: '', task: 'all', setting: 'all' }};
    const grid = document.getElementById('grid');
    const resultLine = document.getElementById('result-line');
    const taskSelect = document.getElementById('task');
    const settingSelect = document.getElementById('setting');
    const queryInput = document.getElementById('query');

    function esc(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[ch]));
    }}

    function fillSelects() {{
      taskSelect.innerHTML = '<option value="all">全部任务</option>' +
        summary.tasks.map(task => `<option value="${{esc(task)}}">${{esc(task)}}</option>`).join('');
      const settingIds = [...new Set(rows.map(row => row.light_setting_index).filter(v => v !== null && v !== undefined))].sort((a, b) => a - b);
      settingSelect.innerHTML = '<option value="all">全部光照设置</option>' +
        settingIds.map(id => `<option value="${{id}}">setting ${{id}}</option>`).join('');
    }}

    function renderStats() {{
      const skipped = summary.skipped || {{}};
      document.getElementById('stats').innerHTML = [
        `${{summary.total}} paired videos`,
        `${{summary.tasks.length}} tasks`,
        `${{summary.light_settings.length}} light settings`,
        `${{skipped.missing_inference_video || 0}} skipped missing inference`
      ].map(text => `<span class="pill">${{esc(text)}}</span>`).join('');
    }}

    function rowMatches(row) {{
      if (state.task !== 'all' && row.task !== state.task) return false;
      if (state.setting !== 'all' && String(row.light_setting_index) !== state.setting) return false;
      if (!state.query) return true;
      const haystack = [
        row.task, row.run, row.episode, row.source_episode, row.source_seed,
        row.light_setting_index, row.light_label, row.source_path, row.inference_path
      ].join(' ').toLowerCase();
      return haystack.includes(state.query.toLowerCase());
    }}

    function card(row) {{
      const title = `${{row.task}} / ${{row.run}} / episode${{row.episode}}`;
      const details = {{
        task: row.task,
        run: row.run,
        episode: row.episode,
        source_episode: row.source_episode,
        source_seed: row.source_seed,
        length: row.length,
        supervised_light_setting_index: row.light_setting_index,
        light_setting: row.light_setting,
        scene_info: row.scene_info,
        robotwin_video: row.source_path,
        inference_video: row.inference_path
      }};
      return `<article class="card" data-task="${{esc(row.task)}}" data-setting="${{esc(row.light_setting_index)}}">`
        + `<div class="card-head">`
        + `<div><div class="title"><span>${{esc(title)}}</span><span class="tag">light setting ${{esc(row.light_setting_index)}}</span></div>`
        + `<div class="meta">source episode: ${{esc(row.source_episode)}} · source seed: ${{esc(row.source_seed)}} · frames: ${{esc(row.length)}}</div>`
        + `<div class="pair-actions"><button type="button" data-pair-action="play">播放这对</button><button type="button" class="secondary" data-pair-action="pause">暂停这对</button><button type="button" class="secondary" data-pair-action="restart">从头同步</button></div></div>`
        + `<div class="setting">${{esc(row.light_label)}}</div>`
        + `</div>`
        + `<div class="videos">`
        + `<div class="video-pane"><div class="video-label"><span>RoboTwin light intervention</span><span>episode${{esc(row.episode)}}</span></div><video controls muted loop preload="metadata" src="${{esc(row.source_video)}}"></video></div>`
        + `<div class="video-pane"><div class="video-label"><span>Boundless World Model inference</span><span>episode${{esc(row.episode)}}</span></div><video controls muted loop preload="metadata" src="${{esc(row.inference_video)}}"></video></div>`
        + `</div>`
        + `<details><summary>路径和完整参数</summary><pre>${{esc(JSON.stringify(details, null, 2))}}</pre></details>`
        + `</article>`;
    }}

    function render() {{
      const filtered = rows.filter(rowMatches);
      resultLine.textContent = `显示 ${{filtered.length}} / ${{rows.length}} 条成对视频`;
      if (!filtered.length) {{
        grid.innerHTML = '<div class="empty">没有匹配的任务、episode 或扰动参数。</div>';
        return;
      }}
      grid.innerHTML = filtered.map(card).join('');
    }}

    function visibleVideos() {{
      return [...document.querySelectorAll('video')];
    }}

    function pairVideos(card) {{
      return [...card.querySelectorAll('video')];
    }}

    function playPair(card, restart) {{
      const videos = pairVideos(card);
      if (!videos.length) return;
      const targetTime = restart ? 0 : Math.min(...videos.map(video => video.currentTime || 0));
      videos.forEach(video => {{
        try {{ video.currentTime = targetTime; }} catch (error) {{}}
      }});
      requestAnimationFrame(() => {{
        videos.forEach(video => video.play().catch(() => {{}}));
      }});
    }}

    function pausePair(card) {{
      pairVideos(card).forEach(video => video.pause());
    }}

    queryInput.addEventListener('input', event => {{
      state.query = event.target.value.trim();
      render();
    }});
    taskSelect.addEventListener('change', event => {{
      state.task = event.target.value;
      render();
    }});
    settingSelect.addEventListener('change', event => {{
      state.setting = event.target.value;
      render();
    }});
    document.getElementById('play-visible').addEventListener('click', () => {{
      visibleVideos().forEach(video => video.play().catch(() => {{}}));
    }});
    document.getElementById('pause-visible').addEventListener('click', () => {{
      visibleVideos().forEach(video => video.pause());
    }});
    grid.addEventListener('click', event => {{
      const button = event.target.closest('button[data-pair-action]');
      if (!button) return;
      const card = button.closest('.card');
      if (!card) return;
      const action = button.dataset.pairAction;
      if (action === 'play') playPair(card, false);
      if (action === 'pause') pausePair(card);
      if (action === 'restart') playPair(card, true);
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
    parser.add_argument("--robotwin-data-root", type=Path, default=DEFAULT_ROBOTWIN_DATA_ROOT)
    parser.add_argument("--light-config", type=Path, default=DEFAULT_LIGHT_CONFIG)
    parser.add_argument("--html-path", type=Path, default=DEFAULT_HTML_PATH)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument(
        "--materialize-source-videos",
        choices=("hardlink", "copy", "none"),
        default="hardlink",
        help="Place RoboTwin source videos under asset-root for relative playback.",
    )
    parser.add_argument(
        "--embed-videos",
        action="store_true",
        help="Embed all paired videos as base64 data URIs so the HTML is fully standalone.",
    )
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
