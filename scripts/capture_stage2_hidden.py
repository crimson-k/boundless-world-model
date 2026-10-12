"""Capture noisy DiT block states and matching GT batches for PEFM Phase B."""

import hashlib
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train import build_training_module, wan_parser
from wan_video_action.data import build_train_dataset
from wan_video_action.loss import flow_match_forward
from wan_video_action.parsers import merge_yaml_and_args, prepare_model_config, resolve_data_keys
from wan_video_action.stage2_semantic import build_teacher_batch, noise_sampling_contract
from wan_video_action.utils import set_global_seed


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return value


def main():
    parser = wan_parser()
    parser.add_argument("--capture_output", required=True)
    parser.add_argument("--capture_selected_block", required=True, type=int)
    parser.add_argument("--capture_max_samples", type=int, default=0)
    parser.add_argument("--capture_timestep_id", type=int, default=None)
    cli = parser.parse_args()
    cli_args = __import__("sys").argv[1:]
    args = merge_yaml_and_args(cli.config, parser, cli, cli_args)
    args.stage2_capture = True
    args.stage2_loss_weight = 0.0
    args.evaluator_loss_weight = 0.0
    args = resolve_data_keys(args, stage="train")
    model_config = prepare_model_config(args)
    set_global_seed(args.seed, deterministic=args.deterministic)

    dataset = build_train_dataset(args, repeat=1)
    model = build_training_module(args, model_config, device=torch.device("cuda"))
    model.eval()
    pipe = model.pipe

    output = Path(args.capture_output)
    output.mkdir(parents=True, exist_ok=True)
    action_state = _cpu(pipe.action_encoder.state_dict())
    action_digest = hashlib.sha256()
    for key, value in sorted(action_state.items()):
        action_digest.update(key.encode())
        action_digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    torch.save(action_state, output / "action_encoder_initial.pt")
    manifest_path = output / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing capture manifest: {manifest_path}")
    sample_count = len(dataset)
    if args.capture_max_samples:
        sample_count = min(sample_count, int(args.capture_max_samples))

    with manifest_path.open("w", encoding="utf-8") as manifest, torch.no_grad():
        for index in range(sample_count):
            sample = dataset[index]
            inputs = model.get_pipeline_inputs(sample)
            inputs = model.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
            for unit in pipe.units:
                inputs = pipe.unit_runner(unit, pipe, *inputs)
            shared, positive, _ = inputs
            result = flow_match_forward(
                pipe,
                return_hidden_block=args.capture_selected_block,
                timestep_id=args.capture_timestep_id,
                **shared,
                **positive,
            )
            model_output = result["model_output"]
            teacher_batch = build_teacher_batch(
                shared["stage2_rgb"], shared["stage2_eef"],
            )
            payload = {
                "task": sample["task"],
                "sample_index": index,
                "source_video": dataset.data[index]["video"],
                "source_action": dataset.data[index]["action"],
                "frame_indices": sample.get("frame_indices"),
                "metadata_path": str(Path(args.dataset_metadata_path).resolve()),
                "model_paths": str(Path(args.model_paths).resolve()),
                "model_seed": args.seed,
                "action_encoder_sha256": action_digest.hexdigest(),
                "hidden": model_output["hidden"],
                "dit_grid": tuple(model_output["dit_grid"]),
                "valid_len": int(model_output["valid_len"]),
                "num_views": 1,
                "teacher_batch": teacher_batch,
                "timestep_id": result["timestep_id"],
                "noise_sigma": result["sigma"],
                "selected_block": int(args.capture_selected_block),
                "noise_sampling": noise_sampling_contract(
                    args.min_timestep_boundary, args.max_timestep_boundary,
                ),
            }
            path = output / f"sample_{index:06d}.pt"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing hidden batch: {path}")
            torch.save(_cpu(payload), path)
            manifest.write(json.dumps({"path": path.name}) + "\n")
            manifest.flush()
            print(
                f"[capture] {index + 1}/{sample_count} path={path} "
                f"hidden={tuple(payload['hidden'].shape)} sigma={float(result['sigma'].mean()):.6f}",
                flush=True,
            )

    print(f"[capture] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
