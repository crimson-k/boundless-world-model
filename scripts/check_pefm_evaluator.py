"""One-real-batch acceptance check for the frozen PEFM loss."""

import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train import WanTrainingModule, wan_parser
from wan_video_action import loss as loss_module
from wan_video_action.data import build_train_dataset
from wan_video_action.evaluator_loss import prior_feature_matching_loss
from wan_video_action.parsers import merge_yaml_and_args, prepare_model_config, resolve_data_keys


def clone_tensors(values):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in values.items()}


def main():
    parser = wan_parser()
    cli = parser.parse_args()
    cli_args = __import__("sys").argv[1:]
    args = merge_yaml_and_args(cli.config, parser, cli, cli_args)
    model_config = prepare_model_config(args)
    args = resolve_data_keys(args, stage="train")
    args.evaluator_loss_weight = 1.0

    model = WanTrainingModule(
        model_paths=json.dumps(model_config["model_paths_list"]),
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=model_config["tokenizer_path"],
        trainable_models=",".join(args.trainable),
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=torch.device("cuda"),
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_history_frames=args.num_history_frames,
        history_template_sampling=args.history_template_sampling,
        action_dim=args.action_dim,
        action_mode=args.modes["action"],
        args=args,
    )
    pipe = model.pipe
    sample = build_train_dataset(args)[0]

    video = sample["video"].unsqueeze(0).to(pipe.device, pipe.torch_dtype)
    action = torch.as_tensor(sample["action"], device=pipe.device, dtype=pipe.torch_dtype)
    identity = prior_feature_matching_loss(
        pipe.evaluator, video, video, action, args.num_history_frames,
    )["loss"]
    assert identity.item() < 1e-5, identity.item()

    inputs = model.get_pipeline_inputs(sample)
    inputs = model.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
    for unit in pipe.units:
        inputs = pipe.unit_runner(unit, pipe, *inputs)
    shared, positive, _ = inputs
    shared_for_zero = clone_tensors(shared)
    captured = {}

    original_model_fn = pipe.model_fn
    original_target = pipe.scheduler.training_target
    original_weight = pipe.scheduler.training_weight
    original_decode = loss_module.decode_clean_video

    def capture_model_fn(**kwargs):
        velocity = original_model_fn(**kwargs)
        captured["velocity"] = velocity
        return velocity

    def capture_target(*values, **kwargs):
        target = original_target(*values, **kwargs)
        captured["target"] = target
        return target

    def capture_weight(*values, **kwargs):
        weight = original_weight(*values, **kwargs)
        captured["weight"] = weight
        return weight

    def capture_decode(pipe_, history, future, num_views):
        if future.requires_grad and "pred_clean" not in captured:
            future.retain_grad()
            captured["pred_clean"] = future
        return original_decode(pipe_, history, future, num_views)

    pipe.model_fn = capture_model_fn
    pipe.scheduler.training_target = capture_target
    pipe.scheduler.training_weight = capture_weight
    loss_module.decode_clean_video = capture_decode
    seed = 123
    torch.manual_seed(seed)
    total = loss_module.FlowMatchSFTLossWanAction(pipe, **shared, **positive)

    history_t = int(shared.get("fused_condition_latent_frames") or 0)
    velocity = captured["velocity"][:, :, history_t:]
    target = captured["target"][:, :, history_t:]
    flow = torch.nn.functional.mse_loss(velocity.float(), target.float()) * captured["weight"]
    interaction = total - flow
    interaction.backward()

    evaluator_grad = any(parameter.grad is not None for parameter in pipe.evaluator.parameters())
    pred_grad = captured["pred_clean"].grad
    dit_grad = [parameter.grad for parameter in pipe.dit.parameters() if parameter.grad is not None]
    assert not evaluator_grad
    assert pred_grad is not None and torch.count_nonzero(pred_grad) > 0
    assert dit_grad and any(torch.count_nonzero(grad) > 0 for grad in dit_grad)

    pipe.evaluator_loss_weight = 0.0
    pipe.model_fn = lambda **kwargs: captured["velocity"].detach()
    torch.manual_seed(seed)
    zero_weight = loss_module.FlowMatchSFTLossWanAction(pipe, **shared_for_zero, **positive)
    assert torch.equal(zero_weight, flow.detach()), (zero_weight.item(), flow.item())

    result = {
        "sample": int(sample["source_episode_index"]),
        "identity_loss": identity.item(),
        "flow_loss": flow.item(),
        "interaction_loss": interaction.item(),
        "pred_clean_grad_norm": pred_grad.float().norm().item(),
        "dit_grad_tensors": len(dit_grad),
        "evaluator_has_grad": evaluator_grad,
        "zero_weight_equals_flow": True,
    }
    output = Path(args.output_path) / "pefm_evaluator_acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
