"""两个训练阶段、最终向量场恢复 latent、冻结 evaluator 的梯度边界。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.models.wan_video_dit import WanModel
from pefm.export import teacher_contract
from pefm.models import LatentSemanticDistillation, build_evaluator
from wan_video_action.loss import FlowMatchSFTLossWanAction, flow_match_forward
from wan_video_action.parsers import add_stage2_config, resolve_data_keys
from wan_video_action.pipelines.wan_video_action import model_fn_wan_video_action
from wan_video_action.stage2_semantic import (
    build_teacher_batch, initialize_stage2_distiller, stage2_prior_loss,
)


ROOT = Path(__file__).resolve().parents[1]


def evaluator_config():
    return OmegaConf.create({
        "device": "cpu", "height": 192, "width": 256, "resize_mode": "crop",
        "model": {"rssm": {
            "stoch": 2, "deter": 64, "hidden": 32, "discrete": 4,
            "img_layers": 2, "obs_layers": 1, "dyn_layers": 1, "blocks": 8,
            "act": "SiLU", "unimix_ratio": 0.01, "initial": "zeros", "device": "cpu",
        }},
    })


def make_distiller():
    torch.manual_seed(4)
    return LatentSemanticDistillation(build_evaluator(evaluator_config()))


def make_scheduler():
    scheduler = FlowMatchScheduler(template="Wan")
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def make_pipe(model_fn, distiller=None):
    return SimpleNamespace(
        scheduler=make_scheduler(), model_fn=model_fn, in_iteration_models=[],
        torch_dtype=torch.float32, device=torch.device("cpu"), action_injection_mode="adaln",
        stage2_distiller=distiller, stage2_loss_weight=0.5 if distiller is not None else 0.0,
        stage2_weight_schedule="constant",
    )


def test_oracle_final_velocity_recovers_gt_with_bf16_timestep_rounding():
    clean = torch.randn(1, 48, 21, 2, 2)
    noise = torch.randn_like(clean)
    pipe = make_pipe(lambda **kwargs: noise - clean)
    pipe.torch_dtype = torch.bfloat16
    times = pipe.scheduler.timesteps
    rounded_indices = (times[:, None] - times.to(torch.bfloat16)[None, :]).abs().argmin(0)
    index = int(torch.nonzero(rounded_indices != torch.arange(len(times)))[0])
    output = flow_match_forward(pipe, timestep_id=index, input_latents=clean, noise=noise)
    torch.testing.assert_close(output["clean_latents"], clean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(output["flow_loss"], torch.zeros_like(output["flow_loss"]))
    torch.testing.assert_close(output["sigma"][0], pipe.scheduler.sigmas[rounded_indices[index]])


def test_history_is_copied_and_final_velocity_shape_is_checked():
    clean = torch.randn(1, 48, 21, 2, 2)
    noise = torch.randn_like(clean)
    velocity = torch.randn_like(clean, requires_grad=True)
    history = clean[:, :, :3].clone()
    pipe = make_pipe(lambda **kwargs: velocity)
    output = flow_match_forward(
        pipe, timestep_id=100, input_latents=clean, noise=noise,
        history_condition_latents=history, fused_condition_latent_frames=3,
    )
    assert output["clean_latents"].shape == clean.shape
    torch.testing.assert_close(output["clean_latents"][:, :, :3], history)
    output["clean_latents"].sum().backward()
    assert velocity.grad[:, :, :3].count_nonzero() == 0
    assert velocity.grad[:, :, 3:].abs().sum() > 0
    pipe.model_fn = lambda **kwargs: velocity[:, :47]
    with pytest.raises(ValueError, match="Final DiT velocity must match"):
        flow_match_forward(pipe, input_latents=clean, noise=noise)


def test_frozen_evaluator_prior_gradient_has_correct_time_alignment():
    distiller = make_distiller().train()
    target = torch.randn(1, 48, 21, 2, 2, requires_grad=True)
    generated = torch.randn_like(target, requires_grad=True)
    pipe = make_pipe(None, distiller)
    loss, weight, per_step = stage2_prior_loss(
        pipe, generated, {"input_latents": target, "stage2_eef": torch.randn(1, 81, 14)},
        sigma=torch.tensor([0.6]),
    )
    assert per_step.shape == (1, 17) and float(weight) == 0.5
    loss.backward()
    assert target.grad is None
    assert generated.grad[:, :, :3].count_nonzero() == 0
    assert generated.grad[:, :, 19].abs().sum() > 0
    assert generated.grad[:, :, 20].count_nonzero() == 0
    assert all(p.grad is None and not p.requires_grad for p in distiller.parameters())
    assert all(not module.training for module in distiller.teacher.modules())


def test_bwm_autocast_preserves_fp32_evaluator_and_input_gradients():
    distiller = make_distiller()
    pipe = make_pipe(None, distiller)
    generated = torch.randn(1, 48, 21, 2, 2, requires_grad=True)
    inputs = {"input_latents": torch.randn_like(generated), "stage2_eef": torch.randn(1, 81, 14)}
    dtypes = []
    hook = distiller.teacher.prior_prediction_head.register_forward_hook(
        lambda module, args, output: dtypes.append(output.dtype),
    )
    sigma = torch.tensor([0.6])
    baseline = stage2_prior_loss(pipe, generated, inputs, sigma)[0]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = stage2_prior_loss(pipe, generated, inputs, sigma)[0]
    hook.remove()
    torch.testing.assert_close(loss, baseline)
    assert dtypes == [torch.float32] * 4
    loss.backward()
    assert torch.isfinite(generated.grad).all() and generated.grad.abs().sum() > 0


@pytest.mark.parametrize("include_flow", [False, True])
def test_real_dit_final_head_backpropagates_through_real_latent_evaluator(include_flow):
    dit = WanModel(
        dim=32, in_dim=48, ffn_dim=64, out_dim=48, text_dim=32, freq_dim=16,
        eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2,
        has_image_input=False, seperated_timestep=True,
        require_vae_embedding=False, require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
    )
    distiller = make_distiller()
    pipe = make_pipe(
        lambda **kwargs: model_fn_wan_video_action(dit, **kwargs), distiller,
    )
    clean = torch.randn(1, 48, 21, 4, 6)
    inputs = dict(
        input_latents=clean, noise=torch.randn_like(clean), stage2_eef=torch.randn(1, 81, 14),
        history_condition_latents=clean[:, :, :3], fused_condition_latent_frames=3,
        fuse_vae_embedding_in_latents=True,
        action_emb=torch.zeros(1, 21, 32), action_mod_emb=torch.zeros(1, 21, 32),
    )
    seen = []
    hook = distiller.teacher.token_aggregator.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].shape),
    )
    if include_flow:
        loss = FlowMatchSFTLossWanAction(pipe, **inputs)
        assert {"flow_loss", "noise_sigma", "prior_kl", "stage2_weight"} <= set(pipe.loss_metrics)
    else:
        output = flow_match_forward(pipe, timestep_id=500, **inputs)
        assert output["clean_latents"].shape == clean.shape
        loss = stage2_prior_loss(pipe, output["clean_latents"], inputs, output["sigma"])[0]
    loss.backward()
    hook.remove()
    assert seen == [torch.Size([1, 21, 24, 48]), torch.Size([1, 21, 24, 48])]
    assert torch.isfinite(loss)
    assert dit.head.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in distiller.parameters())


def test_gt_and_recovered_latents_must_have_identical_vae_shape():
    distiller = make_distiller()
    latents = torch.randn(1, 48, 21, 2, 2)
    actions = torch.randn(1, 21, 14)
    reset = torch.zeros(1, 21, dtype=torch.bool)
    with pytest.raises(ValueError, match="match GT VAE latents exactly"):
        distiller(latents[..., :1], latents, actions, reset)
    with pytest.raises(ValueError, match="Expected clean latents"):
        distiller(latents[:, :47], latents[:, :47], actions, reset)


def test_actions_use_group_endpoints_without_rgb_input():
    latents = torch.zeros(1, 48, 21, 2, 2)
    eef = torch.zeros(81, 14)
    eef[:, 0] = torch.arange(81).float()
    batch = build_teacher_batch(latents, eef)
    assert "rgb" not in batch
    torch.testing.assert_close(batch["action_delta"][0, 1:, 0], torch.full((20,), 4.0))
    grouped = build_teacher_batch(latents, eef[::4])
    torch.testing.assert_close(grouped["action_delta"], batch["action_delta"])
    assert batch["reset"].shape == (1, 21) and batch["reset"][0, 0]


def test_initializer_loads_only_frozen_evaluator_without_registering_it(tmp_path):
    cfg = evaluator_config()
    teacher = build_evaluator(cfg)
    path = tmp_path / "frozen.pt"
    torch.save({
        "format": "pefm-frozen-evaluator-v1", "config": OmegaConf.to_container(cfg),
        "model": teacher.state_dict(), "semantic_teacher": teacher_contract(cfg), "epoch": 1,
    }, path)
    pipe = nn.Module()
    pipe.device = "cpu"
    initialize_stage2_distiller(pipe, teacher_path=path, loss_weight=0.1)
    assert pipe.stage2_distiller is not None
    assert not hasattr(pipe.stage2_distiller, "adapter")
    assert list(pipe.named_parameters()) == []
    assert all(not p.requires_grad for p in pipe.stage2_distiller.parameters())
    initialize_stage2_distiller(pipe, teacher_path=None, loss_weight=0)
    assert pipe.stage2_distiller is None


@pytest.mark.parametrize("vae,expected", [
    ("emb", ["latents", "action", "stage2_eef"]),
    ("raw", ["video", "action", "stage2_eef"]),
])
def test_stage2_dataset_keys_need_only_latents_and_raw_eef(vae, expected):
    args = argparse.Namespace(modes={"vae": vae, "action": "adaln", "text": "none"}, stage2_loss_weight=1)
    assert resolve_data_keys(args).data_keys == expected
    parser = add_stage2_config(argparse.ArgumentParser())
    assert not hasattr(parser.parse_args([]), "stage2_adapter_path")
    assert not hasattr(parser.parse_args([]), "stage2_selected_block")


def test_shell_entry_runs_exactly_evaluator_then_bwm(tmp_path):
    calls = tmp_path / "calls.jsonl"
    executable = tmp_path / "python_stub"
    executable.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "with open(os.environ['STAGE2_TEST_CALLS'],'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:])+'\\n')\n",
    )
    executable.chmod(0o755)
    bundle = tmp_path / "frozen.pt"
    bundle.touch()
    env = dict(os.environ, PYTHON_BIN=str(executable), TEACHER_BUNDLE=str(bundle), STAGE2_TEST_CALLS=str(calls),
               BWM_BATCH_SIZE="4", BWM_GRADIENT_ACCUMULATION_STEPS="8")
    subprocess.run(["bash", str(ROOT / "scripts/train_stage2.sh"), "all"], env=env, check=True, capture_output=True)
    arguments = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(arguments) == 2
    assert "pefm.training.train" in arguments[0]
    assert "scripts/train.py" in arguments[1]
    assert arguments[1][arguments[1].index("--batch_size") + 1] == "1"
    assert arguments[1][arguments[1].index("--gradient_accumulation_steps") + 1] == "8"
    assert not any("adapter" in arg or "selected_block" in arg for args in arguments for arg in args)
    for old_stage in ("adapter", "capture"):
        assert subprocess.run(["bash", str(ROOT / "scripts/train_stage2.sh"), old_stage], env=env, capture_output=True).returncode == 2
