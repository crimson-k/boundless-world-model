"""CPU acceptance checks for the BWM-to-PEFM Stage 2 boundary."""

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from pefm.models import SpatiallyAlignedAdapter, TemporalSemanticDistillation, TokenAggregator, ActionPredictor
from pefm.training.train_adapter import HiddenBatchDataset

from wan_video_action.loss import FlowMatchSFTLossWanAction
from wan_video_action.parsers import resolve_data_keys
from wan_video_action.pipelines.wan_video_action import model_fn_wan_video_action
from wan_video_action.stage2_semantic import stage2_embedding_loss


class TinyVisual(nn.Module):
    def forward(self, rgb, group_ids):
        signal = rgb.float().mean(dim=(1, 2, 4, 5))
        ids = group_ids[0]
        grouped = signal.new_zeros(signal.shape[0], 21)
        grouped.index_add_(1, ids, signal)
        grouped /= torch.bincount(ids).to(grouped)
        return grouped[:, :, None, None].expand(-1, -1, 300, 4)


class FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.vjepa_adapter = TinyVisual()
        self.token_aggregator = TokenAggregator(token_dim=4, embed_dim=3, num_heads=1)
        self.rgb_grid = self.aligned_grid = (15, 20)


class AddBlock(nn.Module):
    def forward(self, x, context, t_mod, freqs):
        del context, t_mod, freqs
        return x + 1


class TestHead(nn.Module):
    def forward(self, x, timestep):
        del timestep
        return x


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 4
        self.freq_dim = 4
        self.seperated_timestep = True
        self.has_image_input = False
        self.require_vae_embedding = False
        self.require_clip_embedding = False
        self.use_text_embedding = False
        self.has_text_input = True
        self.time_embedding = nn.Linear(4, 4)
        self.time_projection = nn.Linear(4, 24)
        self.blocks = nn.ModuleList([AddBlock(), AddBlock(), AddBlock()])
        self.head = TestHead()
        self.freqs = [torch.zeros(4, 1), torch.zeros(4, 1), torch.zeros(4, 1)]

    def patchify(self, latents):
        return latents.expand(-1, 4, -1, -1, -1)[:, :, :, :1, :1]

    def unpatchify(self, tokens, grid):
        batch, _, dim = tokens.shape
        frames, height, width = grid
        return tokens.transpose(1, 2).reshape(batch, dim, frames, height, width)


class Stage2BoundaryTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def make_distiller():
        teacher = FakeTeacher()
        adapter = SpatiallyAlignedAdapter(4, vjepa_dim=4, hidden_dim=4)
        return teacher, adapter, TemporalSemanticDistillation(teacher, adapter, ActionPredictor(3, 8)).freeze_adapter()

    def test_model_fn_returns_selected_block_hidden(self):
        dit = TinyDiT()
        latents = torch.zeros(1, 1, 2, 2, 2, requires_grad=True)
        output = model_fn_wan_video_action(
            dit,
            latents=latents,
            timestep=torch.ones(1),
            action_emb=torch.zeros(1, 2, 4),
            action_mod_emb=torch.zeros(1, 2, 4),
            fuse_vae_embedding_in_latents=True,
            fused_condition_latent_frames=1,
            return_hidden_block=1,
        )
        self.assertEqual(output["dit_grid"], (2, 1, 1))
        self.assertEqual(output["valid_len"], 2)
        self.assertEqual(tuple(output["hidden"].shape), (1, 2, 4))
        torch.testing.assert_close(output["hidden"], torch.full_like(output["hidden"], 2.0))
        output["velocity"].sum().backward()
        self.assertIsNotNone(latents.grad)
        plain = model_fn_wan_video_action(
            dit,
            latents=latents.detach(),
            timestep=torch.ones(1),
            action_emb=torch.zeros(1, 2, 4),
            action_mod_emb=torch.zeros(1, 2, 4),
            fuse_vae_embedding_in_latents=True,
            fused_condition_latent_frames=1,
        )
        self.assertIsInstance(plain, torch.Tensor)
        self.assertEqual(tuple(plain.shape), (1, 4, 2, 1, 1))

    def test_bwm_loss_reaches_hidden_through_real_pefm_distiller(self):
        torch.manual_seed(4)
        teacher, adapter, distiller = self.make_distiller()
        pipe = SimpleNamespace(
            stage2_distiller=distiller,
            stage2_loss_weight=0.5,
            stage2_align_scale=1.0,
            stage2_pred_scale=1.0,
            stage2_weight_schedule="constant",
        )
        hidden = torch.randn(1, 6300, 4, requires_grad=True)
        rgb = torch.randn(1, 1, 3, 81, 2, 2)
        raw_eef = torch.randn(1, 81, 14)
        loss, metrics = stage2_embedding_loss(
            pipe,
            {
                "hidden": hidden,
                "dit_grid": (21, 15, 20),
                "valid_len": 6300,
            },
            {"stage2_rgb": rgb, "stage2_eef": raw_eef},
            sigma=torch.tensor([0.6]),
        )
        self.assertEqual(float(metrics["stage2_weight"]), 0.5)
        self.assertIn("align_mse_t03", metrics)
        self.assertIn("pred_mse_t20", metrics)
        loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(float(hidden.grad.reshape(1, 21, -1).norm(dim=-1)[0, 19]), 0)
        self.assertEqual(float(hidden.grad.reshape(1, 21, -1).norm(dim=-1)[0, 20]), 0)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in adapter.parameters()))

    def test_flow_loss_and_embedding_losses_share_one_backward(self):
        class Scheduler:
            timesteps = torch.tensor([0.5])
            sigmas = torch.tensor([0.6])

            @staticmethod
            def add_noise(clean, noise, timestep):
                del timestep
                return 0.4 * clean + 0.6 * noise

            @staticmethod
            def training_target(clean, noise, timestep):
                del timestep
                return noise - clean

            @staticmethod
            def training_weight(timestep):
                return torch.ones_like(timestep)

        _, _, distiller = self.make_distiller()
        dit_weight = nn.Parameter(torch.tensor(0.2))

        def model_fn(latents, return_hidden_block, **kwargs):
            del kwargs
            self.assertEqual(return_hidden_block, 1)
            return {
                "velocity": latents * dit_weight,
                "hidden": dit_weight.expand(1, 6300, 4),
                "dit_grid": (21, 15, 20),
                "valid_len": 6300,
            }

        pipe = SimpleNamespace(
            scheduler=Scheduler(),
            torch_dtype=torch.float32,
            device=torch.device("cpu"),
            in_iteration_models=[],
            model_fn=model_fn,
            action_injection_mode="none",
            stage2_distiller=distiller,
            stage2_loss_weight=0.5,
            stage2_align_scale=1.0,
            stage2_pred_scale=1.0,
            stage2_weight_schedule="constant",
            stage2_selected_block=1,
        )
        total = FlowMatchSFTLossWanAction(
            pipe,
            input_latents=torch.randn(1, 1, 21, 1, 1),
            noise=torch.randn(1, 1, 21, 1, 1),
            stage2_rgb=torch.randn(1, 1, 3, 81, 2, 2),
            stage2_eef=torch.randn(1, 81, 14),
        )
        total.backward()
        self.assertIsNotNone(dit_weight.grad)
        self.assertTrue(torch.isfinite(dit_weight.grad))
        self.assertEqual(
            set(pipe.loss_metrics),
            {"flow_loss", "noise_sigma", "align_mse", "pred_mse", "stage2_weight", "stage2_active"}
            | {f"pred_mse_t{step:02d}" for step in range(4, 21)}
            | {f"align_mse_t{step:02d}" for step in range(3, 20)},
        )

    def test_auxiliary_ablation_and_noise_gate(self):
        _, _, distiller = self.make_distiller()
        hidden = torch.randn(1, 6300, 4, requires_grad=True)
        output = dict(hidden=hidden, dit_grid=(21, 15, 20), valid_len=6300)
        inputs = dict(stage2_rgb=torch.randn(1, 1, 3, 81, 2, 2), stage2_eef=torch.randn(1, 81, 14))
        pipe = SimpleNamespace(stage2_distiller=distiller, stage2_loss_weight=.3,
                               stage2_align_scale=1., stage2_pred_scale=1., stage2_weight_schedule="constant")
        # 外层 bf16 autocast 不应改变辅助目标和 Adapter 的计算精度。
        with torch.autocast("cpu", dtype=torch.bfloat16):
            total, metrics = stage2_embedding_loss(pipe, output, inputs, torch.tensor([.2]))
        torch.testing.assert_close(total.detach(), .3 * (metrics["align_mse"] + metrics["pred_mse"]))
        pipe.stage2_pred_scale = 0.
        align, _ = stage2_embedding_loss(pipe, output, inputs, torch.tensor([.2]))
        torch.testing.assert_close(align.detach(), .3 * metrics["align_mse"])
        distiller.sigma_max = .1
        masked, metrics = stage2_embedding_loss(pipe, output, inputs, torch.tensor([.2]))
        self.assertEqual(float(masked.detach()), 0.)
        self.assertEqual(float(metrics["stage2_active"]), 0.)
        masked.backward()
        self.assertEqual(float(hidden.grad.norm()), 0.)

    def test_transfer_keeps_raw_auxiliary_inputs_in_float32(self):
        from scripts.train import WanTrainingModule
        module = WanTrainingModule.__new__(WanTrainingModule)
        value = torch.tensor([1.0001, 1.0002])
        result = module.transfer_data_to_device(
            ({"stage2_eef": value, "stage2_rgb": value, "action": value},),
            torch.device("cpu"), torch.bfloat16,
        )[0]
        self.assertEqual(result["action"].dtype, torch.bfloat16)
        torch.testing.assert_close(result["stage2_eef"], value)
        torch.testing.assert_close(result["stage2_rgb"], value)

    def test_stage2_dataset_keys_include_rgb_and_raw_eef(self):
        args = argparse.Namespace(
            modes={"vae": "emb", "action": "adaln", "text": "none"},
            stage2_loss_weight=1.0,
            stage2_capture=False,
        )
        resolve_data_keys(args)
        self.assertEqual(args.data_keys, ["latents", "action", "video", "stage2_eef"])

    def test_capture_payload_is_consumed_by_pefm_phase_b(self):
        hidden = torch.randn(1, 6300, 4)
        teacher_batch = {
            "rgb": torch.randn(1, 1, 3, 81, 2, 2),
            "group_ids": torch.cat((
                torch.zeros(1, dtype=torch.long),
                torch.arange(1, 21).repeat_interleave(4),
            )).view(1, 81),
            "action_delta": torch.randn(1, 21, 14),
            "reset": torch.tensor([[True] + [False] * 20]),
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            torch.save({
                "hidden": hidden,
                "dit_grid": (21, 15, 20),
                "valid_len": 6300,
                "num_views": 1,
                "teacher_batch": teacher_batch,
            }, directory / "sample.pt")
            (directory / "manifest.jsonl").write_text(
                json.dumps({"path": "sample.pt"}) + "\n", encoding="utf-8",
            )
            payload = HiddenBatchDataset(directory / "manifest.jsonl")[0]
        self.assertEqual(tuple(payload["hidden"].shape), (1, 6300, 4))
        self.assertEqual(tuple(payload["dit_grid"]), (21, 15, 20))


if __name__ == "__main__":
    unittest.main()
