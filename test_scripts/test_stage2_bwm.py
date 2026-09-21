"""CPU acceptance checks for the BWM-to-PEFM Stage 2 boundary."""

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from pefm.models import SpatiallyAlignedAdapter, TemporalSemanticDistillation, TokenAggregator
from pefm.training.train_adapter import HiddenBatchDataset, _unpack_batch

from wan_video_action.loss import FlowMatchSFTLossWanAction
from wan_video_action.parsers import resolve_data_keys
from wan_video_action.pipelines.wan_video_action import model_fn_wan_video_action
from wan_video_action.stage2_semantic import stage2_prior_loss


class FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_aggregator = TokenAggregator(token_dim=4, embed_dim=3, num_heads=1)
        self.state_head = nn.Linear(3, 6)
        self.action_head = nn.Linear(14, 6)

    def _posterior(self, embed):
        return self.state_head(embed).reshape(embed.shape[0], embed.shape[1], 2, 3)

    def forward(self, batch):
        batch_size = batch["rgb"].shape[0]
        frame_signal = batch["rgb"].float().mean(dim=(1, 2, 4, 5))
        ids = batch["group_ids"][0].to(frame_signal.device)
        grouped = frame_signal.new_zeros(batch_size, 21)
        grouped.index_add_(1, ids, frame_signal)
        grouped = grouped / torch.bincount(ids).to(grouped).view(1, -1)
        visual_grid = grouped[:, :, None, None, None, None].expand(-1, -1, 1, 15, 20, 4)
        embed = self.token_aggregator(visual_grid.flatten(2, 4))
        return {
            "visual_grid": visual_grid,
            "embed": embed,
            "posterior_logits": self._posterior(embed),
        }

    def forward_embeddings(self, embed, actions, reset):
        del reset
        posterior = self._posterior(embed)
        action_logits = self.action_head(actions).reshape(actions.shape[0], actions.shape[1], 2, 3)
        prior = torch.cat((posterior[:, :1], posterior[:, :-1] + action_logits[:, :-1]), dim=1)
        return {"embed": embed, "posterior_logits": posterior, "prior_logits": prior}


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
    @staticmethod
    def make_distiller():
        teacher = FakeTeacher()
        adapter = SpatiallyAlignedAdapter(4, vjepa_dim=4, hidden_dim=4)
        return teacher, adapter, TemporalSemanticDistillation(teacher, adapter).freeze_adapter()

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
            stage2_weight_schedule="constant",
        )
        hidden = torch.randn(1, 6300, 4, requires_grad=True)
        rgb = torch.randn(1, 1, 3, 81, 2, 2)
        raw_eef = torch.randn(1, 81, 14)
        loss, weight, per_step = stage2_prior_loss(
            pipe,
            {
                "hidden": hidden,
                "dit_grid": (21, 15, 20),
                "valid_len": 6300,
            },
            {"stage2_rgb": rgb, "stage2_eef": raw_eef},
            sigma=torch.tensor([0.6]),
        )
        self.assertEqual(tuple(per_step.shape), (1, 17))
        self.assertEqual(float(weight), 0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(float(hidden.grad.reshape(1, 21, -1).norm(dim=-1)[0, 19]), 0)
        self.assertEqual(float(hidden.grad.reshape(1, 21, -1).norm(dim=-1)[0, 20]), 0)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in adapter.parameters()))

    def test_flow_loss_and_pefm_prior_loss_share_one_backward(self):
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
            {"flow_loss", "noise_sigma", "prior_kl", "stage2_weight"}
            | {f"prior_kl_t{step:02d}" for step in range(4, 21)},
        )

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
            unpacked = _unpack_batch(payload)
        self.assertEqual(tuple(unpacked[0].shape), (1, 6300, 4))
        self.assertEqual(unpacked[1:4], ((21, 15, 20), 6300, 1))


if __name__ == "__main__":
    unittest.main()
