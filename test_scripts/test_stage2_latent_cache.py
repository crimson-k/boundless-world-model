"""缓存保留原始窗口、独立历史编码和每次重新采样的噪声。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from wan_video_action.data.wan_dataset import RoboTwinUnifiedDataset
from wan_video_action.data.latent_cache import metadata_digest, source_fingerprint
from wan_video_action.pipelines.wan_video_action import WanVideoUnit_ImageEmbedderFused


class LatentCacheTest(unittest.TestCase):
    def test_dataset_skips_video_and_rejects_changed_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "train.jsonl"
            metadata.write_text(json.dumps({"video": "unused.mp4", "start_frame": 0, "end_frame": 80}) + "\n")
            folder = root / "cache" / "train"
            folder.mkdir(parents=True)
            (root / "unused.mp4").write_bytes(b"test source")
            settings = {"test": True}
            (folder / "manifest.json").write_text(json.dumps({"settings": settings, "metadata_sha256": metadata_digest(metadata), "source_fingerprint": source_fingerprint(metadata, root)}))
            payload = {"input_latents": torch.ones(1, 48, 21, 2, 2), "history_latents": torch.full((1, 48, 3, 2, 2), 2.), "frame_indices": None}
            torch.save(payload, folder / "000000.pt")
            def forbidden_video_decode(data):
                raise AssertionError("Cached dataset must not decode video")
            dataset = RoboTwinUnifiedDataset(str(root), str(metadata), data_file_keys=["video"], main_data_operator=forbidden_video_decode, vae_latent_cache_dir=root / "cache", vae_cache_settings=settings)
            torch.testing.assert_close(dataset[0]["cached_vae_latents"]["history_latents"], payload["history_latents"])
            metadata.write_text(metadata.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                RoboTwinUnifiedDataset(str(root), str(metadata), vae_latent_cache_dir=root / "cache", vae_cache_settings=settings)

    def test_history_uses_separate_encoding_and_fresh_noise(self):
        unit = WanVideoUnit_ImageEmbedderFused()
        pipe = SimpleNamespace(dit=SimpleNamespace(fuse_vae_embedding_in_latents=True), device="cpu", torch_dtype=torch.float32, action_injection_mode="adaln", scheduler=SimpleNamespace(training=True, sigmas=torch.linspace(1., 0., 1000)))
        full = torch.zeros(1, 48, 21, 2, 2)
        history = torch.full((1, 48, 3, 2, 2), 2.)
        outputs = []
        for noise_value in [0., 1.]:
            outputs.append(unit.process(pipe, input_video=None, precomputed_latents=full, precomputed_history_latents=history, input_latents=full, latents=torch.zeros_like(full), noise=torch.full_like(full, noise_value), num_history_frames=9, tiled=False, tile_size=None, tile_stride=None))
        torch.testing.assert_close(outputs[0]["history_condition_latents"][:, :, :1], history[:, :, :1])
        self.assertFalse(torch.equal(outputs[0]["history_condition_latents"][:, :, 1:], outputs[1]["history_condition_latents"][:, :, 1:]))
        torch.testing.assert_close(history, torch.full_like(history, 2.))


if __name__ == "__main__":
    unittest.main()
