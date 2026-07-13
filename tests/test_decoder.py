import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ghostbuster.decoder import decode_ghost_video
from tests.test_generator import generate_motion_masked_video


def iou(ground_truth, predicted):
    inter = np.logical_and(ground_truth, predicted).sum()
    union = np.logical_or(ground_truth, predicted).sum()
    return inter / union if union else 0.0


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("videos") / "clip.mp4"
    mask = generate_motion_masked_video(path, text="GHOST", velocity=2, num_frames=60, seed=7)
    return path, mask


def test_output_is_binary(clip):
    path, _ = clip
    detected, _ = decode_ghost_video(path)
    assert set(np.unique(detected)).issubset({0, 255})


def test_recovers_text(clip):
    path, mask = clip
    detected, _ = decode_ghost_video(path, velocity=2, num_frames=60)
    assert iou(mask, detected > 0) > 0.8


def test_deterministic(clip):
    path, _ = clip
    a, _ = decode_ghost_video(path)
    b, _ = decode_ghost_video(path)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("velocity", [1, 2, 3])
def test_recovers_across_velocities(tmp_path, velocity):
    path = tmp_path / f"v{velocity}.mp4"
    mask = generate_motion_masked_video(path, text="OK", velocity=velocity, num_frames=60, seed=1)
    detected, _ = decode_ghost_video(path, velocity=velocity, num_frames=60)
    assert iou(mask, detected > 0) > 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
