import numpy as np
import pytest

from ghostbuster.decoder import decode_ghost_video
from ghostbuster.encoder import generate_motion_masked_video


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
    detected = decode_ghost_video(path)
    assert set(np.unique(detected)).issubset({0, 255})


def test_recovers_text(clip):
    path, mask = clip
    detected = decode_ghost_video(path, velocity=2, num_frames=60)
    assert iou(mask, detected > 0) > 0.8


def test_deterministic(clip):
    path, _ = clip
    assert np.array_equal(decode_ghost_video(path), decode_ghost_video(path))


@pytest.mark.parametrize("velocity", [1, 2, 3])
def test_recovers_across_velocities(tmp_path, velocity):
    path = tmp_path / f"v{velocity}.mp4"
    mask = generate_motion_masked_video(path, text="OK", velocity=velocity, num_frames=60, seed=1)
    detected = decode_ghost_video(path, velocity=velocity, num_frames=60)
    assert iou(mask, detected > 0) > 0.7
