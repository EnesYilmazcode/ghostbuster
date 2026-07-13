import pytest
import numpy as np
from pathlib import Path
from ghostbuster.decoder import decode_ghost_video
from test_generator import generate_motion_masked_video


@pytest.fixture(scope="session")
def test_videos(tmp_path_factory):
    """Generate test videos once per session."""
    tmpdir = tmp_path_factory.mktemp("videos")

    videos = {
        "simple": tmpdir / "simple.mp4",
        "long": tmpdir / "long.mp4",
        "fast": tmpdir / "fast.mp4",
    }

    generate_motion_masked_video(
        videos["simple"], text="HI", velocity=2, num_frames=60, seed=42
    )
    generate_motion_masked_video(
        videos["long"], text="GHOSTBUSTER", velocity=2, num_frames=60, seed=43
    )
    generate_motion_masked_video(
        videos["fast"], text="GO", velocity=3, num_frames=60, seed=44
    )

    return videos


def test_decoder_runs(test_videos):
    """Smoke test: decoder should run without error."""
    mask, debug = decode_ghost_video(test_videos["simple"], num_frames=30)

    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    assert debug["frame_count"] == 30


def test_decoder_output_range(test_videos):
    """Output should be binary (0 or 255)."""
    mask, _ = decode_ghost_video(test_videos["simple"])

    unique_vals = np.unique(mask)
    assert set(unique_vals).issubset({0, 255})


def test_decoder_with_velocity_tuning(test_videos):
    """Decoder should work with different velocity settings."""
    # Fast velocity video should use velocity=3
    mask_fast, _ = decode_ghost_video(
        test_videos["fast"], velocity=3, num_frames=30
    )

    # Should have some text pixels (non-zero)
    assert mask_fast.sum() > 0


def test_decoder_consistency(test_videos):
    """Same video should produce same output."""
    mask1, _ = decode_ghost_video(test_videos["simple"])
    mask2, _ = decode_ghost_video(test_videos["simple"])

    assert np.array_equal(mask1, mask2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
