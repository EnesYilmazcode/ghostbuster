import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _text_mask(text, width, height, font_size):
    """Boolean mask (H, W), True inside the glyphs."""
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2 - bbox[0]
    y = (height - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=255, font=font)

    return np.array(img) > 128


def _binary_noise(rng, height, width, grain):
    """Black/white noise field in square chunks of `grain` pixels."""
    if grain <= 1:
        return rng.integers(0, 2, (height, width), dtype=np.uint8) * 255
    gh, gw = math.ceil(height / grain), math.ceil(width / grain)
    small = rng.integers(0, 2, (gh, gw), dtype=np.uint8) * 255
    big = np.repeat(np.repeat(small, grain, axis=0), grain, axis=1)
    return big[:height, :width]


def render_frames(text="GHOST", width=640, height=480, num_frames=60,
                  velocity=2, font_size=150, grain=2, seed=42):
    """
    Render a motion-masked clip. Returns (frames, text_mask).

    A single noise canvas scrolls behind the frame: up inside the glyphs, down
    everywhere else. Any single frame is uniform noise, so the text is invisible
    in a screenshot and only the motion reveals it.
    """
    rng = np.random.default_rng(seed)
    mask = _text_mask(text, width, height, font_size)

    scroll = velocity * (num_frames - 1)
    canvas = _binary_noise(rng, height + scroll + 1, width, grain)

    frames = []
    for t in range(num_frames):
        up = velocity * t                # glyphs scroll up
        down = scroll - velocity * t     # background scrolls down
        frames.append(np.where(mask, canvas[up:up + height], canvas[down:down + height]))

    return frames, mask


def generate_motion_masked_video(output_path, **kwargs):
    """Write a motion-masked video and return its ground-truth text mask."""
    frames, mask = render_frames(**kwargs)
    height, width = frames[0].shape

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return mask
