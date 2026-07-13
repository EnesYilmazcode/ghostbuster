"""Render the demo assets used in the README."""

import sys
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ghostbuster.decoder import decode_ghost_video
from tests.test_generator import render_frames

TEXT = "GHOST"
VELOCITY = 2
FRAMES = 60
GIF_SCALE = 0.5
GIF_FRAMES = 30


def main():
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)

    frames, _ = render_frames(text=TEXT, velocity=VELOCITY, num_frames=FRAMES, seed=7)
    height, width = frames[0].shape

    video = assets / "demo.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()

    cv2.imwrite(str(assets / "frame.png"), frames[len(frames) // 2])

    size = (int(width * GIF_SCALE), int(height * GIF_SCALE))
    gif = [Image.fromarray(f).resize(size, Image.NEAREST).convert("P", palette=Image.ADAPTIVE, colors=2)
           for f in frames[:GIF_FRAMES]]
    gif[0].save(assets / "noise.gif", save_all=True, append_images=gif[1:],
                duration=50, loop=0, optimize=True)

    detected, _ = decode_ghost_video(str(video), velocity=VELOCITY, num_frames=FRAMES)
    cv2.imwrite(str(assets / "detected.png"), detected)

    print("wrote:", ", ".join(sorted(p.name for p in assets.iterdir())))


if __name__ == "__main__":
    main()
