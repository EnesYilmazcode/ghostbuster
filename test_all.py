from ghostbuster.decoder import decode_ghost_video
from pathlib import Path
import cv2

test_videos = [
    ("simple", "tests/videos/simple.mp4", 2),
    ("long", "tests/videos/long.mp4", 2),
    ("fast", "tests/videos/fast.mp4", 3),
    ("slow", "tests/videos/slow.mp4", 1),
    ("many_frames", "tests/videos/many_frames.mp4", 2),
]

print("Testing decoder on all synthetic videos...\n")

for name, path, velocity in test_videos:
    mask, debug = decode_ghost_video(path, velocity=velocity, num_frames=60)

    percent_text = 100 * (mask > 0).sum() / mask.size
    print(f"{name:15} | Shape: {mask.shape} | Text pixels: {percent_text:.1f}% | Score: [{debug['score_min']:.0f}, {debug['score_max']:.0f}]")

    cv2.imwrite(f"result_{name}.png", mask)

print("\nAll results saved.")
