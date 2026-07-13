import cv2
import numpy as np


def decode_ghost_video(video_path, block_size=12, velocity=2, num_frames=None):
    """
    Decode motion-masked text from video using block matching.

    Args:
        video_path: path to video file
        block_size: correlation block size (12-16 typical)
        velocity: expected pixel velocity per frame (1-3 typical)
        num_frames: max frames to read (None = all)

    Returns:
        segmentation mask (0/255, H×W), and debug dict
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames.append(gray)
        frame_count += 1
        if num_frames and frame_count >= num_frames:
            break

    cap.release()

    if len(frames) < 2:
        raise ValueError("Need at least 2 frames")

    frames = np.stack(frames)  # (T, H, W)
    T, H, W = frames.shape

    # Center frames to remove DC component
    frames = frames - frames.mean(axis=(1, 2), keepdims=True)

    score = np.zeros((H, W), np.float32)

    # Accumulate motion evidence across frame pairs
    for t in range(T - 1):
        a, b = frames[t], frames[t + 1]

        # If pattern moves UP (text region), pixel at y correlates with y-velocity
        # If pattern moves DOWN (background), opposite
        shift = velocity

        # UP motion correlation
        if shift < H:
            up_corr = a[shift:, :] * b[:-shift, :]
            score[shift:, :] += up_corr

        # DOWN motion correlation (subtract to get opposite sign)
        if shift < H:
            down_corr = a[:-shift, :] * b[shift:, :]
            score[:-shift, :] -= down_corr

    # Spatial pooling with box filter
    score = cv2.blur(score, (block_size, block_size))

    # Threshold: positive score = UP motion (text), negative = DOWN motion (background)
    mask = ((score > 0) * 255).astype(np.uint8)

    # Median filter to clean up noise
    mask = cv2.medianBlur(mask, 5)

    debug = {
        "frame_count": T,
        "shape": (H, W),
        "score_min": score.min(),
        "score_max": score.max(),
        "score_mean": score.mean(),
    }

    return mask, debug
