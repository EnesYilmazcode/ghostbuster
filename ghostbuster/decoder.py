import cv2
import numpy as np


def decode_ghost_video(video_path, block_size=12, velocity=2, num_frames=None):
    """
    Recover motion-masked text from a video.

    For each pair of consecutive frames, correlate them shifted up and down by
    `velocity` pixels. Upward motion scores positive, downward negative. Summed
    over the clip and pooled over `block_size` blocks, the sign of the score
    separates the text (scrolling up) from the background (scrolling down).

    Returns a binary mask (0/255, H x W) and a dict of debug stats.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while num_frames is None or len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()

    if len(frames) < 2:
        raise ValueError("need at least 2 frames to decode")

    frames = np.stack(frames)
    T, H, W = frames.shape

    # Remove each frame's DC term, otherwise it swamps the correlation.
    frames -= frames.mean(axis=(1, 2), keepdims=True)

    s = velocity
    score = np.zeros((H, W), np.float32)
    for t in range(T - 1):
        a, b = frames[t], frames[t + 1]
        score[s:, :] += a[s:, :] * b[:-s, :]   # matches if content moved up
        score[:-s, :] -= a[:-s, :] * b[s:, :]  # matches if content moved down

    score = cv2.blur(score, (block_size, block_size))
    mask = ((score > 0) * 255).astype(np.uint8)
    mask = cv2.medianBlur(mask, 5)

    debug = {
        "frame_count": T,
        "shape": (H, W),
        "score_min": float(score.min()),
        "score_max": float(score.max()),
    }
    return mask, debug
