import cv2
import numpy as np


def decode_ghost_video(video_path, block_size=12, velocity=2, num_frames=None):
    """
    Recover motion-masked text from a video.

    For each pair of consecutive frames, correlate them shifted up and down by
    `velocity` pixels. Upward motion scores positive, downward negative. Summed
    over the clip and pooled over `block_size` blocks, the sign of the score
    separates the text (scrolling up) from the background (scrolling down).

    Returns a binary mask (0/255, H x W), white where text was detected.
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
    num, height, width = frames.shape

    # Remove each frame's DC term, otherwise it swamps the correlation.
    frames -= frames.mean(axis=(1, 2), keepdims=True)

    v = velocity
    score = np.zeros((height, width), np.float32)
    for t in range(num - 1):
        a, b = frames[t], frames[t + 1]
        score[v:, :] += a[v:, :] * b[:-v, :]   # matches if content moved up
        score[:-v, :] -= a[:-v, :] * b[v:, :]  # matches if content moved down

    score = cv2.blur(score, (block_size, block_size))
    mask = (score > 0).astype(np.uint8) * 255
    return cv2.medianBlur(mask, 5)
