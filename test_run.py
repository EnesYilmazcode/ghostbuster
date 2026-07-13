from ghostbuster.decoder import decode_ghost_video
import cv2

video = "tests/videos/simple.mp4"
mask, debug = decode_ghost_video(video, num_frames=30)

print("OK Decoder ran successfully")
print(f"  Shape: {mask.shape}")
print(f"  Frames read: {debug['frame_count']}")
print(f"  Score range: [{debug['score_min']:.1f}, {debug['score_max']:.1f}]")
print(f"  Non-zero pixels: {(mask > 0).sum()}")

cv2.imwrite("result_simple.png", mask)
print("OK Saved result to result_simple.png")
