import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def generate_motion_masked_video(
    output_path,
    text="GHOST",
    width=640,
    height=480,
    num_frames=60,
    velocity=2,
    font_size=100,
    seed=42,
):
    """
    Generate a synthetic video with motion-masked text.

    Text region: noise moving UP at velocity v
    Background: noise moving DOWN at velocity v
    Result: text is only readable in video, not in single frames.

    Args:
        output_path: where to save video
        text: text to hide
        width, height: video dimensions
        num_frames: number of frames
        velocity: pixels per frame
        font_size: text size
        seed: random seed for reproducibility
    """
    np.random.seed(seed)

    # Create text mask using PIL
    pil_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(pil_img)

    # Use default font; try to load a better one if available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Get text bounding box to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill=255, font=font)
    text_mask = np.array(pil_img) > 128  # (H, W) boolean

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))

    # Generate frames
    for t in range(num_frames):
        # Random noise for this frame
        noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)

        # Shift noise for text region (UP = increasing indices wrap around)
        # and background (DOWN = decreasing indices wrap around)
        shift = (velocity * t) % height

        frame = noise.copy()

        # Text region: shift noise UP (subtract shift)
        for y in range(height):
            src_y = (y + shift) % height
            frame[y, text_mask[y]] = noise[src_y, text_mask[y]]

        # Background region: shift noise DOWN (add shift)
        for y in range(height):
            src_y = (y - shift) % height
            frame[y, ~text_mask[y]] = noise[src_y, ~text_mask[y]]

        # Convert to BGR for video output
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Generated video: {output_path}")


def generate_test_suite():
    """Generate a suite of test videos."""
    test_dir = Path(__file__).parent / "videos"
    test_dir.mkdir(exist_ok=True)

    # Test case 1: simple text, standard velocity
    generate_motion_masked_video(
        test_dir / "simple.mp4",
        text="HELLO",
        velocity=2,
        num_frames=60,
        seed=42,
    )

    # Test case 2: longer text
    generate_motion_masked_video(
        test_dir / "long.mp4",
        text="GHOSTBUSTER",
        velocity=2,
        num_frames=60,
        seed=43,
    )

    # Test case 3: faster velocity
    generate_motion_masked_video(
        test_dir / "fast.mp4",
        text="FAST",
        velocity=3,
        num_frames=60,
        seed=44,
    )

    # Test case 4: slower velocity
    generate_motion_masked_video(
        test_dir / "slow.mp4",
        text="SLOW",
        velocity=1,
        num_frames=60,
        seed=45,
    )

    # Test case 5: more frames (better SNR)
    generate_motion_masked_video(
        test_dir / "many_frames.mp4",
        text="CLEAR",
        velocity=2,
        num_frames=120,
        seed=46,
    )


if __name__ == "__main__":
    generate_test_suite()
