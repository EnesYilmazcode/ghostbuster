# ghostbuster
Decodes "human-only" motion-masked fonts by tracking per-pixel velocity across frames.

## What is motion-masked text?

A text image is encoded into video by making pixels in the text region move **up** at velocity _v_, and pixels in the background move **down** at velocity _v_. The noise in each region is random, so any single frame looks like pure noise. But across frames, the temporal derivative reveals the mask.

## How it works

**Block matching + accumulation:**
- Stack N frames (30–60 typical)
- For each pixel, correlate frame pairs with ±velocity shifts
- Accumulate evidence across time: upward motion → text (positive), downward → background (negative)
- Threshold and cleanup

**Why it works:** A single frame has terrible SNR (~coin flip). But integrating over T frames and spatial blocks (12×12) pools ~4000 samples, and error falls off like 1/√(samples).

## Quick start

```bash
pip install -r requirements.txt

# Generate test videos
python tests/test_generator.py

# Run tests
pytest tests/ -v

# Start web server
python app.py
# Visit http://localhost:8000
```

## API

### Python
```python
from ghostbuster.decoder import decode_ghost_video

mask, debug = decode_ghost_video("video.mp4", velocity=2, num_frames=60)
# mask: binary segmentation (H×W, 0/255)
# debug: {frame_count, shape, score stats}
```

### Web
```
POST /decode
  file: video file (multipart)
  velocity: int (default 2)
  num_frames: int (default 60)
→ PNG mask
```

## Parameters

- **velocity**: pixels/frame the text/background move in opposite directions. Typically 1–3. Sweep if unsure.
- **num_frames**: frames to read from video. More = better SNR. 30–120 typical.
- **block_size**: spatial pooling kernel. Default 12.

## Gotchas

- **Video compression kills temporal structure.** H.264 MP4s smear noise. Test on originals.
- **Aliasing:** if velocity × frame_rate > 1px, you get temporal aliasing. Sub-sample or downscale if needed.
- **SNR is terrible per-pixel per-frame.** The approach works because it integrates massively. 2–3 frames won't work.

## Why this matters

This isn't an AI-proof font. It's a defense against *single-frame sampling*. Any model trained on temporal video reads it trivially. The fun is showing that concretely, and understanding motion segmentation as a primitive.
