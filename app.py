import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, Response

from ghostbuster.decoder import decode_ghost_video

app = FastAPI(title="Ghostbuster")

PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ghostbuster</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 760px; margin: 48px auto; padding: 0 16px; }
        form { display: flex; flex-direction: column; gap: 12px; max-width: 360px; }
        label { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
        input, button { padding: 10px; font-size: 15px; }
        button { background: #111; color: #fff; border: 0; cursor: pointer; }
        #result { margin-top: 28px; }
        img { max-width: 100%; background: #000; }
    </style>
</head>
<body>
    <h1>Ghostbuster</h1>
    <p>Upload a motion-masked video to recover the hidden text.</p>

    <form id="form">
        <input type="file" id="video" accept="video/*" required>
        <label>Velocity (px/frame) <input type="number" id="velocity" min="1" max="5" value="2"></label>
        <label>Frames to read <input type="number" id="frames" min="10" max="500" value="60"></label>
        <button type="submit">Decode</button>
    </form>

    <div id="result"></div>

    <script>
        form.onsubmit = async (e) => {
            e.preventDefault();
            const data = new FormData();
            data.append('file', video.files[0]);
            data.append('velocity', velocity.value);
            data.append('num_frames', frames.value);

            result.textContent = 'Decoding...';
            const resp = await fetch('/decode', { method: 'POST', body: data });
            if (!resp.ok) { result.textContent = 'Decode failed.'; return; }
            const url = URL.createObjectURL(await resp.blob());
            result.innerHTML = `<img src="${url}">`;
        };
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/decode")
async def decode(file: UploadFile = File(...), velocity: int = 2, num_frames: int = 60):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        mask = decode_ghost_video(path, velocity=velocity, num_frames=num_frames)
    finally:
        Path(path).unlink(missing_ok=True)

    _, png = cv2.imencode(".png", mask)
    return Response(content=png.tobytes(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
