import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import cv2
from ghostbuster.decoder import decode_ghost_video

app = FastAPI(title="Ghostbuster")


@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve simple HTML upload form."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ghostbuster</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 50px auto; }
            h1 { color: #333; }
            form { display: flex; flex-direction: column; gap: 10px; }
            input[type="file"] { padding: 10px; border: 1px solid #ccc; }
            button { padding: 10px; background: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            #result { margin-top: 30px; }
            img { max-width: 100%; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>🐻 Ghostbuster</h1>
        <p>Upload a motion-masked video to reveal the hidden text.</p>

        <form id="uploadForm">
            <input type="file" id="videoFile" accept="video/*" required>
            <input type="number" id="velocity" min="1" max="5" value="2" placeholder="Velocity (pixels/frame)">
            <input type="number" id="frames" min="10" max="500" value="60" placeholder="Frames to read">
            <button type="submit">Decode</button>
        </form>

        <div id="result"></div>

        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData();
                formData.append('file', document.getElementById('videoFile').files[0]);
                formData.append('velocity', document.getElementById('velocity').value);
                formData.append('num_frames', document.getElementById('frames').value);

                const result = document.getElementById('result');
                result.innerHTML = '<p>Processing...</p>';

                try {
                    const resp = await fetch('/decode', { method: 'POST', body: formData });
                    if (!resp.ok) throw new Error('Decode failed');

                    const blob = await resp.blob();
                    const url = URL.createObjectURL(blob);
                    result.innerHTML = `<h2>Result:</h2><img src="${url}">`;
                } catch (err) {
                    result.innerHTML = `<p style="color: red;">Error: ${err.message}</p>`;
                }
            };
        </script>
    </body>
    </html>
    """


@app.post("/decode")
async def decode(
    file: UploadFile = File(...),
    velocity: int = 2,
    num_frames: int = 60,
):
    """Decode a motion-masked video and return segmentation mask."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "video.mp4"
        output_path = Path(tmpdir) / "result.png"

        # Save uploaded file
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)

        # Decode
        mask, _ = decode_ghost_video(
            str(video_path),
            velocity=velocity,
            num_frames=num_frames,
        )

        # Save result
        cv2.imwrite(str(output_path), mask)

        return FileResponse(output_path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
