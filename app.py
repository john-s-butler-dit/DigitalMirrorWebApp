from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from sklearn.decomposition import PCA
import base64
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")


def frame_to_base64(img_array: np.ndarray) -> str:
    """Convert a numpy grayscale image to a base64 PNG string."""
    # Normalise to 0–255
    norm = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    _, buf = cv2.imencode(".png", norm)
    return base64.b64encode(buf).decode("utf-8")


@app.post("/process")
async def process_video(
    file: UploadFile = File(...),
    n_components: int = 10,
    skip_frames: int = 15,
):
    # Save upload to a temp file
    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # --- Extract frames ---
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()

        if len(frames) < n_components + skip_frames:
            return JSONResponse(
                {"error": f"Video too short. Got {len(frames)} frames, need at least {n_components + skip_frames}."},
                status_code=400,
            )

        images_array = np.array(frames)[skip_frames:]
        n_samples, h, w = images_array.shape
        X = images_array.reshape(n_samples, h * w).astype(np.float32)

        # --- PCA ---
        n_components = min(n_components, n_samples - 1)
        pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X)
        eigenfaces = pca.components_.reshape((n_components, h, w))
        mean_face = pca.mean_.reshape(h, w)

        # --- Encode images ---
        result_images = [{"title": "Mean Face", "data": frame_to_base64(mean_face)}]
        for i, ef in enumerate(eigenfaces):
            result_images.append({
                "title": f"Eigenface {i + 1}",
                "data": frame_to_base64(ef),
            })

        variance = (pca.explained_variance_ratio_ * 100).tolist()

        return JSONResponse({
            "images": result_images,
            "variance": variance,
            "n_frames": n_samples,
            "frame_size": [h, w],
        })

    finally:
        os.unlink(tmp_path)
