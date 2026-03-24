"""
FastAPI backend for UNet satellite image segmentation.
Run with:  uvicorn main:app --host 127.0.0.1 --port 8001
"""

import io
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from constants import INPUT_SIZE, NUM_CLASSES
from model import build_unet

# ── Load Keras model ──────────────────────────────────────────────────────────
WEIGHTS_PATH = Path(__file__).parent.parent / "unet.weights.h5"

print("Building model…")
model = build_unet(INPUT_SIZE, NUM_CLASSES)
model.load_weights(str(WEIGHTS_PATH))
print(f"✓ Loaded weights from {WEIGHTS_PATH}")

# warm-up: run inference once so TF kernels are compiled
_dummy = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
model.predict(_dummy, verbose=0)
print("✓ Model warmed up")

# Thread pool so inference doesn't block the async event loop
_executor = ThreadPoolExecutor(max_workers=1)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="UNet Satellite Segmentation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]  # (1, H, W, 3)


def postprocess(prediction: np.ndarray) -> Image.Image:
    pred = prediction[0]  # remove batch dim → (H, W, NUM_CLASSES)

    if NUM_CLASSES == 1:
        mask = (pred[:, :, 0] * 255).astype(np.uint8)
        return Image.fromarray(mask, mode="L").convert("RGB")
    else:
        class_map = np.argmax(pred, axis=-1).astype(np.uint8)
        palette = np.array([
            [0,   0,   0],
            [0, 200, 100],
            [255, 100,  0],
            [0, 100, 255],
            [255, 255,  0],
            [200,   0, 200],
            [0,  200, 200],
            [255, 200,   0],
            [100,  50,   0],
            [150, 150, 150],
            [255,   0,   0],
            [0,   0, 255],
            [0, 255,   0],
            [255, 128,   0],
            [128,   0, 255],
            [0, 128, 255],
            [255,   0, 128],
            [128, 255,   0],
            [0, 255, 128],
            [64,  64,  64],
            [192, 192, 192],
            [255, 255, 128],
            [128, 255, 255],
        ], dtype=np.uint8)
        rgb = palette[class_map % len(palette)]
        return Image.fromarray(rgb, mode="RGB")


def _run_inference(inp: np.ndarray) -> np.ndarray:
    import time
    t = time.time()
    print(f"[inference] started, input shape={inp.shape}")
    result = model.predict(inp, verbose=0)
    print(f"[inference] done in {time.time() - t:.2f}s")
    return result


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    import time
    t0 = time.time()

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    data = await file.read()
    print(f"[predict] file read in {time.time() - t0:.2f}s, size={len(data)} bytes")

    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Could not decode image.")

    inp = preprocess(image)
    print(f"[predict] preprocess done in {time.time() - t0:.2f}s")

    loop = asyncio.get_event_loop()
    pred = await loop.run_in_executor(_executor, _run_inference, inp)
    print(f"[predict] inference done in {time.time() - t0:.2f}s")

    result_img = postprocess(pred)
    print(f"[predict] postprocess done in {time.time() - t0:.2f}s")

    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    print(f"[predict] total={time.time() - t0:.2f}s")
    return StreamingResponse(buf, media_type="image/png")
