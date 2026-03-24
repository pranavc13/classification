"""
One-time conversion: unet.weights.h5 → unet.onnx
Run once from the backend folder:
    python convert_to_onnx.py
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import tf2onnx

sys.path.insert(0, str(Path(__file__).parent))
from model import build_unet, INPUT_SIZE, NUM_CLASSES

WEIGHTS_PATH = Path(__file__).parent.parent / "unet.weights.h5"
OUTPUT_PATH  = Path(__file__).parent.parent / "unet.onnx"

print("Building model and loading weights…")
model = build_unet(INPUT_SIZE, NUM_CLASSES)
model.load_weights(str(WEIGHTS_PATH))

print("Converting to ONNX…")
input_signature = [tf.TensorSpec((1, INPUT_SIZE, INPUT_SIZE, 3), tf.float32, name="input")]
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=17)

with open(OUTPUT_PATH, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"✓ Saved to {OUTPUT_PATH}")
