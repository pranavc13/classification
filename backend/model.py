"""
ResNet50-UNet architecture reconstructed from unet.weights.h5.

Inspected weight shapes confirm:
  - 7x7 initial conv (64 filters)
  - ResNet50 encoder: stages 1-3 (64->256, 128->512, 256->1024)
  - UNet decoder with 3 skip connections + UpSampling2D
  - 23 output classes (softmax)

INPUT_SIZE can be None (fully-convolutional) or a fixed int like 256/512.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

INPUT_SIZE = 256    # fixed input size — avoids graph retracing on variable shapes
NUM_CLASSES = 23


# ── Bottleneck residual block ─────────────────────────────────────────────────

def bottleneck(x, filters, expansion, stride=1, projection=False):
    """
    Standard ResNet bottleneck: 1x1 -> 3x3 -> 1x1.
    Main path is built first so layer numbering matches saved weights.
    """
    out_ch = filters * expansion

    # Main path (created before shortcut to preserve default naming order)
    m = layers.Conv2D(filters, 1, use_bias=True)(x)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('relu')(m)

    m = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=True)(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('relu')(m)

    m = layers.Conv2D(out_ch, 1, use_bias=True)(m)
    m = layers.BatchNormalization()(m)

    # Shortcut path (after main path so numbering is correct)
    if projection:
        s = layers.Conv2D(out_ch, 1, strides=stride, use_bias=True)(x)
        s = layers.BatchNormalization()(s)
    else:
        s = x

    x = layers.Add()([m, s])
    x = layers.Activation('relu')(x)
    return x


# ── Full model ────────────────────────────────────────────────────────────────

def build_unet(input_size=INPUT_SIZE, num_classes=NUM_CLASSES):
    shape = (input_size, input_size, 3) if input_size else (None, None, 3)
    inputs = layers.Input(shape=shape)

    # ── Encoder ───────────────────────────────────────────────────────────────

    # Initial conv block
    x = layers.Conv2D(64, 7, padding='same', use_bias=True)(inputs)  # conv2d
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    skip0 = x                                          # 64-ch skip (before padding)
    x = layers.ZeroPadding2D(1)(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Stage 1 — output 256 channels (3 blocks, stride=1)
    x = bottleneck(x, 64, 4, stride=1, projection=True)   # block 1
    x = bottleneck(x, 64, 4, stride=1, projection=False)  # block 2
    x = bottleneck(x, 64, 4, stride=1, projection=False)  # block 3
    skip1 = x                                          # 256-ch skip

    # Stage 2 — output 512 channels (4 blocks, stride=2 on first)
    x = bottleneck(x, 128, 4, stride=2, projection=True)
    x = bottleneck(x, 128, 4, stride=1, projection=False)
    x = bottleneck(x, 128, 4, stride=1, projection=False)
    x = bottleneck(x, 128, 4, stride=1, projection=False)
    skip2 = x                                          # 512-ch skip

    # Stage 3 — output 1024 channels (6 blocks, stride=2 on first)
    x = bottleneck(x, 256, 4, stride=2, projection=True)
    for _ in range(5):
        x = bottleneck(x, 256, 4, stride=1, projection=False)
    # x = 1024 channels

    # ── Decoder ───────────────────────────────────────────────────────────────

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(512, 3, padding='same', use_bias=True)(x)   # conv2d_43
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Concatenate()([x, skip2])               # 512 + 512 = 1024
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=True)(x)   # conv2d_44
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Concatenate()([x, skip1])               # 256 + 256 = 512
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', use_bias=True)(x)   # conv2d_45
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Concatenate()([x, skip0])               # 128 + 64 = 192
    x = layers.Conv2D(64, 3, padding='same', use_bias=True)(x)    # conv2d_46
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Output — 23 classes
    outputs = layers.Conv2D(num_classes, 3, padding='same',        # conv2d_47
                            activation='softmax', use_bias=True)(x)

    return Model(inputs, outputs)
