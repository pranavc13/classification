# Satellite Image Segmentation

A full-stack web app that performs semantic segmentation on satellite images using a ResNet50-UNet deep learning model. Upload an image, get a color-coded segmentation mask back instantly.

## Architecture

- **Frontend** — React 19 + TypeScript + Vite. Drag-and-drop image upload, side-by-side result view, mask download.
- **Backend** — FastAPI (Python) serving a ResNet50-UNet model trained on 23 classes via TensorFlow/Keras.
- **Model** — Pre-trained weights in `unet.weights.h5`. Accepts any image size (fully-convolutional) and outputs a 23-class softmax segmentation mask.

```
classification/
├── backend/
│   ├── main.py          # FastAPI app — /predict endpoint
│   ├── model.py         # ResNet50-UNet architecture
│   └── requirements.txt
├── src/
│   └── App.tsx          # React frontend
├── unet.weights.h5      # Pre-trained model weights
└── package.json
```

## Prerequisites

- Node.js 18+
- Python 3.10+

## Running the project

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

The API will be available at `http://localhost:8001`. The `/predict` endpoint accepts a `multipart/form-data` POST with an `image/*` file and returns a PNG segmentation mask.

### 2. Frontend

From the project root:

```bash
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## Usage

1. Drop a satellite image (PNG, JPG, or TIF) onto the upload zone, or click to browse.
2. The image is sent to the backend, which runs inference and returns a segmentation mask.
3. View the original and mask side-by-side, then download the mask as `mask.png`.

## Other frontend commands

| Command | Description |
|---|---|
| `npm run build` | Production build |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |