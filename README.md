# Crop Disease Detection AI System

> End-to-end AI system for real-time crop disease classification using Deep Learning and Machine Learning, deployed via FastAPI.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)
![Accuracy](https://img.shields.io/badge/AlexNet%20Accuracy-97.96%25-brightgreen?style=flat-square)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat-square)

---

## Overview

This project presents a full comparative study between classical Machine Learning and modern Deep Learning approaches for plant disease classification. Six models were trained and evaluated on the PlantVillage dataset. The best-performing model — **AlexNet at 97.96% accuracy** — was selected and deployed as a production FastAPI backend, optimized with INT8 quantization for efficient cloud deployment.

---

## Accuracy Results

| Model | Type | Accuracy |
|---|---|---|
| **AlexNet** | Deep Learning | **97.96%** ← deployed |
| SVM + ResNet50 features | Machine Learning | 91.93% |
| MobileNetV2 | Deep Learning | 84.57% |
| ResNet50 | Deep Learning | 83.82% |
| Random Forest + ResNet50 | Machine Learning | 83.18% |
| k-NN + ResNet50 | Machine Learning | 78.66% |

![Accuracy Chart](results/professional_accuracy_chart.png)

---

## Dataset

| Property | Value |
|---|---|
| Source | PlantVillage |
| Total Images | 20,638 |
| Classes | 15 |
| Crops | Tomato, Potato, Bell Pepper |

**Classes included:**
`Bacterial Spot` · `Early Blight` · `Late Blight` · `Leaf Mold` · `Septoria Leaf Spot` · `Spider Mites` · `Target Spot` · `Yellow Leaf Curl Virus` · `Mosaic Virus` · `Healthy`

---

## System Architecture

```
Image Input
    │
    ▼
FastAPI Backend (api/main.py)
    │
    ▼
Preprocessing (Resize 224×224 → ToTensor)
    │
    ▼
AlexNet (INT8 Quantized · 64 MB)
    │
    ▼
JSON Response
{
  "final_prediction": "Tomato_Early_blight",
  "confidence": "97.43%",
  "best_model": "AlexNet"
}
```

---

## Project Structure

```
crop-disease-detection-ai/
├── api/
│   └── main.py               ← FastAPI app
├── models/
│   ├── dl/
│   │   ├── alexnet.pth       ← deployed (64 MB, INT8 quantized)
│   │   ├── mobilenet.pth     ← float16 compressed
│   │   └── resnet.pth        ← float16 compressed
│   └── ml/
│       ├── svm_resnet.pkl    ← lz4 compressed
│       ├── rf_resnet.pkl     ← lz4 compressed
│       ├── knn_resnet.pkl    ← lz4 compressed
│       └── scaler.pkl
├── notebooks/
│   └── training.ipynb        ← full training pipeline
├── results/
│   ├── ml_results.csv
│   ├── comparison_graph.m    ← MATLAB visualization
│   └── professional_accuracy_chart.png
├── compress_models.py        ← model compression utility
├── render.yaml               ← Render deployment config
├── requirements.txt
└── README.md
```

---

## API Endpoints

### `GET /`
Health check.
```json
{ "message": "Crop Disease Detection API is running" }
```

### `POST /predict`
Upload a plant leaf image and get a disease prediction.

**Request:** `multipart/form-data` with `file` field (JPG/PNG)

**Response:**
```json
{
  "final_prediction": "Tomato_Early_blight",
  "confidence": "97.43%",
  "predictions": {
    "alexnet": "Tomato_Early_blight"
  },
  "model_accuracy": {
    "alexnet": 97.96,
    "svm": 91.93,
    "mobilenet": 84.57,
    "resnet": 83.82,
    "random_forest": 83.18,
    "knn": 78.66
  },
  "best_model": "AlexNet"
}
```

---

## Run Locally

**1. Clone the repo:**
```bash
git clone https://github.com/Muhammad-Ameen86/crop-disease-detection-ai.git
cd crop-disease-detection-ai
```

**2. Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Start the API:**
```bash
uvicorn api.main:app --reload
```

**5. Open Swagger UI:**
```
http://127.0.0.1:8000/docs
```

---

## Model Optimization

All models are compressed for GitHub and cloud deployment:

| Model | Original | Compressed | Method |
|---|---|---|---|
| AlexNet | 228 MB | 64 MB | INT8 quantization |
| ResNet50 | 94 MB | 47 MB | float16 |
| MobileNetV2 | 9 MB | 4.7 MB | float16 |
| SVM | 166 MB | 79 MB | lz4 compression |
| Random Forest | 198 MB | 26 MB | lz4 compression |
| k-NN | 112 MB | 87 MB | lz4 compression |
| **Total** | **~800 MB** | **~308 MB** | |

To recompress after retraining:
```bash
python compress_models.py
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Deep Learning | PyTorch, torchvision |
| Machine Learning | scikit-learn |
| API | FastAPI, Uvicorn |
| Visualization | Matplotlib, MATLAB |
| Notebook | Jupyter |
| Deployment | Render |
| Version Control | Git, GitHub |

---

## Deployment

This project is configured for **Render** deployment via `render.yaml`.

Start command:
```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Compatible with: **Render · Railway · AWS EC2 · Google Cloud · Azure**

---

## Future Improvements

- Top-3 prediction confidence scores
- Disease treatment recommendations
- Android mobile frontend
- Farmer advisory system in Urdu
- Multilingual support
- Live cloud inference endpoint
- EfficientNet fine-tuning for higher accuracy

---

## Author

**Muhammad Ameen Rajper**
AI · ML · Android Developer · Final Year IT Student

[![GitHub](https://img.shields.io/badge/GitHub-Muhammad--Ameen86-181717?style=flat-square&logo=github)](https://github.com/Muhammad-Ameen86)
