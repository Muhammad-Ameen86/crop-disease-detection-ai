# 🌿 Crop Disease Detection AI System

> AI-powered crop disease classification using **Machine Learning, Deep Learning, FastAPI, MATLAB, Python Visualization, and the PlantVillage Dataset**

---

## 🚀 Project Overview
This project is an **end-to-end AI-based crop disease detection system** developed using both **Machine Learning (ML)** and **Deep Learning (DL)** approaches.

The system uses the **PlantVillage dataset** with **20,638 images across 15 classes** and performs a full comparative study between traditional ML classifiers and modern DL architectures.

The final deployed backend is powered by **FastAPI**, making it ready for **Android, Web, and production integration**.

---

## 📊 Dataset Information
- 📁 Dataset: **PlantVillage**
- 🖼️ Total Images: **20,638**
- 🌱 Total Classes: **15**
- 🍅 Crops Included:
  - Tomato
  - Potato
  - Bell Pepper
- 🔍 Classes include:
  - Early Blight
  - Late Blight
  - Bacterial Spot
  - Leaf Mold
  - Healthy leaves
  - Mosaic Virus
  - Yellow Leaf Curl Virus

---

## 🧠 Models Used

### 🔹 Machine Learning (Hybrid Pipeline)
Deep features were extracted using **ResNet50** and passed into classical ML models.

- ⭐ **SVM + ResNet50 Features → 91.93%**
- 🌲 Random Forest → 83.40%
- 📍 k-NN → 78.66%

---

### 🔹 Deep Learning
The following CNN architectures were trained and compared:

- ⭐ **AlexNet → 98.16%**
- 🔥 ResNet50 → 83.40%
- ⚡ MobileNetV2 → 84.11%

---

## 🏆 Final Model Selection
The **AlexNet model achieved the highest validation accuracy of 98.16%** and was selected as the **final deployment model**.

### ✅ Why AlexNet was selected
- highest accuracy
- stable training performance
- excellent leaf texture learning
- best disease classification consistency
- ideal for FastAPI deployment

---

## 🌐 FastAPI Backend
A real-time backend API was developed using **FastAPI**.

### ✅ Features
- image upload endpoint
- real-time disease prediction
- ML vs DL grouped response
- JSON output
- Swagger UI documentation
- Android/Web ready integration

### ▶️ Run locally
```bash
uvicorn api.main:app --reload
```

### 📘 Swagger Docs
```text
http://127.0.0.1:8000/docs
```

---

## 📈 Results Visualization
The project includes both **MATLAB and Python-based result visualizations**.

### 📊 Included Charts
- ML vs DL all model comparison
- average ML vs DL performance
- publication-ready bar charts
- GitHub-ready PNG visualizations

### 🖼️ Accuracy Comparison
![Accuracy Chart](results/professional_accuracy_chart.png)

---

## ☁️ Deployment Ready
The project includes:
- ✅ `render.yaml`
- ✅ FastAPI backend
- ✅ trained AlexNet weights
- ✅ production-ready API routes

This makes the project ready for deployment on:
- Render ⭐
- Railway
- AWS EC2
- Google Cloud
- Azure

---

## 📂 Project Structure
```text
crop-disease-detection-ai/
├── api/
│   └── main.py
├── dataset/
├── models/
│   └── dl/
│       └── alexnet.pth
├── notebooks/
├── results/
│   └── professional_accuracy_chart.png
├── README.md
├── requirements.txt
├── render.yaml
└── .gitignore
```

---

## 🛠 Tech Stack
- Python
- PyTorch
- Scikit-learn
- FastAPI
- MATLAB
- Matplotlib
- Jupyter Notebook
- VS Code
- PlantVillage Dataset

---

## 🎯 Key Outcomes
✅ Built full ML pipeline  
✅ Built full DL pipeline  
✅ Compared 6 total models  
✅ Selected best deployment model  
✅ Developed real-time API backend  
✅ Created MATLAB + Python visualizations  
✅ GitHub portfolio ready  
✅ Cloud deployment ready  

---

## 🚀 Future Improvements
- confidence score
- top-3 predictions
- disease treatment recommendations
- mobile frontend
- farmer advisory system
- multilingual support
- better ResNet fine-tuning
- live cloud inference

---

## 👨‍💻 Author
**Muhammad Ameen Rajper**  
AI / ML / Backend Developer  
GitHub: https://github.com/Muhammad-Ameen86
