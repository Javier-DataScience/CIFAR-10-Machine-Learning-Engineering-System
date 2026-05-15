# 🧠 CIFAR-10 Machine Learning Engineering System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-orange)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-purple)

---

## 🚀 Project Overview (PORTFOLIO STORYTELLING)

This project implements a complete Machine Learning Engineering system for CIFAR-10 image classification.

Instead of focusing only on model training, the system is designed as an end-to-end ML pipeline including:

- Model development (CNN_V1 → CNN_V6)
- Experiment tracking (MLflow)
- Model versioning and selection
- Production inference pipeline
- REST API deployment (FastAPI)
- Interactive UI (Streamlit)
- Containerized deployment (Docker)

The system simulates a real-world ML production environment where models are trained, tracked, selected, and deployed as services.

---

## 🏗️ Architecture

The system is designed as a modular ML pipeline:

```
flowchart LR
A[Dataset CIFAR-10] --> B[Data Loader]
B --> C[Training Pipeline]
C --> D[MLflow Tracking]
C --> E[Model Registry]
E --> F[Best Model Artifact]

F --> G[Inference Layer]
G --> H[FastAPI Service]
G --> I[Streamlit UI]

H --> J[User Image Upload]
I --> J
J --> G
```

## 🧩 Project Structure
```
Cifar_10_Computer_Vision_Project_System/
│
├── app/
│   ├── fastapi_app.py
│   └── streamlit_app.py
│
├── src/
│   ├── train.py
│   ├── models.py
│   ├── inference.py
│   ├── production_inference.py
│   ├── data_loader.py
│   ├── model_registry.py
│   └── save_best_model.py
│
├── models/
│   └── best_model.pt
│
├── notebooks/
│
├── Dockerfile
├── Dockerfile.streamlit
├── requirements.txt
└── README.md
```

## 🔬 What This Project Demonstrates
End-to-end ML system design
Separation of training and inference logic
Model versioning strategy
Production-ready API deployment
UI integration with Streamlit
Docker-based containerization
MLflow experiment tracking


## 🧠 Model Development

Multiple CNN architectures were developed and evaluated:

CNN_V1 → baseline model
CNN_V2–V5 → incremental improvements
CNN_V6 → best-performing production model

All experiments were tracked using MLflow.

##  🔬  Machine Learning Pipeline
- Data preprocessing
- CNN training
- Experiment tracking (MLflow)
- Best model selection
- Export to production artifact
- Deployment via API + UI

# 🌐 Deployment

## ⚡ FastAPI Inference Service
- uvicorn app.fastapi_app:app --reload
- Endpoint: /predict
- Accepts image upload
- Returns predicted class

## 🎨 Streamlit UI
streamlit run app/streamlit_app.py

Interactive interface for real-time predictions.

## 🐳 Docker Deployment
docker build -t cifar10-api .
docker run -p 8000:8000 cifar10-api
docker build -f Dockerfile.streamlit -t cifar10-streamlit .
docker run -p 8501:8501 cifar10-streamlit

## 📊 Key Features
- Modular ML architecture
- Multiple CNN experiments (V1–V6)
- Model registry system
- Production inference pipeline
- REST API + Web UI
- Dockerized deployment
- MLflow experiment tracking

## 🎯 Key Learning Outcomes
- End-to-end ML system design
- Separation of training and inference
- Model versioning strategy
- API deployment with FastAPI
- UI integration with Streamlit
- Containerization with Docker
- Production-style ML architecture

## 🚀 Future Improvements

- CI/CD pipeline (GitHub Actions)
- Kubernetes deployment
- Cloud deployment (AWS / Azure)
- Model monitoring system

## 👨‍💻 Author

Machine Learning Engineering Project
Built for portfolio and production-style demonstration