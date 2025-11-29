# MLOps-pipeline
An end-to-end MLOps pipeline for predicting student academic risk (Graduate, Dropout, Enrolled). Features data versioning, experiment tracking (MLflow), hyperparameter tuning, FastAPI deployment, Docker containerization, and CI/CD automation with GitHub Actions.

## Student Academic Risk Predictor: End-to-End MLOps Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview

This project implements a complete, production-grade machine learning pipeline to predict student academic risk in higher education. The model classifies students into three categories: **Graduate**, **Dropout**, or **Enrolled**.

It is built with a focus on MLOps best practices, demonstrating how to move from a raw dataset to a deployable, scalable API. The system includes automated training, hyperparameter tuning, experiment tracking, and containerized deployment.

### Key Features
* **Modular Codebase:** Clean separation of concerns (data loading, preprocessing, training, tuning, deployment).
* **Robust Preprocessing:** Custom feature engineering and `scikit-learn` pipelines for data transformation.
* **Experiment Tracking:** Integration with **MLflow** to log parameters, metrics, and model artifacts.
* **Hyperparameter Tuning:** Automated optimization using `RandomizedSearchCV`.
* **REST API:** A high-performance **FastAPI** application for real-time predictions.
* **Containerization:** Fully Dockerized application for consistent deployment.
* **CI/CD:** Automated build and push workflows using **GitHub Actions**.

## 📂 Project Structure
student_risk_predictor/ 
├── .github/ 
│ └── workflows/ 
│ └── ci-cd.yml # GitHub Actions workflow for CI/CD 
├── app/ # FastAPI Application 
│ ├── init.py 
│ ├── main.py # API server logic 
│ └── schemas.py # Pydantic models for data validation 
├── artifacts/ # Generated files (models, encoders, metrics) 
│ └── (Populated automatically by scripts) 
├── data/ # Raw Data 
│ ├── train.csv # Training dataset 
│ └── test.csv # Test dataset (optional) 
├── mlruns/ # MLflow tracking data (auto-generated) 
├── notebooks/ # Jupyter Notebooks 
│ └── 1-Data-Exploration.ipynb 
├── src/ # Core ML Source Code 
│ ├── init.py 
│ ├── data_loader.py # Data loading and splitting logic 
│ ├── preprocessor.py # Preprocessing pipeline definition 
│ ├── train.py # Model training and selection script 
│ ├── tune.py # Hyperparameter tuning script 
│ └── utils.py # Helper functions 
├── .gitignore 
├── Dockerfile # Docker image configuration 
├── params.yaml # Configuration file for parameters 
├── requirements.txt # Python dependencies 
└── README.md # Project documentation

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Git
* Docker (optional for local dev, required for containerization)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/student-risk-predictor.git](https://github.com/yourusername/student-risk-predictor.git)
    cd student-risk-predictor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Setup:**
    Ensure you have the `train.csv` file placed inside the `data/` directory.


## 🛠️ Usage Pipeline

Follow these steps to reproduce the entire training and deployment process.

### 1. Data Exploration (Optional)
Run the Jupyter notebook to understand the dataset distribution and correlations.
```bash
# Open the notebook in your editor or Jupyter Lab
notebooks/1-Data-Exploration.ipynb
```