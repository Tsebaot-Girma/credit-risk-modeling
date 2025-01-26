Credit Risk Modeling for Bati Bank
Overview
This repository contains the implementation of a Credit Scoring Model developed for Bati Bank's buy-now-pay-later service in collaboration with an eCommerce company. The primary goal is to assess customer creditworthiness and enable a risk-aware lending process through advanced machine learning techniques and data-driven insights.

Features
Credit Risk Analysis: Proxy variable definition to classify users as high-risk or low-risk.
Feature Engineering: Data preprocessing, transformation, and selection of predictive features.
Model Development: Building machine learning models for:
Risk probability estimation.
Credit score assignment.
Loan amount and duration prediction.
REST API: Deployment of the model as a REST API for real-time credit scoring.
MLOps Integration: Model versioning and deployment pipelines using MLFlow and CI/CD practices.
Technologies Used
Programming Languages: Python
Libraries:
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: scikit-learn, xverse
Model Deployment: Flask, FastAPI
MLOps Tools: MLFlow, CI/CD pipelines
Other Tools: Weight of Evidence (WoE) for feature engineering
Project Structure
plaintext
Copy
Edit
credit-risk-modeling/
├── data/                # Data files (input and processed datasets)
├── notebooks/           # Jupyter notebooks for analysis and model development
├── models/              # Saved models
├── script/               # Source code for data preprocessing, modeling, and deployment
│   ├── preprocessing/   # Data preprocessing scripts
│   ├── features/        # Feature engineering scripts
│   ├── models/          # Model training and evaluation scripts
│   └── api/             # REST API implementation
├── tests/               # Unit and integration tests
├── .gitignore            
└── README.md            # Project overview and details
Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/Tsebaot-Girma/credit-risk-modeling
cd credit-risk-modeling
2. Install Dependencies
Use the requirements.txt file to install the necessary Python packages:

bash
Copy
Edit
pip install -r requirements.txt
3. Data Preparation
Place the dataset in the data/ directory.
Run the preprocessing script to clean and transform the data:
bash
Copy
Edit
python src/preprocessing/data_preprocessing.py
4. Train Models



