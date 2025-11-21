# 🌟 Bank Marketing Classification Using Residual Neural Networks

[![Made with Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Tabular%20DL-red)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model%20Evaluation-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

This project implements a **binary classifier** for the UCI Bank Marketing Dataset using a **Residual MLP (Multi-Layer Perceptron)** built in PyTorch.  
The goal is to predict whether a bank customer will **subscribe to a term deposit**.

The model includes:

- Residual skip connections  
- Batch Normalization + Dropout  
- AdamW optimizer  
- Cosine Annealing LR scheduler  
- Class-weighted BCEWithLogitsLoss for imbalance  

This is a complete end-to-end Data Science + Deep Learning project.

---

## 🚀 Highlights

- End-to-end pipeline: preprocessing → training → evaluation  
- Handles numerical + categorical features  
- Residual architecture improves gradient flow  
- Robust evaluation: F1, ROC-AUC, confusion matrix, loss curves  
- Clean Jupyter notebook + modular Python scripts (`model.py`, `data.py`, `train.py`)  

---

## 📂 Project Structure

bank-marketing-classification/
│── README.md
│── requirements.txt
│
├── notebooks/
│ └── CIS602_Project1_PERFECT.ipynb
│
├── src/
│ ├── model.py
│ ├── data.py
│ └── train.py
│
├── data/
│ └── bank-additional-full.csv
│
├── results/
│ ├── loss_curve.png
│ ├── f1_curve.png
│ ├── confusion_matrix.png
│ ├── confusion_matrix_eval.png
│ └── roc_curve.png
│
└── report/
└── Project_Report.docx (optional)

---

## 📊 Dataset Description

- **Source:** UCI Machine Learning Repository  
- **Size:** ~41,000 rows  
- **Target:** `y` (yes/no → deposit subscription)  
- **Features:**  
  - Numerical: age, duration, campaign, pdays, balance, etc.  
  - Categorical: job, education, contact type, month, poutcome, etc.  

Data preprocessing includes:

- StandardScaler for numerical features  
- OneHotEncoder for categorical variables  
- Stratified train/val/test split: 70/15/15  

---

## 🧠 Model Architecture (Residual MLP)

nput Layer
|
Dense → BatchNorm → ReLU → Dropout
| (skip connection)
Dense → BatchNorm → ReLU → Dropout
| (skip connection)
Output Layer (1 logit)

Why Residual MLP?

- Improves training stability  
- Helps deeper models on tabular data  
- Works better on imbalanced datasets  

---

## ⚙️ Training Configuration

- **Optimizer:** AdamW  
- **Learning Rate:** 1e-3  
- **Weight Decay:** 1e-4  
- **Scheduler:** CosineAnnealingLR  
- **Loss:** BCEWithLogitsLoss with class weights  
- **Batch Size:** 256  
- **Epochs:** 30–50  

---

## 📈 Results (Test Set)

| Metric       | Score |
|--------------|-------|
| Accuracy     | 0.88  |
| Precision    | 0.82  |
| Recall       | 0.79  |
| F1 Score     | 0.80  |
| ROC-AUC      | 0.91  |

### Sample Visualizations (inside `results/`)
- `loss_curve.png`
- `f1_curve.png`
- `confusion_matrix.png`
- `confusion_matrix_eval.png`
- `roc_curve.png`

---

## 🏃‍♀️ How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
2️⃣ Run the notebook:
jupyter notebook notebooks/CIS602_Project1_PERFECT.ipynb
3️⃣ Run the Python training script:
python src/train.py
This runs the entire training loop and saves all plots to:
results/
🧪 Evaluation
The project includes:
Confusion matrix
ROC curve
Precision, Recall, F1 Score
Full classification report
Residual networks + weighted BCE loss achieve balanced performance on an imbalanced dataset.
🚀 Future Improvements
Add SHAP feature interpretations
Try XGBoost / CatBoost / LightGBM baselines
Add categorical embeddings instead of one-hot encoding
Hyperparameter tuning using Optuna
👤 Author
Harshith Santosh Sathyasai Atla
Master’s Student, Data Science
UMass Dartmouth
📄 License
This project is licensed under the MIT License.