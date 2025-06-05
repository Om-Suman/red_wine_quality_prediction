# 🍷 Red Wine Quality Prediction

This is a web-based machine learning application that predicts the quality of red wine based on its physicochemical properties. The model is built using **XGBoost** and served via a **Flask** web interface.

## 🚀 Features

- Clean and responsive UI
- Input validation and smooth UX animations
- Trained XGBoost classifier with optimized hyperparameters
- Real-time prediction with JSON response
- Labels: **Excellent** or **Poor** quality

---

## 🧪 Dataset

The project uses the **Red Wine Quality** dataset from the UCI Machine Learning Repository:

- https://archive.ics.uci.edu/ml/datasets/wine+quality

### Features used for prediction:

| Feature                 | Description                          |
|------------------------|--------------------------------------|
| `volatile acidity`     | Amount of acetic acid in wine        |
| `residual sugar`       | Amount of remaining sugar            |
| `chlorides`            | Salt content                         |
| `total sulfur dioxide` | Total SO₂ levels                     |
| `density`              | Density of the wine                  |
| `sulphates`            | Sulfate content                      |
| `alcohol`              | Alcohol percentage                   |

The quality scores are binarized:
- `quality >= 7` → **Excellent (1)**
- `quality < 7`  → **Poor (0)**

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/wine-quality-predictor.git
cd wine-quality-predictor
