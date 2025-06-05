# 🍷 Red Wine Quality Prediction

This is a web-based machine learning application that predicts the quality of red wine based on its physicochemical properties. I trained many models but one which performed the best was  **XGBoost** . I used flask for web interface

## 🚀 Features

- Clean and responsive UI
- Input validation and smooth UX animations
- Trained XGBoost classifier with optimized hyperparameters
- Real-time prediction with JSON response
- Labels: **Excellent** or **Poor** quality

---

## 🧪 Dataset

The project uses the **Red Wine Quality** dataset from Kaggle

- https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

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

Dropped low-impact features ('pH', 'fixed acidity', 'citric acid', 'free sulfur dioxide') based on their correlation with the target variable.

The quality scores are binarized:
- `quality >= 7` → **Excellent (1)**
- `quality < 7`  → **Poor (0)**
---


---
### Accuracy of diffrent models

| Models                 | Accuracy  |
|------------------------|-----------|
| `SVC`                  | 86.63     |
|`DecisionTreeClassifier`| 86.66     |
|`RandomForestClassifier`| 93.02     |
| `XGBClassifier`        | 93.60     |
| `KNeighborsClassifier` | 90.70     |

After calculating cross validation score of each , I figured out that XGBclassifier is the best fit model for this project



