# FME-Datathon2025

**Machine Learning System for Schneider Sales Prediction**

## 🚀 Description

This repository contains a complete Machine Learning system developed for the FME Datathon 2025. The system includes advanced capabilities for training, evaluation, and explainability of predictive models, with special focus on result interpretability.

## ✨ Key Features

### 🎯 Modeling Capabilities
- **Two training modes**:
  - `ModelTrainHalf()`: Training with 50% of dataset
  - `ModelTrainFull()`: Training with 100% of dataset and evaluation of another dataset for maximum performance
- **Class balancing**: SMOTE-Tomek for imbalanced datasets
- **Feature selection**: RandomForest-based feature selection
- **Optimized model**: RandomForest with fine-tuned hyperparameters

### 📊 Evaluation & Testing
- `testModelHalf_fixed()`: Robust evaluation on second half of dataset
- `testModelFull()`: Complete dataset evaluation
- **Comprehensive metrics**: Accuracy, ROC-AUC, classification reports, confusion matrices
- **Cross-validation**: 5-fold stratified cross-validation

### 🔍 Model Explainability
- **Global importance**: Feature importance rankings
- **SHAP analysis**: Shapley values for individual interpretation
- **Partial Dependence Plots**: Feature-target relationships
- **Local explanations**: Per-instance analysis

### 📦 Core Dependencies
- `pandas`, `numpy`, `scikit-learn`
- `imbalanced-learn` (SMOTE-Tomek)
- `matplotlib`, `seaborn` (visualizations)
- `joblib` (model serialization)
- `shap` (advanced explainability)

## 🎮 Quick Start

### Basic Training
```python
from TestModelFixed import ModelTrainHalf, ModelTrainFull

# Train with half data
model_data = ModelTrainHalf("dataset.csv")

# Train with all data
model_data_full = ModelTrainFull("dataset.csv")
```

### Model Evaluation
```python
from TestModelFixed import testModelHalf_fixed, testModelFull

# Evaluate on second half
testModelHalf_fixed("dataset.csv")

# Evaluate on complete dataset
testModelFull("dataset.csv")
```

### Complete Pipeline
```python
from TestModelFixed import run_complete_pipeline

# Run training + testing + explainability
run_complete_pipeline("dataset.csv")
```

## 📁 Project Structure

```
FME-Datathon2025/
│
├── TestModelFixed.py          # 🎯 Main file with all functions
├── requirements.txt           # 📦 Project dependencies
├── modelo_entrenado_avanzado.pkl  # 💾 Saved model (50% training)
├── modelo_entrenado_full.pkl      # 💾 Saved model (100% training)
└── README.md                  # 📚 This file
```

## 🔧 Available Functions

### Training
- `ModelTrainHalf()`: Training with 50/50 split
- `ModelTrainFull()`: Training with complete dataset

### Evaluation  
- `testModelHalf_fixed()`: Test on second half
- `testModelFull()`: Complete test
- `preprocess_new_data()`: Preprocessing for new data

### Explainability
- `explain_model_global()`: Global feature importance
- `explain_with_shap()`: Advanced SHAP analysis
- `explain_with_pdp()`: Partial Dependence Plots
- `explain_local_prediction()`: Per-instance explanations

## 📈 Metrics & Results

The system provides:
- ✅ **Accuracy** in training and test
- ✅ **ROC-AUC** for binary problems
- ✅ **Detailed classification reports**
- ✅ **Confusion matrices**
- ✅ **Cross-validation scores**
- ✅ **OOB scores** (Out-of-Bag)

## 🎯 Use Cases

### For Development & Experimentation
```python
# Iterative development
model_data = ModelTrainHalf("dataset.csv")
testModelHalf_fixed("dataset.csv")
explain_model_global()
```

### For Production & Deployment
```python
# Final model for production
model_data = ModelTrainFull("dataset.csv")
# Use preprocess_new_data() for new predictions
```

### For Analysis & Interpretation
```python
# Comprehensive analysis
run_complete_pipeline("dataset.csv")
```

## 🔍 Sample Output

```
=== testModelHalf_fixed ===
ROC AUC: 0.9345
Classification report:
               precision    recall  f1-score   support
           0       0.92      0.95      0.93       245
           1       0.94      0.91      0.92       255
Confusion matrix:
[[232  13]
 [ 23 232]]
```

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Authors

- **François Liraud** - (https://github.com/francois0405)
- **Ignacio DM** - (https://github.com/IgnaciDM)

## 🙏 Acknowledgments

- FME for organizing Datathon 2025
- Contributors of the open-source libraries used
- Data Science community for the implemented techniques

---

Readme written with help of Artificial Intelligence

**Questions or issues?** Open an *issue* in the repository for support.