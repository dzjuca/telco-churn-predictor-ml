# 📊 Telco Customer Churn Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ha42z3yhgydqdb9ncnjoh7.streamlit.app)

A comprehensive machine learning application for predicting customer churn in telecommunications companies using ensemble models.

🔗 **[Launch Live App](https://ha42z3yhgydqdb9ncnjoh7.streamlit.app)**

---

## 🎯 Project Overview

This project implements an end-to-end machine learning system for predicting customer churn using the Telco Customer Churn dataset. The application features three ensemble models (Logistic Regression, Stacking Classifier, and Voting Classifier), each with two versions: one using all features and another using optimally selected features.

**Key Features:**
- 🔮 Individual customer churn prediction with probability scores
- 📊 Interactive model performance dashboard
- 📈 Comprehensive exploratory data analysis (EDA)
- ⚡ Real-time predictions through web interface
- 🎨 Professional UI with Material Design icons

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/dzjuca/telco-churn-predictor-ml.git
cd telco-churn-predictor-ml
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`

---

## 📁 Project Structure
```
telco-churn-predictor-ml/
├── app.py                          # Main Streamlit application
├── utils.py                        # Custom transformer classes
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── models/                         # Trained models and pipelines
│   ├── final_models_for_streamlit.pkl
│   ├── complete_preprocessing_pipeline.pkl
│   └── feature_importance_data.pkl
├── data/                          # Processed data and metrics
│   ├── telcoCustomer.csv          # Original dataset
│   ├── final_metrics_for_streamlit.pkl
│   ├── feature_config_for_streamlit.pkl
│   ├── final_summary_table.pkl
│   └── telco_data_prepared.pkl
└── assets/                        # Static resources (icons, images)
```

---

## 🤖 Models

### Implemented Models

1. **Logistic Regression**
   - Baseline linear model
   - Two versions: All features (22) and Selected features (12)
   - Optimized with Optuna (Hyperband pruner)

2. **Stacking Classifier**
   - Meta-learner combining multiple base estimators
   - Improved generalization through model stacking
   - Optimized hyperparameters

3. **Voting Classifier (Soft)**
   - Ensemble averaging of probability predictions
   - Combines strengths of multiple models
   - Robust performance across different scenarios

### Feature Engineering

- **All Features Version:** Uses complete feature set (22 features after preprocessing)
- **Selected Features Version:** Uses top 12 features identified through Random Forest feature importance
- **Preprocessing Pipeline:** 
  - Custom transformers for data cleaning
  - OneHotEncoding with drop='first'
  - StandardScaler for numerical features
  - Handles categorical redundancies

---

## 📊 Application Features

### 1. Make Prediction Tab
- Input form for customer information (18 fields)
- Real-time churn prediction with probability scores
- Risk assessment (High/Low)
- Model confidence display

### 2. Model Dashboard Tab
- **Performance Metrics Comparison:** Visual comparison of Accuracy, AUC, and F1-Score
- **Confusion Matrices:** Side-by-side matrices for both model versions
- **Feature Importance:** Top features ranked by importance score
- Interactive visualizations with Plotly

### 3. Dataset Exploration Tab
- **Churn Distribution:** Pie chart with class balance metrics
- **Numerical Features Distribution:** Histograms showing data patterns
- **Churn Analysis by Key Features:** Contract type and Internet service impact
- **Correlation Matrix:** Heatmap showing feature relationships

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Streamlit-AntD-Components, Plotly |
| **ML/Data Science** | scikit-learn, imbalanced-learn, pandas, numpy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git, GitHub |

---

## 📈 Model Performance

| Model | Version | Accuracy | AUC | F1-Score |
|-------|---------|----------|-----|----------|
| Logistic Regression | All Features | 0.7395 | 0.8282 | 0.6149 |
| Logistic Regression | Selected (Optimized) | 0.7409 | 0.8294 | 0.6179 |
| Stacking Classifier | All Features | 0.7888 | 0.8460 | 0.6815 |
| Stacking Classifier | Selected (Optimized) | 0.7866 | 0.8469 | 0.6783 |
| Voting Classifier | All Features | 0.7866 | 0.8455 | 0.6783 |
| Voting Classifier | Selected (Optimized) | 0.7866 | 0.8469 | 0.6783 |

*Note: Metrics calculated on test set*

---

## 🎓 Academic Context

This project was developed as part of the **Machine Learning Module - Final Exam** for the Master's in Data Science program at **Yachay Tech University**.

**Key Learning Outcomes:**
- End-to-end ML pipeline development
- Ensemble methods implementation
- Hyperparameter optimization with Optuna
- Model deployment and productionization
- Interactive dashboard creation

---

## 📝 Usage Example
```python
# Example: Making a prediction programmatically

import pandas as pd
import pickle

# Load the preprocessing pipeline
with open('models/complete_preprocessing_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Load a trained model
with open('models/final_models_for_streamlit.pkl', 'rb') as f:
    models = pickle.load(f)
    model = models['stacking_selected_optimized']

# Create input data
customer_data = pd.DataFrame({
    'customerID': ['CUST-001'],
    'gender': ['Female'],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['No'],
    'tenure': [12],
    # ... (other features)
})

# Preprocess and predict
X_processed = pipeline.transform(customer_data)
prediction = model.predict(X_processed)
probability = model.predict_proba(X_processed)

print(f"Churn Prediction: {prediction[0]}")
print(f"Churn Probability: {probability[0][1]:.2%}")
```

---

## 🤝 Contributing

This is an academic project, but suggestions and feedback are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add some improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is developed for academic purposes as part of a Master's program at **Yachay Tech University**.

---

## 👤 Author

**Daniel Zambrano Juca**
- GitHub: [@dzjuca](https://github.com/dzjuca)
- Project: [telco-churn-predictor-ml](https://github.com/dzjuca/telco-churn-predictor-ml)

---

## 🙏 Acknowledgments

- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Framework: [Streamlit](https://streamlit.io/)
- ML Library: [scikit-learn](https://scikit-learn.org/)
- Optimization: [Optuna](https://optuna.org/)

---

## 📞 Support

For questions or issues, please open an issue in the GitHub repository or contact through the university platform.

---

**⭐ If you find this project useful, please consider giving it a star!**