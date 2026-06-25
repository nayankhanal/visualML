# Visual ML

A simple Streamlit app to train and evaluate classical ML algorithms without writing code.

Upload a CSV, handle missing values, encode categorical columns, scale features, pick an algorithm (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Linear Regression), tune its hyperparameters, and see live results — accuracy/F1/confusion matrix for classification, or MSE/R² for regression.

## Run locally

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — Streamlit UI
- `ml_logic.py` — preprocessing and model logic
