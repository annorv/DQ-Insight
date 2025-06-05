# utils/ml_utils.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_ml_workflow(uploaded_file):
    df = pd.read_csv(uploaded_file)
    st.write("Initial Data Preview:", df.head())

    # Impute missing values, handle data quality, etc.
    # EDA and other steps

    # Data split
    train_size = st.slider("Select train size", 0.1, 0.9, 0.8)
    target_column = st.selectbox("Select the column to predict", df.columns)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    # Model training
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Further steps like retraining, model improvement suggestions, etc.
