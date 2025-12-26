import pandas as pd
import numpy as np
import os

def test_data_existence():
    assert os.path.exists("data/heart_disease.csv")

def test_data_columns():
    df = pd.read_csv("data/heart_disease.csv")
    expected_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    assert all(col in df.columns for col in expected_columns)

def test_target_values():
    df = pd.read_csv("data/heart_disease.csv")
    # UCI target is 0-4, we convert to 0-1 in training
    assert df['target'].min() >= 0
    assert df['target'].max() <= 4
