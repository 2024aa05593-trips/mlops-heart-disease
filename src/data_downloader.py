import pandas as pd
import os

def download_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    
    print(f"Downloading data from {url}...")
    df = pd.read_csv(url, names=columns, na_values="?")
    
    # Save to data folder
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/heart_disease.csv", index=False)
    print("Data saved to data/heart_disease.csv")
    print(f"Dataset shape: {df.shape}")

if __name__ == "__main__":
    download_data()
