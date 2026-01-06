import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
<<<<<<< HEAD
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
=======
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
>>>>>>> upstream/master
import mlflow
import mlflow.sklearn
import os
import pickle

def train():
    # Load data
    df = pd.read_csv('data/heart_disease.csv')
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    df = df.fillna(df.median())

    X = df.drop('target', axis=1)
    y = df['target']

    # Numerical and Categorical columns
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MLflow tracking
    mlflow.set_experiment("Heart_Disease_Prediction")
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
<<<<<<< HEAD
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
=======
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        # Plotting metrics
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
        plt.title('Model Metrics Comparison')
        plt.ylim(0, 1)
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        plot_path = "models/metrics_plot.png"
        os.makedirs('models', exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(plot_path)
>>>>>>> upstream/master
        mlflow.sklearn.log_model(pipeline, "model")
        
        print("Metrics:", metrics)
        
        # Save model locally for API
        os.makedirs('models', exist_ok=True)
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
        print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train()
