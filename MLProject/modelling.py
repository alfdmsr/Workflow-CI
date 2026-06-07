import pandas as pd
import os
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("CI_Automated_Training")

with mlflow.start_run():
    print("Memulai proses pelatihan CI...")

    DATA_PATH = os.path.join("..", "ai4i2020_preprocessing", "ai4i2020_processed.csv")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=['Machine failure'])
    y = df['Machine failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    mlflow.sklearn.log_model(rf, "model")

    print("Pelatihan CI selesai dan model berhasil disimpan!")