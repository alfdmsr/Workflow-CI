import pandas as pd
import os
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run():
    print("Memulai proses pelatihan CI...")

    DATA_PATH = os.path.join("..", "ai4i2020_preprocessing", "ai4i2020_processed.csv")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=['Machine failure'])
    y = df['Machine failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    custom_env = {
        "name": "mlflow-env",
        "channels": ["conda-forge", "nodefaults"], # nodefaults adalah kunci ajaibnya!
        "dependencies": [
            "python=3.10",
            "pip",
            {"pip": ["mlflow==2.19.0", "pandas", "scikit-learn"]}
        ]
    }

    mlflow.sklearn.log_model(sk_model=rf,
                             artifact_path= "model",
                             conda_env=custom_env
                             )

    print("Pelatihan CI selesai dan model berhasil disimpan!")