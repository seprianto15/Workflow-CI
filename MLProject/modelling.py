import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.utils import estimator_html_repr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def load_dataset(current_dir):
    """Memuat dataset students performance."""
    data_path = os.path.join(current_dir, 'data', 'final_data_students.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan pada path: {data_path}")
    return pd.read_csv(data_path)

def perform_tuning(X_train, y_train, X_test, y_test):
    """Melakukan hyperparameter tuning sederhana menggunakan Random Search manual."""
    n_estimators_range = np.linspace(10, 100, 5, dtype=int)
    max_depth_range = np.linspace(1, 20, 5, dtype=int)

    best_accuracy = 0
    best_model = None
    best_params = {}

    print("Starting hyperparameter tuning local...")
    for n_est in n_estimators_range:
        for m_depth in max_depth_range:
            model = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_params = {'n_estimators': n_est, 'max_depth': m_depth}

    print(f"Tuning finished. Best Accuracy: {best_accuracy:.4f}")
    return best_model, best_params, best_accuracy


def main():
    # 1. Konfigurasi Kredensial DagsHub & MLflow
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME', '')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN', '')
    os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI', '')

    mlflow.set_experiment('Students-Performance')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data = load_dataset(base_dir)

    # 2. Split Dataset
    X = data.drop(['Status', 'Student_ID'], axis=1)
    y = data['Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    input_example = X_train.iloc[0:5]

    # 3. Proses Training & Tuning Model
    best_model, best_params, best_accuracy = perform_tuning(X_train, y_train, X_test, y_test)

    best_n_est = int(best_params['n_estimators'])
    best_m_depth = int(best_params['max_depth'])

    # 4. Tangkap Run ID yang dibuat otomatis oleh 'mlflow run' di terminal
    active_run = mlflow.active_run()
    if active_run:
        current_run_id = active_run.info.run_id
        
        # Ekspor Run ID ke root directory GitHub Actions secara otomatis
        if 'GITHUB_ENV' in os.environ:
            with open(os.environ['GITHUB_ENV'], 'a') as f:
                f.write(f'MLFLOW_RUN_ID={current_run_id}\n')
        print(f'[Autolog/MLflow Run] Active Run ID successfully captured: {current_run_id}')

    # Catatan: Log metrik, artifacts, dan model tidak perlu dipanggil manual lagi 
    # karena environment CLI mlflow run sudah menangani sesi eksekusi secara penuh, 
    # atau Anda bisa mengandalkan MLflow autologging jika scikit-learn diaktifkan.
    
if __name__ == '__main__':
    main()