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

    mlflow.log_params({
        'n_estimators': best_n_est,
        'max_depth': best_m_depth,
        'random_state': 42
    })
    
    # Prediksi Evaluasi
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')

    mlflow.log_metric('train_accuracy', accuracy_score(y_train, y_pred_train))
    mlflow.log_metric('accuracy', float(best_accuracy))
    mlflow.log_metric('test_f1_score', float(test_f1))

    # ARTIFACT 1: Representasi HTML Model
    mlflow.log_text(estimator_html_repr(best_model), artifact_file='estimator.html')

    # ARTIFACT 2: JSON Informasi Metrik
    mlflow.log_dict({'accuracy': float(best_accuracy), 'test_f1_score': float(test_f1)}, artifact_file='metric_info.json')

    # ARTIFACT 3: Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_train, y_pred_train, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.4g', cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix (n={best_n_est}, depth={best_m_depth})")
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    mlflow.log_figure(fig, artifact_file='training_confusion_matrix.png')
    plt.close(fig)

    # ARTIFACT 4: Klasifikasi Report
    os.makedirs('reports', exist_ok=True)
    mlflow.log_text(classification_report(y_test, y_pred_test), artifact_file='reports/classification_report.txt')

    # ARTIFACT 5: Filter Feature Importance Plot
    importances = best_model.feature_importances_
    threshold = 0.03
    important_idx = np.where(importances >= threshold)[0]
    sorted_idx = important_idx[np.argsort(importances[important_idx])[::-1]]
            
    if len(sorted_idx) > 0:
        os.makedirs('plots', exist_ok=True)
        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=importances[sorted_idx],
            y=X_train.columns[sorted_idx],
            ax=ax_fi,
            color='teal' 
        )
        ax_fi.set_title(f"Feature Importance >= {threshold} (n={best_n_est}, depth={best_m_depth})")
        ax_fi.set_xlabel('Relative Importance')
        ax_fi.set_ylabel('Features')
        plt.tight_layout()
        mlflow.log_figure(fig_fi, artifact_file='plots/feature_importance.png')
        plt.close(fig_fi)
    
    # Logging Model Utama
    mlflow.sklearn.log_model(sk_model=best_model, artifact_path='model', input_example=input_example)

    # 5. Menangkap ID Run yang sedang aktif secara global dan mengekspornya ke GitHub Actions
    active_run = mlflow.active_run()
    if active_run:
        run_id = active_run.info.run_id
        if 'GITHUB_ENV' in os.environ:
            with open(os.environ['GITHUB_ENV'], 'a') as f:
                f.write(f'MLFLOW_RUN_ID={run_id}\n')
        print(f'Run ID successfully captured: {run_id}')
    
if __name__ == '__main__':
    main()