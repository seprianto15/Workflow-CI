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

def main():
    # 1. Configure environment variables and DagsHub credentials
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME', '')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN', '')

    # 2. Setup MLflow tracking remote URI and experiment
    mlflow.set_tracking_uri('https://dagshub.com/seprianto15/Workflow-CI.mlflow')
    mlflow.set_experiment('Students-Performance')

    # 3. Load dataset using safe relative pathing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'final_data_students.csv')
    data = pd.read_csv(data_path)

    # 4. Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['Status', 'Student_ID'], axis=1),
        data['Status'],
        random_state=42,
        test_size=0.2
    )
    input_example = X_train.iloc[0:5]

    # 5. Inisialisasi model dengan parameter tetap (n_estimators=70, max_depth=15)
    n_est = 70
    m_depth = 15
    
    print(f"Initializing RandomForest with n_estimators={n_est} and max_depth={m_depth}")
    model = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi model langsung ke data test
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"Training finished. Test Accuracy: {test_accuracy}")

    # Menangkap wadah run yang sudah dibuat secara otomatis oleh eksekutor mlflow run
    active_run = mlflow.active_run()
    if active_run:
        best_run_id = active_run.info.run_id
        
        # Berikan label/nama pada run yang aktif
        mlflow.set_tag("mlflow.runName", f"run_rf_{n_est}_{m_depth}")

        # 6. Catat parameter, metrik, dan artefak langsung ke active run context
        mlflow.log_params({
            'n_estimators': n_est,
            'max_depth': m_depth,
            'random_state': 42
        })

        y_pred_train = model.predict(X_train)
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')

        mlflow.log_metric('train_accuracy', float(accuracy_score(y_train, y_pred_train)))
        mlflow.log_metric('accuracy', float(test_accuracy))
        mlflow.log_metric('test_f1_score', float(test_f1))

        # ARTIFACT 1: estimator.html
        html_repr = estimator_html_repr(model)
        mlflow.log_text(html_repr, artifact_file='estimator.html')

        # ARTIFACT 2: metric_info.json 
        metric_info = {'accuracy': float(test_accuracy), 'test_f1_score': float(test_f1)}
        mlflow.log_dict(metric_info, artifact_file='metric_info.json')

        # ARTIFACT 3: Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_train, y_pred_train, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.4g', cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix (n={n_est}, depth={m_depth})")
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()
        mlflow.log_figure(fig, artifact_file='training_confusion_matrix.png')
        plt.close(fig)

        # ARTIFACT 4: Classification Report 
        report = classification_report(y_test, y_pred_test)
        mlflow.log_text(report, artifact_file='reports/classification_report.txt')

        # ARTIFACT 5: Filtered Feature Importance Plot
        importances = model.feature_importances_
        feature_names = X_train.columns  
        threshold = 0.03
        important_features_indices = np.where(importances >= threshold)[0]
        sorted_important_features_indices = important_features_indices[np.argsort(importances[important_features_indices])[::-1]]
                        
        if len(sorted_important_features_indices) > 0:
            fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=importances[sorted_important_features_indices],
                y=feature_names[sorted_important_features_indices],
                ax=ax_fi,
                palette='viridis',
                hue=feature_names[sorted_important_features_indices],
                legend=False
            )
            ax_fi.set_title(f"Feature Importance >= {threshold} (n={n_est}, depth={m_depth})")
            ax_fi.set_xlabel('Relative Importance')
            ax_fi.set_ylabel('Features')
            plt.tight_layout()
                            
            mlflow.log_figure(fig_fi, artifact_file='plots/feature_importance.png')
            plt.close(fig_fi)
        
        # Model Logging
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            input_example=input_example
        )

        # 7. Tulis Run ID ke file teks untuk validasi downstream GitHub Actions
        with open('run_id.txt', 'w') as f:
            f.write(best_run_id)
        print(f"Run ID successfully written to run_id.txt: {best_run_id}")
    else:
        print("Error: No active MLflow run detected. Ensure 'mlflow run' initiates the environment.")

if __name__ == '__main__':
    main()