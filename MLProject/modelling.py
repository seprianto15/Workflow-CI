import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils import estimator_html_repr

def main():
    # 1. Pastikan Kredensial DagsHub Ditarik dari Environment Variables GitHub Secrets
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME', '')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN', '')
    os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI', '')

    # Define or select the MLflow experiment
    mlflow.set_experiment('Students-Performance')

    # Load the preprocessing dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(base_dir, 'data', 'final_data_students.csv'))

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['Status', 'Student_ID'], axis=1),
        data['Status'],
        random_state=42,
        test_size=0.2
    )
    input_example = X_train.iloc[0:5] # Menggunakan iloc agar lebih stabil

    # Set up parameter random search
    n_estimators_range = np.linspace(10, 100, 5, dtype=int) 
    max_depth_range = np.linspace(1, 20, 5, dtype=int) 

    best_accuracy = 0
    best_params = {}

    # Execute Random Search
    for n_estimators in n_estimators_range:
        for max_depth in max_depth_range:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                criterion='gini',
                random_state=42
            )
            model.fit(X_train, y_train)

            test_accuracy = model.score(X_test, y_test)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                }
                print(f'Accuracy: {best_accuracy:.4f} -> n_estimators: {n_estimators}, max_depth: {max_depth}')

    best_n_est = best_params['n_estimators']
    best_m_depth = best_params['max_depth']

    # 2. Tangkap Sesi Run yang sudah dibuat otomatis oleh 'mlflow run' (Tanpa with mlflow.start_run)
    active_run = mlflow.active_run()
    if active_run:
        current_run_id = active_run.info.run_id
        
        # Lakukan Logging Hyperparameters Manual
        mlflow.log_params({
            'n_estimators': best_n_est,
            'max_depth': best_m_depth,
            'criterion': 'gini',
        })
        
        # Latih Ulang Model Terbaik
        best_model = RandomForestClassifier(
            n_estimators=best_n_est,
            max_depth=best_m_depth,
            criterion='gini',
            random_state=42,
        )
        best_model.fit(X_train, y_train)

        # Metrics evaluation
        train_accuracy = best_model.score(X_train, y_train)
        y_pred_train = best_model.predict(X_train)

        test_accuracy = best_model.score(X_test, y_test)
        y_pred_test = best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred_test, average="weighted")

        mlflow.log_metrics({
            'train_accuracy': train_accuracy,
            'accuracy': test_accuracy,
            'test_f1_score': test_f1
        })
        
        # ARTIFACT 1: HTML model representation
        html_repr = estimator_html_repr(best_model)
        mlflow.log_text(html_repr, artifact_file='estimator.html')

        # ARTIFACT 2: Summary metric info json
        metric_info = {'accuracy': test_accuracy, 'test_f1_score': test_f1}
        mlflow.log_dict(metric_info, artifact_file='metric_info.json')

        # ARTIFACT 3: Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_train, y_pred_train, normalize='true')
        sns.heatmap(cm, annot=True, fmt=".4g", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix (n={best_n_est}, depth={best_m_depth})")
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()

        mlflow.log_figure(fig, artifact_file='training_confusion_matrix.png')
        plt.close(fig)

        # ARTIFACT 4: Classification Report Text File
        os.makedirs('reports', exist_ok=True)
        report = classification_report(y_test, y_pred_test)
        mlflow.log_text(report, artifact_file='reports/classification_report.txt')

        # ARTIFACT 5: Filtered Feature Importance Plot
        importances = best_model.feature_importances_
        feature_names = X_train.columns  
                                
        threshold = 0.03
        important_features_indices = np.where(importances >= threshold)[0]
        sorted_important_features_indices = important_features_indices[np.argsort(importances[important_features_indices])[::-1]]
                                
        if len(sorted_important_features_indices) > 0:
            os.makedirs('plots', exist_ok=True)
            fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=importances[sorted_important_features_indices],
                y=feature_names[sorted_important_features_indices],
                ax=ax_fi,
                palette='viridis',
                hue=feature_names[sorted_important_features_indices],
                legend=False
            )
            ax_fi.set_title(f"Feature Importance >= {threshold} (n={best_n_est}, depth={best_m_depth})")
            ax_fi.set_xlabel('Relative Importance')
            ax_fi.set_ylabel('Features')
            plt.tight_layout()
            mlflow.log_figure(fig_fi, artifact_file='plots/feature_importance.png')
            plt.close(fig_fi)
        
        # Model Logging
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path='model',
            input_example=input_example,
        )

        # 3. Ekspor Run ID ke root directory agar dapat ditarik oleh CI/CD GitHub Actions
        if 'GITHUB_ENV' in os.environ:
            with open(os.environ['GITHUB_ENV'], 'a') as f:
                f.write(f'MLFLOW_RUN_ID={current_run_id}\n')
                
        print(f"\nRun ID successfully captured: {current_run_id}")
        print(f"Final Accuracy in Dashboard: {best_accuracy:.4f}")

if __name__ == '__main__':
    main()