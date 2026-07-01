import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.utils import estimator_html_repr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # 1. Init path dataset
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/final_data_students.csv')
    data = pd.read_csv(file_path)

    # 2. Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['Status', 'Student_ID'], axis=1),
        data['Status'],
        random_state=42,
        test_size=0.2
    )
    
    input_example = X_train.iloc[0:5]
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 70
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    # 3. Initialize and fit the Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion='gini',
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4. Evaluasi Metrik
    train_accuracy = model.score(X_train, y_train)
    y_pred_train = model.predict(X_train)

    test_accuracy = model.score(X_test, y_test)
    y_pred_test = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred_test, average="weighted")

    # Log metrics ke MLflow
    mlflow.log_metrics({
        'train_accuracy': train_accuracy,
        'accuracy': test_accuracy,
        'test_f1_score': test_f1
    })

    # 5. ARTIFACTS
    # Artifact 1: estimator.html
    mlflow.log_text(estimator_html_repr(model), artifact_file='estimator.html')

    # Artifact 2: metric_info.json 
    mlflow.log_dict({'accuracy': test_accuracy, 'test_f1_score': test_f1}, artifact_file='metric_info.json')

    # Artifact 3: Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_train, y_pred_train, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.4g', cmap="Blues", ax=ax)
    ax.set_title(f'Confusion Matrix (n={n_estimators}, depth={max_depth})')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    mlflow.log_figure(fig, artifact_file='training_confusion_matrix.png')
    plt.close(fig)

    # Artifact 4: Classification Report 
    mlflow.log_text(classification_report(y_test, y_pred_test), artifact_file='reports/classification_report.txt')

    # Artifact 5: Filtered Feature Importance Plot
    importances = model.feature_importances_
    feature_names = X_train.columns  
                    
    threshold = 0.03
    important_features_indices = np.where(importances >= threshold)[0]
    sorted_idx = important_features_indices[np.argsort(importances[important_features_indices])[::-1]]
                    
    if len(sorted_idx) > 0:
        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=importances[sorted_idx],
            y=feature_names[sorted_idx],
            ax=ax_fi,
            palette='viridis',
            hue=feature_names[sorted_idx],
            legend=False
        )
        ax_fi.set_title(f'Feature Importance >= {threshold} (n={n_estimators}, depth={max_depth})')
        ax_fi.set_xlabel('Relative Importance')
        ax_fi.set_ylabel('Features')
        plt.tight_layout()
                        
        mlflow.log_figure(fig_fi, artifact_file='plots/feature_importance.png')
        plt.close(fig_fi)
    
    # 6. Model Logging (Dikeluarkan dari pengecekan plot agar selalu tercatat)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='model',
        input_example=input_example
    )

    # 7. Save Run ID for CI/CD pipeline automation
    run_id = mlflow.active_run().info.run_id
    print(f"MLFLOW_RUN_ID: {run_id}")
    
    # Write the active run ID to a text file in current working directory
    with open(os.path.join(os.getcwd(), "run_id.txt"), "w") as f:
        f.write(run_id)
        
    print(f"Run ID {run_id} successfully saved")

    