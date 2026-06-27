import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
# 1. Activate MLautolog
mlflow.autolog(log_models=True)

# 2. Define or select the MLflow experiment
mlflow.set_experiment('Students-Performance')

# 3. Data Preparation
data = pd.read_csv('data/final_data_students.csv')

# 4. Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Status', 'Student_ID'], axis=1),
    data['Status'],
    random_state=42,
    test_size=0.2
)
input_example = X_train[0:5]

# 5. Running MLflow
with mlflow.start_run() as activate_run:
    # Log parameters
    n_estimators = 70
    max_depth = 15
    
    # Initialize and fit the Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion='gini',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Metrics evaluation
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    # Log metrics
    mlflow.log_metric('accuracy', accuracy)

    # Artifact Classification Report
    report = classification_report(y_test, y_pred)
    mlflow.log_text(report, artifact_file='reports/classification_report.txt')

    # Artifact Feature Importance Plot
    importance = model.feature_importances_
    feature_names = X_train.columns
    threshold = 0.03
    important_features_indices = np.where(importance >= threshold)[0]
    sorted_important_features_indices = important_features_indices[np.argsort(importance[important_features_indices])[::-1]]

    if len(sorted_important_features_indices) > 0:
        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))

        sns.barplot(
            x=importance[sorted_important_features_indices],
            y=feature_names[sorted_important_features_indices],
            ax=ax_fi,
            palette='viridis',
            hue=feature_names[sorted_important_features_indices],
            legend=False
        )
        ax_fi.set_title(f'Feature Importance >= {threshold} (n={n_estimators}, depth={max_depth})')
        ax_fi.set_xlabel('Relative Importance')
        ax_fi.set_ylabel('Features')
        plt.tight_layout()
                                    
        mlflow.log_figure(fig_fi, artifact_file='plots/feature_importance.png')
        plt.close(fig_fi)
