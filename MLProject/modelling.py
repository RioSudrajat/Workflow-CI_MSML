import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import os

# === 1. Setup DagsHub & MLflow ===
# === 1. Setup DagsHub & MLflow ===
print("Initializing DagsHub & MLflow...")
if not os.getenv("MLFLOW_TRACKING_URI"):
    dagshub.init(repo_owner='RioSudrajat', repo_name='MSML', mlflow=True)
    mlflow.set_experiment("Bank Marketing Experiment")
else:
    print("CI Environment detected. Using existing MLFLOW_TRACKING_URI.")

# === 2. Load Data ===
print("Loading data...")
# Asumsi script dijalankan dari dalam folder 'Membangun_model'
if os.path.exists('dataset/train.csv'):
    train_path = 'dataset/train.csv'
    test_path = 'dataset/test.csv'
else:
    # Fallback paths
    train_path = 'dataset/train.csv'
    test_path = 'dataset/test.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Separate Features and Target
X_train = df_train.drop('y', axis=1)
y_train = df_train['y']
X_test = df_test.drop('y', axis=1)
y_test = df_test['y']

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# === 3. Train & Log ===
print("Starting training...")
with mlflow.start_run(run_name="RandomForest_Baseline"):
    # Define Hyperparameters
    n_estimators = 100
    max_depth = 10
    random_state = 42
    
    # Log Parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForestClassifier")
    
    # Train Model
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    
    # Predict (Class and Proba)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1] # Probability for class 1
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) # ROC AUC calc
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}") 
    
    # Log Metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc) # Log ROC AUC
    
    # Log Model
    mlflow.sklearn.log_model(clf, "model")
    
    # === 4. Create & Log Artifacts ===
    # A. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    
    # B. Feature Importance
    importances = clf.feature_importances_
    feature_names = X_train.columns
    feature_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_df = feature_df.sort_values(by='importance', ascending=False).head(20) # Top 20
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_df, palette='viridis')
    plt.title('Top 20 Feature Importance')
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    
    # C. ROC Curve (New Artifact)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    plt.close()
    
    print("Training complete. Artifacts (including ROC Curve) logged to DagsHub.")
