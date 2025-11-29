import joblib
import json
import os
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    
def save_metrics(metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"mterics saved to {filepath}")

def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    report_dict = classification_report(y_test, y_pred, target_names = class_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=class_names)
    
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1 Score (Weighted): {f1:.4f}")
    print(f"    Precision (Weighted): {precision:.4f}")
    print(f"    Recall (Weighted): {recall:.4f}")
    print("\n   Classification Report:")
    print(report_str)
    
    metrics = {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "classification_report": report_dict
    }
    
    return metrics