import yaml
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from data_loader import load_data
from preprocessor import create_preprocessor
from utils import save_model, evaluate_model, save_metrics

print("Loading configuration from params.yaml...")
with open('/home/pravakar/git_repo/ml_pipelines/params.yaml', 'r') as f:
    params = yaml.safe_load(f)

mlflow.set_experiment("Student Risk Prediction")
mlflow.sklearn.autolog()
print("MLflow experiment set.")

print("Loading and splitting data...")
split_params = params['data_split']

X_train, X_val, y_train, y_val, _, _ = load_data(
    '/home/pravakar/git_repo/ml_pipelines/data/train.csv',
    test_size = split_params['test_size'],
    random_state=split_params['random_state']
)

print("Encoding target variable (y)...")

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

save_model(le, '/home/pravakar/git_repo/ml_pipelines/artifacts/label_encoder.pkl')
print("LabelEncoder saved to artifacts/label_encoder.pkl")

print("Initializing models from params.yaml...")
models_config = params['models']

models = {
    'LogisticRegression': LogisticRegression(**models_config.get('LogisticRegression', {})),
    'RandomForest': RandomForestClassifier(**models_config.get('RandomForest', {})),
    'SVC': SVC(**models_config.get('SVC', {})),
    'DecisionTree': DecisionTreeClassifier(**models_config.get('DecisionTree', {})),
    'GradientBoosting': GradientBoostingClassifier(**models_config.get('GradientBoosting', {})),
    'AdaBoost': AdaBoostClassifier(**models_config.get('AdaBoost', {})),
    'KNN': KNeighborsClassifier(**models_config.get('KNN', {})),
    'GaussianNB': GaussianNB(**models_config.get('GaussianNB', {}))
}

preprocessor_pipeline = create_preprocessor()
print('Preprocessing pipeline created.')

best_model_pipeline = None
best_accuracy = 0.0
best_model_name = ""

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"Train_{model_name}") as run:
        print(f"\n--- Training Model: {model_name} ---")
        
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor_pipeline),
            ('model', model)
        ])
        
        print("Training full pipeline...")
        full_pipeline.fit(X_train, y_train_encoded)
        
        print("Evaluating pipeline...")
        metrics = evaluate_model(full_pipeline, X_val, y_val_encoded, le.classes_)
        
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        
        save_metrics(metrics, f"/home/pravakar/git_repo/ml_pipelines/artifacts/metrics_{model_name}.json")
        mlflow.log_artifact(f"/home/pravakar/git_repo/ml_pipelines/artifacts/metrics_{model_name}.json")
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model_pipeline = full_pipeline
            best_model_name = model_name
            print(f"New best model found: {model_name} with accuracy: {best_accuracy:.4f}")
            
print("\nTraining complete.")
if best_model_pipeline:
    print(f"Saving best model ({best_model_name}) to 'artifacts/pipeline.pkl'...")
    save_model(best_model_pipeline, 'artifacts/pipeline.pkl')
    mlflow.log_artifact('artifacts/pipeline.pkl', artifact_path="best_model")
else:
    print("No models were trained successfully.")
    
print("Process finished.")