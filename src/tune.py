import yaml
import mlflow
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from data_loader import load_data
from preprocessor import create_preprocessor
from utils import save_model, evaluate_model, save_metrics

MODEL_TO_TUNE = 'GradientBoosting'

def run_tuning():
    print("Loading params.yaml...")
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    tuning_config = params['tuning']
    model_mapping = {
        'RandomForest': RandomForestClassifier(n_jobs=-1, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    
    if MODEL_TO_TUNE not in model_mapping:
        raise ValueError(f"Model '{MODEL_TO_TUNE}' is not defined in src/tune.py mapping.")

    mlflow.set_experiment("Student Risk - Hyperparameter Tuning")
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    print("Loading data...")
    split_params = params['data_split']
    X_train, X_val, y_train, y_val, _, _ = load_data(
        'data/train.csv', 
        test_size=split_params['test_size'], 
        random_state=split_params['random_state']
    )

    print("Encoding target variable...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)

    print(f"Preparing pipeline for {MODEL_TO_TUNE}...")
    preprocessor = create_preprocessor()
    model_instance = model_mapping[MODEL_TO_TUNE]
    param_grid = tuning_config[MODEL_TO_TUNE]['param_grid']

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_instance)
    ])

    print(f"Starting RandomizedSearchCV ({tuning_config['n_iter']} iterations)...")
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=tuning_config['n_iter'],
        cv=tuning_config['cv'],
        scoring=tuning_config['scoring'],
        n_jobs=tuning_config['n_jobs'],
        random_state=42,
        verbose=1
    )

    with mlflow.start_run(run_name=f"Tune_{MODEL_TO_TUNE}"):
        search.fit(X_train, y_train_encoded)
        
        print(f"\nBest CV Score: {search.best_score_:.4f}")
        print(f"Best Parameters: {search.best_params_}")

        print("Evaluating best model on validation set...")
        best_model = search.best_estimator_
        metrics = evaluate_model(best_model, X_val, y_val_encoded, le.classes_)
        
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        
        print("Saving optimized model to artifacts/pipeline.pkl...")
        save_model(best_model, 'artifacts/pipeline.pkl')
        save_model(le, 'artifacts/label_encoder.pkl')

if __name__ == "__main__":
    run_tuning()
