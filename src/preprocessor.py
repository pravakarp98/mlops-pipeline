import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = [
    'Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

CATEGORICAL_FEATURES = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
]

NEW_ENGINEERED_FEATURES = [
    'total_approved', 'total_enrolled', 'total_evaluations', 
    'total_credited', 'average_grade', 'grade_per_approved', 'pass_rate'
]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input to FeatureEngineer must be a pandas DataFrame")
        
        X_transformed = X.copy()
        X_transformed['total_approved'] = X['Curricular units 1st sem (approved)'] + X['Curricular units 2nd sem (approved)']
        X_transformed['total_enrolled'] = X['Curricular units 1st sem (enrolled)'] + X['Curricular units 2nd sem (enrolled)']
        X_transformed['total_evaluations'] = X['Curricular units 1st sem (evaluations)'] + X['Curricular units 2nd sem (evaluations)']
        X_transformed['total_credited'] = X['Curricular units 1st sem (credited)'] + X['Curricular units 2nd sem (credited)']
        X_transformed['average_grade'] = (X['Curricular units 1st sem (grade)'] + X['Curricular units 2nd sem (grade)']) / 2
        X_transformed['grade_per_approved'] = X_transformed['average_grade'] / (X_transformed['total_approved'] + 1)
        X_transformed['pass_rate'] = X_transformed['total_approved'] / (X_transformed['total_enrolled'] + 1)

        return X_transformed

def create_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES + NEW_ENGINEERED_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
    
    full_pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

if __name__ == "__main__":
    df = pd.read_csv('/home/pravakar/git_repo/ml_pipelines/data/train.csv')
    X = df.drop(columns=["Target", 'id'])
    y = df['Target']

    my_pipeline = create_preprocessor()
    print("Fiting and Transforing the data with full pipeline...")
    X_transformed = my_pipeline.fit_transform(X, y)
    
    print("\nData transformation complete...")
    print(f"Original number of features: {len(X.columns)}")
    print(f"Transformed number of features: {X_transformed.shape[1]}")
    
    print("\nPipeline steps:")
    print(my_pipeline.steps)