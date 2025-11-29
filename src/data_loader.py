import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath, test_size, random_state):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None, None, None, None, None
    
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_val, y_train, y_val, X, y