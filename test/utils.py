
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error

def save_json(data, filepath):
    with open(filepath, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def train_and_predict(df, features, target_col, model_pipeline):
    X = df[features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    return y_test, y_pred

def evaluate_model(y_test, y_pred):
    mape_score = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    mae_score = mean_absolute_error(y_test, y_pred)
    return mape_score, mae_score
