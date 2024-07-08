import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV  
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.feature_selection import SelectKBest, f_regression 
from sklearn.impute import SimpleImputer
import json
for MTU_Project import Constants as const

# Define MAPE
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Function to train and evaluate model
def train_and_evaluate(df, features, target_col, model_pipeline):
    X = df[features]
    y = df[target_col]

    # Split data into train, validation, and test sets (80-10-10)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1111, random_state=42)  # 0.1111 * 0.9 = 0.1

    # Train model on training set
    model_pipeline.fit(X_train, y_train)

    # Validate model on validation set
    y_val_pred = model_pipeline.predict(X_val)
    val_score = mean_squared_error(y_val, y_val_pred)

    # Test model on test set
    y_test_pred = model_pipeline.predict(X_test)
    test_mape_score = mape(y_test, y_test_pred)
    test_mae_score = mean_absolute_error(y_test, y_test_pred)

    return val_score, test_mape_score, test_mae_score

# Function to perform Randomized Search CV and get the best model
def perform_random_search(X_train, y_train, model, param_distributions):
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=3, n_jobs=-1, random_state=42, scoring='neg_mean_squared_error')
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

# Main function to train models
def train_models():
    target_column = 'litres'
    lag = 52

    # Load data
    full_df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')

    # Data cleaning and preprocessing
    full_df = full_df.drop(columns=['EU_milk_price_without UK', 'feed_ex_port', 'Malta_milk_price', 'Croatia_milk_price', 'Malta_Milk_Price'])
    imputer = SimpleImputer(strategy='mean')
    full_df['yield_per_supplier'] = imputer.fit_transform(full_df[['yield_per_supplier']])
    columns_with_nan = full_df.columns[full_df.isna().any()].tolist()
    full_df = full_df.drop(columns=columns_with_nan)

    df = full_df.copy()

    # Lag features
    for i in range(1, lag+1):
        df[f'{target_column}_next_{i}weeks'] = df[target_column].shift(-i)
    df = df.dropna()

    # Features and target
    features = [col for col in df.columns if not col.startswith(f'{target_column}_next_') and col != 'Date']
    
    # Model and parameter configurations
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'RandomForest': RandomForestRegressor()
    }

    param_distributions = {
        'LinearRegression': {},
        'Ridge': {'alpha': np.logspace(-4, 4, 20)},
        'RandomForest': {'n_estimators': [10, 50, 100, 200], 'max_features': ['auto', 'sqrt', 'log2']}
    }

    scalers = {
        'StandardScaler': StandardScaler()
    }

    scoring_methods = {
        'f_regression': f_regression,
        None: None
    }

    results = []

    # Training, validation, and testing over multiple iterations
    for iteration in range(30):
        for model_name, model in models.items():
            for scaler_name, scaler in scalers.items():
                for scoring_name, scoring_func in scoring_methods.items():
                    pipeline_steps = [('scaler', scaler)]
                    if scoring_func:
                        pipeline_steps.insert(0, ('selectkbest', SelectKBest(score_func=scoring_func, k=10)))
                    pipeline_steps.append(('model', model))
                    model_pipeline = Pipeline(pipeline_steps)

                    # Perform random search to get the best model
                    best_model = perform_random_search(df[features], df[f'{target_column}_next_1weeks'], model_pipeline, param_distributions[model_name])

                    mape_scores = []
                    mae_scores = []

                    # Evaluate over multiple lags
                    for week in range(1, lag + 1):
                        target_col = f'{target_column}_next_{week}weeks'
                        val_score, test_mape_score, test_mae_score = train_and_evaluate(df, features, target_col, best_model)

                        mape_scores.append(test_mape_score)
                        mae_scores.append(test_mae_score)
                        print(f'Iteration {iteration + 1}, {model_name}, {scaler_name}, {scoring_name}, Week {week}, Validation Score: {val_score:.2f}, Test MAPE: {test_mape_score:.2f}, Test MAE: {test_mae_score:.2f}')

                    results.append({
                        'Iteration': iteration + 1,
                        'Model': model_name,
                        'Scaler': scaler_name,
                        'Feature Selection': scoring_name,
                        'MAPE': mape_scores,
                        'MAE': mae_scores
                    })

    # Save results to JSON
    with open('result/final_results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

train_models()
