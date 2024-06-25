# ___________________________________________________________________________________________________
#                                        IMPORTS
#____________________________________________________________________________________________________

import pandas as pd  # For data manipulation with Pandas
import numpy as np   # For numerical calculations with NumPy
import matplotlib.pyplot as plt  # For plotting
import tensorflow as tf  # For using TensorFlow for deep learning
from tensorflow.keras.models import Sequential  # Keras Sequential model for neural networks
from tensorflow.keras.layers import Dense, Dropout  # Dense and Dropout layers for neural networks
from sklearn.model_selection import cross_validate, train_test_split, RandomizedSearchCV  # Tools for cross-validation and hyperparameter search
from sklearn.pipeline import Pipeline  # For creating preprocessing and modeling pipelines
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler  # Different scalers for data normalization
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error  # Model evaluation metrics
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge, PassiveAggressiveRegressor  # Linear and Bayesian regression models
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor  # Decision tree models
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor  # Ensemble models like random forests and boosting
from sklearn.neighbors import KNeighborsRegressor  # K-nearest neighbors regression model
from sklearn.svm import LinearSVR, NuSVR, SVR  # Support Vector Machine models for regression
from sklearn.neural_network import MLPRegressor  # Multi-layer Perceptron neural network for regression
from sklearn.gaussian_process import GaussianProcessRegressor  # Gaussian Process regression model
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression  # Feature selection methods
from statsmodels.tsa.seasonal import seasonal_decompose  # For seasonal decomposition of time series
from I_dataset_creation import visualization_sorted_tab  # Specific functions for dataset creation
from II_preprocessing import check_tab, feature_selection_correlation, scale_data, time_series_analysis, feature_selection  # Data preprocessing functions
from III_train_models import NARX_model, train_arima_model, SANN_model, knn_regressor, linear_regression_sklearn, PolynomialRegression, Neural_Network_Pytorch, ANN_model, random_forest, gradient_boosting_regressor  # Functions for model training
import openpyxl  # For handling Excel files with openpyxl
from openpyxl.styles import PatternFill  # To add styles to Excel cells
from openpyxl.utils.dataframe import dataframe_to_rows  # To convert DataFrame to Excel rows

import constants as const  # Constants file containing predefined models, scalers, and hyperparameters

# ___________________________________________________________________________________________________
#                                        FUNCTIONS
#____________________________________________________________________________________________________

def mape(y_true, y_pred):
    """
    Function to calculate Mean Absolute Percentage Error (MAPE).

    Args:
    - y_true (array-like): Actual values.
    - y_pred (array-like): Predicted values.

    Returns:
    - float: MAPE as a percentage.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ___________________________________________________________________________________________________
#                                        CODE
#____________________________________________________________________________________________________

data_creation = False  # Dataset creation flag

if data_creation:
    # Read data
    df_prices = pd.read_excel('spreadsheet/Data_prices_monthly.xlsx')
    df_yield = pd.read_csv('initial_datas/Data_weekly_yield.csv')

    # Data preprocessing steps
    df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    df_yield['year_month'] = df_yield['year_month'].astype(str)
    df_yield['year'] = df_yield['year_month'].str[:4].astype(int)
    df_yield['month'] = df_yield['year_month'].str[4:6].astype(int)
    df_yield['Date'] = pd.to_datetime(df_yield['year'].astype(str) + df_yield['month'].astype(str).str.zfill(2), format='%Y%m') + pd.to_timedelta((df_yield['week'] - 1) * 7, unit='D')
    df_yield['yield_per_supplier'] = df_yield['litres'] / df_yield['num_suppliers']
    df_yield['cos_week'] = np.cos(df_yield['week'] * (2 * np.pi / 52))
    df_merged = df_yield.merge(df_prices, on='Date', how='left')
    df_merged.drop(df_merged.columns[4:50], axis=1, inplace=True)
    df_merged = df_merged[(df_merged['Date'] >= '2009-01-01') & (df_merged['Date'] <= '2021-11-30')].copy()
    df_merged.dropna(axis=1, how='all', inplace=True)
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
    df_merged.dropna(subset=['num_suppliers', 'feed_bag', 'feed_bulk', 'feed_totals'], inplace=True)
    df_merged.sort_values(by='Date', inplace=True)
    columns_to_drop = ['Unnamed: 0', 'feed_ex_port', 'Croatia_milk_price', 'Malta_milk_price', 'EU_milk_price_without UK']
    df_merged.drop(columns=columns_to_drop, inplace=True)
    columns_to_drop = df_merged.columns[-27:]
    df_merged.drop(columns=columns_to_drop, inplace=True)
    df = df_merged.drop(columns=['year_month', 'week', 'year', 'month'])
    milk_price_columns = [col for col in df.columns if col.endswith('_milk_price')]
    df.dropna(subset=milk_price_columns, inplace=True)

    # Fill NaN values with column mean
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    df['Date'] = pd.to_datetime(df['Date'])

    # Add 'Year' and 'Month' columns
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df_monthly = df.groupby(['Year', 'Month']).mean().reset_index()
    print(df_monthly.head().to_markdown(index=False, numalign="left", stralign="left"))

    # Time series analysis
    target_column = 'litres'
    past_time = 7
    df = time_series_analysis(past_time, df, target_column)

    # Save DataFrame to Excel
    df.reset_index(inplace=True)
    df.to_excel('spreadsheet/test.xlsx', index=False)

# Read preprocessed data from Excel
df = pd.read_excel('spreadsheet/test.xlsx')
df_pre = df[df['Date'] < '2015-04-01']
df_post = df[df['Date'] >= '2015-04-01']

X_train = df_pre.drop(columns=['litres', 'Date'])
y_train = df_pre['litres']

X_test = df_post.drop(columns=['litres', 'Date'])
y_test = df_post['litres']

mape_scorer = make_scorer(mape, greater_is_better=False)

iterations = 2 # Number of iterations
all_results = []

# Loop through iterations
for iteration in range(iterations):
    # Loop through different models
    for name, model in const.models.items():
        # Loop through different scalers
        for scaler_name, scaler in const.scalers.items():
            pipe = Pipeline([('scaler', scaler), ('model', model)])
            param_grid = const.hyperparameters.get(name, {})  # Get hyperparameters

            # Split data into train and validation sets
            X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            # Loop through different feature scoring methods
            for scoring_name in ['f_regression', 'mutual_info_regression', 'pearson', None]:
                if scoring_name:
                    weights = feature_selection(X_train_split, y_train_split, method=scoring_name)
                    selected_features = weights.nlargest(10, 'weight')['feature']
                    X_train_split_selected = X_train_split[selected_features]
                    X_val_selected = X_val[selected_features]
                else:
                    X_train_split_selected = X_train_split
                    X_val_selected = X_val
                    scoring_name = 'None'

                # Randomized hyperparameter search
                random_search = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=3, scoring=mape_scorer, cv=3, random_state=42)
                random_search.fit(X_train_split_selected, y_train_split)

                # Validation set prediction
                y_pred_val = random_search.predict(X_val_selected)
                val_mape = mape(y_val, y_pred_val)

                # Store results in dictionary
                result = {
                    'Model': name,
                    'Feature Scoring': scoring_name,
                    'Feature Scaling': scaler_name,
                    'Best Parameters': random_search.best_params_,
                    'Validation MAPE': val_mape,
                    'Model Instance': random_search.best_estimator_,
                    'Iteration': iteration + 1
                }
                all_results.append(result)

# Convert list of dictionaries to DataFrame
results_df = pd.DataFrame(all_results)

# Calculate mean MAPE for each model combination
mean_results = results_df.groupby(['Model', 'Feature Scoring', 'Feature Scaling'])['Validation MAPE'].mean().reset_index()

# Find best parameters for each model
best_results = mean_results.loc[mean_results.groupby('Model')['Validation MAPE'].idxmin()]
best_results = best_results.sort_values(by='Validation MAPE', ascending=True)

# Print and save results
print(results_df.to_markdown(index=False))
print(mean_results.to_markdown(index=False))
print(best_results.to_markdown(index=False))

results_df.to_excel('all_models_results.xlsx', index=False)
mean_results.to_excel('mean_models_results.xlsx', index=False)
best_results.to_excel('best_models_results.xlsx', index=False)

# Initialize a dictionary to store MAPE results for each model and prediction horizon
mape_results = {model: [] for model in best_results['Model']}

# Define the prediction horizons
prediction_horizons = range(1, 16)  # 1 to 15 weeks ahead

# Loop through the best models and calculate MAPE for each prediction horizon
for index, row in best_results.iterrows():
    model_name = row['Model']
    best_model_instance = row['Best Parameters']
    scaler_name = row['Feature Scaling']
    scoring_name = row['Feature Scoring']

    # Fit the best model on the entire training data
    pipe = Pipeline([('scaler', const.scalers[scaler_name]), ('model', const.models[model_name].set_params(**best_model_instance))])
    if scoring_name != 'None':
        weights = feature_selection(X_train, y_train, method=scoring_name)
        selected_features = weights.nlargest(10, 'weight')['feature']
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
    else:
        X_train_selected = X_train
        X_test_selected = X_test

    pipe.fit(X_train_selected, y_train)

    for horizon in prediction_horizons:
        # Shift the test data by the horizon to create a prediction set
        X_test_horizon = X_test_selected.shift(-horizon)
        y_test_horizon = y_test.shift(-horizon)

        # Drop NaN values resulting from the shift
        X_test_horizon = X_test_horizon.dropna()
        y_test_horizon = y_test_horizon.iloc[:len(X_test_horizon)]

        # Make predictions
        y_pred_horizon = pipe.predict(X_test_horizon)

        # Calculate MAPE
        mape_horizon = mape(y_test_horizon, y_pred_horizon)
        mape_results[model_name].append(mape_horizon)

# Plot the MAPE results for each model over the prediction horizons
plt.figure(figsize=(12, 8))
for model_name, mape_values in mape_results.items():
    plt.plot(prediction_horizons, mape_values, marker='o', label=model_name)

plt.xlabel('Prediction Time Horizon (weeks ahead)')
plt.ylabel('MAPE')
plt.title('MAPE for Different Models Over Prediction Horizons')
plt.legend()
plt.grid(True)
plt.show()
