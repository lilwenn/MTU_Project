import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from II_visualization import plot_correlation_matrix  
from III_preprocessing import time_series_analysis, determine_lags 
import Constants as const


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def iteration_mean(mean_results, result, iteration):
        
    for model_name in result.keys():
        mape_lists = []
        prediction_lists = []

        # Collect all MAPE and Prediction lists from all iterations
        for i in range(iteration + 1):
            mape_lists.append(result[model_name][i]['MAPE'])
            prediction_lists.append(result[model_name][i]['Prediction'])

        # Transpose lists to calculate mean for each week
        transposed_mape_lists = list(zip(*mape_lists))
        transposed_prediction_lists = list(zip(*prediction_lists))

        mean_mape = [np.mean(week_mape) for week_mape in transposed_mape_lists]
        mean_prediction = [np.mean(week_prediction) for week_prediction in transposed_prediction_lists]

        mean_results[model_name] = {
            'MAPE': mean_mape,
            'Prediction': mean_prediction
        }

    return mean_results



def train_and_predict(df, features, target_col, model_pipeline):
    X = df[features]
    y = df[target_col]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model pipeline
    model_pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model_pipeline.predict(X_test)

    # Evaluate the model using MAPE and MAE
    mape_score = mape(y_test, y_pred)
    mae_score = mae(y_test, y_pred)

    # Get the selected features from SelectKBest
    selected_features = model_pipeline.named_steps['selectkbest'].get_support(indices=True)
    selected_feature_names = [features[i] for i in selected_features]

    # Predict future values
    last_row = df.iloc[-1][features]
    future = pd.DataFrame([last_row])
    prediction = model_pipeline.predict(future)

    return mape_score, mae_score, prediction[0], selected_feature_names

def determine_lags(df, target_column, max_lag=40):
    """
    Determine the number of lags based on the autocorrelation function (ACF) and partial autocorrelation function (PACF).
    
    Args:
    df (pd.DataFrame): DataFrame containing the time series data.
    target_column (str): The name of the target column in the DataFrame.
    max_lag (int): The maximum number of lags to consider.
    
    Returns:
    int: The optimal number of lags.
    """
    series = df[target_column]
    
    # Plot ACF and PACF
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, lags=max_lag, ax=ax[0])
    plot_pacf(series, lags=max_lag, ax=ax[1])
    
    ax[0].set_title('Autocorrelation Function (ACF)')
    ax[1].set_title('Partial Autocorrelation Function (PACF)')
    plt.savefig(f"visualization/Autocorrelation_{target_column}.png")
    
    # Determine the optimal number of lags using acf function
    acf_values = acf(series, nlags=max_lag, alpha=0.05)[0]
    
    for lag in range(1, max_lag + 1):
        if abs(acf_values[lag]) < 1.96/np.sqrt(len(series)):
            optimal_lag = lag
            break
    else:
        optimal_lag = max_lag
    
    print(f"Optimal number of lags: {optimal_lag}")
    return optimal_lag


def load_and_preprocess_data():
    """
    Load data from Excel file, perform data cleaning by dropping specified columns,
    impute missing values in specified columns, and create lagged features.

    Args:
    - file_path (str): Path to the Excel file containing the data.
    - columns_to_drop (list): List of columns to drop from the DataFrame.
    - columns_to_impute (list): List of columns to impute missing values.

    Returns:
    - df (DataFrame): Preprocessed DataFrame ready for model training.
    - result_file (str): File path where lagged results will be saved.
    """

    # Load your data
    full_df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')

    # Clean your data (if necessary)
    full_df = full_df.drop(columns=['Week','EU_milk_price_without UK', 'feed_ex_port', 'Malta_milk_price', 'Croatia_milk_price', 'Malta_Milk_Price'])

    # Define columns to impute
    columns_to_impute = ['yield_per_supplier']

    # Utilize SimpleImputer to impute missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    full_df[columns_to_impute] = imputer.fit_transform(full_df[columns_to_impute])

    columns_with_nan = full_df.columns[full_df.isna().any()].tolist()
    full_df = full_df.drop(columns=columns_with_nan)

    df = full_df

    for i in range(1, const.forecast_weeks + 1):
        df.loc[:, f'{const.target_column}_next_{i}weeks'] = df[const.target_column].shift(-i)

    df = df.dropna()

    past_time = determine_lags(df, const.target_column, max_lag=40)

    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)

    df = time_series_analysis(past_time, df, const.target_column)
    df.reset_index(inplace=True)

    df.to_excel('spreadsheet/lagged_results.xlsx', index=False)
    return df


def train_models(df, model_name):
    """
    Train multiple machine learning models on preprocessed data, evaluate their performance,
    and save results including MAPE, MAE, and predictions for each model, scaler, and scoring method combination.

    Args:
    - df (DataFrame): Preprocessed DataFrame obtained from load_and_preprocess_data().
    - result_file (str): File path where lagged results were saved.
    - features (list): List of feature columns for model training.
    - const (object): Object containing constants such as models, scalers, scoring methods, etc.

    Outputs:
    - Saves JSON files with evaluation metrics ('result/week_without_scale_weeks.json').
    - Saves correlation matrices for selected features as PNG files in 'visualization/correlation/'.
    """

    # Define features (use all columns except 'Date' and target columns)
    features = [col for col in df.columns if not col.startswith(f'{const.target_column}_next_') and col != 'Date']

    result = {}
    result[model_name] = {}
    
    for scaler_name, scaler in const.scalers.items():
        result[model_name][scaler_name] = {}

        for scoring_name, scoring_func in const.scoring_methods.items():
            result[model_name][scaler_name][scoring_name] = {}
            result[model_name][scaler_name][scoring_name]['MAPE'] = {}
            result[model_name][scaler_name][scoring_name]['MAE'] = {}
            result[model_name][scaler_name][scoring_name]['Prediction'] = {}

            # Default scoring function and k value if scoring_func is None
            default_scoring_func = f_regression
            default_k = 5

            pipeline = Pipeline([
                ('selectkbest', SelectKBest(score_func=scoring_func if scoring_func else default_scoring_func, k=const.k_values.get(scoring_name, default_k))),
                ('scaler', scaler),
                ('model', model)
            ])

            selected_features_set = set()

            for week in range(1, const.forecast_weeks + 1):
                target_col = f'{const.target_column}_next_{week}weeks'
                mape_score, mae_score, prediction, selected_features = train_and_predict(df, features, target_col, pipeline)
                print(f'MAPE for {model_name} with {target_col}, scaler {scaler_name}, scoring {scoring_name}: {mape_score:.2f}')

                result[model_name][scaler_name][scoring_name]['MAPE'][f'week_{week}'] = mape_score
                result[model_name][scaler_name][scoring_name]['MAE'][f'week_{week}'] = mae_score
                result[model_name][scaler_name][scoring_name]['Prediction'][f'week_{week}'] = prediction

                selected_features_set.update(selected_features)

            plot_correlation_matrix(df, const.target_column , list(selected_features_set), output_file=f'visualization/correlation/correlation_matrix_{model_name}_{week}{scaler_name}_{scoring_name}.png')

            with open(f'result/by_model/{model_name}_{week}week.json', 'w') as json_file:
                json.dump(result[model_name], json_file, indent=4)


def find_best_combination(model_data):
    best_mape = float('inf')
    best_combination = None
    for scaler_name, scaler_data in model_data.items():
        for scoring_name, metrics in scaler_data.items():
            if 'MAPE' in metrics:
                mape = sum(metrics["MAPE"].values()) / len(metrics["MAPE"])
                if mape < best_mape:
                    best_mape = mape
                    best_combination = (scaler_name, scoring_name, metrics)
    return best_combination


if __name__ == "__main__":

    df = load_and_preprocess_data()
    for model_name, model in const.models.items():
        train_models(df, model)




    with open('result/week_without_scale_52weeks.json', 'r') as json_file:
        data = json.load(json_file)


    best_combinations = {}

    folder_path = 'result/by_model'
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        # Determinar el nombre del modelo a partir del nombre de archivo
        model_name = file_name.split('_')[0]  # Obtener el nombre del modelo

        # Obtener datos específicos del modelo del archivo JSON
        model_data = data.get(model_name, {})  # Obtener datos del modelo del JSON

        # Encontrar la mejor combinación para el modelo actual
        best_combination = find_best_combination(model_data)

        # Guardar la mejor combinación y su MAPE asociado
        best_combinations[model_name] = {
            'best_combination': best_combination[:2],  # Solo scaler_name y scoring_name
            'best_mape': best_combination[2]['MAPE']  # MAPE asociado a la mejor combinación
        }

    # Guardar los resultados en un archivo JSON
    with open("result/best_week_with_scale_52weeks.json", 'w') as json_file:
        json.dump(best_combinations, json_file, indent=4)



    # Print structure of the JSON data to debug
    print(json.dumps(data, indent=4))



    # Find the best combinations for each model
    best_combinations = {}
    for model, model_data in data.items():
        best_combinations[model] = find_best_combination(model_data)

    # Prepare data for the table
    best_combinations_json = []
    for model, (scaler_name, best_method, metrics) in best_combinations.items():
        best_combinations_json.append({
            "Model": model,
            "Scaler": scaler_name,
            "Best Feature Selection": best_method,
            "MAPE": metrics["MAPE"],
            "MAE": metrics["MAE"],
            "Prediction": metrics["Prediction"]
        })

    # Save data to a JSON file
    with open("result/best_week_with_scale_52weeks.json", 'w') as json_file:
        json.dump(best_combinations_json, json_file, indent=4)


    # Charger les données depuis le fichier JSON
    with open('result/best_week_with_scale_52weeks.json', 'r') as json_file:
        data = json.load(json_file)

    # Transformer les données en DataFrame pandas
    df = pd.DataFrame(data)

    # Extraire les valeurs MAPE et MAE de la première semaine
    df['MAPE'] = df['MAPE'].apply(lambda x: x['week_1'])
    df['MAE'] = df['MAE'].apply(lambda x: x['week_1'])

    # Trier les lignes en fonction du score MAPE, du plus petit au plus grand
    df = df.sort_values(by='MAPE')

    # Réorganiser les colonnes
    df = df[['Model', 'Scaler', 'MAPE', 'MAE', 'Best Feature Selection', 'Prediction']]

    # Enregistrer le DataFrame en un fichier Excel
    df.to_excel('result/best_models_sorted_with_scale.xlsx', index=False)


    # Charger les données depuis le fichier JSON
    with open('result/best_week_with_scale_52weeks.json', 'r') as json_file:
        data = json.load(json_file)

    # Initialiser les figures et axes pour les trois graphiques
    fig_mape, ax_mape = plt.subplots(figsize=(10, 6))
    fig_mae, ax_mae = plt.subplots(figsize=(10, 6))
    fig_pred, ax_pred = plt.subplots(figsize=(10, 6))

    # Parcourir chaque modèle pour tracer les courbes
    for model_data in data:
        model = model_data["Model"]
        weeks = list(model_data["MAPE"].keys())
        weeks_int = [int(week.split('_')[1]) for week in weeks]  # Convertir les semaines en entiers

        # Récupérer les valeurs de MAPE, MAE et Prediction
        mape_values = list(model_data["MAPE"].values())
        mae_values = list(model_data["MAE"].values())
        pred_values = list(model_data["Prediction"].values())

        # Tracer les courbes
        ax_mape.plot(weeks_int, mape_values, label=model)
        ax_mae.plot(weeks_int, mae_values, label=model)
        ax_pred.plot(weeks_int, pred_values, label=model)

    # Ajouter des titres et des légendes
    ax_mape.set_title('MAPE par Modèle au fil du temps')
    ax_mape.set_xlabel('Semaine')
    ax_mape.set_ylabel('MAPE')
    ax_mape.legend()

    ax_mae.set_title('MAE par Modèle au fil du temps')
    ax_mae.set_xlabel('Semaine')
    ax_mae.set_ylabel('MAE')
    ax_mae.legend()

    ax_pred.set_title('Prédictions par Modèle au fil du temps')
    ax_pred.set_xlabel('Semaine')
    ax_pred.set_ylabel('Prédiction')
    ax_pred.legend()

    # Afficher les graphiques
    #plt.show()
