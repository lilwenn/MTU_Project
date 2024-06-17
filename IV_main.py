import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import altair as alt

from sklearn.model_selection import cross_validate, train_test_split


from I_dataset_creation import visualization_sorted_tab
from II_preprocessing import check_tab, correlation_matrix, scale_data, time_series_analysis
from III_train_models import linear_regression_sklearn,PolynomialRegression, Neural_Network_Pytorch, ANN_model, random_forest, gradient_boosting_regressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from statsmodels.tsa.seasonal import seasonal_decompose

def Performance(y_test, predictions, name, metrics_data, predictions_data):
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    metrics_data.append({'Model': name,
                         'R^2': round(r2,4),
                         'MSE': round(mse,4),
                         'MAE': round(mae,4)})

    predictions_data[name] = predictions

def main():

    print("---------------- DATASET CREATION ------------------")



    visualization_sorted_tab()

    data = pd.read_excel('spreadsheet/Data.xlsx')
    colonne_cible = 'Ireland_Milk_Price'


    data = data.iloc[:, 1:]
    data = data.sort_values('Date')


    data.set_index('Date', inplace=True)
    data.index = pd.to_datetime(data.index)


    past_time = 3
    time_series_analysis(past_time, data)


    X, y = time_series_to_tabular(data)

    print(f'Shape of X: {X.shape}')
    print(f'Shape of y: {y.shape}')

    print(" --------------- PREPROCESSING ------------------")

    # Le tableau est-il exploitable
    exploitable = check_tab(data, colonne_cible)
    print("Le fichier est exploitable :", exploitable)

    # Corrélation
    # data = correlation_matrix(data, colonne_cible, seuil_corr)

    # Normalisation / standardisation
    # data = scale_data(data, method='normalisation')
    # data = scale_data(data, method='standardisation')

    print("Tableau final :")
    print(data.head())

    print("----------------- TRAIN MODELS -----------------")


    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    test_size = 0.2
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Vérification des tailles des ensembles
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(X_test.head(), X_train.head()) 

    # List to store the performance metrics
    metrics_data = []
    predictions_data = {}


    # Linear Regression w/ sklearn
    predictions = linear_regression_sklearn(X_train, X_test, y_train)
    Performance(y_test, predictions, 'Linear Regression', metrics_data, predictions_data)



    data_y_test = pd.DataFrame(y_test)
    data_y_test['Date'] = data_y_test.index
    data_y_test['Date'] = pd.to_datetime(data_y_test['Date'], format='%Y-%m-%d')

    data_predictions = pd.DataFrame(predictions, index=y_test.index, columns=[f'Prediction_{i}' for i in range(predictions.shape[1])])

    data_combined = pd.concat([data_y_test, data_predictions], axis=1)
    data_combined.drop(columns='Date', inplace=True)
    excel_file = 'predictions_vs_ytest.xlsx'

    # Écrire le DataFrame combiné dans le fichier Excel
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='w') as writer:
        data_combined.to_excel(writer, sheet_name='Predictions vs y_test', index=True)

    # Polynomial regression
    degree = 2
    predictions = PolynomialRegression(X_train, X_test, y_train, degree)
    Performance(y_test, predictions, 'PolynomialRegression', metrics_data, predictions_data)
    print(metrics_data)

    """

    # ANN model w/ tensorflow.keras
    epochs = 100
    batch_size = 10
    validation_split = 0.2
    predictions = ANN_model(X_train, y_train, X_test, epochs, batch_size, validation_split)
    #Performance(y_test, predictions, 'ANN_Model', metrics_data, predictions_data)

    # Random Forest model
    n_estimators = 100
    random_state = 42
    predictions = random_forest(X_train, X_test, y_train, n_estimators, random_state)
    Performance(y_test, predictions, 'Random Forest', metrics_data, predictions_data)

    print(metrics_data)

    # Create the DataFrame from the list of dictionaries
    metrics_data = pd.DataFrame(metrics_data)
    metrics_data.to_excel('spreadsheet/Model_Performance_Metrics.xlsx', index=False)


    # Gradient Boosting Model
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 3
    random_state = 42

    # Sélectionner la première colonne de y_train si c'est un DataFrame avec plusieurs colonnes
    y_train_single_column = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train

    predictions = gradient_boosting_regressor(X_train, y_train_single_column, X_test, n_estimators, learning_rate, max_depth, random_state)
    Performance(y_test, predictions, 'Gradient Boosting Model', metrics_data, predictions_data)

    # Neural Network with PyTorch
    predictions, y_test = Neural_Network_Pytorch(X_train, X_test, y_train, y_test, epochs=50, learning_rate=0.01)
    #Performance(y_test, predictions, 'Neural Network PyTorch', metrics_data, predictions_data)

    for model, values in predictions_data.items():
        if isinstance(values, np.ndarray):
            predictions_data[model] = np.round(values, 2)
        elif isinstance(values[0], (int, float)):
            predictions_data[model] = [round(value, 2) for value in values]

    data = pd.DataFrame(predictions_data)
    data.insert(0, 'Actual', y_test)

    print(data)
    data.to_excel('spreadsheet/predictions.xlsx', index=False)
    print("Les prédictions ont été écrites dans 'Predictions.xlsx'")

    """





if __name__ == '__main__':
    main()
