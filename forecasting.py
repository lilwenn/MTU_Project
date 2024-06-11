import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Progress bar for training loops

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import rmse



class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def visualization_sorted_tab():
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    # Read milk prices data
    price_data = pd.read_excel('Ire_EU_Milk_Prices.xlsx', sheet_name=0, skiprows=6, index_col=0)

    # Clean data
    columns_to_delete = [col for col in price_data.columns if 'Unnamed' in str(col)]
    price_data.drop(columns=columns_to_delete, inplace=True)

    price_data = price_data.iloc[:, :-3]
    price_data.replace('c', np.nan, inplace=True)
    price_data = price_data[:-641]

    # Convert to datetime 
    price_data.index = pd.to_datetime(price_data.index, format='%Ym%m')

    # Visualize milk price evolution by country
    plt.figure(figsize=(30, 13))
    for country in price_data.columns:
        filtered_data = price_data[price_data[country] != 0]
        plt.plot(filtered_data.index, filtered_data[country], label=country)
    plt.xlabel('Year')
    plt.ylabel('Milk Price')
    plt.title('Milk Price Evolution by Country')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('milk_price_evolution.png')

    # Visualize milk price evolution Ireland vs EU
    average_prices = price_data.mean(axis=1)
    ireland_data = price_data[price_data['Ireland'] != 0]
    plt.figure(figsize=(10, 6))
    plt.plot(price_data.index, average_prices, label='Average Price', color='red')
    plt.plot(ireland_data.index, ireland_data['Ireland'], label='Ireland', color='blue')
    plt.xlabel('Year')
    plt.ylabel('Milk Price')
    plt.title('Average Milk Price Evolution with Ireland')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('milk_price_evolution_with_ireland.png')

    # Convert index to year-month format and remove Luxembourg column
    price_data.index = price_data.index.strftime('%Y-%m')
    price_data = price_data.drop(columns=['Luxembourg'])
    price_data.dropna(how='any', inplace=True)
    price_data = price_data[~(price_data == 0).any(axis=1)]
    price_data.index.name = 'Date'
    price_data.reset_index(inplace=True)

    # Rename columns
    for col in price_data.columns:
        if col != 'Date':
            price_data = price_data.rename(columns={str(col): str(col) + '_Milk_Price'})

    print(price_data)

    # Load grass growth data

    file_path = '4 Data Grass Growth Yearly & Monthly 2013-2024.xlsx'
    grass_data = pd.read_excel(file_path)


    grass_data = grass_data.iloc[:-17]

    # Plotting grass growth
    plt.figure(figsize=(15, 10))
    for year in grass_data.columns:
        plt.plot(grass_data.index, grass_data[year], label=year)

    plt.title('Grass Growth Over Time')
    plt.xlabel('Date')
    plt.ylabel('Grass Quantity')
    plt.legend(title='Year')
    plt.grid(True)

    plt.savefig('Grass_growth_plot.png')

    # Melt grass growth data and modify date format, Put informations in 1 column

    merged_column = grass_data.melt()['value']
    dates_column = grass_data['Date']

    for year in [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
        modified_dates_column = pd.to_datetime(dates_column)
        modified_dates_column = modified_dates_column.apply(lambda x: x.replace(year=year))
        grass_data['Modified_' + str(year)] = modified_dates_column
    grass_data = grass_data.drop(columns=['Date'])

    # Prepare grass growth data for merging
    values = grass_data[['2013', '2014', '2015', '2016', '2017', '2018', 2019, 2020, 2021, 2022, 2023, 2024]]
    values_column = values.melt()['value']

    dates = grass_data[
        ['Modified_2013', 'Modified_2014', 'Modified_2015', 'Modified_2016', 'Modified_2017', 'Modified_2018',
         'Modified_2019', 'Modified_2020', 'Modified_2021', 'Modified_2022', 'Modified_2023', 'Modified_2024']]
    dates_column = dates.melt()['value']

    # Create DataFrame for merging
    merged_df = pd.DataFrame({'Date': dates_column, 'Value': values_column})
    merged_df['Mois'] = merged_df['Date'].dt.to_period('M')

    # Calculate monthly average grass growth
    monthly_avg = merged_df.groupby('Mois')['Value'].mean().reset_index()
    monthly_avg = monthly_avg.rename(columns={'Mois': 'Date', 'Value': 'Average_grass_growth/week'})
    grass_data = monthly_avg.iloc[:-8]

    print(grass_data)

    ## Creation of a table with grass information + price
    print("Columns in price_data:", price_data.columns)
    print("Columns in herbe_data:", monthly_avg.columns)

    # Convert the data types of the 'Date' column to ensure consistency
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    monthly_avg['Date'] = monthly_avg['Date'].dt.to_timestamp()
    print("Data type in price_data - Date:", price_data['Date'].dtype)
    print("Data type in monthly_avg - Date:", monthly_avg['Date'].dtype)

    # Merge the datasets based on the 'Date' column
    data = pd.merge(price_data, monthly_avg, on='Date')
    print(data)

    # Save the merged data to an Excel file
    data.to_excel('Data.xlsx', index=True)


def verifier_tableau_excel(data, colonne_cible):

    if data.empty:
        print("Le tableau est vide.")
        return False

    if not all(isinstance(col, str) for col in data.columns):
        print("Le tableau doit avoir des en-têtes de colonnes valides.")
        return False

    # types de données
    types_colonnes = data.dtypes
    for col, dtype in types_colonnes.items():
        if dtype == 'object' and not data[col].apply(lambda x: isinstance(x, (str, type(None)))).all():
            print(f"La colonne '{col}' contient des types de données hétérogènes.")
            return False

    # présence de valeurs manquantes
    valeurs_manquantes = data.isnull().sum().sum()
    if valeurs_manquantes > 0:
        print(f"Le tableau contient {valeurs_manquantes} valeurs manquantes.")
        return False

    # taille du tableau
    if len(data) < 10:  # Ce seuil peut être ajusté selon les besoins
        print("Le tableau contient trop peu de données pour un traitement IA efficace.")
        return False

    # colonne cible présente
    if colonne_cible not in data.columns:
        print(f"La colonne cible '{colonne_cible}' n'est pas présente dans le tableau.")
        return False

    return True


def correlation_matrix(data, colonne_cible, seuil_corr):

    print(f"Nombre total de features initiales : {len(data.columns)}")

    corr_matrix = data.corr()

    print("Matrice de corrélation :")
    print(corr_matrix)

    # Enregistrer la matrice de corrélation
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title('Matrice de Corrélation')
    plt.savefig('correlation_matrix.png')
    plt.close()

    features_importantes = corr_matrix[colonne_cible][abs(corr_matrix[colonne_cible]) >= seuil_corr].index.tolist()
    if colonne_cible in features_importantes:
        features_importantes.remove(colonne_cible)

    # Supprimer les features inutiles
    sorted_data = data[features_importantes + [colonne_cible]]

    # Afficher les features conservées et leur nombre
    print("Features conservées :")
    print(features_importantes)
    print(f"Nombre de features conservées : {len(features_importantes)}")

    sorted_corr_matrix = sorted_data.corr()

    # Afficher la matrice de corrélation
    print("Matrice de corrélation :")
    print(corr_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title('Matrice de Corrélation')
    plt.savefig('sorted_corr_matrix.png')
    plt.close()

    return sorted_data


def scale_data(df, method='standardisation'):

    if method not in ['standardisation', 'normalisation']:
        raise ValueError("La méthode doit être 'standardisation' ou 'normalisation'")

    # Sélectionner les colonnes numériques
    colonnes_numeriques = df.select_dtypes(include=['float64', 'int64']).columns

    # Appliquer la normalisation ou standardisation
    if method == 'standardisation':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'normalisation':
        scaler = MinMaxScaler()


    df_copy = df.copy()  
    df_copy.loc[:, colonnes_numeriques] = scaler.fit_transform(df_copy[colonnes_numeriques])

    return df_copy


def linear_regression_sklearn(X_train, X_test, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return predictions


def PolynomialRegression(X_train, X_test, y_train, degree):
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    predictions = model.predict(X_test_poly)

    return predictions

def ANN_model(X_train, y_train, X_test, epochs, batch_size, validation_split):
    input_dim = X_train.shape[1]
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    predictions = model.predict(X_test)

    predictions = predictions.tolist()
    lst= []
    for elem in predictions:
        lst.append(elem[0])

    tensor = torch.tensor(lst, dtype=torch.float32)
    np_array = tensor.numpy()
    lst = np_array.astype(np.float64)
    return lst

def random_forest(X_train, X_test, y_train, n_estimators, random_state):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)

    return predictions


def gradient_boosting_regressor(X_train, y_train, X_test, n_estimators=100, learning_rate=0.1, max_depth=3,
                                random_state=42):
    gb_model = GradientBoostingRegressor(n_estimators = n_estimators, learning_rate = learning_rate, max_depth = max_depth, random_state = random_state)
    gb_model.fit(X_train, y_train)
    predictions = gb_model.predict(X_test)

    return predictions

def Neural_Network_Pytorch(X_train, X_test, y_train, y_test, epochs=100, learning_rate=0.001):

    # Convert data to NumPy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Initialize model
    input_dim = X_train.shape[1]
    model = NeuralNetwork(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()

    predictions = [round(float(pred[0]), 2) for pred in predictions]

    return predictions, y_test



def Performance(y_test, predictions, name, metrics_data, predictions_data):

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    metrics_data.append({'Model': name,
                         'R^2': round(r2,3),
                         'MSE': round(mse,3),
                         'MAE': round(mae,3)})

    predictions_data[name] = predictions

def main():

    # visualization_sorted_tab()  # Uncomment this if you need to visualize and prepare the data again

    data = pd.read_excel('Data.xlsx')
    colonne_cible = 'Ireland_Milk_Price'
    seuil_corr = 0.9
    data = data.iloc[:, 1:]  
    data['Mois de l\'année'] = data['Date'].dt.month

    # Le tableau est-il exploitable
    exploitable = verifier_tableau_excel(data, colonne_cible)
    print("Le fichier est exploitable :", exploitable)

    # Corrélation
    # data = correlation_matrix(data, colonne_cible, seuil_corr)

    # Normalisation / standardisation
    # data = scale_data(data, method='normalisation')
    # data = scale_data(data, method='standardisation')

    print("Tableau final :")
    print(data.head())

    # Split data based on time series
    date_split_index = int(0.8 * len(data))
    train = data.iloc[:date_split_index]
    test = data.iloc[date_split_index:]

    X_train = train.drop(columns=[colonne_cible, 'Date'])
    y_train = train[colonne_cible]
    X_test = test.drop(columns=[colonne_cible, 'Date'])
    y_test = test[colonne_cible]

    # Vérification des tailles des ensembles
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(X_test.head(), X_train.head())

    # List to store the performance metrics
    metrics_data = []
    predictions_data = {}

    # Linear Regression w/ sklearn
    predictions = linear_regression_sklearn(X_train, X_test, y_train)
    Performance(y_test, predictions, 'LinearRegression', metrics_data, predictions_data)

    # Polynomial regression
    degree = 2
    predictions = PolynomialRegression(X_train, X_test, y_train, degree)
    Performance(y_test, predictions, 'PolynomialRegression', metrics_data, predictions_data)

    # ANN model w/ tensorflow.keras
    epochs = 100
    batch_size = 10
    validation_split = 0.2
    predictions = ANN_model(X_train, y_train, X_test, epochs, batch_size, validation_split)
    Performance(y_test, predictions, 'ANN_Model', metrics_data, predictions_data)

    # Random Forest model
    n_estimators = 100
    random_state = 42
    predictions = random_forest(X_train, X_test, y_train, n_estimators, random_state)
    Performance(y_test, predictions, 'Random Forest', metrics_data, predictions_data)

    # Gradient Boosting Model
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 3
    random_state = 42
    predictions = gradient_boosting_regressor(X_train, y_train, X_test, n_estimators, learning_rate, max_depth, random_state)
    Performance(y_test, predictions, 'Gradient Boosting Model', metrics_data, predictions_data)

    # Neural Network with PyTorch
    predictions, y_test = Neural_Network_Pytorch(X_train, X_test, y_train, y_test, epochs=50, learning_rate=0.01)
    Performance(y_test, predictions, 'Neural Network PyTorch', metrics_data, predictions_data)

    # Create the DataFrame from the list of dictionaries
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df)

    metrics_df.to_excel('Model_Performance_Metrics.xlsx', index=False)

    for model, values in predictions_data.items():
        if isinstance(values, np.ndarray):
            predictions_data[model] = np.round(values, 2)
        elif isinstance(values[0], (int, float)):
            predictions_data[model] = [round(value, 2) for value in values]

    df = pd.DataFrame(predictions_data)
    df.insert(0, 'Actual', y_test)

    print(df)
    df.to_excel('predictions.xlsx', index=False)
    print("Les prédictions ont été écrites dans 'Predictions.xlsx'")

if __name__ == '__main__':
    main()