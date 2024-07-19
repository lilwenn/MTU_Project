from matplotlib import pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
import Constants as const
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



"""   A essayer sous linux !!


import autosklearn.regression

# Charger les données
df = pd.read_excel('spreadsheet/lagged_results.xlsx')

# Exploration des données
print(df.head())
print(df.info())
print(df.describe())

# Préparation des données
X = df.drop(columns=const.target_column)
y = df[const.target_column]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utiliser auto-sklearn pour entraîner un modèle
automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=3600, per_run_time_limit=360)
automl.fit(X_train, y_train)

# Prédictions
y_pred = automl.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Erreur quadratique moyenne (MSE): {mse}')
print(f'Coefficient de détermination (R^2): {r2}')

# Afficher les statistiques de l'entraînement
print(automl.sprint_statistics())

# Afficher le modèle final
print(automl.show_models())


"""


import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tpot import TPOTRegressor
import Constants as const
import joblib

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Charger les données
df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')

# Exploration des données
print(df.head())
print(df.info())
print(df.describe())

# Prétraitement des données
# Conversion des colonnes datetime en format numérique
for col in df.select_dtypes(include=['datetime64']).columns:
    df[col] = df[col].astype('int64') // 10**9  # Convertir en secondes depuis l'époque Unix

# Remplir les valeurs manquantes si nécessaire
df.fillna(method='ffill', inplace=True)

# Normaliser les données (optionnel mais souvent utile)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df)

# Préparation des données
X = df.drop(columns=const.target_column)
y = df[const.target_column]

# Utiliser TimeSeriesSplit pour diviser les données
tscv = TimeSeriesSplit(n_splits=5)
best_model = None
best_score = float('inf')
best_model_performance = {}

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Utiliser TPOT pour entraîner un modèle
    tpot = TPOTRegressor(generations=10, population_size=100, verbosity=2, random_state=42, cv=5, scoring='neg_mean_squared_error')
    tpot.fit(X_train, y_train)

    # Prédictions
    y_pred = tpot.predict(X_test)

    # Évaluation du modèle
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    score_mape = mape(y_test, y_pred)

    print(f'Erreur quadratique moyenne (MSE): {mse}')
    print(f'Coefficient de détermination (R^2): {r2}')
    print(f'Erreur absolue moyenne (MAE): {mae}')
    print(f'Erreur absolue moyenne en pourcentage (MAPE): {score_mape}')

    # Mémoriser le meilleur modèle basé sur MSE
    if mse < best_score:
        best_score = mse
        best_model = tpot.fitted_pipeline_

        # Enregistrer les performances du meilleur modèle
        best_model_performance = {
            'MSE': mse,
            'R2': r2,
            'MAE': mae,
            'MAPE': score_mape
        }

        # Enregistrer le meilleur modèle
        joblib.dump(best_model, 'best_tpot_model.pkl')

# Charger le meilleur modèle
best_model = joblib.load('best_tpot_model.pkl')

# Entraîner le meilleur modèle sur toutes les données
best_model.fit(X, y)

# Faire des prédictions pour les 52 semaines
if len(X) >= 52:
    y_pred_52_weeks = best_model.predict(X[:52])
    print('Prédictions pour les 52 semaines:', y_pred_52_weeks)
else:
    print("Pas assez de données pour les prédictions sur 52 semaines")

# Faire des prédictions sur l'ensemble des données
y_pred = best_model.predict(X)

# Calculer MAE et MAPE
mae = mean_absolute_error(y, y_pred)
score_mape = mape(y, y_pred)

print(f'Erreur absolue moyenne (MAE): {mae}')
print(f'Erreur absolue moyenne en pourcentage (MAPE): {score_mape}')

# Afficher les performances du meilleur modèle
print("Performances du meilleur modèle :")
print(best_model_performance)



"""



def preprocess_exog_data(df, forecast_weeks):
    # Create a new dataframe exog_data by dropping columns that contain the string 'litres'
    exog_data = df.drop(columns=[col for col in df.columns if 'litres' in col])

    # Split exog_data into exog_train and exog_future
    exog_train = exog_data.iloc[:-forecast_weeks]
    exog_future = exog_data.iloc[-forecast_weeks:]

    # Ensure the number of columns in exog_future matches exog_train
    exog_future = exog_future.reindex(columns=exog_train.columns, fill_value=0)

    return exog_train, exog_future

def train_and_predict_arima_corrected(df, exog_future, order=(1, 1, 1)):
    # Check if 'Date' is in the columns
    if 'Date' in df.columns:
        # Set the date as index
        df_copy = df.set_index('Date')
    else:
        # Assume 'Date' is the index
        df_copy = df.copy()

    # Infer the frequency
    inferred_freq = pd.infer_freq(df_copy.index)

    # Resample if not weekly on Sundays
    if inferred_freq != 'W-SUN':
        df_copy = df_copy.resample('W-SUN').mean().fillna(df_copy.mean())

    # Create a new dataframe exog_data by dropping columns that contain the string 'litres'
    exog_data = df_copy.drop(columns=[col for col in df_copy.columns if 'litres' in col])

    # Ensure the number of columns in exog_future matches exog_data
    exog_future = exog_future.reindex(columns=exog_data.columns, fill_value=0)

    # Train the ARIMA model
    model = ARIMA(df_copy['litres'], exog=exog_data, order=order, dates=df_copy.index, freq='W-SUN')
    model_fit = model.fit()

    # Predict the next const.forecast_weeks weeks
    predictions = model_fit.forecast(steps=const.forecast_weeks, exog=exog_future)

    # Get the last date in the training data
    last_date = df_copy.index.max()

    # Reset the index
    df_copy = df_copy.reset_index()

    return predictions, last_date

def calculate_mape(y_true, y_pred):
    y_true, y_pred = pd.Series(y_true).reset_index(drop=True), pd.Series(y_pred).reset_index(drop=True)
    mask = y_true != 0
    y_true, y_pred = y_true[mask], y_pred[mask]
    return (abs(y_true - y_pred) / y_true).mean() * 100




ARIMA QUI FONCTIONNE

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

full_df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')

# Clean your data (if necessary)
full_df = full_df.drop(columns=['year_week','Week','EU_milk_price_without UK', 'feed_ex_port', 'Malta_milk_price', 'Croatia_milk_price', 'Malta_Milk_Price'])

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

past_time = 13

if 'Date' in df.columns:
    df.set_index('Date', inplace=True)

df.reset_index(inplace=True)



#df = pd.read_excel('spreadsheet/lagged_results.xlsx')
model_name = 'ARIMA'

result = {}
result[model_name] = {}

result[model_name] = {}

# Preprocess exogenous data
exog_train, exog_future = preprocess_exog_data(df, const.forecast_weeks)

# Check if 'Date' is in the columns
if 'Date' in df.columns:
    # Set the date as index
    df_copy = df.set_index('Date')
else:
    # Assume 'Date' is the index
    df_copy = df.copy()

best_order = (1, 1, 1)  # Hardcoding the ARIMA order as (1, 1, 1)

# Call the function to get predictions and last date
predictions, last_date = train_and_predict_arima_corrected(df, exog_future, best_order)

# Get actual values for comparison
actual_values = df_copy['litres'].iloc[-const.forecast_weeks:].values

# Calculate MAE and MAPE
mae = mean_absolute_error(actual_values, predictions)
mape = calculate_mape(actual_values, predictions)

# Store predictions, last date, MAE, and MAPE in results
result[model_name]['Predictions'] = predictions.tolist()
result[model_name]['Last_Date'] = last_date.strftime('%Y-%m-%d')
result[model_name]['MAE'] = mae
result[model_name]['MAPE'] = mape

# Print the predictions, the last date, MAE, and MAPE for verification
print("Predictions for the next 52 weeks:")
print(predictions)
print("Last date in the training data:", last_date)
print("MAE:", mae)
print("MAPE:", mape)
"""


"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from datetime import datetime

# Charger les données
df = pd.read_excel('spreadsheet/lagged_results.xlsx')

# Assumons que la colonne 'litres' est celle à prédire et qu'il y a d'autres colonnes pour les caractéristiques
features = df.drop(columns=['litres'])
target = df['litres']

# Convertir les colonnes datetime en nombres de jours depuis une date de référence (par exemple, la première date dans le dataset)
date_columns = features.select_dtypes(include=['datetime64']).columns

for col in date_columns:
    features[col] = (features[col] - features[col].min()).dt.days

# Normaliser les caractéristiques
scaler_features = StandardScaler()
features = scaler_features.fit_transform(features)

scaler_target = StandardScaler()
target = scaler_target.fit_transform(target.values.reshape(-1, 1)).flatten()

# Préparer les séquences de données
def create_sequences(features, target, seq_length):
    sequences = []
    targets = []
    for i in range(len(features) - seq_length):
        seq = features[i:i+seq_length]
        label = target[i+seq_length]
        sequences.append(seq)
        targets.append(label)
    return np.array(sequences), np.array(targets)

seq_length = 52
X, y = create_sequences(features, target, seq_length)

# Diviser les données en ensembles d'entraînement et de test
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Définir le modèle RNN
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_dim = X_train.shape[2]
hidden_dim = 50
output_dim = 1
num_layers = 2

model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)

# Définir la perte et l'optimiseur
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
num_epochs = 100

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# Évaluer le modèle
model.eval()
test_loss = 0.0
predictions = []
actuals = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        test_loss += loss.item()
        predictions.append(outputs.squeeze().numpy())
        actuals.append(targets.numpy())

test_loss /= len(test_loader)
predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# Inverser la normalisation des cibles pour l'interprétation
predictions = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()
actuals = scaler_target.inverse_transform(actuals.reshape(-1, 1)).flatten()

# Calculer le MAPE
mape = mean_absolute_percentage_error(actuals, predictions)

# Afficher quelques exemples de prédictions et valeurs réelles pour vérification
for i in range(10):
    print(f'Actual: {actuals[i]}, Predicted: {predictions[i]}')

print(f'Test Loss: {test_loss}')
print(f'R^2 Score: {r2_score(actuals, predictions)}')
print(f'Mean Squared Error: {mean_squared_error(actuals, predictions)}')
print(f'MAPE: {mape}')


# Visualiser les prédictions et les valeurs réelles
plt.figure(figsize=(15, 6))

# Tracer les valeurs réelles
plt.plot(actuals, label='Valeurs Réelles')

# Tracer les prédictions
plt.plot(predictions, label='Prédictions', linestyle='--')

# Ajouter des titres et des légendes
plt.title('Prédictions vs Valeurs Réelles')
plt.xlabel('Temps')
plt.ylabel('Litres')
plt.legend()

# Afficher la courbe
plt.show()"""