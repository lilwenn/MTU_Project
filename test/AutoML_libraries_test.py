import pmdarima as pm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pmdarima import model_selection

# Charger les données
df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')

# Assumons que 'litres' est la colonne que nous voulons utiliser
data = df['litres']
# Déterminer la taille totale des données
total_size = len(data)

# Calculer la taille de l'ensemble d'entraînement
train_size = total_size - 52

# Diviser les données en ensembles d'entraînement et de test
train, test = model_selection.train_test_split(data, train_size=train_size)

# Ajuster un modèle auto_arima simple
arima = pm.auto_arima(train, error_action='ignore', trace=True,
                      suppress_warnings=True, seasonal=True, m=7)

# Faire des prévisions sur l'ensemble de test
forecast = arima.predict(n_periods=len(test))


print(forecast)

# Plot actual test vs. forecasts
x = np.arange(len(test))
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(train)), train, label='Train', color='blue')
plt.plot(np.arange(len(train), len(train) + len(test)), test, marker='x', linestyle='none', label='Test')
plt.plot(np.arange(len(train), len(train) + len(test)), forecast, color='red', linestyle='--', label='Forecast')
plt.title('Actual test samples vs. forecasts')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.show()
