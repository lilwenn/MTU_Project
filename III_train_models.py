import pandas as pd
import numpy as np


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel  


from statsmodels.tsa.arima.model import ARIMA

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Dense, Dropout



import torch
import torch.nn as nn
import torch.optim as optim

from statsmodels.tsa.arima.model import ARIMA



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


    input_dim = X_train.shape[1]
    model = NeuralNetwork(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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


def gradient_boosting_regressor(X_train, X_test, y_train, n_estimators, learning_rate):
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


def train_arima_model(y_train, y_test):
    forecast_index = y_test.index  

    y_train = pd.Series(y_train.values, index=y_train.index)

    model = ARIMA(y_train, order=(5, 1, 0))
    model_fit = model.fit()
   
    predictions = model_fit.get_forecast(steps=len(y_test)).predicted_mean

    return predictions.values 

def knn_regressor(X_train, X_test, y_train, n_neighbors):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

def SANN_model(X_train, y_train, X_test, epochs, batch_size):

    model = Sequential()


    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear')) 

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    predictions = model.predict(X_test).flatten() 

    return predictions

def NARX_model(X_train, y_train, X_test):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear') 
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
    predictions = model.predict(X_test).flatten() 

    return predictions
