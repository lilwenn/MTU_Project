from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, hidden_size)

    def forward(self, lstm_output, hidden):
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        
        output = torch.cat((hidden, attn_applied), 1)
        output = self.attn_combine(output)
        return output

class Model(nn.Module):
    def __init__(self, model_class, input_size, hidden_size, num_layers, output_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.5)
        self.attention = Attention(hidden_size)
        self.lstm = model_class(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if isinstance(self.lstm, nn.LSTM):
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        else:
            lstm_out, hn = self.lstm(x, h0)
        
        lstm_out = self.dropout(lstm_out)
        attn_out = self.attention(lstm_out, hn[-1])
        out = self.fc(attn_out)
        return out

class ModelRun(BaseEstimator, RegressorMixin):
    def __init__(self, model_class, hidden_size=100, num_layers=2, output_size=1, num_epochs=200, learning_rate=0.0005):
        self.model_class = model_class
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model = None

    def _initialize_model(self, input_size):
        self.model = Model(self.model_class, input_size, self.hidden_size, self.num_layers, self.output_size).to(device)

    def fit(self, X, y):
        input_size = X.shape[1]
        x_train_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        y_train_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

        self._initialize_model(input_size)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        start_time = datetime.now()

        for epoch in range(self.num_epochs):
            self.model.train()
            outputs = self.model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')
        
        self.duration = (datetime.now() - start_time).total_seconds()
        return self

    def predict(self, X):
        self.model.eval()
        x_test_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            predictions = self.model(x_test_tensor).cpu().numpy()
        return predictions

    def score(self, X, y):
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)

    def mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_elements = y_true != 0
        if np.any(non_zero_elements):
            return np.mean(np.abs((y_true[non_zero_elements] - y_pred[non_zero_elements]) / y_true[non_zero_elements])) * 100
        else:
            return np.inf

    def get_params(self, deep=True):
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

class LSTM_Model(ModelRun):
    def __init__(self, **kwargs):
        super().__init__(model_class=nn.LSTM, **kwargs)

class GRU_Model(ModelRun):
    def __init__(self, **kwargs):
        super().__init__(model_class=nn.GRU, **kwargs)

class RNN_Model(ModelRun):
    def __init__(self, **kwargs):
        super().__init__(model_class=nn.RNN, **kwargs)



# Boucle pour la recherche des meilleurs hyperparamètres
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Charger les données
df = pd.read_excel('spreadsheet/lagged_results.xlsx')

# Convertir les colonnes de dates en nombres de jours depuis une date de référence, si nécessaire
date_columns = df.select_dtypes(include=['datetime64']).columns
for col in date_columns:
    df[col] = (df[col] - df[col].min()).dt.days

# Supprimer les colonnes non numériques si elles ne sont pas pertinentes
df = df.select_dtypes(include=[np.number])

# Supposons que les colonnes d'entrée sont toutes sauf la dernière, et la dernière colonne est la cible
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Prétraiter les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
ModelClass = LSTM_Model  # Remplacez par GRU_Model ou RNN_Model si nécessaire

hidden_sizes = [150]
num_layers_list = [2]
num_epochs_list = [275]
learning_rates = [0.001]

best_score = float('inf')
best_params = None

for hidden_size in hidden_sizes:
    for num_layers in num_layers_list:
        for num_epochs in num_epochs_list:
            for learning_rate in learning_rates:
                print(f'Testing hyperparameters: hidden_size={hidden_size}, num_layers={num_layers}, num_epochs={num_epochs}, learning_rate={learning_rate}')
                
                model = ModelClass(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=1,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse_score = model.score(X_test, y_test)
                mape_score = model.mean_absolute_percentage_error(y_test, y_pred)
                
                print(f'MSE Score: {-mse_score:.4f}')
                print(f'MAPE Score: {mape_score:.4f}')
                
                if mape_score < best_score:
                    best_score = mape_score
                    best_params = {
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "num_epochs": num_epochs,
                        "learning_rate": learning_rate
                    }

print(f'Best MAPE Score: {best_score:.4f}')
print(f'Best Parameters: {best_params}')