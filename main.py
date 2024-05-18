import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor


# Charging datas
data = pd.read_excel('Ire_EU_Milk_Prices.xlsx', sheet_name=0, skiprows=6, index_col=0)

# Clean datas
columns_to_delete = [col for col in data.columns if 'Unnamed' in str(col)]
data.drop(columns=columns_to_delete, inplace=True)
data = data.iloc[:, :-3]
data.replace('c', np.nan, inplace=True)
data.dropna(inplace=True)
data.index = pd.to_datetime(data.index, format='%Ym%m')

# Graph of the evolution of milk prices by country
plt.figure(figsize=(30, 13))
for country in data.columns:
    filtered_data = data[data[country] != 0]
    plt.plot(filtered_data.index, filtered_data[country], label=country)
plt.xlabel('Année')
plt.ylabel('Prix du lait')
plt.title('Évolution du prix du lait par pays')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('milk_price_evolution.png')

# Irland / EU
average_prices = data.mean(axis=1)
ireland_data = data[data['Ireland'] != 0]
plt.figure(figsize=(10, 6))
plt.plot(data.index, average_prices, label='Prix moyen', color='red')
plt.plot(ireland_data.index, ireland_data['Ireland'], label='Irlande', color='blue')
plt.xlabel('Année')
plt.ylabel('Prix du lait')
plt.title('Évolution du prix moyen du lait avec l\'Irlande')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('milk_price_evolution_with_ireland.png')


# remove columns with values equal to zero
data = data.loc[:, (data != 0).any(axis=0)]

# Scikit-Learn : Linear regression model
X = data.drop(columns='Ireland')
y = data['Ireland']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# TensorFlow : Neural network

# Resize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
y_pred = model.predict(X_test_scaled).flatten()

# Metrics
r2_tf = r2_score(y_test, y_pred)
mse_tf = mean_squared_error(y_test, y_pred)
mae_tf = mean_absolute_error(y_test, y_pred)


# PyTorch: Neural network

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

# Define the neural network architecture
class MilkPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(MilkPricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MilkPricePredictor(X_train_scaled.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50
batch_size = 32
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        labels = y_train_tensor[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(X_train_tensor)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
y_pred = y_pred_tensor.numpy().flatten()
y_test = y_test_tensor.numpy().flatten()
r2_pt = r2_score(y_test, y_pred)
mse_pt = mean_squared_error(y_test, y_pred)
mae_pt = mean_absolute_error(y_test, y_pred)

# Scikit-Learn : Random Forest Regression
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)



print("Scikit-Learn :")
print("Coefficient de détermination (R²) :", r2_score(y_test, y_pred))
print("Erreur quadratique moyenne (MSE) :", mean_squared_error(y_test, y_pred))
print("Erreur absolue moyenne (MAE) :", mean_absolute_error(y_test, y_pred))

print("\nTensorFlow :")
print("Coefficient de détermination (R²) :", r2_tf)
print("Erreur quadratique moyenne (MSE) :", mse_tf)
print("Erreur absolue moyenne (MAE) :", mae_tf)

print("\nPyTorch :")
print("Coefficient de détermination (R²) :", r2_pt)
print("Erreur quadratique moyenne (MSE) :", mse_pt)
print("Erreur absolue moyenne (MAE) :", mae_pt)


print("\nRandom Forest :")
print("Coefficient de détermination (R²) :", r2_rf)
print("Erreur quadratique moyenne (MSE) :", mse_rf)
print("Erreur absolue moyenne (MAE) :", mae_rf)


