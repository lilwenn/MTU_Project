import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import warnings
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns


## Data exploration and visualization

# Suppressing warnings for better clarity
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Read milk prices data
price_data = pd.read_excel('Ire_EU_Milk_Prices.xlsx', sheet_name=0, skiprows=6, index_col=0)

# Clean data
columns_to_delete = [col for col in price_data.columns if 'Unnamed' in str(col)]
price_data.drop(columns=columns_to_delete, inplace=True)

price_data = price_data.iloc[:, :-3]
price_data.replace('c', np.nan, inplace=True)
price_data = price_data[:-641]

# Convert index to datetime using correct format
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
        price_data = price_data.rename(columns={str(col): str(col)+'_Milk_Price'})

print(price_data)

# Load grass growth data

file_path = '4 Data Grass Growth Yearly & Monthly 2013-2024.xlsx'
grass_data = pd.read_excel(file_path)

# Remove last 17 rows
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

for year in [2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]:
    modified_dates_column = pd.to_datetime(dates_column)
    modified_dates_column = modified_dates_column.apply(lambda x: x.replace(year=year))
    grass_data['Modified_'+str(year)] = modified_dates_column
grass_data = grass_data.drop(columns=['Date'])

# Prepare grass growth data for merging
values = grass_data[['2013','2014','2015','2016','2017','2018',2019,2020,2021,2022,2023,2024]]
values_column = values.melt()['value']

dates = grass_data[['Modified_2013','Modified_2014','Modified_2015','Modified_2016','Modified_2017','Modified_2018','Modified_2019','Modified_2020','Modified_2021','Modified_2022','Modified_2023','Modified_2024']]
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

# Calculate correlation matrix
correlation_matrix = data.corr()

# Plotting correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.xlabel('Features')
plt.ylabel('Features')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig('Correlation_matrix.png')

# Create a list to store the performance metrics
metrics_data = []

## Linear Regression with sklearn

features = data.drop(columns=['Date', 'Ireland_Milk_Price'])
target = data['Ireland_Milk_Price']

sorted_features = data.drop(columns=['Date','Average_grass_growth/week', 'Ireland_Milk_Price','Malta_Milk_Price','Finland_Milk_Price','Portugal_Milk_Price','Cyprus_Milk_Price', 'Croatia_Milk_Price', 'Spain_Milk_Price', 'Greece_Milk_Price'])

# Calculate correlation matrix with sorted features
correlation_matrix = sorted_features.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.xlabel('Features')
plt.ylabel('Features')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig('Correlation_matrix.png')

# Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's performance
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# Append the metrics for linear regression to the list
metrics_data.append({'Model': 'Linear Regression',
                     'R^2': r2,
                     'MSE': mse,
                     'MAE': mae})

## Polynomial regression
# Define the degree of the polynomial
degree = 2

# Create polynomial features
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_poly)

# Evaluate the model's performance
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# Append the metrics for polynomial regression to the list
metrics_data.append({'Model': 'Polynomial Regression (Degree {})'.format(degree),
                     'R^2': r2,
                     'MSE': mse,
                     'MAE': mae})


#ANN model w/ tensorflow.keras

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, validation_split=0.2)

predictions = model.predict(X_test_scaled)

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

metrics_data.append({'Model': 'Artificial Neural Network (tensorflow.keras)',
                     'R^2': r2,
                     'MSE': mse,
                     'MAE': mae})

# Plotting the loss during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('ann_loss_plot.png')

# - continuous decrease so the model learns well
# - no overfitting

# Random Forest model

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
rf_predictions = rf_model.predict(X_test)

# Evaluate the model's performance
rf_r2 = r2_score(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)

# Append the metrics for Random Forest to the list
metrics_data.append({'Model': 'Random Forest',
                     'R^2': rf_r2,
                     'MSE': rf_mse,
                     'MAE': rf_mae})

# Gradient Boosting Model

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_test)

gb_r2 = r2_score(y_test, gb_predictions)
gb_mse = mean_squared_error(y_test, gb_predictions)
gb_mae = mean_absolute_error(y_test, gb_predictions)

metrics_data.append({'Model': 'Gradient Boosting',
                     'R^2': gb_r2,
                     'MSE': gb_mse,
                     'MAE': gb_mae})

## Neural Network with PyTorch


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
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Append the metrics for the PyTorch model to the list
metrics_data.append({'Model': 'PyTorch Neural Network',
                     'R^2': r2,
                     'MSE': mse,
                     'MAE': mae})

# Create the DataFrame from the list of dictionaries
metrics_df = pd.DataFrame(metrics_data)
print(metrics_df)

# Sauvegarder les métriques dans un fichier Excel
metrics_df.to_excel('Model_Performance_Metrics.xlsx', index=False)

