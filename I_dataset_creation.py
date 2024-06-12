import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def visualization_sorted_tab(data):
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    # Read milk prices data
    price_data = pd.read_excel('initial_datas/Ire_EU_Milk_Prices.xlsx', sheet_name=0, skiprows=6, index_col=0)

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
    plt.savefig('visualization/milk_price_evolution.png')

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
    plt.savefig('visualization/milk_price_evolution_with_ireland.png')

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

    file_path = 'initial_datas/4 Data Grass Growth Yearly & Monthly 2013-2024.xlsx'
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

    plt.savefig('visualization/Grass_growth_plot.png')

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
    data.to_excel('spreadsheet/Data.xlsx', index=True)


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


def time_series_to_tabular(data):
    target = 'Ireland_Milk_Price'  # The column in data we want to forecast
    past_time = 6  # This is how far back we want to look for features
    futur_time = 3  # This is how far forward we want to forecast

    # Separate datetime columns from others
    date_col = data.select_dtypes(include=[np.datetime64]).columns
    other_cols = data.columns.difference(date_col)

    # Fill in missing values for non-datetime columns
    data[other_cols] = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data[other_cols])
    data[other_cols] = pd.DataFrame(data[other_cols], columns=other_cols, index=data.index)

    # Recombine data with datetime columns
    data = pd.concat([data[date_col], data[other_cols]], axis=1)

    print('\nInitial data shape:', data.shape)

    # Create feature data (X)
    data = create_lag_features(data, past_time)
    print('\ndata shape with feature columns:', data.shape)

    # Create targets to forecast (y)
    data, targets = create_future_values(data, target, futur_time)
    print('\ndata shape with target columns:', data.shape)

    # Separate features (X) and targets (y)
    y = data[targets]
    X = data.drop(targets, axis=1)
    print('\nShape of X (features):', X.shape)
    print('Shape of y (target(s)): ', y.shape)
    # Saving the features and targets to CSV
    X.to_excel('spreadsheet/features.xlsx')
    y.to_excel('spreadsheet/targets.xlsx')

    return X, y


def create_lag_features(data, lag):
    """Create features for our ML model (X matrix).

    :param pd.DataFrame data: DataFrame
    :param str target: Name of target column (int)
    :param int lag: Lookback window (int)
    """
    lagged_data = []
    for col in data.columns:
        for i in range(1, lag + 1):
            lagged_data.append(data[col].shift(i).rename(f'{col}-{i}'))

    lagged_df = pd.concat(lagged_data, axis=1)
    data = pd.concat([data, lagged_df], axis=1)

    # Drop first N rows where N = lag
    data = data.iloc[lag:]
    return data

def create_future_values(data, target, futur_time):
    """Create target columns for futur_times greater than 1"""
    targets = [target]
    future_data = {}
    for i in range(1, futur_time):
        col_name = f'{target}+{i}'
        future_data[col_name] = data[target].shift(-i)
        targets.append(col_name)

    future_df = pd.DataFrame(future_data)
    data = pd.concat([data, future_df], axis=1)

    # Optional: Drop rows missing future target values
    data = data[data[targets[-1]].notna()]
    return data, targets

