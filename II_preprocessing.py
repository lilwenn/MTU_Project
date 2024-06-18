from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose




def create_lag_features(data, lag):
    new_data = data.copy()  

    lagged_cols = []
    for col in data.columns:
        # Create lagged columns for each original column
        for i in range(1, lag + 1):
            lagged_col = data[col].shift(i)
            lagged_col.name = f'{col}-{i}'
            lagged_cols.append(lagged_col)

    lagged_data = pd.concat(lagged_cols, axis=1)
    data = pd.concat([data, lagged_data], axis=1)

    data = data.iloc[lag:]

    return data

def create_MA(data, past_time):
    new_data = data.copy() 
    
    ma_cols = []
    for col in new_data.columns:
        # Create a moving average column for each original column
        ma_col = new_data[col].rolling(window=past_time).mean()
        ma_col.name = f'{col}_MA_{past_time}'
        ma_cols.append(ma_col)
    
    # Concatenate MA columns with the original data
    ma_data = pd.concat(ma_cols, axis=1)
    new_data = pd.concat([new_data, ma_data], axis=1)

    # Drop the first 'past_time' rows to avoid NaN values
    new_data = new_data.iloc[past_time:]
    
    return new_data



def time_series_analysis(past_time, data):


    date_col = data.select_dtypes(include=[np.datetime64]).columns
    other_cols = data.columns.difference(date_col)

    data[other_cols] = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data[other_cols])
    data[other_cols] = pd.DataFrame(data[other_cols], columns=other_cols, index=data.index)

    data = pd.concat([data[date_col], data[other_cols]], axis=1)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)


    data_ireland = data[['Ireland_Milk_Price']].copy()

    all_periods = pd.date_range(start=data_ireland.index.min(), end=data_ireland.index.max(), freq='MS')
    data_all_periods = pd.DataFrame(index=all_periods)


    data_ireland = data_ireland.merge(data_all_periods, how='outer', left_index=True, right_index=True).fillna(0)
    decomposition = seasonal_decompose(data_ireland['Ireland_Milk_Price'], model='additive', period=12)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    decomposition.observed.plot(ax=ax1, title='Série temporelle originale: Prix du lait en Irlande')
    decomposition.trend.plot(ax=ax2, title='Tendance: Prix du lait en Irlande')
    decomposition.seasonal.plot(ax=ax3, title='Saisonnalité: Prix du lait en Irlande')
    decomposition.resid.plot(ax=ax4, title='Résidu: Prix du lait en Irlande')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig('visualization/analyse temporelle.png')

    # Statistiques descriptives
    print("\nDescriptive statistics for original time series:")
    print(data_ireland['Ireland_Milk_Price'].describe().to_markdown(numalign="left", stralign="left"))

    print("\nDescriptive statistics for trend component:")
    print(decomposition.trend.describe().to_markdown(numalign="left", stralign="left"))

    print("\nDescriptive statistics for seasonal component:")
    print(decomposition.seasonal.describe().to_markdown(numalign="left", stralign="left"))

    print("\nDescriptive statistics for residual component:")
    print(decomposition.resid.describe().to_markdown(numalign="left", stralign="left"))

    # Lissage

    data_MA = create_MA(data, past_time)
    data_lagged = create_lag_features(data, past_time)


    data_final = pd.concat([data_lagged, data_MA], axis=1)


    data_final = data_final.loc[:,~data_final.columns.duplicated()]
    data_final = data_final.reindex(sorted(data_final.columns), axis=1)

    data_final.to_excel('spreadsheet/data_final.xlsx', index=True)

    return data_final


def check_tab(data, target_column):
    """Validates a dataset for basic quality checks and suitability for ML.
    
    Args:
        df (pandas.DataFrame): The dataset to be validated.
        target_col (str): The name of the target column.

    Returns:
        bool: True if the dataset is valid, False otherwise.
    """

    if data.empty:
        print("The dataframe is empty.")
        return False

    if not all(isinstance(col, str) for col in data.columns):
        print("The dataframe must have valid column headers")
        return False

    # types de données
    column_types = data.dtypes
    for col, dtype in column_types.items():
        if dtype == 'object' and not data[col].apply(lambda x: isinstance(x, (str, type(None)))).all():
            print(f"Column '{col}' contains mixed data types.")
            return False

    # présence de valeurs manquantes
    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        print(f"The dataframe contains {missing_values} missing values.")
        return False

    # taille du tableau
    if len(data) < 10:  
        print("The dataframe contains too few rows for effective AI processing.")
        return False

    # colonne cible présente
    if target_column not in data.columns:
        print(f"The target column '{target_column}' is not present in the dataframe.")
        return False

    return True


def feature_selection_correlation(data, target_column, seuil_corr):
    """Selects features based on correlation with the target variable.

    Args:
        df (pandas.DataFrame): The dataset.
        target_col (str): The name of the target column.
        corr_threshold (float): The minimum absolute correlation threshold.

    Returns:
        pandas.DataFrame: The dataset with selected features.
    """

    print(f"Nombre total de features initiales : {len(data.columns)}")

    corr_matrix = data.corr()

    print("Matrice de corrélation :")
    print(corr_matrix)

    # Save Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title('Correlation Matrix')
    plt.savefig('visualization/correlation_matrix.png')
    plt.close()

    features_importantes = corr_matrix[target_column][abs(corr_matrix[target_column]) >= seuil_corr].index.tolist()
    if target_column in features_importantes:
        features_importantes.remove(target_column)

    # Delete features
    sorted_data = data[features_importantes + [target_column]]

    sorted_corr_matrix = sorted_data.corr()

    # Afficher la matrice de corrélation
    print('Correlation Matrix')
    print(corr_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title('Correlation Matrix')
    plt.savefig('visualization/sorted_corr_matrix.png')
    plt.close()

    return sorted_data


def scale_data(df, method):
    """Scales numerical features using the specified method.

    Args:
        df (pandas.DataFrame): The dataset.
        method (str): The scaling method ('standard', 'minmax', 'robust').

    Returns:
        pandas.DataFrame: The dataset with scaled features.
    """

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

