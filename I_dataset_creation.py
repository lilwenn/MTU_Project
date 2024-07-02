import os

from sklearn.ensemble import RandomForestRegressor
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
from collections import Counter
from itertools import product




def price_data_creation():

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

    # Convert index to year-month format and remove Luxembourg column
    price_data.index = price_data.index.strftime('%Y-%m')
    price_data = price_data.drop(columns=['Luxembourg','Croatia'])

    price_data.index.name = 'Date'
    price_data.reset_index(inplace=True)

    
    price_data = price_data[(price_data['Date'] >= '2009-01-01') & (price_data['Date'] <= '2021-12-31')]

    # Rename columns
    for col in price_data.columns:
        if col != 'Date':
            price_data = price_data.rename(columns={str(col): str(col) + '_Milk_Price'})

    price_data.to_excel('spreadsheet/Prices_datas_monthly_2009-2021.xlsx', index=False)

    # changing mounth to weeks
    repetitions = [0] * len(price_data) 
    for i in range(len(price_data)):
        if (i+3) % 3 == 0 :
            repetitions[i] = 5 
        else :
            repetitions[i] = 4 

    price_data["repetitions"] = repetitions

    repeated_rows = []
    for i in range(len(price_data)):
        row = price_data.iloc[i]
        num_repeats = row['repetitions']
        for _ in range(num_repeats):
            repeated_rows.append(row)
    new_df_prices = pd.DataFrame(repeated_rows).reset_index(drop=True)
    new_df_prices = new_df_prices.drop(columns=["repetitions"])

    new_df_prices.to_excel('spreadsheet/Prices_datas_weekly_2009-2021.xlsx', index=False)
    new_df_prices['Date'] = pd.to_datetime(new_df_prices['Date'])

    # Add 'week' column
    weeks = []
    week = 5
    for i in range(len(new_df_prices)):
        weeks.append(week)
        week += 1
        if week > 52:
            week = 1

    new_df_prices['Week'] = weeks
    new_df_prices['year_week'] = new_df_prices['Date'].dt.year.astype(str)+ '_' +new_df_prices['Week'].astype(str)  

    new_df_prices.to_excel('spreadsheet/Prices_datas_weekly_2009-2021.xlsx', index=False)


def grass_data_creation():

    file_path = 'initial_datas/4 Data Grass Growth Yearly & Monthly 2013-2024.xlsx'
    grass_data = pd.read_excel(file_path)

    grass_data = grass_data.iloc[:-17]
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

    grass_df = pd.DataFrame({'Date': dates_column, 'Grass_growth': values_column})
    grass_df = pd.DataFrame(grass_df)
    grass_df['Date'] = pd.to_datetime(grass_df['Date']) 

    grass_df.to_excel('spreadsheet/Grass_datas_weekly_2013_2024.xlsx', index=False)


    # Filtrer les dates
    df_filtered_dates = grass_df[(grass_df['Date'] >= '2009-01-01') & (grass_df['Date'] <= '2024-12-31')]
    weekly_dates = pd.date_range(start='2009-01-01', end='2024-12-31', freq='W-MON')

    # Fusionner avec les dates hebdomadaires
    df_full = pd.DataFrame({'Date': weekly_dates})
    df_full = df_full.merge(df_filtered_dates, on='Date', how='left')

    df_date = df_full[['Date']]

    df_date.loc[:, 'Date'] = pd.to_datetime(df_date['Date'])
    df_date.loc[:, 'Year'] = df_date['Date'].dt.year.astype(str)
    df_date.loc[:, 'Week'] = df_date['Date'].dt.strftime('%W').astype(int)
    df_date.loc[:, 'year_week'] = df_date['Year'].astype(str) + '_' + df_date['Week'].astype(str)

    grass_df.loc[:, 'Date'] = pd.to_datetime(grass_df['Date'])
    grass_df.loc[:, 'Year'] = grass_df['Date'].dt.year.astype(str)
    grass_df.loc[:, 'Week'] = grass_df['Date'].dt.strftime('%W').astype(int)
    grass_df.loc[:, 'year_week'] = grass_df['Year'].astype(str) + '_' + grass_df['Week'].astype(str)

    # Supprimer les colonnes inutiles
    df_date = df_date.drop(columns=['Year'])
    grass_df = grass_df.drop(columns=['Date', 'Year', 'Week'])

    # Fusionner les deux DataFrames sur year_week
    merged_df = pd.merge(df_date, grass_df, on='year_week', how='outer')
    merged_df = merged_df.sort_values(by='Date')


    # Initialize the `IterativeImputer`

    df_week = merged_df[['Week', 'Grass_growth']].copy()

    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=50, random_state=42, tol=1e-4)
    imputed_values = imputer.fit_transform(df_week)
    merged_df['Grass_growth'] = imputed_values[:, 1]
    merged_df.to_excel('spreadsheet/Grass_data_weekly_2009-2024.xlsx', index=False)



def yield_data_creation():
    df_yield = pd.read_csv('initial_datas/Data_weekly_yield.csv')
    df_yield['year_month'] = df_yield['year_month'].astype(str)
    df_yield['year'] = df_yield['year_month'].str[:4].astype(int)
    df_yield['month'] = df_yield['year_month'].str[4:6].astype(int)
    df_yield['Date'] = pd.to_datetime(df_yield['year'].astype(str) + df_yield['month'].astype(str).str.zfill(2), format='%Y%m') + pd.to_timedelta((df_yield['week'] - 1) * 7, unit='D')

    df_yield['yield_per_supplier'] = df_yield['litres'] / df_yield['num_suppliers']
    df_yield['cos_week'] = np.cos(df_yield['week'] * (2 * np.pi / 52))
    df_yield['year_week'] = df_yield['year'].astype(str) + '_' + df_yield['week'].astype(str)


    df_yield = df_yield.drop(columns=['Date','year_month','grass_growth','week','month','year'])
    df_yield.to_excel("spreadsheet/Yield_data_weekly_2009_2021.xlsx", index=False)


def final_data_creation():
    grass_data = pd.read_excel('spreadsheet/Grass_data_weekly_2009-2024.xlsx')
    price_data = pd.read_excel('spreadsheet/Prices_datas_weekly_2009-2021.xlsx')
    yield_data = pd.read_excel("spreadsheet/Yield_data_weekly_2009_2021.xlsx")

    grass_data = grass_data.drop(columns=['Date', 'Week'])

    final_data = pd.merge(price_data, grass_data, on='year_week')
    final_data = pd.merge(final_data, yield_data, on='year_week')

    # Mettre 'year_week' en première colonne
    columns = ['year_week'] + [col for col in final_data.columns if col != 'year_week']
    final_data = final_data[columns]

    # Utiliser pandas pour écrire dans un fichier Excel sans index
    final_data.to_excel("spreadsheet/Final_Weekly_2009_2021.xlsx", index=False)

price_data_creation()
#grass_data_creation()
#yield_data_creation()
#final_data_creation()