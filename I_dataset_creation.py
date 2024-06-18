import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def visualization_sorted_tab():
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
