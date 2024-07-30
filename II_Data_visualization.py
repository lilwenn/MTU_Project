import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import altair_saver



# Function to apply moving average
def apply_moving_average(data, window):
    return data.rolling(window=window, min_periods=1).mean()


def plot_over_time ():

    # Plot for Irish Milk Price Over Time
    plt.figure(figsize=(10, 6))
    irish_milk_price = apply_moving_average(milk_price_df['Ireland_Milk_Price'], window=4)
    plt.plot(milk_price_df['Date'], irish_milk_price, label='Ireland', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Milk Price (euro/ 100 kg)')
    plt.title('Irish Milk Price Over Time')
    plt.legend()
    plt.savefig('visualization/irish_milk_price_over_time.png')
    plt.close()
    #plt.show()

    # Plot for Milk Price of all European Countries Over Time
    plt.figure(figsize=(18, 12))
    for country in milk_price_long_df['Country'].unique():
        country_data = milk_price_long_df[milk_price_long_df['Country'] == country]
        smoothed_price = apply_moving_average(country_data['Milk_Price'], window=4)
        plt.plot(country_data['Date'], smoothed_price, label=country)

    plt.xlabel('Date')
    plt.ylabel('Milk Price (euro/ 100 kg)')
    plt.title('European Milk Prices Over Time')
    plt.legend()
    plt.savefig('visualization/european_milk_prices_over_time.png')
    plt.close()

    #plt.show()

    # Plot for Grass Growth Over Time
    plt.figure(figsize=(10, 6))
    smoothed_grass_growth = apply_moving_average(df['Grass_growth'], window=4)
    plt.plot(df['Date'], smoothed_grass_growth, label='Grass Growth', color='green')
    plt.xlabel('Date')
    plt.ylabel('Grass Growth (cm)')
    plt.title('Grass Growth Over Time')
    plt.legend()
    plt.savefig('visualization/grass_growth_over_time.png')
    plt.close()

    #plt.show()

    # Plot for Litres of Milk Over Time
    plt.figure(figsize=(10, 6))
    smoothed_litres = apply_moving_average(df['litres'], window=4)
    plt.plot(df['Date'], smoothed_litres, label='Litres of Milk', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Litres of Milk')
    plt.title('Litres of Milk Over Time')
    plt.legend()
    plt.savefig('visualization/litres_of_milk_over_time.png')
    plt.close()

    #plt.show()

    # Plot for Number of Suppliers Over Time
    plt.figure(figsize=(10, 6))
    smoothed_suppliers = apply_moving_average(df['num_suppliers'], window=4)
    plt.plot(df['Date'], smoothed_suppliers, label='Number of Suppliers', color='purple')
    plt.xlabel('Date')
    plt.ylabel('Number of Suppliers')
    plt.title('Number of Suppliers Over Time')
    plt.legend()
    plt.savefig('visualization/number_of_suppliers_over_time.png')
    plt.close()

    #plt.show()


def plot_change_bar():
    # Plotting average change
    plt.figure(figsize=(12, 8))
    sorted_data = summary_df['Average Change (%)'].sort_values(ascending=False)
    plt.bar(sorted_data.index, sorted_data.values, color='skyblue')
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Average Milk Price Change (%)', fontsize=12)
    plt.title('Average Milk Price Change by Country', fontsize=14)
    for i, v in enumerate(sorted_data):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig('visualization/avg_milk_price_change_bar_chart.png')
    plt.close()

    #plt.show()

    # Plotting total change
    plt.figure(figsize=(12, 8))
    sorted_data = summary_df['Total Change (%)'].sort_values(ascending=False)
    plt.bar(sorted_data.index, sorted_data.values, color='lightgreen')
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Total Milk Price Change (%)', fontsize=12)
    plt.title('Total Milk Price Change by Country', fontsize=14)
    for i, v in enumerate(sorted_data):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig('visualization/total_milk_price_change_bar_chart.png')
    plt.close()

    #plt.show()

    # Plotting mean price
    plt.figure(figsize=(12, 8))
    sorted_data = summary_df['Mean Price (€/L)'].sort_values(ascending=False)
    plt.bar(sorted_data.index, sorted_data.values, color='lightcoral')
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Mean Milk Price (€/L)', fontsize=12)
    plt.title('Mean Milk Price by Country', fontsize=14)
    for i, v in enumerate(sorted_data):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig('visualization/mean_milk_price_bar_chart.png')
    plt.close()

    #plt.show()

    # Plotting median price
    plt.figure(figsize=(12, 8))
    sorted_data = summary_df['Median Price (€/L)'].sort_values(ascending=False)
    plt.bar(sorted_data.index, sorted_data.values, color='lavender')
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Median Milk Price (€/L)', fontsize=12)
    plt.title('Median Milk Price by Country', fontsize=14)
    for i, v in enumerate(sorted_data):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig('visualization/median_milk_price_bar_chart.png')
    plt.close()

    #plt.show()


def plot_after_smoothing(data_ireland, target_column, data_MA):

    plt.figure(figsize=(15, 6))
    plt.plot(data_ireland.index, data_ireland[target_column], label='Original Data', color='blue')
    plt.plot(data_MA.index, data_MA[target_column], label='Smoothed Data', color='red')
    plt.title('Original vs Smoothed Time Series: Milk Price in Ireland')
    plt.xlabel('Date')
    plt.ylabel('Milk Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualization/original_vs_smoothed.png')
    plt.close()


def plot_decomposed_components(decomposition):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    decomposition.observed.plot(ax=ax1, title='Original Time Series: Milk Price in Ireland')
    decomposition.trend.plot(ax=ax2, title='Trend: Milk Price in Ireland')
    decomposition.seasonal.plot(ax=ax3, title='Seasonality: Milk Price in Ireland')
    decomposition.resid.plot(ax=ax4, title='Residual: Milk Price in Ireland')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig('visualization/time_series_analysis.png')
    plt.close()


def plot_correlation_matrix(df, target, selected_features, output_file='correlation_matrix.png'):
    """
    Plots and saves the correlation matrix for the selected features.
    
    Args:
    df (pd.DataFrame): DataFrame containing the time series data.
    selected_features (list): List of selected features after applying SelectKBest.
    output_file (str): Path to save the correlation matrix image.
    """
    # Compute the correlation matrix
    corr_matrix = df[selected_features].corr()
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix of Selected Features for {target}')
    plt.tight_layout()
    
    # Save the heatmap to a file
    plt.savefig(output_file)
    plt.close()


# Read the Excel file into a DataFrame
df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')

# Convert the `Date` column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter the columns that contain 'Milk_Price' or 'Date'
milk_price_df = df.filter(regex='Milk_Price|Date')

# Melt the DataFrame to long format
milk_price_long_df = milk_price_df.melt(id_vars='Date', var_name='Country', value_name='Milk_Price')

# Extract the country name from the column name
milk_price_long_df['Country'] = milk_price_long_df['Country'].apply(lambda x: x[:-10])

# Set the style
sns.set(style='whitegrid')

plot_over_time ()


# Filter columns for milk prices
milk_price_columns = milk_price_df.columns[1:]  # Exclude 'Date' column

# Calculate average percentage change for each country
avg_pct_change = milk_price_df[milk_price_columns].pct_change().mean() * 100

# Calculate total percentage change for each country
total_pct_change = (milk_price_df[milk_price_columns].pct_change() + 1).prod() * 100 - 100

# Calculate mean milk price for each country
mean_price = milk_price_df[milk_price_columns].mean()

# Calculate median milk price for each country
median_price = milk_price_df[milk_price_columns].median()

# Create summary DataFrame
summary_df = pd.DataFrame({
    'Average Change (%)': avg_pct_change,
    'Total Change (%)': total_pct_change,
    'Mean Price (€/L)': mean_price,
    'Median Price (€/L)': median_price
})

# Sort by average change
summary_df = summary_df.sort_values(by='Average Change (%)', ascending=False)
plot_change_bar()
