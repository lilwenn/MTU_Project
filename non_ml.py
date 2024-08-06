def train_darts_modelquifonctionne(df, forecast_weeks):
    """
    Train Darts models on the time series data and evaluate their performance.

    Args:
    - df (DataFrame): Preprocessed DataFrame ready for model training.
    - forecast_weeks (int): Number of weeks to forecast.

    Returns:
    - None
    """
    # Ensure the Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Remove duplicate dates
    df = df.drop_duplicates(subset='Date')

    # Sort the DataFrame by date
    df = df.sort_values(by='Date')

    # Set the Date column as the index
    df.set_index('Date', inplace=True)

    # Reindex the DataFrame to fill in missing dates with a specified frequency (e.g., 'W' for weekly)
    df = df.asfreq('W', method='ffill')

    # Convert the DataFrame to a TimeSeries object, only using the target column
    series = TimeSeries.from_dataframe(df[[const.TARGET_COLUMN]], fill_missing_dates=True)

    # Split the data into training and validation sets
    train, val = darts_train_test_split(series, test_size=forecast_weeks)

    # Optionally scale the time series
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)

    # Define models to train
    models = {
        "NBEATSModel": NBEATSModel(input_chunk_length=24, output_chunk_length=forecast_weeks),
        "NaiveSeasonal": NaiveSeasonal(K=52),  # Example of a simple seasonal naive model
        "ExponentialSmoothing": ExponentialSmoothing()
    }

    # Train and evaluate each model
    results = {}
    for model_name, model in models.items():
        model.fit(train_transformed)
        prediction = model.predict(len(val))
        mape_score = darts_mape(val_transformed, prediction)

        # Inverse transform the prediction to get actual scale
        prediction = transformer.inverse_transform(prediction)

        results[model_name] = {
            "MAPE": mape_score,
            "Prediction": prediction.values().flatten().tolist()
        }

        print(f"{model_name} - MAPE: {mape_score:.2f}")

    # Save the results to a JSON file
    os.makedirs('result', exist_ok=True)
    with open(f'result/darts_models_{forecast_weeks}weeks.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results

def train_darts_model(df, forecast_weeks):
    # Convert the Date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    # Remove duplicate dates and sort the DataFrame by date
    df = df.drop_duplicates(subset='Date').sort_values(by='Date')
    # Set the Date column as the index
    df.set_index('Date', inplace=True)
    # Fill missing dates with the specified frequency
    df = df.asfreq('W', method='ffill')
    # Convert the DataFrame to a TimeSeries object
    series = TimeSeries.from_dataframe(df[[const.TARGET_COLUMN]], fill_missing_dates=True)

    # Split the data into training and validation sets
    train, val = darts_train_test_split(series, test_size=forecast_weeks)
    # Optionally scale the time series
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)

    # Define models to train
    models = {
        "NaiveSeasonal": NaiveSeasonal(K=52),
    }
    
    for model_name, model in models.items():
        # Fit the model on the transformed training data
        model.fit(train_transformed)
        # Predict the future values
        prediction = model.predict(len(val_transformed))
        # Inverse transform the prediction to get actual scale
        prediction = transformer.inverse_transform(prediction)

    # Return the predictions as a flattened list
    return prediction.values().flatten().tolist()

def train_prophet_model(df, forecast_weeks):
    """
    Train Prophet models on the time series data and evaluate their performance.

    Args:
    - df (DataFrame): Preprocessed DataFrame ready for model training.
    - forecast_weeks (int): Number of weeks to forecast.

    Returns:
    - dict: Results with MAPE and predictions from Prophet.
    """
    # Ensure the Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Remove duplicate dates
    df = df.drop_duplicates(subset='Date')

    # Sort the DataFrame by date
    df = df.sort_values(by='Date')

    # Set the Date column as the index
    df = df.set_index('Date')

    # Reindex the DataFrame to fill in missing dates with a specified frequency (e.g., 'W' for weekly)
    df = df.asfreq('W', method='ffill')

    # Prepare data for Prophet
    df_prophet = df.reset_index().rename(columns={'Date': 'ds', const.TARGET_COLUMN: 'y'})

    # Split the data into training and validation sets
    train = df_prophet[:-forecast_weeks]
    val = df_prophet[-forecast_weeks:]

    # Define and train the Prophet model
    model = Prophet()
    model.fit(train)

    # Create a DataFrame for future dates
    future = model.make_future_dataframe(periods=forecast_weeks, freq='W')

    # Predict future values
    forecast = model.predict(future)

    # Extract predictions and ensure alignment with validation data
    predictions = forecast[['ds', 'yhat']].tail(forecast_weeks).set_index('ds').rename(columns={'yhat': 'Prediction'})

    # Ensure that predictions and actual values align
    actual = val.set_index('ds')['y']

    # Align predictions with actual values
    aligned_predictions = predictions.reindex(actual.index)

    # Calculate MAPE
    mape = np.mean(np.abs((actual - aligned_predictions['Prediction']) / actual)) * 100

    # Prepare results for JSON (no dates, only predictions and MAPE)
    predictions_list = aligned_predictions['Prediction'].tolist()

    results = {
        "MAPE": mape,
        "Predictions": predictions_list
    }

    print(f"Prophet - MAPE: {mape:.2f}")

    # Save the results to a JSON file
    os.makedirs('result', exist_ok=True)
    with open(f'result/prophet_model_{forecast_weeks}weeks.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results

def train_TPOT_model(df, TARGET_COLUMN):

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)

    # Préparation des données
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    # Utiliser TimeSeriesSplit pour diviser les données
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_score = float('inf')
    best_model_performance = {}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Utiliser TPOT pour entraîner un modèle
        tpot = TPOTRegressor(generations=10, population_size=100, verbosity=2, random_state=42, cv=5, scoring='neg_mean_squared_error')
        tpot.fit(X_train, y_train)

        # Prédictions
        y_pred = tpot.predict(X_test)

        # Évaluation du modèle
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print(f'Erreur quadratique moyenne (MSE): {mse}')
        print(f'Coefficient de détermination (R^2): {r2}')
        print(f'Erreur absolue moyenne (MAE): {mae}')
        print(f'Erreur absolue moyenne en pourcentage (MAPE): {score_mape}')

        # Mémoriser le meilleur modèle basé sur MSE
        if mse < best_score:
            best_score = mse
            best_model = tpot.fitted_pipeline_

            # Enregistrer les performances du meilleur modèle
            best_model_performance = {
                'MSE': mse,
                'R2': r2,
                'MAE': mae,
                'MAPE': score_mape
            }

            # Enregistrer le meilleur modèle
            joblib.dump(best_model, 'best_tpot_model.pkl')

    # Charger le meilleur modèle
    best_model = joblib.load('best_tpot_model.pkl')

    # Entraîner le meilleur modèle sur toutes les données
    best_model.fit(X, y)

    # Faire des prédictions pour les 52 semaines
    if len(X) >= 52:
        y_pred_52_weeks = best_model.predict(X.head(52))
        print('Prédictions pour les 52 semaines:', y_pred_52_weeks)
    else:
        print("Pas assez de données pour les prédictions sur 52 semaines")

    # Faire des prédictions sur l'ensemble des données
    y_pred = best_model.predict(X)

    # Calculer MAE et MAPE pour les prédictions sur l'ensemble des données
    mae = mean_absolute_error(y, y_pred)
    score_mape = mape(y, y_pred)

    print(f'Erreur absolue moyenne (MAE): {mae}')
    print(f'Erreur absolue moyenne en pourcentage (MAPE): {score_mape}')

    # Afficher les performances du meilleur modèle
    print("Performances du meilleur modèle :")
    print(best_model_performance)

def train_and_predict_arima(df, exog_future, order=(1, 1, 1)):
    # Check if 'Date' is in the columns
    if 'Date' in df.columns:
        # Set the date as index
        df_copy = df.set_index('Date')
    else:
        # Assume 'Date' is the index
        df_copy = df.copy()

    # Infer the frequency
    inferred_freq = pd.infer_freq(df_copy.index)

    # Resample if not weekly on Sundays
    if inferred_freq != 'W-SUN':
        df_copy = df_copy.resample('W-SUN').mean().fillna(df_copy.mean())

    # Create a new dataframe exog_data by dropping columns that contain the string 'litres'
    exog_data = df_copy.drop(columns=[col for col in df_copy.columns if 'litres' in col])

    # Ensure the number of columns in exog_future matches exog_data
    exog_future = exog_future.reindex(columns=exog_data.columns, fill_value=0)

    # Train the ARIMA model
    model = ARIMA(df_copy['litres'], exog=exog_data, order=order, dates=df_copy.index, freq='W-SUN')
    model_fit = model.fit()

    # Predict the next const.FORECAST_WEEKS weeks
    predictions = model_fit.forecast(steps=const.FORECAST_WEEKS, exog=exog_future)

    # Get the last date in the training data
    last_date = df_copy.index.max()

    # Reset the index
    df_copy = df_copy.reset_index()

    return predictions, last_date

def train_pmdarima_model(df, target_column, forecast_weeks):
    # Préparation des données
    data = df[target_column]

    # Diviser les données en ensemble d'entraînement et de test
    train_size = len(data) - forecast_weeks
    train, test = data[:train_size], data[train_size:]

    # Ajuster le modèle auto_arima avec différents paramètres
    model = auto_arima(train, seasonal=True, m=12, stepwise=True, trace=True)
    
    # Faire des prévisions
    forecast = model.predict(n_periods=len(test))
    return forecast, test



def train_models(df, model_name, model, w):
    features = [col for col in df.columns if not col.startswith(f'{const.TARGET_COLUMN}_next_') and col != 'Date']
    result = {}

    if model_name in ['ARIMA', 'Pmdarima', 'Darts']:
        if model_name == 'ARIMA':
            exog_train, exog_future = preprocess_arima_data(df, const.FORECAST_WEEKS)
            df_copy = df.set_index('Date') if 'Date' in df.columns else df.copy()
            best_order = (1, 1, 1)
            start_time = time.time()
            predictions, last_date = train_and_predict_arima(df, exog_future, best_order)
            end_time = time.time()
            execution_time = end_time - start_time
            actual_values = df_copy[const.TARGET_COLUMN].iloc[-const.FORECAST_WEEKS:].values
            result = calc_scores(actual_values, predictions, execution_time)

        elif model_name == 'Pmdarima':
            start_time = time.time()
            predictions, test = train_pmdarima_model(df, const.TARGET_COLUMN, const.FORECAST_WEEKS)
            predictions = predictions.tolist()
            end_time = time.time()
            execution_time = end_time - start_time
            actual_values = df[const.TARGET_COLUMN].iloc[-const.FORECAST_WEEKS:].values
            result = calc_scores(actual_values, predictions, execution_time)

        elif model_name == 'Darts':
            start_time = time.time()
            predictions = train_darts_model(df, const.FORECAST_WEEKS)
            end_time = time.time()
            execution_time = end_time - start_time
            actual_values = df[const.TARGET_COLUMN].iloc[-const.FORECAST_WEEKS:].values
            result = calc_scores(actual_values, predictions, execution_time)

    else:
        for scaler_name, scaler in const.SCALERS.items():
            result[scaler_name] = {}

            for scoring_name, scoring_func in const.SCORING_METHODS.items():
                result[scaler_name][scoring_name] = {}
                default_scoring_func = f_regression
                default_k = 5

                pipeline = Pipeline([
                    ('scaler', scaler),
                    ('selectkbest', SelectKBest(score_func=scoring_func if scoring_func else default_scoring_func, k=const.K_VALUES.get(scoring_name, default_k))),
                    ('model', model)
                ])
                random_search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=const.HYPERPARAMETERS[model_name],
                    n_iter=10,
                    cv=5,
                    verbose=2,
                    n_jobs=-1
                )

                start_time = time.time()
                random_search.fit(df[features], df[const.TARGET_COLUMN])
                end_time = time.time()
                total_execution_time = end_time - start_time

                best_pipeline = random_search.best_estimator_

                # Get predictions on the entire dataset
                y_test, y_pred, execution_time = train_and_predict(df, features, const.TARGET_COLUMN, best_pipeline)

                result[scaler_name][scoring_name] = calc_scores(y_test, y_pred, total_execution_time)

    print(f"All combinations of {model_name} for window ={w} saved in json")
    with open(f'result/by_model/{model_name}_{w}_model_{const.FORECAST_WEEKS}.json', 'w') as f:
        json.dump(result, f, indent=4)


