# Iterate over the models
for model in models:
  model_name = model.__class__.__name__

  # Reset index before converting to datetime
  X_train_reset = X_train.reset_index(drop=True)

  r2, mse, mae = model_evaluation(model, X_train_reset, X_test, y_train, y_test)
  results[model_name] = {'R2': r2, 'MSE': mse, 'MAE': mae}

  # Generate predictions for 1, 2, and 3 months after the last date in the training data
  def prediction_generation(model, X_train, X_test, y_train, y_test):
    # Fit the model
    model.fit(X_train, y_train)

    # Get last date from the original X_train
    last_date = X_train.index.max()

    # Generate future dates, adding 1 day to start from the next month
    future_dates = pd.date_range(last_date + pd.DateOffset(days=1), periods=3, freq='MS')
    future_X = X.loc[future_dates]
    future_predictions = model.predict(future_X)

    return future_predictions

  predictions = prediction_generation(model, X_train_reset, X_test, y_train, y_test)
  predictions_dict[model_name] = predictions

# Create dataframes from the results
results_df = pd.DataFrame(results).transpose()
predictions_df = pd.DataFrame(predictions_dict)

# Print the results
print("Evaluation metrics for different models:")
print(results_df.to_markdown(numalign="left", stralign="left"))
print("\nPredictions for the next 3 months for different models:")
print(predictions_df.to_markdown(numalign="left", stralign="left"))

# Iterate over the models
for model in models:
  model_name = model.__class__.__name__

  # Reset index before converting to datetime
  X_train_reset = X_train.reset_index(drop=True)

  r2, mse, mae = model_evaluation(model, X_train_reset, X_test, y_train, y_test)
  results[model_name] = {'R2': r2, 'MSE': mse, 'MAE': mae}

  # Generate predictions for 1, 2, and 3 months after the last date in the training data
  def prediction_generation(model, X_train, X_test, y_train, y_test):
    # Fit the model
    model.fit(X_train, y_train)

    # Get last date from the original X_train
    last_date = X_train.index.max()

    # Convert last_date to pandas Timestamp before adding DateOffset
    last_date = pd.Timestamp(last_date)

    # Generate future dates, adding 1 day to start from the next month
    future_dates = pd.date_range(last_date + pd.DateOffset(days=1), periods=3, freq='MS')
    future_X = X.loc[future_dates]
    future_predictions = model.predict(future_X)

    return future_predictions

  predictions = prediction_generation(model, X_train_reset, X_test, y_train, y_test)
  predictions_dict[model_name] = predictions

# Create dataframes from the results
results_df = pd.DataFrame(results).transpose()
predictions_df = pd.DataFrame(predictions_dict)

# Print the results
print("Evaluation metrics for different models:")
print(results_df.to_markdown(numalign="left", stralign="left"))
print("\nPredictions for the next 3 months for different models:")
print(predictions_df.to_markdown(numalign="left", stralign="left"))
# Iterate over the models
for model in models:
  model_name = model.__class__.__name__

  # Reset index before converting to datetime
  X_train_reset = X_train.reset_index(drop=True)

  r2, mse, mae = model_evaluation(model, X_train_reset, X_test, y_train, y_test)
  results[model_name] = {'R2': r2, 'MSE': mse, 'MAE': mae}

  # Generate predictions for 1, 2, and 3 months after the last date in the training data
  def prediction_generation(model, X_train, X_test, y_train, y_test):
    # Fit the model
    model.fit(X_train, y_train)

    # Get last date from the original X_train and normalize to midnight
    last_date = X_train.index.max().normalize()

    # Convert last_date to pandas Timestamp before adding DateOffset
    last_date = pd.Timestamp(last_date)

    # Generate future dates, adding 1 day to start from the next month
    future_dates = pd.date_range(last_date + pd.DateOffset(days=1), periods=3, freq='MS')
    future_X = X.loc[future_dates]
    future_predictions = model.predict(future_X)

    return future_predictions

  predictions = prediction_generation(model, X_train_reset, X_test, y_train, y_test)
  predictions_dict[model_name] = predictions

# Create dataframes from the results
results_df = pd.DataFrame(results).transpose()
predictions_df = pd.DataFrame(predictions_dict)

# Print the results
print("Evaluation metrics for different models:")
print(results_df.to_markdown(numalign="left", stralign="left"))
print("\nPredictions for the next 3 months for different models:")
print(predictions_df.to_markdown(numalign="left", stralign="left"))

# Convert the index of X to datetime
X.index = pd.to_datetime(X.index)

# Split the data into training and testing sets
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Iterate over the models
for model in models:
  model_name = model.__class__.__name__

  # Reset index before converting to datetime
  X_train_reset = X_train.reset_index(drop=True)

  r2, mse, mae = model_evaluation(model, X_train_reset, X_test, y_train, y_test)
  results[model_name] = {'R2': r2, 'MSE': mse, 'MAE': mae}

  # Generate predictions for 1, 2, and 3 months after the last date in the training data
  def prediction_generation(model, X_train, X_test, y_train, y_test):
    # Fit the model
    model.fit(X_train, y_train)

    # Get last date from the original X_train and normalize to midnight
    last_date = X_train.index.max().normalize()

    # Convert last_date to pandas Timestamp before adding DateOffset
    last_date = pd.Timestamp(last_date)

    # Generate future dates, adding 1 day to start from the next month
    future_dates = pd.date_range(last_date + pd.DateOffset(days=1), periods=3, freq='MS')
    future_X = X.loc[future_dates]
    future_predictions = model.predict(future_X)

    return future_predictions

  predictions = prediction_generation(model, X_train_reset, X_test, y_train, y_test)
  predictions_dict[model_name] = predictions

# Create dataframes from the results
results_df = pd.DataFrame(results).transpose()
predictions_df = pd.DataFrame(predictions_dict)

# Print the results
print("Evaluation metrics for different models:")
print(results_df.to_markdown(numalign="left", stralign="left"))
print("\nPredictions for the next 3 months for different models:")
print(predictions_df.to_markdown(numalign="left", stralign="left"))


# Split the data into training and testing sets
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Iterate over the models
for model in models:
  model_name = model.__class__.__name__

  # Reset index before converting to datetime
  X_train_reset = X_train.reset_index(drop=True)

  r2, mse, mae = model_evaluation(model, X_train_reset, X_test, y_train, y_test)
  results[model_name] = {'R2': r2, 'MSE': mse, 'MAE': mae}

  # Generate predictions for 1, 2, and 3 months after the last date in the training data
  def prediction_generation(model, X_train, X_test, y_train, y_test):
    # Fit the model
    model.fit(X_train, y_train)

    # Convert X_train index to datetime
    X_train.index = pd.to_datetime(X_train.index)

    # Get last date from X_train and normalize to midnight
    last_date = X_train.index.max().normalize()

    # Convert last_date to pandas Timestamp before adding DateOffset
    last_date = pd.Timestamp(last_date)

    # Generate future dates, adding 1 day to start from the next month
    future_dates = pd.date_range(last_date + pd.DateOffset(days=1), periods=3, freq='MS')
    future_X = X.loc[future_dates]
    future_predictions = model.predict(future_X)

    return future_predictions

  predictions = prediction_generation(model, X_train_reset, X_test, y_train, y_test)
  predictions_dict[model_name] = predictions

# Create dataframes from the results
results_df = pd.DataFrame(results).transpose()
predictions_df = pd.DataFrame(predictions_dict)

# Print the results
print("Evaluation metrics for different models:")
print(results_df.to_markdown(numalign="left", stralign="left"))
print("\nPredictions for the next 3 months for different models:")
print(predictions_df.to_markdown(numalign="left", stralign="left"))


# Split the data into training and testing sets
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
  # Convert X_train index to datetime before reset
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  X_train.index = pd.to_datetime(X_train.index)
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  model_name = model.__class__.__name__

  # Reset index before converting to datetime
  X_train_reset = X_train.reset_index(drop=True)

  r2, mse, mae = model_evaluation(model, X_train_reset, X_test, y_train, y_test)
  results[model_name] = {'R2': r2, 'MSE': mse, 'MAE': mae}

  # Generate predictions for 1, 2, and 3 months after the last date in the training data
  def prediction_generation(model, X_train, X_test, y_train, y_test):
    # Fit the model
    model.fit(X_train, y_train)

    # Get last date from the original X_train and normalize to midnight
    last_date = X_train.index.max().normalize()

    # Convert last_date to pandas Timestamp before adding DateOffset
    last_date = pd.Timestamp(last_date)

    # Generate future dates, adding 1 day to start from the next month
    future_dates = pd.date_range(last_date + pd.DateOffset(days=1), periods=3, freq='MS')
    future_X = X.loc[future_dates]
    future_predictions = model.predict(future_X)

    return future_predictions

  predictions = prediction_generation(model, X_train_reset, X_test, y_train, y_test)
  predictions_dict[model_name] = predictions

# Create dataframes from the results
results_df = pd.DataFrame(results).transpose()
predictions_df = pd.DataFrame(predictions_dict)

# Print the results
print("Evaluation metrics for different models:")
print(results_df.to_markdown(numalign="left", stralign="left"))
print("\nPredictions for the next 3 months for different models:")
print(predictions_df.to_markdown(numalign="left", stralign="left"))
