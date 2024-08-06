import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error
import Constants as const

# Custom function to extract datetime features
def extract_date_features(df):
    df = df.copy()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.isocalendar().week
    df['day'] = df['Date'].dt.day
    df = df.drop(columns=['Date'])  # Drop original Date column if no longer needed
    return df

# Load the data
df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply Date extraction function
X = df.drop(columns=['litres'])
X = extract_date_features(X)
y = df['litres']

# Split the data into training and future sets
n_future_weeks = 52  # Adjust this as needed
X_train, X_future = X[:-n_future_weeks], X[-n_future_weeks:]
y_train, y_future = y[:-n_future_weeks], y[-n_future_weeks:]

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Evaluate models
results = {}
best_estimators = {}
mapes = {}

for name, model in const.MODELS.items():
    print(f"Training {name}...")

    for scaler_name, scaler in const.SCALERS.items():

        for scoring_name, scoring_func in const.SCORING_METHODS.items():

            default_scoring_func = f_regression
            default_k = 5

            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
                ('scaler', scaler),
                ('selectkbest', SelectKBest(score_func=scoring_func if scoring_func else default_scoring_func, k=const.K_VALUES.get(scoring_name, default_k))),
                ('model', model)
            ])

            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=const.HYPERPARAMETERS[name],
                n_iter=10,
                cv=tscv,
                verbose=2,
                n_jobs=-1
            )
            random_search.fit(X_train, y_train)
            
            best_model = random_search.best_estimator_
            best_score = -random_search.best_score_
            
            results[name] = best_score
            best_estimators[name] = best_model

    # Display results
    for name, score in results.items():
        print(f"{name}: Mean Squared Error = {score}")

    # Train the best model on all training data
    best_model_name = min(results, key=results.get)
    best_model = best_estimators[best_model_name]

    # Use the pipeline to transform and predict future data
    X_future_transformed = best_model.transform(X_future)  # Apply transformations in the pipeline
    pred = best_model.predict(X_future_transformed)
    
    print(f"Predicted future prices: {pred}")
    print(f"Actual future prices: {y_future.values}")

    # Calculate and print MAPE
    mape = mean_absolute_percentage_error(y_future, pred)
    mapes[name] = mape
    print(f"{name}: Mean Absolute Percentage Error (MAPE) = {mape}")

# Print the MAPE of each model
for name, mape in mapes.items():
    print(f"{name}: Mean Absolute Percentage Error (MAPE) = {mape}")
