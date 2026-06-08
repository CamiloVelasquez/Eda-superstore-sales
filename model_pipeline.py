import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='utf-8')
    logging.info("Dataset loaded successfully.")
    return df

def preprocess_data(df):
    """Applies preprocessing steps to the dataset."""
    # Convert date columns
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y')
    logging.info("Date columns converted to datetime format.")

    # Feature Engineering
    df['Order_Year'] = df['Order Date'].dt.year
    df['Order_Month'] = df['Order Date'].dt.month
    df['Order_Day'] = df['Order Date'].dt.day
    df['Ship_Year'] = df['Ship Date'].dt.year
    df['Ship_Month'] = df['Ship Date'].dt.month
    df['Ship_Day'] = df['Ship Date'].dt.day
    df['Order_Processing_Time'] = (df['Ship Date'] - df['Order Date']).dt.days
    logging.info("Date-based features and order processing time engineered.")

    # Drop unnecessary columns
    columns_to_drop = [
        'Row ID', 'Order ID', 'Customer ID', 'Customer Name',
        'Product ID', 'Product Name', 'Postal Code', 'Order Date', 'Ship Date'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    logging.info(f"Dropped columns: {columns_to_drop}")

    return df

def train_and_evaluate(model_name, model, param_distributions, X_train, y_train, X_test, y_test):
    """Trains, tunes, and evaluates a model, logging results to MLflow."""
    with mlflow.start_run(run_name=model_name):
        logging.info(f"Starting MLflow run for {model_name}")

        # Log model parameters
        mlflow.log_param("model_name", model_name)

        # Setup preprocessing pipelines for numerical and categorical features
        numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough' # Keep other columns (if any)
        )

        # Create the full pipeline
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Hyperparameter tuning
        logging.info(f"Starting RandomizedSearchCV for {model_name}...")
        random_search = RandomizedSearchCV(
            full_pipeline,
            param_distributions,
            n_iter=10,  # Number of parameter settings that are sampled
            cv=3,       # Number of folds for cross-validation
            verbose=1,
            random_state=42,
            n_jobs=-1   # Use all available cores
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        logging.info(f"RandomizedSearchCV completed for {model_name}. Best parameters: {random_search.best_params_}")

        # Log best hyperparameters
        mlflow.log_params(random_search.best_params_)

        # Evaluate the best model
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Log metrics
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2_score": r2})

        # Log the model
        mlflow.sklearn.log_model(best_model, "model")
        logging.info(f"Model '{model_name}' logged to MLflow.")

def main():
    file_path = 'sample_-_superstore.csv'
    df = load_data(file_path)
    df = preprocess_data(df)

    # Define features (X) and target (y)
    X = df.drop('Profit', axis=1)
    y = df['Profit']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets.")

    # MLflow experiment setup
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Superstore Profit Prediction")

    # --- Linear Regression Model ---
    lr_param_dist = {
        'regressor__fit_intercept': [True, False]
    }
    train_and_evaluate("Linear Regression", LinearRegression(), lr_param_dist, X_train, y_train, X_test, y_test)

    # --- Random Forest Regressor Model ---
    rf_param_dist = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_features': ['auto', 'sqrt', 'log2'],
        'regressor__max_depth': [10, 20, 30, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    train_and_evaluate("Random Forest Regressor", RandomForestRegressor(random_state=42), rf_param_dist, X_train, y_train, X_test, y_test)

    # --- Gradient Boosting Regressor Model ---
    gbr_param_dist = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 8],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__subsample': [0.7, 0.8, 0.9, 1.0]
    }
    train_and_evaluate("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42), gbr_param_dist, X_train, y_train, X_test, y_test)

    logging.info("All models trained and evaluated. Check MLflow UI for results.")

if __name__ == "__main__":
    main()
