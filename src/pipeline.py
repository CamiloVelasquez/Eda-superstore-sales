import json
import logging
import os
import pickle
import socket
import warnings
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

# Shared MLflow constants (reused by the Streamlit app)
EXPERIMENT_NAME = "Superstore Profit Prediction"
REGISTERED_MODEL_NAME = "SuperstoreProfitBestModel"
DEFAULT_REMOTE_TRACKING_URI = "http://127.0.0.1:5000"
LOCAL_TRACKING_URI = "sqlite:///mlflow.db"

# Pickle export — fallback cuando MLflow no está disponible
PICKLE_PATH = "best_model.pkl"
PICKLE_META_PATH = "best_model_meta.json"

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

    # Drop unnecessary columns.
    # 'City': alta cardinalidad (542 valores), infla OHE sin aportar señal.
    # Ship_Year/Month/Day: redundantes con Order_Year/Month + Order_Processing_Time
    #   (VIF > 4M en análisis de supuestos). Order_Day: r≈0 con Profit y VIF alto.
    columns_to_drop = [
        'Row ID', 'Order ID', 'Customer ID', 'Customer Name',
        'Product ID', 'Product Name', 'Postal Code', 'Order Date', 'Ship Date',
        'City',
        'Ship_Year', 'Ship_Month', 'Ship_Day', 'Order_Day',
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    logging.info(f"Dropped columns: {columns_to_drop}")

    return df

def train_and_evaluate(model_name, model, param_distributions, X_train, y_train, X_test, y_test):
    """Trains, tunes, and evaluates a model, logging results to MLflow.

    Returns a dict with the fitted best model and its test metrics so the caller
    can compare experiments and select the best model.
    """
    with mlflow.start_run(run_name=model_name) as run:
        logging.info(f"Starting MLflow run for {model_name}")

        # Log model parameters
        mlflow.log_param("model_name", model_name)

        # Setup preprocessing pipelines for numerical and categorical features
        numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

        # RobustScaler usa IQR en lugar de media/std: menos sensible a outliers extremos.
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler())
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
            n_iter=8,
            cv=2,
            verbose=1,
            random_state=0,
            n_jobs=-1,
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

        # Log the model (with signature and input example for reproducibility)
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            best_model,
            name="model",
            signature=signature,
            input_example=X_test.head(),
        )
        logging.info(f"Model '{model_name}' logged to MLflow.")

        # Return results so the caller can compare and select the best model
        return {
            "model_name": model_name,
            "model": best_model,
            "run_id": run.info.run_id,
            "best_params": random_search.best_params_,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

def _registered_model_exists():
    """Returns True if a version of the best-model registry entry already exists."""
    try:
        from mlflow.tracking import MlflowClient
        return bool(MlflowClient().search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"))
    except Exception:
        return False


def main():
    # MLflow experiment setup (resolve backend before any check)
    setup_tracking_uri()

    # Idempotencia para Docker: si ya hay un modelo registrado y se pide omitir,
    # no se vuelve a entrenar (util para 'docker compose up' repetidos).
    if os.environ.get("SKIP_IF_REGISTERED") == "1" and _registered_model_exists():
        logging.info(
            f"A registered '{REGISTERED_MODEL_NAME}' model already exists; "
            "skipping training (SKIP_IF_REGISTERED=1)."
        )
        return

    mlflow.set_experiment(EXPERIMENT_NAME)

    file_path = 'data/raw/sample_-_superstore.csv'
    df = load_data(file_path)
    df = preprocess_data(df)

    # Define features (X) and target (y)
    X = df.drop('Profit', axis=1)
    y = df['Profit']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets.")

    results = []

    # --- Regresión Lineal (baseline) ---
    # Se incluye como referencia aunque viola varios supuestos (normalidad de
    # residuos, homocedasticidad, VIF en fechas). Su RMSE alto sirve para
    # cuantificar la mejora que aportan los modelos robustos y no lineales.
    lr_param_dist = {
        'regressor__fit_intercept': [True, False],
    }
    results.append(train_and_evaluate("Linear Regression", LinearRegression(), lr_param_dist, X_train, y_train, X_test, y_test))

    # --- Huber Regressor (versión robusta de la regresión lineal) ---
    # HuberRegressor minimiza una función de pérdida mixta L1/L2 que descarta
    # outliers extremos, corrigiendo la violación de normalidad y homocedasticidad
    # detectada en el análisis de supuestos (skewness residuos=15.5, kurtosis=397).
    hr_param_dist = {
        'regressor__epsilon': [1.1, 1.35, 1.5, 2.0],   # umbral outlier
        'regressor__alpha':   [0.0001, 0.001, 0.01],    # regularización L2
    }
    results.append(train_and_evaluate("Huber Regressor", HuberRegressor(max_iter=300), hr_param_dist, X_train, y_train, X_test, y_test))

    # --- Random Forest Regressor Model ---
    # Rangos centrados en los mejores parámetros encontrados: n_estimators=300,
    # max_features=1.0, max_depth=20, min_samples_split=2, min_samples_leaf=1
    rf_param_dist = {
        'regressor__n_estimators': [200, 300],
        'regressor__max_features': ['sqrt', 1.0],
        'regressor__max_depth': [10, 20],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
    }
    results.append(train_and_evaluate("Random Forest Regressor", RandomForestRegressor(random_state=42), rf_param_dist, X_train, y_train, X_test, y_test))

    # --- Gradient Boosting Regressor con pérdida Huber ---
    # loss='huber' es menos sensible a los outliers extremos del target (18.8% IQR)
    # que la pérdida cuadrática por defecto: pondera con L2 errores pequeños y con
    # L1 los grandes, controlado por el parámetro alpha (cuantil de transición).
    gbr_param_dist = {
        'regressor__n_estimators': [200, 300],
        'regressor__learning_rate': [0.1, 0.2],
        'regressor__max_depth': [3, 5],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
        'regressor__subsample': [0.8, 0.9],
        'regressor__alpha': [0.85, 0.90, 0.95],
    }
    results.append(train_and_evaluate("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42, loss='huber'), gbr_param_dist, X_train, y_train, X_test, y_test))

    logging.info("All models trained and evaluated. Check MLflow UI for results.")

    select_best_model(results)


def _split_host_port(uri):
    """Extracts (host, port) from an http(s) tracking URI for a reachability check."""
    parsed = urlparse(uri)
    if not parsed.hostname:
        raise ValueError(f"Not a host-based tracking URI: {uri}")
    return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def setup_tracking_uri():
    """Resolves and sets the MLflow tracking URI, returning the active URI.

    Uses a remote tracking server (MLFLOW_TRACKING_URI env var, or the default
    local server) when reachable; otherwise falls back to a local SQLite store.
    Shared by the training pipeline and the Streamlit app so both point to the
    same backend. SQLite (not the './mlruns' file store, which MLflow 3.x
    rejects) is used locally and also enables the Model Registry.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_REMOTE_TRACKING_URI)
    try:
        with socket.create_connection(_split_host_port(tracking_uri), timeout=2):
            pass
        mlflow.set_tracking_uri(tracking_uri)
        logging.info(f"Using MLflow tracking server at {tracking_uri}")
    except (OSError, ValueError):
        mlflow.set_tracking_uri(LOCAL_TRACKING_URI)
        logging.warning(
            f"MLflow server at {tracking_uri} unreachable; "
            f"falling back to local store at {LOCAL_TRACKING_URI}"
        )
    return mlflow.get_tracking_uri()


def select_best_model(results):
    """Compares all experiments and selects the best model with a justification.

    Criterio de seleccion: menor RMSE en el conjunto de prueba. Se prioriza el
    RMSE porque penaliza con mas fuerza los errores grandes, lo cual es relevante
    en la prediccion de 'Profit', donde subestimar grandes perdidas o ganancias
    tiene mayor impacto en el negocio que un error pequeno y uniforme. Se reporta
    tambien MAE (error medio en dolares, mas interpretable) y R2 (varianza
    explicada) como contexto de apoyo.
    """
    valid = [r for r in results if r is not None]
    if not valid:
        logging.error("No models were trained successfully; cannot select a best model.")
        return None

    # Comparison table
    logging.info("=" * 70)
    logging.info("Model comparison (sorted by RMSE, lower is better):")
    logging.info(f"{'Model':<30}{'RMSE':>12}{'MAE':>12}{'R2':>10}")
    for r in sorted(valid, key=lambda r: r["rmse"]):
        logging.info(f"{r['model_name']:<30}{r['rmse']:>12.4f}{r['mae']:>12.4f}{r['r2']:>10.4f}")
    logging.info("=" * 70)

    best = min(valid, key=lambda r: r["rmse"])
    justification = (
        f"Selected '{best['model_name']}' as the best model: it achieves the lowest "
        f"test RMSE ({best['rmse']:.4f}), with MAE={best['mae']:.4f} and "
        f"R2={best['r2']:.4f}. RMSE was chosen as the primary criterion because it "
        f"penalizes large prediction errors more heavily, which matters most when "
        f"predicting Profit."
    )
    logging.info(justification)

    # Register the winning model in the MLflow Model Registry, tagged with the rationale.
    model_uri = f"runs:/{best['run_id']}/model"
    try:
        registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        logging.info(
            f"Best model registered as '{REGISTERED_MODEL_NAME}' "
            f"version {registered.version}."
        )
    except Exception as exc:  # registry may be unavailable on a file-based store
        logging.warning(f"Could not register model in the MLflow Model Registry: {exc}")

    # Exportar a pickle: permite que Streamlit funcione sin servidor MLflow
    try:
        with open(PICKLE_PATH, "wb") as f:
            pickle.dump(best["model"], f)
        meta = {
            "model_name": best["model_name"],
            "rmse": best["rmse"],
            "mae": best["mae"],
            "r2": best["r2"],
            "params": {k: str(v) for k, v in best["best_params"].items()},
        }
        with open(PICKLE_META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        logging.info(f"Best model exported to '{PICKLE_PATH}' and '{PICKLE_META_PATH}'.")
    except Exception as exc:
        logging.warning(f"Could not export model to pickle: {exc}")

    return best

if __name__ == "__main__":
    main()
