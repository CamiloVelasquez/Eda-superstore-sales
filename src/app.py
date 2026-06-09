"""
App de Streamlit para USAR y visualizar el mejor modelo de prediccion de 'Profit'.

Carga el modelo ganador registrado en MLflow (SuperstoreProfitBestModel) y
permite:
  - 🧮 Predecir el Profit de una transaccion (formulario interactivo).
  - 📁 Predecir por lotes subiendo un CSV (con descarga de resultados).
  - 📊 Comparar los experimentos y ver por que se eligio este modelo.

Requisito previo: haber entrenado los modelos al menos una vez con
    uv run python src/pipeline.py
para que exista el modelo registrado.

Ejecucion:
    uv run streamlit run src/app.py
"""
import io
import json
import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from pipeline import (
    EXPERIMENT_NAME,
    PICKLE_META_PATH,
    PICKLE_PATH,
    REGISTERED_MODEL_NAME,
    load_data,
    preprocess_data,
    setup_tracking_uri,
)

DATA_PATH = "data/raw/sample_-_superstore.csv"
TARGET = "Profit"

# Columnas derivadas de fechas que el pipeline genera internamente.
# Order_Day y Ship_Year/Month/Day se eliminaron (VIF > 4M, r≈0 con Profit).
# El formulario las reemplaza con dos date_input amigables.
DATE_DERIVED = frozenset({
    "Order_Year", "Order_Month",
    "Order_Processing_Time",
})

LABELS_ES = {
    # Numéricas directas
    "Sales":    "Ventas ($)",
    "Quantity": "Cantidad (unidades)",
    "Discount": "Descuento (0.0 – 1.0)",
    # Derivadas de fechas (las que permanecen en el modelo)
    "Order_Year":            "Fecha de pedido → año",
    "Order_Month":           "Fecha de pedido → mes",
    "Order_Processing_Time": "Días hasta el envío",
    # Categóricas
    "Ship Mode":       "Modo de envío",
    "Segment":         "Segmento de cliente",
    "Country/Region":  "País / Región",
    "State/Province":  "Estado / Provincia",
    "Region":          "Región",
    "Category":        "Categoría",
    "Sub-Category":    "Subcategoría",
}

st.set_page_config(
    page_title="Superstore · Usar el mejor modelo",
    page_icon="📈",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Carga de datos y modelo (cacheada)
# --------------------------------------------------------------------------- #
@st.cache_resource
def init_tracking():
    """Apunta MLflow al mismo backend que usa el pipeline de entrenamiento."""
    return setup_tracking_uri()


@st.cache_data
def get_feature_frame():
    """Devuelve (X_train, X_test, y_test) con la misma semilla que el pipeline."""
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_test


@st.cache_resource
def load_best_model_and_meta():
    """Carga el modelo: primero desde MLflow, luego desde pickle como fallback."""
    # 1. Intentar desde MLflow
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{latest.version}")
            run = client.get_run(latest.run_id)
            return model, latest, run
    except Exception:
        pass

    # 2. Fallback a pickle
    if os.path.exists(PICKLE_PATH):
        with open(PICKLE_PATH, "rb") as f:
            model = pickle.load(f)
        meta = {}
        if os.path.exists(PICKLE_META_PATH):
            with open(PICKLE_META_PATH) as f:
                meta = json.load(f)
        version_info = SimpleNamespace(version="local")
        run = SimpleNamespace(
            data=SimpleNamespace(
                params={"model_name": meta.get("model_name", "Desconocido"), **meta.get("params", {})},
                metrics={
                    "rmse": meta.get("rmse", float("nan")),
                    "mae": meta.get("mae", float("nan")),
                    "r2_score": meta.get("r2", float("nan")),
                },
            )
        )
        return model, version_info, run

    return None, None, None


@st.cache_data
def get_runs():
    """Tabla con todos los experimentos del estudio."""
    try:
        return mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    except Exception:
        return pd.DataFrame()


def prepare_features(raw_df, feature_columns):
    """Aplica el mismo preprocesamiento del pipeline a un CSV en formato Superstore.

    Devuelve (X alineado a feature_columns, y_real o None).
    """
    proc = preprocess_data(raw_df.copy())
    y_real = proc[TARGET] if TARGET in proc.columns else None
    missing = [c for c in feature_columns if c not in proc.columns]
    if missing:
        raise ValueError(
            "El CSV no tiene las columnas necesarias tras el preprocesamiento. "
            f"Faltan: {missing}"
        )
    return proc[feature_columns], y_real


# --------------------------------------------------------------------------- #
# Inicializacion
# --------------------------------------------------------------------------- #
tracking_uri = init_tracking()

st.title("📈 Predicción de *Profit* · Mejor modelo")

def _run_training():
    """Lanza model_pipeline.py en un subprocess y retorna (ok, stderr)."""
    env = os.environ.copy()
    env.pop("SKIP_IF_REGISTERED", None)  # forzar re-entrenamiento aunque ya exista modelo
    result = subprocess.run(
        [sys.executable, "src/pipeline.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode == 0, result.stderr


with st.sidebar:
    st.header("⚙️ MLflow")
    if tracking_uri.startswith("http"):
        # En Docker (local o AWS) el servidor MLflow corre en el mismo host público.
        # MLFLOW_TRACKING_URI apunta al hostname interno (mlflow:5000); para mostrar
        # la URL pública reemplazamos el host interno por el del navegador.
        public_host = os.environ.get("PUBLIC_HOST", "localhost")
        st.success("Servidor MLflow activo")
        st.write(f"[Abrir MLflow UI](http://{public_host}:5000)")
    else:
        st.warning("Modo local — sin servidor MLflow")
        st.caption(f"`{tracking_uri}`")
    st.write(f"**Experimento:** {EXPERIMENT_NAME}")
    st.write(f"**Modelo registrado:** {REGISTERED_MODEL_NAME}")
    if st.button("🔄 Recargar datos y modelo"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.divider()
    if st.button("🚀 Re-entrenar modelos", help="Entrena los 3 modelos y registra el mejor"):
        with st.spinner("Entrenando modelos... (~3-5 minutos)"):
            ok, stderr = _run_training()
        if ok:
            st.success("Entrenamiento completado.")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error("Error durante el entrenamiento.")
            st.code(stderr)

model, version_info, best_run = load_best_model_and_meta()

if model is None:
    st.warning("No se encontró ningún modelo entrenado.")
    st.info(
        "Entrena los modelos para comenzar a usar la aplicación. "
        "El proceso toma ~3-5 minutos la primera vez."
    )
    if st.button("🚀 Entrenar modelo ahora", type="primary"):
        with st.spinner("Entrenando 4 modelos (Linear Regression, Huber, Random Forest, Gradient Boosting)..."):
            ok, stderr = _run_training()
        if ok:
            st.success("¡Entrenamiento completado! Recargando...")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error("Error durante el entrenamiento.")
            st.code(stderr)
    st.stop()

X_train, X_test, y_test = get_feature_frame()

# Detectar si el modelo guardado fue entrenado con un conjunto de features distinto
# (ocurre cuando se despliega código nuevo sin re-entrenar).
try:
    y_pred = model.predict(X_test)
except ValueError as _feat_err:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.error(
        "**El modelo guardado es incompatible con el conjunto de variables actual.** "
        "Esto ocurre cuando se actualiza el pipeline sin re-entrenar. "
        f"Detalle: `{_feat_err}`"
    )
    st.warning(
        "Haz clic en **Re-entrenar modelos** en la barra lateral para generar un modelo "
        "con las variables actuales y resolver el problema."
    )
    st.stop()

best_name = best_run.data.params.get("model_name", "Desconocido")
best_metrics = best_run.data.metrics

# Encabezado compacto: que modelo se esta usando
st.success(
    f"**Modelo en uso:** {best_name} (v{version_info.version}) — "
    f"RMSE {best_metrics.get('rmse', float('nan')):.2f} · "
    f"MAE {best_metrics.get('mae', float('nan')):.2f} · "
    f"R² {best_metrics.get('r2_score', float('nan')):.3f}"
)

tab_pred, tab_batch, tab_cmp = st.tabs(
    [
        "🧮 Predicción individual",
        "📁 Predicción por lotes",
        "📊 Comparación de modelos",
    ]
)

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

# Mappings geográficos derivados del dataset
_GEO_ORDER = ["Country/Region", "State/Province", "Region"]
_geo_cols   = [c for c in _GEO_ORDER if c in categorical_cols]
_other_cats = [c for c in categorical_cols if c not in _geo_cols]

# País → lista de estados/provincias
country_to_states: dict = (
    X_train.groupby("Country/Region")["State/Province"]
    .apply(lambda s: sorted(s.dropna().unique().tolist()))
    .to_dict()
    if "Country/Region" in categorical_cols and "State/Province" in categorical_cols
    else {}
)
# Estado → región (relación 1:1 en el dataset)
state_to_region: dict = (
    X_train[["State/Province", "Region"]]
    .drop_duplicates()
    .set_index("State/Province")["Region"]
    .to_dict()
    if "State/Province" in categorical_cols and "Region" in categorical_cols
    else {}
)

# --------------------------------------------------------------------------- #
# 1) Prediccion individual  (usar el modelo, transaccion a transaccion)
# --------------------------------------------------------------------------- #
with tab_pred:
    st.markdown("### Estima el *Profit* de una transacción")
    st.caption("Completa los datos de la transacción y el modelo predice la utilidad esperada.")

    other_numeric = [c for c in numeric_cols if c not in DATE_DERIVED]

    # --- Geográficas FUERA del form: cada cambio filtra en tiempo real ---
    st.markdown("**Ubicación**")
    geo_cols_ui = st.columns(len(_geo_cols))
    _geo_inputs: dict = {}
    _selected_country = sorted(X_train["Country/Region"].dropna().unique())[0] if "Country/Region" in _geo_cols else None
    _selected_state   = None
    for i, col in enumerate(_geo_cols):
        all_options = sorted(X_train[col].dropna().unique().tolist())
        with geo_cols_ui[i]:
            if col == "Country/Region":
                _selected_country = st.selectbox(LABELS_ES.get(col, col), all_options, key="geo_country")
                _geo_inputs[col] = _selected_country
            elif col == "State/Province":
                state_options = country_to_states.get(_selected_country, all_options)
                _selected_state = st.selectbox(LABELS_ES.get(col, col), state_options, key="geo_state")
                _geo_inputs[col] = _selected_state
            elif col == "Region":
                _auto_region = state_to_region.get(_selected_state, all_options[0])
                st.selectbox(
                    LABELS_ES.get(col, col) + " (automática)",
                    options=[_auto_region],
                    disabled=True,
                    key="geo_region",
                    help="Se determina automáticamente según el Estado / Provincia seleccionado",
                )
                _geo_inputs[col] = _auto_region

    # --- Resto del formulario ---
    with st.form("prediction_form"):
        inputs = {}

        # Otras categóricas (sin geo)
        st.markdown("**Clasificación del producto y cliente**")
        ccols_form = st.columns(3)
        for i, col in enumerate(_other_cats):
            options = sorted(X_train[col].dropna().unique().tolist())
            with ccols_form[i % 3]:
                inputs[col] = st.selectbox(LABELS_ES.get(col, col), options)

        # Numéricas
        st.markdown("**Datos de la transacción**")
        ncols_form = st.columns(3)
        NUM_CONFIG = {
            "Sales":    dict(min_value=0.0, step=1.0, format="%.2f"),
            "Quantity": dict(min_value=1.0, step=1.0, format="%.0f"),
            "Discount": dict(min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
        }
        for i, col in enumerate(other_numeric):
            with ncols_form[i % 3]:
                cfg = NUM_CONFIG.get(col, {})
                inputs[col] = st.number_input(
                    LABELS_ES.get(col, col),
                    value=float(X_train[col].median()),
                    **cfg,
                )

        # Fechas
        st.markdown("**Fechas de la transacción**")
        dcols = st.columns(3)
        with dcols[0]:
            order_date = st.date_input("Fecha de pedido", value=pd.Timestamp("2024-01-15"))
        with dcols[1]:
            ship_date = st.date_input("Fecha de envío", value=pd.Timestamp("2024-01-18"))
        with dcols[2]:
            _proc_days = (pd.Timestamp(ship_date) - pd.Timestamp(order_date)).days
            st.number_input(
                "Días hasta el envío (calculado)",
                value=float(max(_proc_days, 0)),
                disabled=True,
                help="Calculado automáticamente: Fecha de envío − Fecha de pedido",
            )

        submitted = st.form_submit_button("🚀 Predecir Profit", type="primary")

    if submitted:
        order_dt = pd.Timestamp(order_date)
        ship_dt  = pd.Timestamp(ship_date)

        if ship_dt < order_dt:
            st.error("La fecha de envío no puede ser anterior a la fecha de pedido.")
        else:
            inputs.update(_geo_inputs)
            inputs["Order_Year"]            = order_dt.year
            inputs["Order_Month"]           = order_dt.month
            inputs["Order_Processing_Time"] = (ship_dt - order_dt).days

            row  = pd.DataFrame([inputs])[X_train.columns]
            pred = float(model.predict(row)[0])

            st.metric("Profit estimado", f"$ {pred:,.2f}")
            if pred < 0:
                st.warning("El modelo predice una **pérdida**.")
            else:
                st.success("Transacción **rentable**.")

            with st.expander("Ver todos los valores enviados al modelo"):
                records = [
                    {
                        "Campo (modelo)":     c,
                        "Campo (formulario)": LABELS_ES.get(c, c),
                        "Valor":              str(row.iloc[0][c]),
                    }
                    for c in row.columns
                ]
                st.table(pd.DataFrame(records).set_index("Campo (modelo)"))

# --------------------------------------------------------------------------- #
# 2) Prediccion por lotes (subir CSV -> predecir -> descargar)
# --------------------------------------------------------------------------- #
with tab_batch:
    st.markdown("### Predicción por lotes")
    st.caption(
        "Sube un CSV en el **formato original del Superstore** (mismas columnas que "
        "`sample_-_superstore.csv`). Se aplican el mismo preprocesamiento y modelo, "
        "y podrás descargar el resultado con la columna `Profit_predicho`."
    )

    uploaded = st.file_uploader("Archivo CSV", type=["csv"])
    use_sample = st.checkbox("…o usar el dataset de ejemplo del proyecto", value=not bool(uploaded))

    raw_df = None
    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded, encoding="latin1")
        except UnicodeDecodeError:
            uploaded.seek(0)
            raw_df = pd.read_csv(uploaded, encoding="utf-8")
    elif use_sample:
        raw_df = load_data(DATA_PATH)

    if raw_df is not None:
        try:
            X_new, y_real = prepare_features(raw_df, X_train.columns)
            preds = model.predict(X_new)

            result = raw_df.copy()
            result["Profit_predicho"] = preds
            st.success(f"Predicciones generadas para {len(result):,} filas.")

            if y_real is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_real, preds)):.2f}")
                m2.metric("MAE", f"{mean_absolute_error(y_real, preds):.2f}")
                m3.metric("R²", f"{r2_score(y_real, preds):.3f}")

            st.dataframe(result.head(50), width="stretch")

            buf = io.StringIO()
            result.to_csv(buf, index=False)
            st.download_button(
                "⬇️ Descargar predicciones (CSV)",
                data=buf.getvalue(),
                file_name="predicciones_profit.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"No se pudieron generar predicciones: {exc}")
    else:
        st.info("Sube un CSV o marca la casilla para usar el dataset de ejemplo.")

# --------------------------------------------------------------------------- #
# 3) Comparacion de modelos
# --------------------------------------------------------------------------- #
with tab_cmp:
    st.markdown("### Comparación de todos los experimentos")
    runs = get_runs()
    if runs.empty:
        st.info("No hay runs registrados en el experimento.")
    else:
        metric_cols = [c for c in ["metrics.rmse", "metrics.mae", "metrics.r2_score"] if c in runs.columns]
        table = runs[["params.model_name", "start_time", *metric_cols]].copy()
        table.columns = ["Modelo", "start_time"] + [c.replace("metrics.", "").upper() for c in metric_cols]
        table = (
            table.dropna(subset=["Modelo"])
            .sort_values("start_time", ascending=False)   # más reciente primero
            .groupby("Modelo", sort=False)
            .first()                                       # run más reciente por modelo
            .reset_index()
            .drop(columns=["start_time"])
            .sort_values("RMSE")
            .reset_index(drop=True)
        )
        st.dataframe(table, width="stretch")

        metric_defs = [
            ("RMSE",     "RMSE (menor es mejor)",  False),
            ("MAE",      "MAE (menor es mejor)",   False),
            ("R2_SCORE", "R² (mayor es mejor)",    True),
        ]
        available = [(col, label, higher) for col, label, higher in metric_defs if col in table.columns]
        if available:
            n_models = len(table)
            # Ancho dinámico: más espacio por modelo para evitar solapamiento de etiquetas
            fig_w = max(4, n_models * 1.1)
            chart_cols = st.columns(len(available))
            for (col, label, higher), ch_col in zip(available, chart_cols):
                with ch_col:
                    st.markdown(f"**{label}**")
                    fig_m, ax_m = plt.subplots(figsize=(fig_w, 3.5))
                    models = table["Modelo"].tolist()
                    values = table[col].tolist()
                    x = list(range(len(models)))
                    ax_m.plot(x, values, marker="o", linewidth=2, markersize=7, color="#1f77b4")
                    for xi, yi in zip(x, values):
                        ax_m.annotate(
                            f"{yi:.3f}", (xi, yi),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8,
                        )
                    short_labels = [
                        m.replace("Gradient Boosting Regressor", "Gradient\nBoosting")
                         .replace("Random Forest Regressor", "Random\nForest")
                         .replace("Huber Regressor", "Huber\nRegressor")
                         .replace("Linear Regression", "Linear\nRegression")
                        for m in models
                    ]
                    ax_m.set_xticks(x)
                    ax_m.set_xticklabels(short_labels, fontsize=8)
                    ax_m.set_ylabel(col)
                    ax_m.margins(y=0.25)
                    fig_m.tight_layout()
                    st.pyplot(fig_m)
                    plt.close(fig_m)

        with st.expander("Hiperparámetros del modelo ganador"):
            params = {k: v for k, v in best_run.data.params.items() if k != "model_name"}
            st.table(pd.DataFrame(sorted(params.items()), columns=["Hiperparámetro", "Valor"]))

        st.caption(
            "Se comparan 4 modelos: Regresión Lineal (baseline), Huber Regressor (robusto a outliers), "
            "Random Forest y Gradient Boosting (con pérdida Huber). "
            "Criterio de selección: menor **RMSE**, porque penaliza con más fuerza los errores grandes "
            "— relevante al predecir *Profit*, donde un error grande importa más que muchos errores pequeños. "
            "Los modelos basados en árboles no requieren normalidad ni linealidad, cumpliendo mejor los "
            "supuestos estadísticos del dataset (skewness=7.6, 18.8% outliers por IQR)."
        )

