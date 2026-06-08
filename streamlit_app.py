"""
App de Streamlit para USAR y visualizar el mejor modelo de prediccion de 'Profit'.

Carga el modelo ganador registrado en MLflow (SuperstoreProfitBestModel) y
permite:
  - 🧮 Predecir el Profit de una transaccion (formulario interactivo).
  - 📁 Predecir por lotes subiendo un CSV (con descarga de resultados).
  - 📊 Comparar los experimentos y ver por que se eligio este modelo.
  - 🔍 Diagnosticar el modelo (predicho vs real, residuos).
  - ⭐ Ver la importancia de las variables.

Requisito previo: haber entrenado los modelos al menos una vez con
    uv run python model_pipeline.py
para que exista el modelo registrado.

Ejecucion:
    uv run streamlit run streamlit_app.py
"""
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model_pipeline import (
    load_data,
    preprocess_data,
    setup_tracking_uri,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)

DATA_PATH = "sample_-_superstore.csv"
TARGET = "Profit"

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
    """Carga el modelo registrado mas reciente y los metadatos de su run."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        return None, None, None
    latest = max(versions, key=lambda v: int(v.version))
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{latest.version}")
    run = client.get_run(latest.run_id)
    return model, latest, run


@st.cache_data
def get_runs():
    """Tabla con todos los experimentos del estudio."""
    return mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])


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

with st.sidebar:
    st.header("⚙️ MLflow")
    st.write("**Tracking URI**")
    st.code(tracking_uri, language=None)
    st.write(f"**Experimento:** {EXPERIMENT_NAME}")
    st.write(f"**Modelo registrado:** {REGISTERED_MODEL_NAME}")
    if st.button("🔄 Recargar datos y modelo"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

model, version_info, best_run = load_best_model_and_meta()

if model is None:
    st.error(
        "No se encontró ningún modelo registrado como "
        f"`{REGISTERED_MODEL_NAME}`.\n\n"
        "Entrena primero los modelos ejecutando:\n\n"
        "```bash\nuv run python model_pipeline.py\n```"
    )
    st.stop()

X_train, X_test, y_test = get_feature_frame()
y_pred = model.predict(X_test)
residuals = y_test.values - y_pred

best_name = best_run.data.params.get("model_name", "Desconocido")
best_metrics = best_run.data.metrics

# Encabezado compacto: que modelo se esta usando
st.success(
    f"**Modelo en uso:** {best_name} (v{version_info.version}) — "
    f"RMSE {best_metrics.get('rmse', float('nan')):.2f} · "
    f"MAE {best_metrics.get('mae', float('nan')):.2f} · "
    f"R² {best_metrics.get('r2_score', float('nan')):.3f}"
)

tab_pred, tab_batch, tab_cmp, tab_diag, tab_imp = st.tabs(
    [
        "🧮 Predicción individual",
        "📁 Predicción por lotes",
        "📊 Comparación de modelos",
        "🔍 Diagnóstico",
        "⭐ Importancia de variables",
    ]
)

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

# --------------------------------------------------------------------------- #
# 1) Prediccion individual  (usar el modelo, transaccion a transaccion)
# --------------------------------------------------------------------------- #
with tab_pred:
    st.markdown("### Estima el *Profit* de una transacción")
    st.caption("Ajusta los valores y el modelo ganador predice la utilidad esperada.")

    with st.form("prediction_form"):
        inputs = {}
        st.markdown("**Variables numéricas**")
        ncols = st.columns(3)
        for i, col in enumerate(numeric_cols):
            with ncols[i % 3]:
                inputs[col] = st.number_input(col, value=float(X_train[col].median()))

        st.markdown("**Variables categóricas**")
        ccols = st.columns(3)
        for i, col in enumerate(categorical_cols):
            options = sorted(X_train[col].dropna().unique().tolist())
            with ccols[i % 3]:
                inputs[col] = st.selectbox(col, options)

        submitted = st.form_submit_button("🚀 Predecir Profit", type="primary")

    if submitted:
        row = pd.DataFrame([inputs])[X_train.columns]  # respeta el orden original
        pred = float(model.predict(row)[0])
        st.metric("Profit estimado", f"$ {pred:,.2f}")
        if pred < 0:
            st.warning("El modelo predice una **pérdida** para esta transacción.")
        else:
            st.success("El modelo predice una transacción **rentable**.")
        with st.expander("Ver la fila enviada al modelo"):
            st.dataframe(row, use_container_width=True)

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

            st.dataframe(result.head(50), use_container_width=True)

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
        table = runs[["params.model_name", *metric_cols]].copy()
        table.columns = ["Modelo"] + [c.replace("metrics.", "").upper() for c in metric_cols]
        table = table.dropna(subset=["Modelo"]).sort_values("RMSE").reset_index(drop=True)
        st.dataframe(table, use_container_width=True)

        if "RMSE" in table.columns:
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**RMSE / MAE por modelo** (menor es mejor)")
                err = table.set_index("Modelo")[[c for c in ["RMSE", "MAE"] if c in table.columns]]
                st.bar_chart(err)
            with colB:
                if "R2_SCORE" in table.columns:
                    st.markdown("**R² por modelo** (mayor es mejor)")
                    st.bar_chart(table.set_index("Modelo")[["R2_SCORE"]])

        with st.expander("Hiperparámetros del modelo ganador"):
            params = {k: v for k, v in best_run.data.params.items() if k != "model_name"}
            st.table(pd.DataFrame(sorted(params.items()), columns=["Hiperparámetro", "Valor"]))

        st.caption(
            "Criterio de selección: menor RMSE en test, porque penaliza con más "
            "fuerza los errores grandes — relevante al predecir *Profit*, donde un "
            "error grande en una transacción importa más que muchos errores pequeños."
        )

# --------------------------------------------------------------------------- #
# 4) Diagnostico: predicho vs real y residuos
# --------------------------------------------------------------------------- #
with tab_diag:
    st.markdown("### Diagnóstico sobre el conjunto de prueba")
    colL, colR = st.columns(2)

    with colL:
        st.markdown("**Predicho vs. Real**")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_test, y_pred, s=8, alpha=0.3, edgecolors="none")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=1, label="Predicción perfecta")
        ax.set_xlabel("Profit real")
        ax.set_ylabel("Profit predicho")
        ax.legend()
        st.pyplot(fig)

    with colR:
        st.markdown("**Residuos vs. Predicho**")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.scatter(y_pred, residuals, s=8, alpha=0.3, edgecolors="none")
        ax2.axhline(0, color="r", ls="--", lw=1)
        ax2.set_xlabel("Profit predicho")
        ax2.set_ylabel("Residuo (real − predicho)")
        st.pyplot(fig2)

    st.markdown("**Distribución de residuos**")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.hist(residuals, bins=60)
    ax3.set_xlabel("Residuo")
    ax3.set_ylabel("Frecuencia")
    st.pyplot(fig3)

# --------------------------------------------------------------------------- #
# 5) Importancia de variables
# --------------------------------------------------------------------------- #
with tab_imp:
    st.markdown("### Importancia de las variables")
    try:
        pre = model.named_steps["preprocessor"]
        reg = model.named_steps["regressor"]
        feat_names = pre.get_feature_names_out()

        if hasattr(reg, "feature_importances_"):
            importances = reg.feature_importances_
            label = "Importancia (impurity-based)"
        elif hasattr(reg, "coef_"):
            importances = np.abs(np.ravel(reg.coef_))
            label = "|Coeficiente| (estandarizado)"
        else:
            importances = None

        if importances is None:
            st.info("El modelo ganador no expone importancias de variables.")
        else:
            imp = (
                pd.DataFrame({"Variable": feat_names, label: importances})
                .sort_values(label, ascending=False)
                .head(20)
                .set_index("Variable")
            )
            st.bar_chart(imp)
            st.caption(f"Top 20 variables más influyentes — {best_name}.")
    except Exception as exc:  # pragma: no cover - defensivo
        st.warning(f"No se pudo calcular la importancia de variables: {exc}")
