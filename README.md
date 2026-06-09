# EDA - Superstore Sales

Análisis Exploratorio de Datos (EDA) y pipeline de MLOps sobre el dataset **Superstore Sales**.
El objetivo es entender los patrones de ventas, identificar los factores que influyen en la
rentabilidad y exponer un modelo predictivo en producción.

- **Variable objetivo:** `Profit` (ganancia).
- **Dataset:** 10.194 transacciones de una supertienda (9.994 de EE. UU. y 200 de Canadá),
  con 21 columnas: datos de pedido, cliente, producto, geografía y métricas financieras
  (`Sales`, `Quantity`, `Discount`, `Profit`).
- **Equipo:** Camilo Velásquez Restrepo · Mateo · Daniela.

El proyecto tiene dos fases:

1. **EDA** — todo el análisis vive en el notebook `notebooks/superstore_eda.ipynb`.
2. **Modelado y despliegue** — un pipeline que entrena y compara 4 modelos de regresión,
   los registra en **MLflow**, selecciona el mejor y lo expone en una app de **Streamlit**.
   Todo se levanta con Docker y se despliega automáticamente en AWS via GitHub Actions.

## Estructura del repositorio

```
├── src/
│   ├── app.py           # App Streamlit: predicción individual/lotes, comparación, diagnóstico
│   └── pipeline.py      # Entrena 4 modelos, registra en MLflow y selecciona el mejor
├── data/
│   └── raw/
│       ├── sample_-_superstore.csv   # Dataset de origen (encoding latin1)
│       └── us-states.json            # GeoJSON de EE.UU. para mapas coropléticos
├── notebooks/
│   └── superstore_eda.ipynb          # Notebook con el EDA completo
├── infra/
│   └── cloudformation.yml            # EC2 + SG + Elastic IP + IAM (IaC)
├── .github/workflows/
│   ├── 1-infra.yml      # Crea/destruye infraestructura AWS (manual)
│   └── 2-deploy.yml     # Build imagen Docker → push GHCR → deploy en EC2 (auto en push)
├── Dockerfile
├── docker-compose.yml   # mlflow + streamlit (el entrenamiento se dispara desde la UI)
└── pyproject.toml / uv.lock
```

## Inicio rápido con Docker (local)

Solo necesitas **Docker** y **Docker Compose** (incluidos en *Docker Desktop*).

```bash
# 1. Clonar el repositorio
git clone <URL-del-repo> && cd Eda-superstore-sales

# 2. Levantar MLflow + Streamlit
docker compose up --build
```

| Servicio | URL | Descripción |
|----------|-----|-------------|
| `mlflow` | http://localhost:5000 | Servidor de tracking + UI |
| `streamlit` | http://localhost:8501 | App de predicción |

> La primera vez no habrá modelo entrenado. La app muestra un botón **"Entrenar modelo ahora"**
> — haz clic y el pipeline entrena los 4 modelos (~5-10 min) y registra el mejor en MLflow.
> Los datos quedan en un volumen Docker (`mlflow-data`) y persisten entre reinicios.

Comandos útiles:

```bash
docker compose up                 # levantar de nuevo (sin reconstruir)
docker compose up -d              # en segundo plano
docker compose logs -f            # ver logs en tiempo real
docker compose down               # detener contenedores
docker compose down -v            # detener + borrar volumen (fuerza reentrenar)
```

## Uso local con uv (sin Docker)

```bash
# 1. Instalar dependencias
uv sync

# 2. Entrenar modelos (usa SQLite local si no hay servidor MLflow)
uv run python src/pipeline.py

# 3. Lanzar la app
uv run streamlit run src/app.py
```

Para ver la UI de MLflow en otra terminal:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

## ¿Qué hace el pipeline de modelado?

- Entrena y compara **4 modelos de regresión** para predecir `Profit`:
  Regresión Lineal (baseline), Huber Regressor, Random Forest y Gradient Boosting.
- Optimiza hiperparámetros con **`RandomizedSearchCV`** (`n_iter=8`, `cv=2`).
- Registra cada experimento en **MLflow**: hiperparámetros, métricas (RMSE, MAE, R²) y el
  modelo como artefacto.
- **Selecciona el mejor modelo** por menor **RMSE** y lo publica en el *Model Registry*.

## Supuestos estadísticos y decisiones de diseño

Se verificaron los supuestos de cada modelo sobre los datos reales antes de fijar la arquitectura
del pipeline.

### Hallazgos sobre el target `Profit`

| Estadístico | Valor |
|---|---|
| Skewness | 7.6 (muy sesgado a la derecha) |
| Kurtosis | 401.7 (colas extremadamente pesadas) |
| Outliers (IQR) | 18.8 % — rango [−6 600, +8 400] |
| Shapiro-Wilk p | ≈ 0 → distribución no normal |

### Supuestos por modelo

| Supuesto | Reg. Lineal (baseline) | Huber Regressor | Random Forest | Gradient Boosting |
|---|---|---|---|---|
| Normalidad de residuos | ❌ Skewness=15.5 | ⚠️ Mejorado con pérdida Huber | no aplica | no aplica |
| Linealidad | ⚠️ Solo `Sales` (r=0.48) y `Discount` (r=−0.22) | ⚠️ Igual | no aplica | no aplica |
| Homocedasticidad | ❌ Heteroced. (p=0.003) | ⚠️ Parcial | no aplica | no aplica |
| Multicolinealidad | ❌ VIF > 4M en fechas → corregido | ✅ Corregido | ⚠️ Sesga importancias | ⚠️ Sesga importancias |
| Outliers | ❌ Severo (18.8% IQR) | ✅ Pérdida Huber | ✅ Aislados por splits | ✅ Pérdida Huber en GBR |
| Independencia (Durbin-Watson) | ✅ DW = 1.99 | ✅ | ✅ | ✅ |

### Ajustes implementados

**1. Eliminación de variables de fecha redundantes**
`Ship_Year`, `Ship_Month`, `Ship_Day` y `Order_Day` presentaban VIF > 4 000 000: eran
derivables de las variables de fecha del pedido más `Order_Processing_Time`. Se eliminaron
del preprocesamiento para eliminar la multicolinealidad perfecta.

**2. `RobustScaler` en lugar de `StandardScaler`**
Escala usando mediana e IQR, ignorando los valores extremos del target. Beneficia al
Huber Regressor y evita que los outliers distorsionen la normalización.

**3. `HuberRegressor` como modelo lineal robusto adicional**
Se agrega junto a la Regresión Lineal ordinaria (que se conserva como baseline).
Minimiza pérdida L2 para errores pequeños y L1 para los grandes, recortando el impacto
de los outliers que producían residuos con skewness = 15.5 en la regresión ordinaria.

**4. `GradientBoostingRegressor(loss='huber')`**
Evita que los árboles sucesivos se concentren en los outliers extremos que el árbol
anterior no pudo ajustar bien.

### Resultados antes y después de los ajustes

| Modelo | RMSE antes | RMSE después | R² antes | R² después |
|---|---|---|---|---|
| Gradient Boosting | 135.95 | **114.08** | 0.794 | **0.855** |
| Random Forest | 147.87 | **145.87** | 0.756 | **0.763** |
| Huber Regressor | — | **195.73** | — | **0.573** |
| Regresión Lineal (baseline) | 215.15 | ~215 | 0.484 | ~0.484 |

La Regresión Lineal se conserva como baseline para mostrar el impacto de las violaciones
de supuestos. El Gradient Boosting con pérdida Huber obtiene la mayor mejora (−16 % RMSE,
+6 pp R²) y sigue siendo el modelo seleccionado.

## App de Streamlit

La app permite:

- **Predicción individual** — formulario con campos en español, cascada País → Estado → Región,
  date pickers para fechas (días hasta envío calculado automáticamente).
- **Predicción por lotes** — subir un CSV y descargar los resultados.
- **Comparación de modelos** — tabla y gráficas con el mejor run por modelo.
- **Diagnóstico** — predicho vs. real y análisis de residuos.
- **Importancia de variables** — ranking de features del modelo ganador.

## Despliegue en AWS

Ver [DEPLOY.md](DEPLOY.md) para el proceso completo de despliegue en EC2 con GitHub Actions.

## EDA — Notebook

### Requisitos

- **Python 3.12**
- [**uv**](https://docs.astral.sh/uv/) para gestionar el entorno y las dependencias.

### Instalación y ejecución

```bash
uv sync

# Registrar el entorno como kernel de Jupyter
uv run python -m ipykernel install --user \
  --name eda-superstore --display-name "Python (Eda-superstore-sales)"

# Abrir el notebook
uv run jupyter lab
```

Al abrir `notebooks/superstore_eda.ipynb`, selecciona el kernel **"Python (Eda-superstore-sales)"**.

### Contenido del notebook

1. **Documentación del dataset** — contexto, objetivo y descripción de variables.
2. **Inspección inicial** — `head`, `info`, `describe` y dimensiones.
3. **Valores faltantes** — conteo y mapa de calor.
4. **Distribución del target `Profit`** — histograma, boxplot, sesgo y curtosis.
5. **Análisis univariado** — distribuciones numéricas y categóricas.
6. **Análisis bivariado** — correlaciones y relación `Discount` ↔ `Profit`.
7. **Análisis geográfico** — mapas coropléticos y de burbujas por ciudad (Folium).
8. **Análisis temporal** — tendencia mensual, estacionalidad y días de envío.
9. **Umbral de descuento por sub-categoría** — heatmap de Profit por nivel de descuento.
10. **Análisis de clientes** — top clientes, frecuencia de compra y rentabilidad por segmento.
11. **Productos más y menos rentables** — margen neto por sub-categoría.
12. **Correlación avanzada** — pairplot por categoría.
13. **Verificación de supuestos** — Shapiro-Wilk, linealidad (Pearson r), homocedasticidad (Spearman), outliers; tabla resumen por modelo con decisiones del pipeline.
14. **Conclusiones** — hallazgos clave, variables seleccionadas y recomendaciones de negocio.

## Dependencias principales

- **EDA:** `pandas` · `numpy` · `matplotlib` · `seaborn` · `folium` · `pgeocode`
- **Modelado:** `scikit-learn` · `mlflow`
- **App:** `streamlit`

Todas fijadas en `pyproject.toml` / `uv.lock`.
