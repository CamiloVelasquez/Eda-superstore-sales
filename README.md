# EDA - Superstore Sales

Análisis Exploratorio de Datos (EDA) sobre el dataset **Superstore Sales**, desarrollado como
trabajo final de MLOps. El objetivo es entender los patrones de ventas, identificar los factores
que influyen en la rentabilidad y preparar el terreno para un futuro modelado predictivo.

- **Variable objetivo:** `Profit` (ganancia).
- **Dataset:** 10.194 transacciones de una supertienda (9.994 de EE. UU. y 200 de Canadá),
  con 21 columnas: datos de pedido, cliente, producto, geografía y métricas financieras
  (`Sales`, `Quantity`, `Discount`, `Profit`).
- **Equipo:** Camilo Velásquez Restrepo · Mateo · Daniela.

El proyecto tiene dos fases:

1. **EDA** — todo el análisis vive en el notebook **`superstore_eda.ipynb`**.
2. **Modelado y despliegue** — un pipeline que entrena y compara 3 modelos de regresión,
   los registra en **MLflow**, selecciona el mejor y lo expone en una app de **Streamlit**.
   Todo se levanta con un solo comando vía **Docker**.

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `superstore_eda.ipynb` | Notebook principal con todo el EDA y los mapas. |
| `model_pipeline.py` | Entrena y compara 3 modelos, registra todo en MLflow y selecciona el mejor. |
| `streamlit_app.py` | App para **usar** el mejor modelo (predicción individual y por lotes) y visualizarlo. |
| `Dockerfile` / `docker-compose.yml` | Empaquetado y orquestación (MLflow + entrenamiento + app). |
| `sample_-_superstore.csv` | Dataset de origen (se lee con `encoding='latin1'`). |
| `us-states.json` | GeoJSON de los estados de EE. UU. para el mapa coroplético. Se descarga solo si falta. |
| `pyproject.toml` / `uv.lock` | Dependencias del proyecto, gestionadas con `uv`. |
| `main.py` | Placeholder generado por `uv`; no forma parte del análisis. |

## 🚀 Inicio rápido con Docker (recomendado)

La forma más fácil de usar el proyecto. Solo necesitas **Docker** y **Docker Compose**
(incluidos en *Docker Desktop*). No hace falta instalar Python ni nada más.

```bash
# 1. Clonar el repositorio y entrar a la carpeta
git clone <URL-del-repo> && cd Eda-superstore-sales

# 2. Construir y levantar todo (MLflow + entrenamiento + app)
docker compose up --build
```

Eso es todo. Al ejecutarlo, Docker levanta tres servicios en orden:

| Servicio | Qué hace | URL |
|---|---|---|
| `mlflow` | Servidor de tracking + interfaz web de MLflow. | http://localhost:5000 |
| `trainer` | Entrena y compara los 3 modelos y **registra el mejor**. Termina solo. | — |
| `streamlit` | App para usar el modelo ganador. Arranca cuando el entrenamiento termina. | **http://localhost:8501** |

> ⏳ **La primera vez tarda unos minutos** porque entrena los modelos. Cuando veas en los logs
> que `streamlit` está listo, abre **http://localhost:8501**. Las métricas y el modelo
> quedan guardados en un volumen de Docker (`mlflow-data`), así que la próxima vez **no
> vuelve a entrenar** (gracias a `SKIP_IF_REGISTERED=1`) y la app abre de inmediato.

Comandos útiles:

```bash
docker compose up                 # levantar de nuevo (sin reconstruir)
docker compose up -d              # en segundo plano (logs con: docker compose logs -f)
docker compose down               # detener y eliminar los contenedores
docker compose down -v            # además borra el volumen -> fuerza reentrenar la próxima vez
docker compose up --build trainer # forzar un reentrenamiento puntual
```

## 💻 Uso local con uv (sin Docker)

Si prefieres correrlo en tu máquina con [`uv`](https://docs.astral.sh/uv/):

```bash
# 1. Instalar dependencias
uv sync

# 2. Entrenar, comparar y registrar el mejor modelo en MLflow.
#    Sin servidor MLflow corriendo, usa un almacenamiento local (sqlite:///mlflow.db).
uv run python model_pipeline.py

# 3. Lanzar la app de Streamlit
uv run streamlit run streamlit_app.py
```

Abre **http://localhost:8501**. Si quieres además la interfaz de MLflow, en otra terminal:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

> La app detecta sola si todavía no hay un modelo registrado y te indica que ejecutes
> primero `model_pipeline.py`. Tras entrenar, usa el botón **🔄 Recargar** de la barra lateral.

## ¿Qué hace el pipeline de modelado?

- Entrena y compara **3 modelos de regresión** para predecir `Profit`:
  Regresión Lineal, Random Forest y Gradient Boosting.
- Optimiza hiperparámetros con **`RandomizedSearchCV`** (validación cruzada).
- Registra cada experimento en **MLflow**: hiperparámetros, métricas (RMSE, MAE, R²) y el
  modelo como artefacto.
- **Selecciona el mejor modelo** por menor **RMSE** (penaliza más los errores grandes, lo
  relevante al predecir utilidades) y lo publica en el *Model Registry*.
- La app de Streamlit permite **usar el modelo**: predicción de una transacción, predicción
  por lotes (subir CSV y descargar resultados), comparación de modelos, diagnóstico
  (predicho vs. real, residuos) e importancia de variables.

## EDA — Notebook

### Requisitos

- **Python 3.12**
- [**uv**](https://docs.astral.sh/uv/) para gestionar el entorno y las dependencias.
- **Conexión a internet** en la primera ejecución de los mapas: `folium` descarga el GeoJSON de
  estados y `pgeocode` descarga su base de códigos postales (luego quedan cacheados).

### Instalación y ejecución

```bash
# 1. Instalar las dependencias en un entorno local (.venv)
uv sync

# 2. (Una sola vez) Registrar el entorno como kernel de Jupyter,
#    para poder seleccionarlo desde VS Code o Jupyter en el navegador
uv run python -m ipykernel install --user \
  --name eda-superstore --display-name "Python (Eda-superstore-sales)"

# 3. Abrir el notebook
uv run jupyter lab        # o: uv run jupyter notebook
```

Al abrir `superstore_eda.ipynb`, selecciona el kernel **"Python (Eda-superstore-sales)"**
(en VS Code: *Select Kernel* arriba a la derecha; en Jupyter: *Kernel → Change Kernel*).

> Si usas un Jupyter distinto al del proyecto y aparece el error
> `... pip install ipykernel ...`, significa que el kernel apunta al Python del sistema (sin las
> dependencias). La solución es seleccionar el kernel **"Python (Eda-superstore-sales)"**, no
> instalar nada en el Python del sistema.

### Ejecutar el notebook completo desde la terminal

```bash
uv run jupyter nbconvert --to notebook --execute --inplace superstore_eda.ipynb
```

## Cómo funciona el análisis

El notebook está pensado para ejecutarse **de arriba hacia abajo** (las celdas son secuenciales
y comparten estado). Su flujo es:

1. **Documentación del dataset** — contexto, objetivo y descripción de variables.
2. **Inspección inicial** — `head`, `info`, `describe` y dimensiones.
3. **Valores faltantes** — conteo por columna y mapa de calor; conversión de `Order Date` y
   `Ship Date` a tipo fecha (`%m/%d/%Y`).
4. **Distribución del target `Profit`** — histograma, boxplot, sesgo, curtosis y porcentaje de
   transacciones con pérdida.
5. **Análisis univariado** — distribuciones de variables numéricas y categóricas (Top 10).
6. **Análisis bivariado** — correlaciones, scatterplots y boxplots de cada feature frente a
   `Profit` (con foco en la relación `Discount` ↔ `Profit`).
7. **Análisis geográfico interactivo (folium)** — mapas que complementan las gráficas anteriores:
   - **Coropléticos por estado** de `Profit` y `Sales` totales.
   - **Mapa de burbujas por ciudad**: cada ciudad se ubica por su código postal (`pgeocode`);
     el tamaño de la burbuja es proporcional a las ventas y el color indica si es rentable
     (verde) o genera pérdidas (rojo).
8. **Hallazgos y conclusiones** — variables más relacionadas con `Profit`, candidatas a
   transformación, variables redundantes y propuesta inicial de features para modelado.

### Notas sobre los mapas

- El GeoJSON cubre únicamente los estados de EE. UU., por lo que las provincias de Canadá y el
  Distrito de Columbia aparecen en gris en los coropléticos. El mapa de ciudades sí cubre ambos
  países.
- Cerca del 4 % de las filas tienen un código postal que `pgeocode` no resuelve; se excluyen del
  mapa de ciudades y la cantidad se reporta en la propia celda.

## Dependencias principales

- **EDA:** `pandas` · `numpy` · `matplotlib` · `seaborn` · `jupyter` · `folium` · `pgeocode`.
- **Modelado y app:** `scikit-learn` · `mlflow` · `streamlit`.

Todas están fijadas en `pyproject.toml` / `uv.lock`.
