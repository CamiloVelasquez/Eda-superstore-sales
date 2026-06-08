# Imagen única usada por los tres servicios (mlflow, trainer, streamlit).
FROM python:3.12-slim

# uv para instalar dependencias desde el lockfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /app

# 1) Capa de dependencias (se cachea mientras no cambien pyproject.toml / uv.lock).
#    Se incluye README.md porque pyproject.toml lo referencia (readme = "README.md").
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) Código de la aplicación
COPY . .
RUN uv sync --frozen --no-dev

# Streamlit (8501) y MLflow (5000)
EXPOSE 8501 5000

# Por defecto sirve la app; docker-compose sobreescribe 'command' por servicio.
CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
