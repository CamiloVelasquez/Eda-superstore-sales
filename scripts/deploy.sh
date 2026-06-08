#!/bin/bash
# Clona o actualiza el repositorio y levanta los servicios con Docker Compose.
# Espera REPO_URL y opcionalmente GH_TOKEN (repos privados) como variables de entorno.
set -euo pipefail

APP_DIR="/opt/superstore-sales"

echo "=== [1/3] Código fuente ==="
if [ -d "$APP_DIR/.git" ]; then
  echo "Actualizando repositorio existente..."
  cd "$APP_DIR"
  # Para repos privados: git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/..."
  git fetch origin
  git reset --hard origin/main
  echo "Rama: $(git rev-parse --short HEAD)"
else
  echo "Clonando $REPO_URL..."
  git clone "$REPO_URL" "$APP_DIR"
  cd "$APP_DIR"
fi

cd "$APP_DIR"

echo "=== [2/3] Docker Compose ==="
# 'down' sin '-v' para preservar el volumen mlflow-data (modelos entrenados)
docker compose down --remove-orphans 2>/dev/null || true
docker compose up --build -d
echo "Servicios iniciados."

echo "=== [3/3] Estado ==="
sleep 8
docker compose ps
echo ""
echo "Logs recientes (streamlit):"
docker compose logs --tail=10 streamlit 2>/dev/null || true
