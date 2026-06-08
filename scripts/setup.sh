#!/bin/bash
# Instala Docker y swap en Amazon Linux 2023.
# Idempotente: se puede ejecutar múltiples veces sin efectos adversos.
set -euo pipefail

echo "=== [1/3] Swap ==="
if swapon --show | grep -q '/swapfile'; then
  echo "Swap ya existe, se omite."
else
  dd if=/dev/zero of=/swapfile bs=128M count=8 status=progress
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile swap swap defaults 0 0' >> /etc/fstab
  echo "Swap creado: $(free -h | grep Swap)"
fi

echo "=== [2/3] Docker ==="
if command -v docker &>/dev/null; then
  echo "Docker ya instalado: $(docker --version)"
  systemctl start docker || true
else
  echo "Instalando Docker y git..."
  dnf install -y docker git
  systemctl start docker
  systemctl enable docker
  echo "Docker instalado: $(docker --version)"
fi

PLUGINS_DIR="/usr/local/lib/docker/cli-plugins"
mkdir -p "$PLUGINS_DIR"

echo "=== [3/4] Docker Buildx ==="
# Compose reciente requiere buildx >= 0.17; AL2023 trae una versión antigua
BUILDX_VER=$(curl -fsSL https://api.github.com/repos/docker/buildx/releases/latest \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")
echo "Instalando Docker Buildx $BUILDX_VER..."
curl -fsSL "https://github.com/docker/buildx/releases/download/${BUILDX_VER}/buildx-${BUILDX_VER}.linux-amd64" \
  -o "$PLUGINS_DIR/docker-buildx"
chmod +x "$PLUGINS_DIR/docker-buildx"
echo "Buildx instalado: $(docker buildx version)"

echo "=== [4/4] Docker Compose v2 ==="
COMPOSE_PLUGIN="$PLUGINS_DIR/docker-compose"
echo "Instalando Docker Compose v2 (latest)..."
curl -fsSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o "$COMPOSE_PLUGIN"
chmod +x "$COMPOSE_PLUGIN"
echo "Compose instalado: $(docker compose version)"

echo "=== Verificación final ==="
docker info --format 'Docker: {{.ServerVersion}}  Containers: {{.Containers}}'
docker buildx version
docker compose version
echo "Setup completado."
