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
  echo "Instalando Docker..."
  dnf install -y docker git
  systemctl start docker
  systemctl enable docker
  echo "Docker instalado: $(docker --version)"
  echo "Compose: $(docker compose version)"
fi

echo "=== [3/3] Verificación ==="
docker info --format 'Docker: {{.ServerVersion}}  Containers: {{.Containers}}'
docker compose version
echo "Setup completado."
