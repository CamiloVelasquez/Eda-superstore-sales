# Despliegue en AWS EC2 con GitHub Actions

Este documento explica cómo crear la infraestructura en AWS y desplegar la aplicación automáticamente usando GitHub Actions.

---

## Arquitectura

```
GitHub Actions
      │
      ├─ 1-infra.yml ──► CloudFormation ──► EC2 t3.micro (free tier)
      │                                      Security Group
      │                                      Elastic IP
      │                                      IAM Role (SSM)
      │
      └─ 2-deploy.yml ─► SSM Run Command ──► Docker Compose
                          (sin SSH keys)       ├── mlflow  :5000
                                               ├── trainer (run once)
                                               └── streamlit :8501
```

Los workflows se comunican con la EC2 a través de **AWS Systems Manager (SSM)**, sin necesidad de abrir el puerto SSH ni manejar archivos `.pem`.

---

## Prerrequisitos

- Cuenta de AWS activa (free tier disponible los primeros 12 meses)
- Repositorio en GitHub con el código del proyecto
- Permisos para agregar Secrets en el repositorio

---

## Paso 1 — Crear usuario IAM en AWS

Este usuario será el que GitHub Actions usará para operar en tu cuenta.

1. Inicia sesión en [AWS Console](https://console.aws.amazon.com)
2. Ve a **IAM → Users → Create user**
3. Nombre: `github-actions-deploy` (o el que prefieras)
4. En **Permissions**, selecciona **Attach policies directly** y agrega:

| Política | Para qué se usa |
|----------|----------------|
| `AmazonEC2FullAccess` | Crear y gestionar instancias |
| `AWSCloudFormationFullAccess` | Crear el stack de infraestructura |
| `IAMFullAccess` | Crear el rol IAM para SSM |
| `AmazonSSMFullAccess` | Ejecutar comandos en la EC2 sin SSH |

5. Finaliza la creación del usuario
6. Abre el usuario recién creado → **Security credentials → Create access key**
7. Selecciona **Application running outside AWS** → guarda el `Access key ID` y el `Secret access key`

> Estos valores solo se muestran una vez. Guárdalos antes de cerrar la ventana.

---

## Paso 2 — Agregar Secrets en GitHub

1. En tu repositorio, ve a **Settings → Secrets and variables → Actions**
2. Haz clic en **New repository secret** y agrega los tres siguientes:

| Secret | Valor de ejemplo | Descripción |
|--------|-----------------|-------------|
| `AWS_ACCESS_KEY_ID` | `AKIAIOSFODNN7EXAMPLE` | Access Key del usuario IAM |
| `AWS_SECRET_ACCESS_KEY` | `wJalrXUtnFEMI/K7MDENG/...` | Secret Key del usuario IAM |
| `AWS_REGION` | `us-east-1` | Región donde se creará la EC2 |

**Regiones recomendadas (free tier disponible en todas):**
- `us-east-1` — Norte de Virginia (más recursos gratuitos)
- `us-west-2` — Oregón
- `eu-west-1` — Irlanda

---

## Paso 3 — Crear la infraestructura

Solo se ejecuta **una vez** (o cuando necesites recrearla).

1. En GitHub, ve a **Actions → 1 · Infraestructura (CloudFormation)**
2. Haz clic en **Run workflow → Run workflow** (acción: `deploy`)
3. Espera ~2 minutos hasta que termine

Al completarse, el resumen del workflow muestra:

```
✅ Infraestructura lista

Recurso     | Valor
------------|-------------------------
Instancia   | i-0abc123def456789
Streamlit   | http://54.12.34.56:8501
MLflow      | http://54.12.34.56:5000
```

> La IP mostrada es fija (Elastic IP) y no cambia aunque la instancia se reinicie.

---

## Paso 4 — Primer despliegue

### Opción A: Automático
Haz cualquier push a la rama `main` y el workflow **2 · Deploy** se ejecuta solo.

```bash
git add .
git commit -m "feat: primer deploy"
git push origin main
```

### Opción B: Manual
Ve a **Actions → 2 · Deploy a EC2 → Run workflow**.

### Qué hace el workflow de deploy

1. Conecta con la EC2 via SSM (sin SSH)
2. Instala Docker y Docker Compose si no están presentes
3. Configura swap (1 GB extra) para no quedarse sin memoria
4. Clona o actualiza el repositorio en `/opt/superstore-sales`
5. Ejecuta `docker compose up --build -d`
6. La primera vez, el servicio `trainer` entrena los 3 modelos (~3-5 minutos)
7. Streamlit queda disponible cuando el trainer termina

**Duración estimada:**
- Primer deploy: ~8-12 minutos (incluye instalación de Docker y entrenamiento)
- Deploys siguientes: ~3-5 minutos (solo actualiza código, no re-entrena)

---

## Acceder a la aplicación

Una vez completado el deploy:

| Servicio | URL | Descripción |
|----------|-----|-------------|
| Streamlit | `http://<IP>:8501` | App de predicción y análisis |
| MLflow | `http://<IP>:5000` | Experimentos y modelo registrado |

La IP se puede ver en:
- El resumen del workflow en GitHub Actions
- AWS Console → EC2 → Elastic IPs

---

## Flujo de trabajo diario

```
Modificas código
      │
      ▼
git push origin main
      │
      ▼
GitHub Actions dispara "2 · Deploy" automáticamente
      │
      ▼
La EC2 actualiza el código y reinicia los servicios
      │
      ▼
App disponible en la misma URL (IP fija)
```

Los archivos ignorados por el workflow (no disparan deploy):
- `*.md` — documentación
- `infra/**` — cambios de infraestructura
- `*.ipynb` — notebooks de análisis

---

## Estructura de archivos del deploy

```
.github/
  workflows/
    1-infra.yml      ← Crea/destruye infraestructura (manual)
    2-deploy.yml     ← Despliega la app (automático en push a main)
infra/
  cloudformation.yml ← Definición de toda la infraestructura AWS
scripts/
  setup.sh           ← Instala Docker en la EC2 (idempotente)
  deploy.sh          ← Actualiza código y levanta Docker Compose
```

---

## Costos en free tier

| Recurso | Costo | Límite free tier |
|---------|-------|-----------------|
| EC2 t3.micro | $0 | 750 h/mes × 12 meses |
| EBS 30 GB | $0 | 30 GB/mes × 12 meses |
| Elastic IP | $0 | Gratis mientras la instancia corre |
| SSM Run Command | $0 | Siempre gratis |
| CloudFormation | $0 | Siempre gratis |
| **Total** | **$0/mes** | Durante los primeros 12 meses |

> Después de los 12 meses, el costo aproximado es **$8-10 USD/mes**.

---

## Solución de problemas

### El workflow falla en "Esperar SSM agent"
La instancia recién creada tarda ~3 minutos en iniciar el agente SSM. Si falla, vuelve a ejecutar el workflow manualmente desde Actions.

### "No se encontró el stack"
El workflow de deploy requiere que la infraestructura exista. Ejecuta primero el **workflow 1** con la acción `deploy`.

### Streamlit no carga después del deploy
El trainer puede tardar hasta 5 minutos en entrenar los modelos. Espera y recarga la página. También puedes revisar los logs en el resumen del workflow.

### Ver logs en tiempo real
En AWS Console → Systems Manager → Run Command → historial de comandos. Ahí están los logs completos de cada ejecución.

---

## Destruir la infraestructura

Para eliminar **todos** los recursos de AWS (EC2, IP, etc.):

1. Ve a **Actions → 1 · Infraestructura (CloudFormation)**
2. Run workflow → acción: `destroy`

> Esto libera todos los recursos y detiene cualquier posible cobro.
