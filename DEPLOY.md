# Despliegue en AWS EC2 con GitHub Actions

Este documento explica cómo crear la infraestructura en AWS y desplegar la aplicación
automáticamente usando GitHub Actions.

---

## Arquitectura

```
GitHub Actions
      │
      ├─ 1-infra.yml ──► CloudFormation ──► EC2 t2.small
      │                                      Security Group (8501, 5000)
      │                                      Elastic IP (IP fija)
      │                                      IAM Role (SSM)
      │
      └─ 2-deploy.yml
            │
            ├─ Job 1: build ──► Construye imagen Docker (runner GitHub)
            │                   └── Push → ghcr.io/<repo>:latest (GHCR)
            │
            └─ Job 2: deploy ─► SSM Run Command ──► Configura swap 2 GB
                                 (sin SSH keys)       docker pull (desde GHCR)
                                                      docker compose up -d
                                                       ├── mlflow    :5000
                                                       └── streamlit :8501
                                                             └── botón "Entrenar"
                                                                   └── src/pipeline.py
```

Los workflows se comunican con la EC2 a través de **AWS Systems Manager (SSM)**, sin necesidad
de abrir el puerto SSH ni manejar archivos `.pem`.

La imagen Docker se construye en el **runner de GitHub Actions** y se almacena en
**GitHub Container Registry (GHCR)**, gratuito para repositorios públicos.

---

## Prerrequisitos

- Cuenta de AWS activa
- Repositorio en GitHub con el código del proyecto
- Permisos para agregar Secrets en el repositorio

---

## Paso 1 — Crear usuario IAM en AWS

1. Ve a **IAM → Users → Create user**
2. Nombre: `github-actions-deploy`
3. En **Permissions → Attach policies directly**, agrega:

| Política | Para qué se usa |
|----------|----------------|
| `AmazonEC2FullAccess` | Crear y gestionar instancias |
| `AWSCloudFormationFullAccess` | Crear el stack de infraestructura |
| `IAMFullAccess` | Crear el rol IAM para SSM |
| `AmazonSSMFullAccess` | Ejecutar comandos en la EC2 sin SSH |

4. Abre el usuario → **Security credentials → Create access key**
5. Selecciona **Application running outside AWS** → guarda el `Access key ID` y `Secret access key`

> Estos valores solo se muestran una vez. Guárdalos antes de cerrar la ventana.

---

## Paso 2 — Agregar Secrets en GitHub

Ve a **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Descripción |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | Access Key del usuario IAM |
| `AWS_SECRET_ACCESS_KEY` | Secret Key del usuario IAM |
| `AWS_REGION` | Región donde se creará la EC2 (ej. `us-east-1`) |

> El token para publicar en GHCR (`GITHUB_TOKEN`) es automático — GitHub lo genera en cada ejecución.

---

## Paso 3 — Crear la infraestructura

Solo se ejecuta **una vez** (o cuando necesites recrearla).

1. En GitHub → **Actions → 1 · Infraestructura (CloudFormation)**
2. **Run workflow → acción: `deploy`**
3. Espera ~2 minutos

Al completarse, el resumen muestra:

```
✅ Infraestructura lista

Recurso     | Valor
------------|-------------------------
Instancia   | i-0abc123def456789
Streamlit   | http://54.12.34.56:8501
MLflow      | http://54.12.34.56:5000
```

> La IP es fija (Elastic IP) y no cambia aunque la instancia se reinicie.

---

## Paso 4 — Primer despliegue

### Opción A: Automático
Cualquier push a `main` dispara el workflow **2 · Deploy** automáticamente.

### Opción B: Manual
Ve a **Actions → 2 · Deploy a EC2 → Run workflow**.

### Qué hace el workflow de deploy

**Job 1 — build** (runner de GitHub Actions):
1. Construye la imagen Docker con todas las dependencias ML
2. La publica en `ghcr.io/<usuario>/eda-superstore-sales:latest`
3. Usa caché de capas para builds incrementales

**Job 2 — deploy** (EC2 via SSM):
1. Verifica que el agente SSM esté activo
2. Configura swap a 2 GB si es menor (mitiga OOM en instancias con poca RAM)
3. Instala Docker + Compose si no están presentes
4. Clona o actualiza el repositorio en `/opt/superstore-sales`
5. Descarga la imagen desde GHCR (`docker pull`)
6. Reinicia los servicios con `docker compose up -d`

---

## Acceder a la aplicación

| Servicio | URL | Descripción |
|----------|-----|-------------|
| Streamlit | `http://<IP>:8501` | App de predicción y análisis |
| MLflow | `http://<IP>:5000` | Experimentos y modelo registrado |

La IP se puede ver en el resumen del workflow o en **AWS Console → EC2 → Elastic IPs**.

---

## Flujo de trabajo diario

```
Modificas código
      │
      ▼
git push origin main
      │
      ▼
Job 1: GitHub Actions construye imagen → sube a GHCR
      │
      ▼
Job 2: EC2 descarga imagen → reinicia servicios
      │
      ▼
App disponible en la misma URL (IP fija)
```

Cambios que **no** disparan deploy automático:
- `**.md` — documentación
- `infra/**` — cambios de infraestructura
- `*.ipynb` — notebooks
- `.github/workflows/1-infra.yml` — workflow de infraestructura

---

## Costos

| Recurso | Costo/mes |
|---------|-----------|
| EC2 t2.small | ~$17 |
| EBS 30 GB gp2 | ~$3 |
| Elastic IP (activa) | $0 |
| SSM / CloudFormation | $0 |
| **Total** | **~$20/mes** |

> t2.small no está en el free tier. Para uso académico puntual, destruye la infraestructura
> cuando no la uses (`destroy` en el workflow 1).

---

## Solución de problemas

### El workflow falla en "Esperar SSM agent"
La instancia tarda ~3 minutos en iniciar el agente SSM. Vuelve a ejecutar el workflow manualmente.

### "No se encontró el stack"
El workflow de deploy requiere que la infraestructura exista. Ejecuta primero el **workflow 1** con `deploy`.

### Streamlit no carga después del deploy
Streamlit arranca en cuanto MLflow está sano (~60 s). Si no responde, espera un minuto y recarga.

### No hay modelo disponible al entrar a Streamlit
Es el comportamiento esperado en el primer deploy. La app muestra **"Entrenar modelo ahora"** —
haz clic y el pipeline entrena los 4 modelos (~3-5 min en t2.small) y registra el mejor.

### Ver logs en tiempo real
Desde AWS Session Manager:
```bash
sudo docker logs -f $(sudo docker ps -qf name=streamlit)
sudo docker logs -f $(sudo docker ps -qf name=mlflow)
```

---

## Destruir la infraestructura

1. Ve a **Actions → 1 · Infraestructura (CloudFormation)**
2. **Run workflow → acción: `destroy`**

El workflow desactiva automáticamente la protección de terminación de la instancia EC2 antes
de eliminar el stack, por lo que no requiere intervención manual.

> Esto elimina EC2, Elastic IP y Security Group. El volumen EBS se conserva por
> `DeleteOnTermination: false` — elimínalo manualmente desde EC2 → Volumes si ya no lo necesitas.
