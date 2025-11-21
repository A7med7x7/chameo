# mlflow-replay

This repository provides an MLflow server setup that allows you to explore experiment artifacts from public object store containers hosted on [Chameleon Cloud](https://chameleoncloud.org). It's for viewing and analyzing previous experiments.

---

## About Experiment Reproducibility

When researchers complete their experiments, all the critical information gets automatically collected and stored through an MLflow server. This includes:

- Experiment metrics and parameters
- Artifacts and model files  
- GPU utilization info
- Version control status
- Environment configurations

Everything needed to understand and reproduce the experiment is safely stored in persistent object storage. this setup lets you browse these experiments that were previously packaged using [chameo](https://github.com/A7med7x7/chameo/tree/dev), so you can analyze results, compare different runs, and build upon existing work.

> [!IMPORTANT]
> **Sharing Requirements**: to view someone's experiments, the original authors need to:
> 1. Make their storage bucket publicly accessible 
> 2. Provide you with the container links or name

---

## How It Works

Each previous experiment uses two separate containers for storing information:

1. **Metrics Container**(The Backend store) - which stores run metadata, parameters, and metrics
2. **Artifacts Container**(The Artifacts store) - Contains model files, plots, and other experiment output
<img width="1115" height="547" alt="mlflow-server" src="https://github.com/user-attachments/assets/fb67acb6-3755-44d9-91f5-d888e835c67c" />

Our setup scripts use `rclone` to mount the public metrics container locally, while MLflow connects directly to the artifacts container via S3 protocol. This gives you full access to browse experiment history and download any artifacts you need.

---

## What You'll Need

### 1. Container Information
Get the links to both containers from the experiment authors:
- Metrics container URL
- Artifacts container URL

### 2. Access Credentials
We use EC2 credentials by default. generate yours using the [generate_credentials.ipynb](notebooks/generate_credentials.ipynb) notebook. to access Chameleon Cloud expriement

> [!IMPORTANT]
> **Run the notebook from Chameleon JupyterHub** - it won't work from your local machine or other environment.

### 3. Endpoint URL
Choose the correct endpoint based on where the containers are hosted:

- **CHI@TACC**: `https://chi.tacc.chameleoncloud.org:7480`
- **CHI@UC**: `https://chi.uc.chameleoncloud.org:7480`

---

## Setup Instructions

> [!IMPORTANT]
> **Run these commands on a Chameleon Cloud node**.

### Step 1: Clone the Repository from your home directory
```bash
cd ~
git clone --branch mlflow-replay https://github.com/A7med7x7/chameo.git mlflow-read
```

### Step 2: Generate Configuration Scripts
```bash
python3 mlflow-read/main.py
```
You'll be prompted to enter the container information and credentials from above. This creates two setup scripts in the `scripts` directory.

### Step 3: Configure Environment
```bash
bash mlflow-read/scripts/generate_env.sh
```
This script fetches your credentials and container information, storing them in a `.env` file for the next steps.

### Step 4: Mount Storage
```bash
bash mlflow-read/scripts/mount_public.sh
```
Creates and mounts the directory where metrics data will be accessible.

### Step 5: Start MLflow Server
```bash
docker compose --env-file .env2 -f mlflow-read/docker/docker-compose.yml up -d --build
```

### Step 6: Access the Web Interface
Get your MLflow URL:
```bash
echo "http://$(curl -s ifconfig.me):$(docker port mlflow-replay 8000 | cut -d':' -f2)"
```
Copy this URL into your browser to start exploring experiments.

### Step 7: Cleanup (When Finished)
```bash
docker compose -f mlflow-read/docker/docker-compose.yml down
```
