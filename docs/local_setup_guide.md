# 🦙 Llama Stack Local Setup Guide

This guide walks you through running a Llama Stack server locally using **Ollama** and **Podman**.

---

## ✅ 1. Prerequisites

Install the following tools:

- [Podman](https://podman.io/docs/installation)
- Python 3.10 or newer
- [pip](https://pip.pypa.io/en/stable/installation/)
- [Ollama](https://ollama.com/download)

Verify installation:

```bash
podman --version
python3 --version
pip --version
ollama --version
```

---

## 🚀 2. Start Ollama Server

Use the following command to launch the Ollama model:

```bash
ollama run llama3.2:3b-instruct-fp16 --keepalive 60m
```

Note: This will keep the model in memory for 60 minutes.

---

## ⚙️ 3. Configure Environment Variables

Launch a new terminal window to set the necessary environment variables:

```bash
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
export LLAMA_STACK_PORT=8321
```

---

## 🐳 4. Run Llama Stack with Podman

Pull the Docker image:

```bash
podman pull docker.io/llamastack/distribution-ollama
```

Create a local directory for persistent data:

```bash
mkdir -p ~/.llama
```

Run the container:

```bash
podman run -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://host.containers.internal:11434 \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT
```

Optional: Use a custom network:

```bash
podman network create llama-net
podman run --privileged --network llama-net -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT
```

Verify the container is running:

```bash
podman ps
```

---

## 🐍 5. Set Up Python Environment

Install `uv` and sync dependencies:

```bash
pip install uv
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate  # macOS/Linux
# Windows:
# llama-stack-demo\Scripts\activate
```

Check installation:

```bash
pip list | grep llama-stack-client
```

---

## 📡 6. Configure the Client

Point the client to the local Llama Stack server:

```bash
llama-stack-client configure --endpoint http://localhost:$LLAMA_STACK_PORT
```

List models:

```bash
llama-stack-client models list
```

---

## ⚡ 7. Quick Re-Setup

After initial setup, you can restart everything with:

```bash
make setup_local
```

---

## 🧰 8. Troubleshooting Tips

**Check if Podman is running:**

```bash
podman ps
```

**Activate your virtual environment:**

```bash
source llama-stack-demo/bin/activate
```

**Reinstall the client if needed:**

```bash
pip uninstall llama-stack-client
pip install llama-stack-client
```

**Test the client in Python:**

```bash
python -c "from llama_stack_client import LlamaStackClient; print(LlamaStackClient)"
```
