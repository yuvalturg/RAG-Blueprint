VERSION ?= 0.2.5
TAVILY_SEARCH_API_KEY ?= ""
CONTAINER_REGISTRY ?= quay.io/ecosystem-appeng
DIST_UI_DIR := $(abspath $(lastword $(MAKEFILE_LIST)/../../../frontend))

setup_local:
	mkdir -p ~/.local
	ollama run llama3.2:3b-instruct-fp16 --keepalive 160m &

	@if podman inspect -f '{{.State.Running}}' llamastack 2>/dev/null | grep -q true; then \
		echo "llamastack is already running."; \
	else \
		echo "Starting or restarting llamastack ..."; \
		podman rm -fv --depend llamastack 2>/dev/null || true; \
		podman run --platform linux/amd64 -d -v llamastack:/root/.llama \
			-p 8321:8321 \
			--env INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" \
			--env OLLAMA_URL=http://host.containers.internal:11434 \
			--name llamastack \
			llamastack/distribution-ollama:0.2.5; \
	fi

run_ui:
	if podman inspect -f '{{.State.Running}}' ui 2>/dev/null | grep -q true; then \
		echo "ui is already running."; \
	else \
		echo "Starting or restarting ui ...$(PYTHONPATH)"; \
		podman rm -fv --depend ui 2>/dev/null || true; \
		podman run --platform linux/amd64 -it \
			-p 8501:8501 \
			--env LLAMA_STACK_ENDPOINT=http://host.containers.internal:8321 \
			--env TAVILY_SEARCH_API_KEY=$(TAVILY_SEARCH_API_KEY) \
			--name ui \
			llamastack-dist-ui:$(VERSION); \
	fi

run_ui_local:
	cd ../../frontend && \
	python3.11 -m venv .venv && \
	source .venv/bin/activate && \
	pip install -r requirements.txt && \
	streamlit run llama_stack/distribution/ui/app.py

build_ui:
	podman build --platform linux/amd64 -t llamastack-dist-ui:$(VERSION) -f $(DIST_UI_DIR)/Containerfile $(DIST_UI_DIR)

build_and_push_ui: build_ui
	podman login $(CONTAINER_REGISTRY)
	podman tag llamastack-dist-ui:$(VERSION) $(CONTAINER_REGISTRY)/llamastack-dist-ui:$(VERSION)
	podman push $(CONTAINER_REGISTRY)/llamastack-dist-ui:$(VERSION)
