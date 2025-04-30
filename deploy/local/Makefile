VERSION ?= 0.2.4
TAVILY_SEARCH_API_KEY ?= ""
CONTAINER_REGISTRY ?= quay.io/ecosystem-appeng
DIST_UI_DIR := $(abspath $(lastword $(MAKEFILE_LIST)/../../../frontend))

setup_local:
	@if podman inspect -f '{{.State.Running}}' ollama 2>/dev/null | grep -q true; then \
		echo "ollama is already running."; \
	else \
		echo "Starting or restarting ollama ..."; \
		podman rm -fv --depend ollama 2>/dev/null || true; \
		podman run -d -v ollama:/root/.ollama -p 8321:8321 -p 8501:8501 --name ollama ollama/ollama; \
	fi
	@echo "Loading model to memory"
	podman exec -it ollama ollama run llama3.2:3b-instruct-fp16 --keepalive 160m "hi"

	@if podman inspect -f '{{.State.Running}}' llamastack 2>/dev/null | grep -q true; then \
		echo "llamastack is already running."; \
	else \
		echo "Starting or restarting llamastack ..."; \
		podman rm -fv --depend llamastack 2>/dev/null || true; \
		podman run -d -v llamastack:/root/.llama \
			--network container:ollama \
			--env INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" \
			--env OLLAMA_URL=http://localhost:11434 \
			--name llamastack \
			llamastack/distribution-ollama:0.2.3; \
	fi

run_ui: setup_local
	podman run -it -p 8501:8501 \
		--env LLAMA_STACK_ENDPOINT=http://localhost:8321 \
		--env TAVILY_SEARCH_API_KEY=$(TAVILY_SEARCH_API_KEY) \
		--network container:llamastack \
		llamastack-dist-ui:$(VERSION)

run_ui_local: setup_local
	podman run -it -p 8501:8501 \
		-v $(DIST_UI_DIR):/app:Z \
		--env LLAMA_STACK_ENDPOINT=http://localhost:8321 \
		--env TAVILY_SEARCH_API_KEY=$(TAVILY_SEARCH_API_KEY) \
		--network container:llamastack \
		llamastack-dist-ui:$(VERSION)

build_ui:
	podman build -t llamastack-dist-ui:$(VERSION) -f $(DIST_UI_DIR)/Containerfile $(DIST_UI_DIR)

build_and_push_ui: build_ui
	podman login $(CONTAINER_REGISTRY)
	podman tag llamastack-dist-ui:$(VERSION) $(CONTAINER_REGISTRY)/llamastack-dist-ui:$(VERSION)
	podman push $(CONTAINER_REGISTRY)/llamastack-dist-ui:$(VERSION)
