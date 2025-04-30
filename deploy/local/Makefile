VERSION ?= "0.2.4"
LLAMA_STACK_ENDPOINT=http://host.containers.internal:8321
TAVILY_SEARCH_API_KEY ?= ""
CONTAINER_REGISTRY ?= quay.io/ecosystem-appeng

setup_local:
	mkdir -p ~/.local
	ollama run llama3.2:3b-instruct-fp16 --keepalive 160m &
	podman run  --platform linux/amd64 -it -p 8321:8321 -v ~/.llama:/root/.llama llamastack/distribution-ollama:0.2.3 --port 8321 --env INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" --env OLLAMA_URL=http://host.containers.internal:11434

run_ui_local:
	cd ../../frontend && \
	uv run --with ".[ui]" streamlit run llama_stack/distribution/ui/app.py

build_ui:
	cd ../../frontend/llama_stack/distribution/ui && \
	podman build -t llamastack-dist-ui:$(VERSION) --platform="linux/amd64" -f Containerfile .

run_ui:
	podman run -it -p 8501:8501 --env LLAMA_STACK_ENDPOINT=$(LLAMA_STACK_ENDPOINT) --env TAVILY_SEARCH_API_KEY=$(TAVILY_SEARCH_API_KEY) llamastack-dist-ui:$(VERSION)

build_and_push_ui: build_ui
	podman login $(CONTAINER_REGISTRY) 
	podman tag llamastack-dist-ui:$(VERSION) $(CONTAINER_REGISTRY)/llamastack-dist-ui:$(VERSION)
	podman push $(CONTAINER_REGISTRY)/llamastack-dist-ui:$(VERSION)
