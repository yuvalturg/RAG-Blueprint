# Makefile for RAG Deployment
# Replaces the original deploy.sh script with additional uninstall functionality
ifeq ($(NAMESPACE),)
ifeq (,$(filter list-models% help,$(MAKECMDGOALS)))
$(error NAMESPACE is not set)
endif
endif

MAKEFLAGS += --no-print-directory

# Default values
POSTGRES_USER ?= postgres
POSTGRES_PASSWORD ?= rag_password
POSTGRES_DBNAME ?= rag_blueprint
MINIO_USER ?= minio_rag_user
MINIO_PASSWORD ?= minio_rag_password
HF_TOKEN ?= $(shell bash -c 'read -r -p "Enter Hugging Face Token: " HF_TOKEN; echo $$HF_TOKEN')
LLM_SERVICE_CHART := llm-service
LLAMA_STACK_CHART := llama-stack
MCP_SERVERS_CHART := mcp-servers
PGVECTOR_CHART := pgvector
MINIO_CHART := minio
RAG_UI_CHART := rag-ui
INGESTION_PIPELINE_CHART := ingestion-pipeline
CONFIGURE_PIPELINE_SERVER_CHART := configure-pipeline-server
COMPONENTS := $(LLM_SERVICE_CHART) $(MCP_SERVERS_CHART) $(PGVECTOR_CHART) $(MINIO_CHART) $(RAG_UI_CHART) $(INGESTION_PIPELINE_CHART) $(CONFIGURE_PIPELINE_SERVER_CHART) $(LLAMA_STACK_CHART)
TOLERATIONS_TEMPLATE=[{"key":"$(1)","effect":"NoSchedule","operator":"Exists"}]

#ingestion pipeline configuration
SOURCE ?= S3
EMBEDDING_MODEL ?= all-MiniLM-L6-v2
INGESTION_PIPELINE_NAME ?= demo-rag-vector-db
INGESTION_PIPELINE_VERSION ?= 1.0
ACCESS_KEY_ID ?= $(MINIO_USER)
SECRET_ACCESS_KEY ?= $(MINIO_PASSWORD)
BUCKET_NAME ?= documents
ENDPOINT_URL ?= http://minio:9000
REGION ?= us-east-1
# PDF file path variable for upload-pdf target
PDF_DIR = ../../notebooks

S3_TEMPLATE={"access_key_id":"$(1)","secret_access_key":"$(2)","bucket_name":"$(3)","endpoint_url":"$(4)","region":"$(5)"}

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install       - Install the RAG deployment (creates namespace, secrets, and deploys Helm chart)"
	@echo "  install-cpu   - Install the RAG deployment without GPU (creates namespace, secrets, and deploys Helm chart)"
	@echo "  uninstall     - Uninstall the RAG deployment and clean up resources"
	@echo "  status        - Check status of the deployment"
	@echo "  wait          - Wait for all pods to be ready and verify deployment health"
	@echo "  list-models       - List available models for GPU"
	@echo "  list-models-cpu   - List available models for CPU"
	@echo ""
	@echo "Configuration options (set via environment variables or make arguments):"
	@echo "  NAMESPACE                - Target namespace (default: llama-stack-rag)"
	@echo "  HF_TOKEN                 - Hugging Face Token (will prompt if not provided)"
	@echo "  {SAFETY,LLM}             - Model id as defined in values (eg. llama-3-2-1b-instruct)"
	@echo "  {SAFETY,LLM}_URL         - Model URL"
	@echo "  {SAFETY,LLM}_API_TOKEN   - Model API token for remote models"
	@echo "  {SAFETY,LLM}_TOLERATION  - Model pod toleration"

# Create namespace and deploy
namespace:
	@oc create namespace $(NAMESPACE) &> /dev/null && oc label namespace $(NAMESPACE) modelmesh-enabled=false ||:
	@oc project $(NAMESPACE) &> /dev/null ||:

helm-install-%: namespace
	$(if $(HELM_ARGS_FUNC), $(eval HELM_ARGS := $(call $(HELM_ARGS_FUNC))))
	@echo "$*: Deploying Helm chart in namespace $(NAMESPACE)..."
	@helm upgrade --install $* ./$* -n $(NAMESPACE) $(HELM_ARGS) $(EXTRA_HELM_ARGS) > /dev/null && echo "$*: Helm chart installed successfully"

helm-uninstall-%:
	@echo "$*: Uninstalling Helm chart from namespace $(NAMESPACE)..."
	@helm uninstall $* -n $(NAMESPACE) &> /dev/null || echo "$* is not installed or already removed."

list-models-%:
	@helm template dummy-release $(LLM_SERVICE_CHART) --set _debugListModels=true --values $(LLM_SERVICE_CHART)/values-$*.yaml |grep ^model

.PHONY: list-models
list-models: list-models-gpu


helm_llm_service_args = \
    --set secret.hf_token=$(HF_TOKEN) \
    $(if $(LLM),--set models.$(LLM).enabled=true,) \
    $(if $(SAFETY),--set models.$(SAFETY).enabled=true,) \
    $(if $(LLM_TOLERATION),--set-json models.$(LLM).inferenceService.tolerations='$(call TOLERATIONS_TEMPLATE,$(LLM_TOLERATION))',) \
    $(if $(SAFETY_TOLERATION),--set-json models.$(SAFETY).inferenceService.tolerations='$(call TOLERATIONS_TEMPLATE,$(SAFETY_TOLERATION))',)

install-llm-service-%: namespace
	@$(MAKE) helm-install-$(LLM_SERVICE_CHART) HELM_ARGS_FUNC=helm_llm_service_args EXTRA_HELM_ARGS="--values=$(LLM_SERVICE_CHART)/values-$*.yaml "
	@echo "Waiting for model services to deploy. It will take around 10-15 minutes depending on the size of the model..."
	@oc wait -n $(NAMESPACE) --for=condition=Ready --timeout=60m inferenceservice --all ||:

.PHONY: install-llm-service
install-llm-service: install-llm-service-gpu

helm_pgvector_args = \
    --set secret.user=$(POSTGRES_USER) \
    --set secret.password=$(POSTGRES_PASSWORD) \
    --set secret.dbname=$(POSTGRES_DBNAME)

helm_minio_args = \
    --set secret.user=$(MINIO_USER) \
    --set secret.password=$(MINIO_PASSWORD)

helm_llama_stack_args = \
    $(if $(LLM),--set models.$(LLM).enabled=true,) \
    $(if $(SAFETY),--set models.$(SAFETY).enabled=true,) \
    $(if $(LLM_URL),--set models.$(LLM).url='$(LLM_URL)',) \
    $(if $(SAFETY_URL),--set models.$(SAFETY).url='$(SAFETY_URL)',) \
    $(if $(LLM_API_TOKEN),--set models.$(LLM).apiToken='$(LLM_API_TOKEN)',) \
    $(if $(SAFETY_API_TOKEN),--set models.$(SAFETY).apiToken='$(SAFETY_API_TOKEN)',) \
    $(if $(LLAMA_STACK_ENV),--set-json secrets='$(LLAMA_STACK_ENV)',)

set_ingestion_args = \
	$(if $(SOURCE),--set source='$(SOURCE)',) \
	$(if $(EMBEDDING_MODEL),--set embedding_model='$(EMBEDDING_MODEL)',) \
	$(if $(INGESTION_PIPELINE_NAME),--set name='$(INGESTION_PIPELINE_NAME)',) \
	$(if $(INGESTION_PIPELINE_VERSION),--set version='$(INGESTION_PIPELINE_VERSION)',) \
    $(if $(SOURCE),--set-json S3='$(call S3_TEMPLATE,$(ACCESS_KEY_ID),$(SECRET_ACCESS_KEY),$(BUCKET_NAME),$(ENDPOINT_URL),$(REGION))',)

.PHONY: install-rag
install-rag: namespace helm-install-$(MCP_SERVERS_CHART) helm-install-$(PGVECTOR_CHART) helm-install-$(MINIO_CHART)
	@$(MAKE) helm-install-$(PGVECTOR_CHART) HELM_ARGS_FUNC=helm_pgvector_args
	@$(MAKE) helm-install-$(MINIO_CHART) HELM_ARGS_FUNC=helm_minio_args
	@$(MAKE) helm-install-$(LLAMA_STACK_CHART) HELM_ARGS_FUNC=helm_llama_stack_args
	@$(MAKE) helm-install-$(RAG_UI_CHART)
	@$(MAKE) create-minio-bucket
	@$(MAKE) upload-pdf
	@$(MAKE) status
	@$(MAKE) helm-install-$(CONFIGURE_PIPELINE_SERVER_CHART)
	@$(MAKE) helm-install-$(INGESTION_PIPELINE_CHART) HELM_ARGS_FUNC=set_ingestion_args ||:
	@echo "Waiting for deployment to be ready..."
	@$(MAKE) wait

.PHONY: create-minio-bucket
create-minio-bucket:
	oc wait -n $(NAMESPACE) --for=condition=Ready pod/minio-0 --timeout=300s
	sleep 5
	oc exec -n $(NAMESPACE) minio-0 -- bash -c "mc alias set local http://localhost:9000 $(MINIO_USER) $(MINIO_PASSWORD) && mc mb local/$(BUCKET_NAME)" ||:

.PHONY: upload-pdf
upload-pdf:
	@echo "Uploading all PDF files from $(PDF_DIR) to Minio bucket $(BUCKET_NAME)..."
	@echo "Waiting for minio-0 pod to be ready..."
	oc wait -n $(NAMESPACE) --for=condition=Ready pod/minio-0 --timeout=60s
	@echo "Setting up minio client alias..."
	oc exec -n $(NAMESPACE) minio-0 -- bash -c "mc alias set local http://localhost:9000 $(MINIO_USER) $(MINIO_PASSWORD)"
	@find $(PDF_DIR) -name "*.pdf" -type f | while read -r pdf_file; do \
		echo "Uploading $${pdf_file}..."; \
		test -f "$${pdf_file}" || { echo "Error: File $${pdf_file} not found"; continue; }; \
		# Create a safe unique temporary filename using a timestamp and counter to prevent issues with special chars\
		temp_name="tmp_$$(date +%s)_$${RANDOM}.pdf"; \
		echo "Creating temporary file in minio pod with safe name: $${temp_name}"; \
		cat "$${pdf_file}" | oc exec -i -n $(NAMESPACE) minio-0 -- bash -c "cat > /tmp/$${temp_name}"; \
		# Extract original filename \
		orig_name="$$(basename "$${pdf_file}")"; \
		echo "Uploading file to bucket with original name: $${orig_name}"; \
		# Copying file with temporary name, then renaming it to original name \
		oc exec -n $(NAMESPACE) minio-0 -- bash -c "mc cp /tmp/$${temp_name} local/$(BUCKET_NAME)/$${temp_name} && \
		mc mv local/$(BUCKET_NAME)/$${temp_name} \"local/$(BUCKET_NAME)/$${orig_name}\" && \
		rm /tmp/$${temp_name}"; \
		echo "PDF file '$${pdf_file}' uploaded successfully to Minio bucket $(BUCKET_NAME)"; \
	done

install-gpu: install-llm-service-gpu install-rag
install-cpu: install-llm-service-cpu install-rag

.PHONY: install
install: install-gpu

# Uninstall the deployment and clean up
.PHONY: uninstall
uninstall: $(addprefix helm-uninstall-,$(COMPONENTS)) remove-pvcs
	@echo "Deleting remaining pods in namespace $(NAMESPACE)"
	@oc delete pods -n $(NAMESPACE) --all
	@echo "Checking for any remaining resources in namespace $(NAMESPACE)..."
	@echo "If you want to completely remove the namespace, run: oc delete project $(NAMESPACE)"
	@echo "Remaining resources in namespace $(NAMESPACE):"
	@$(MAKE) status

.PHONY: remove-pvcs
remove-pvcs:
	@echo "Removing pgvector and minio PVCs"
	@oc get pvc -n $(NAMESPACE) -o custom-columns=NAME:.metadata.name | grep -E '^(pg|minio)-data' | xargs -I {} oc delete pvc -n $(NAMESPACE) {} ||:

# Check deployment status
.PHONY: status
status:
	@echo "Listing pods..."
	oc get pods -n $(NAMESPACE) || true

	@echo "Listing services..."
	oc get svc -n $(NAMESPACE) || true

	@echo "Listing routes..."
	oc get routes -n $(NAMESPACE) || true

	@echo "Listing secrets..."
	oc get secrets -n $(NAMESPACE) | grep huggingface-secret || true

	@echo "Listing pvcs..."
	oc get pvc -n $(NAMESPACE) || true

# Wait for all pods to be ready
.PHONY: wait
wait:

	@echo "Delete failed jobs in namespace $(NAMESPACE)..."
	oc get pods -n $(NAMESPACE) --field-selector=status.phase=Failed -o jsonpath='{range .items[?(@.metadata.ownerReferences[0].kind=="Job")]}{.metadata.namespace}{";"}{.metadata.name}{"\n"}{end}' | while IFS=";" read ns pod; do \
	  echo "Deleting FAILED pod $$pod from namespace $$ns"; \
	  oc delete pod "$$pod" -n "$$ns"; \
	done

	@echo "Waiting for all pods to be ready in namespace $(NAMESPACE)..."
	@end=$$(($$(date +%s)+60)); \
	while [ $$(date +%s) -lt $$end ]; do \
	  not_ready=$$(kubectl get pods --no-headers | grep -vE 'Running|Succeeded|Completed'); \
	  if [ -z "$$not_ready" ]; then \
	    echo "All pods are Ready or Completed."; \
		break; \
	  fi; \
	  sleep 2; \
	done; \
	echo "Timeout: Some pods are not Ready or Completed."; \

	@echo "Verifying routes are accessible..."
	@for route in $(oc get routes -n $(NAMESPACE) -o name); do \
		echo "Checking route ${route}..."; \
		host=$(oc get ${route} -n $(NAMESPACE) -o jsonpath='{.spec.host}'); \
		if [ -n "${host}" ]; then \
			echo "Route hostname: ${host}"; \
			echo "Note: Manual verification of route accessibility is recommended"; \
		else \
			echo "WARNING: No hostname found for ${route}"; \
		fi; \
	done
