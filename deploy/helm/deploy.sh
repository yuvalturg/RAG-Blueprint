#! /bin/bash

read -r -p "Enter Hugging Face Token: " HF_TOKEN
echo "$HF_TOKEN"

NAMESPACE=llama-stack-rag
PGVECTOR_PASSWORD="$(tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 10)"
PGVECTOR_DBNAME=rag_blueprint
 
oc new-project $NAMESPACE
oc create secret -n $NAMESPACE generic huggingface-secret --from-literal=HF_TOKEN="$HF_TOKEN"
oc create secret -n $NAMESPACE generic vectordb-app \
       --from-literal=username=postgres \
       --from-literal=password=$PGVECTOR_PASSWORD \
       --from-literal=host=pgvector \
       --from-literal=port=5432 \
       --from-literal=dbname=$PGVECTOR_DBNAME

oc annotate secret huggingface-secret -n $NAMESPACE meta.helm.sh/release-name=rag meta.helm.sh/release-namespace=$NAMESPACE
oc annotate secret vectordb-app -n $NAMESPACE meta.helm.sh/release-name=rag meta.helm.sh/release-namespace=$NAMESPACE

# DOMAIN=$(kubectl get Ingress.config.openshift.io/cluster -o jsonpath='{.spec.domain}')

helm upgrade --install rag rag-ui -n $NAMESPACE \
--set-json llama-serve.tolerations='[{"key":"g6e-gpu","effect":"NoSchedule","operator":"Exists"}]' \
--set-json safety-model.tolerations='[{"key":"odh-notebook","effect":"NoSchedule","operator":"Exists"}]'  

echo "Listing pods..."
kubectl get pods -n $NAMESPACE

echo "Listing services..."
kubectl get svc -n $NAMESPACE

echo "Listing routes..."
kubectl get routes -n $NAMESPACE
