from kfp import dsl
from kfp import Client
from kfp import compiler

@dsl.component(
  base_image="python:3.10",
  packages_to_install=[
      "boto3",
      "langchain-community",
      "pdfminer.six",
      "pymupdf",
      "pypdf",
      "tqdm",
      "sentence-transformers",
      "huggingface-hub",
      "llama-stack-client==0.1.9",
      "numpy",
      "pdfplumber"
  ])
def fetch_from_minio_store_pgvector(llamastack_base_url: str):
  import os
  import boto3
  import tempfile
  import numpy as np
  import pdfplumber
  from llama_stack_client import LlamaStackClient, RAGDocument

  temp_dir = tempfile.mkdtemp()
  download_dir = os.path.join(temp_dir, "source_repo")
  os.makedirs(download_dir, exist_ok=True)

  source = os.environ.get('SOURCE')
  name = os.environ.get('NAME')
  version = os.environ.get('VERSION')
  embedding_model = os.environ.get('EMBEDDING_MODEL')
  
  vector_db_name = f"{name}-v{version}".replace(" ", "-").replace(".", "-")

  # S3 Config
  bucket_name = os.environ.get('BUCKET_NAME')
  minio_endpoint = os.environ.get('ENDPOINT_URL')
  minio_access_key = os.environ.get('ACCESS_KEY_ID')
  minio_secret_key = os.environ.get('SECRET_ACCESS_KEY')
  region_name = os.environ.get('REGION')

  s3 = boto3.client(
      "s3",
      endpoint_url=minio_endpoint,
      aws_access_key_id=minio_access_key,
      aws_secret_access_key=minio_secret_key,
      region_name = region_name
  )
  
  paginator = s3.get_paginator("list_objects_v2")
  pages = paginator.paginate(Bucket=bucket_name)

  for page in pages:
      for obj in page.get("Contents", []):
          key = obj["Key"]
          file_path = os.path.join(download_dir, os.path.basename(key))
          s3.download_file(bucket_name, key, file_path)

  rag_documents = []
  rng = np.random.default_rng()
  for filename in os.listdir(download_dir):
      if not filename.endswith(".pdf"):
          continue
      full_path = os.path.join(download_dir, filename)
      full_text = ""
      with pdfplumber.open(full_path) as pdf:
          for page in pdf.pages:
              page_text = page.extract_text()
              if page_text:
                  full_text += page_text
      full_text = full_text.encode("utf-8", "ignore").decode("utf-8").replace("\x00", "")
      if not full_text.strip():
          continue
      rag_documents.append(
          RAGDocument(
              document_id=f"pdf-{rng.integers(1000, 9999)}",
              content=full_text,
              mime_type="application/pdf",
              metadata={"source": "rag-pipeline", "filename": filename}
          )
      )

  client = LlamaStackClient(base_url=llamastack_base_url)
  print("Registering db")
  client.vector_dbs.register(
      vector_db_id=vector_db_name,
      embedding_model=embedding_model,
      embedding_dimension=384,
      provider_id="pgvector",
  )

  try:
      client.tool_runtime.rag_tool.insert(
          documents=rag_documents,
          vector_db_id=vector_db_name,
          chunk_size_in_tokens=512,
      )
  except Exception as e:
      print("Embedding insert failed:", e)


@dsl.pipeline(name="fetch-and-store-pipeline")
def full_pipeline():
  import os
  from kfp import kubernetes
  secret_key_to_env = {
          'SOURCE': 'SOURCE',
          'EMBEDDING_MODEL': 'EMBEDDING_MODEL',
          'NAME': 'NAME',
          'VERSION': 'VERSION',
          'ACCESS_KEY_ID': 'ACCESS_KEY_ID',
          'SECRET_ACCESS_KEY': 'SECRET_ACCESS_KEY',
          'ENDPOINT_URL': 'ENDPOINT_URL',
          'BUCKET_NAME': 'BUCKET_NAME',
          'REGION': 'REGION'
  }

  fetch_task = fetch_from_minio_store_pgvector(
      llamastack_base_url=os.environ["LLAMASTACK_BASE_URL"])

  kubernetes.use_secret_as_env(
      task=fetch_task,
      secret_name="REPLACE_SECRET_NAME",
      secret_key_to_env=secret_key_to_env)

# 1. Compile pipeline to a file
pipeline_yaml = "/tmp/fetch_chunk_embed_pipeline.yaml"

compiler.Compiler().compile(
  pipeline_func=full_pipeline,
  package_path=pipeline_yaml
)

import os
# 2. Connect to KFP
client = Client(
  host=os.environ["DS_PIPELINE_URL"],
  verify_ssl=False
)

# 3. Upload pipeline
uploaded_pipeline = client.upload_pipeline(
  pipeline_package_path=pipeline_yaml,
  pipeline_name="REPLACE_PIPELINE_NAME"
)

# 4. Run the pipeline
run = client.create_run_from_pipeline_package(
  pipeline_file=pipeline_yaml,
  arguments={},
  run_name="fetch-store-run"
)

print(f"Pipeline submitted! Run ID: {run.run_id}")