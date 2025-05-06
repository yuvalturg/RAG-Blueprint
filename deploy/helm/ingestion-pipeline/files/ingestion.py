from kfp import dsl
from kfp import Client
from kfp import compiler

@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "boto3",
        "llama-stack-client==0.2.3", 
        "docling",
        "docling-core"
    ])
def fetch_from_minio_store_pgvector(llamastack_base_url: str):
    import shutil
    import os
    import boto3
    import tempfile    
    
    temp_dir = tempfile.mkdtemp()

    try: 
        # Set EasyOCR path to a writeable directory BEFORE importing docling
        # It runs fine without it in the local environment, but fails in the pipeline container
        os.environ["EASYOCR_MODULE_PATH"] = "/tmp/.EasyOCR"

        from llama_stack_client import LlamaStackClient
        from llama_stack_client.types import Document as LlamaStackDocument
        
        # Import docling libraries
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        from docling_core.types.doc.labels import DocItemLabel

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

        # Step 1: Download files from MinIO
        temp_dir = tempfile.mkdtemp()
        download_dir = os.path.join(temp_dir, "source_repo")
        os.makedirs(download_dir, exist_ok=True)

        # Connect to MinIO
        print(f"Connecting to MinIO at {minio_endpoint}")
        s3 = boto3.client(
            "s3",
            endpoint_url=minio_endpoint,
            aws_access_key_id=minio_access_key,
            aws_secret_access_key=minio_secret_key,
            verify=False
        )

        # List and download objects
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name)

        print(f"Downloading files from bucket: {bucket_name}")
        downloaded_files = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                file_path = os.path.join(download_dir, os.path.basename(key))
                print(f"Downloading: {key} -> {file_path}")
                s3.download_file(bucket_name, key, file_path)
                downloaded_files.append(file_path)
        
        print(f"Downloaded {len(downloaded_files)} files to {download_dir}")
        
        if not downloaded_files:
            raise Exception(f"No files found in bucket: {bucket_name}. Please check your bucket configuration.")
        
        # Step 2: Process the PDFs with docling
        # Setup docling components
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
        )
        chunker = HybridChunker()
        llama_documents = []
        i = 0
        
        # Process each file with docling (chunking)
        for file_path in downloaded_files:
            # Skip empty files
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                print(f"Skipping empty file: {file_path} (0 bytes)")
                continue

            if not file_path.endswith(".pdf"):
                print(f"Skipping non-PDF file: {file_path} (unsupported file type)")
                continue

            print(f"Processing {file_path} with docling...")
            try:
                docling_doc = converter.convert(source=file_path).document
                chunks = chunker.chunk(docling_doc)
                chunk_count = 0

                for chunk in chunks:
                    if any(
                        c.label in [DocItemLabel.TEXT, DocItemLabel.PARAGRAPH]
                        for c in chunk.meta.doc_items
                    ):
                        i += 1
                        chunk_count += 1
                        llama_documents.append(
                            LlamaStackDocument(
                                document_id=f"doc-{i}",
                                content=chunk.text,
                                mime_type="text/plain",
                                metadata={"source": os.path.basename(file_path)},
                            )
                        )
                
                print(f"Created {chunk_count} chunks from {file_path}")
                
            except Exception as e:
                error_message = str(e)
                print(f"Error processing {file_path}: {error_message}")

        total_chunks = len(llama_documents)
        print(f"Total valid chunks prepared: {total_chunks}")

        # Add error handling for zero chunks
        if total_chunks == 0:
            raise Exception("No valid chunks were created. Check document processing errors above.")

        # Step 3: Register vector database and store chunks with embeddings
        client = LlamaStackClient(base_url=llamastack_base_url)
        print("Registering db")
        try:
            client.vector_dbs.register(
                vector_db_id=vector_db_name,
                embedding_model=embedding_model,
                embedding_dimension=384,
                provider_id="pgvector",
            )
            print("Vector DB registered successfully")
        except Exception as e:
            error_message = str(e)
            print(f"Failed to register vector DB: {error_message}")
            print("Continuing with insertion...")

        try:
            print(f"Inserting {total_chunks} chunks into vector database")
            client.tool_runtime.rag_tool.insert(
                documents=llama_documents,
                vector_db_id=vector_db_name,
                chunk_size_in_tokens=512,
            )
            print("Documents successfully inserted into the vector DB")

        except Exception as e:
            print("Embedding insert failed:", e)
            raise Exception(f"Failed to insert documents into vector DB: {e}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Cleaned up temporary directory: {temp_dir}")

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