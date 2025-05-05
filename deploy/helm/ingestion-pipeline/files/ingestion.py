from kfp import dsl
from kfp import Client
from kfp import compiler
import os

@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "boto3"
    ])
def fetch_from_s3(output_dir: dsl.OutputPath()):
    import shutil
    import os
    import boto3
    import tempfile
    import json
    
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
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Print directory contents to debug
    print(f"Contents of temp dir before copy: {os.listdir(temp_dir)}")
    print(f"Contents of download dir before copy: {os.listdir(download_dir)}")
    
    # Copy all downloaded files to the output directory and create a manifest
    file_manifest = []
    for file_path in downloaded_files:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)
            
            # Debug output
            print(f"Copying from {file_path} (exists: {os.path.exists(file_path)}) to {output_path}")
            
            # Copy the file
            shutil.copy2(file_path, output_path)
            file_manifest.append(filename)
            print(f"Successfully copied {file_path} to {output_path}")
    
    # Write a manifest file to track what was downloaded
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(file_manifest, f)
    
    print(f"Created file manifest at {manifest_path} with {len(file_manifest)} files")
    print(f"Final contents of output dir: {os.listdir(output_dir)}")
    
    # Clean up the temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Cleaned up temporary directory: {temp_dir}")

@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "llama-stack-client==0.2.3", 
        "docling",
        "docling-core"
    ])
def process_and_store_pgvector(llamastack_base_url: str, input_dir: dsl.InputPath()):
    import shutil
    import os
    import tempfile
    import json
    
    # Debug input directory
    print(f"Input directory: {input_dir}")
    if os.path.exists(input_dir):
        print(f"Contents of input directory: {os.listdir(input_dir)}")
    else:
        print(f"Input directory does not exist: {input_dir}")
    
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
    
    # Read the manifest to get the list of files
    manifest_path = os.path.join(input_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
        
    with open(manifest_path, "r") as f:
        file_manifest = json.load(f)
    
    print(f"Processing {len(file_manifest)} files from the manifest")
    
    # Get the full paths of all input files
    input_files = [os.path.join(input_dir, filename) for filename in file_manifest]
    
    # Verify files exist
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File doesn't exist: {file_path}")
        else:
            print(f"File exists: {file_path}, size: {os.path.getsize(file_path)} bytes")
    
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
    for file_path in input_files:
        # Skip empty files
        if not os.path.exists(file_path):
            print(f"File doesn't exist: {file_path}")
            continue
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"Skipping empty file: {file_path} (0 bytes)")
            continue

        # Get file extension
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        print(f"Processing {file_path} (extension: {file_extension}) with docling...")
        try:
            # Different format options based on file type
            if file_extension == '.pdf':
                pipeline_options = PdfPipelineOptions()
                pipeline_options.generate_picture_images = True
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
            else:
                # Use generic converter for all other file types
                print(f"Using generic converter for file type: {file_extension}")
                converter = DocumentConverter()
                
            docling_doc = converter.convert(source=file_path).document
            chunks = chunker.chunk(docling_doc)
            chunk_count = 0

            for chunk in chunks:
                if any(
                    c.label in [DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.TABLE, 
                               DocItemLabel.TABLE_CELL, DocItemLabel.HEADER, DocItemLabel.FOOTER, 
                               DocItemLabel.TITLE, DocItemLabel.PICTURE_DESCRIPTION]
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

@dsl.pipeline(name="fetch-and-store-pipeline")
def full_pipeline():
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

  fetch_task = fetch_from_s3()
  
  process_task = process_and_store_pgvector(
      llamastack_base_url=os.environ["LLAMASTACK_BASE_URL"],
      input_dir=fetch_task.outputs["output_dir"]
  )

  kubernetes.use_secret_as_env(
      task=fetch_task,
      secret_name="REPLACE_SECRET_NAME",
      secret_key_to_env=secret_key_to_env)
      
  kubernetes.use_secret_as_env(
      task=process_task,
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