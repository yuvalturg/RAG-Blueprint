## Workbench deployment verification

Navigate to RHOAI dashboard and verify the following -
1. You should be able to see a running workbench with running jupyter notebook.

![Workbench UI](img/workbench.png)

2. Jupyter notebook should have a python script.

![Notebook](img/jupyter-nb.png)

3. Before running that make sure you have your Kubeflow Pipelines configured with your object storage.
   Reference link(configuration) - https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.8/html/working_on_data_science_projects/working-with-data-science-pipelines_ds-pipelines#configuring-a-pipeline-server_ds-pipelines

  For access and secret keys --
  - Navigate to `minio-webui`
  - Login with credentials
  - Create access and secret key in minIO
  - Upload your files in the already created `llama` bucket
  - Now navigate to Kubeflow Pipelines on Openshift AI and configure it with the generated secret and access keys. Configure with the same bucket name as in minIO `llama`.

  ![KFP](img/kfp-configure.png)

4. Once verified, run the python script.
5. This should create `pipelines` and `run` in the Kubeflow pipelines.

![KFP-pipeline](img/kfp-pipeline.png)

![KFP](img/kfp-run.png)

![KFP](img/kfp-logs.png)


## Verifying the embeddings in PGVector

```
psql -d rag_blueprint -U postgres
psql (17.4 (Debian 17.4-1.pgdg120+2))
Type "help" for help.

rag_blueprint=# \dt
               List of relations
 Schema |       Name        | Type  |  Owner
--------+-------------------+-------+----------
 public | metadata_store    | table | postgres
 public | vector_store_test | table | postgres
(2 rows)

rag_blueprint=# \d+ vector_store_test
                                        Table "public.vector_store_test"
  Column   |    Type     | Collation | Nullable | Default | Storage  | Compression | Stats target | Description
-----------+-------------+-----------+----------+---------+----------+-------------+--------------
 id        | text        |           | not null |         | extended |             |              |
 document  | jsonb       |           |          |         | extended |             |              |
 embedding | vector(384) |           |          |         | external |             |              |
Indexes:
    "vector_store_test_pkey" PRIMARY KEY, btree (id)
Access method: heap

rag_blueprint=# \d+ vector_store_test
                                  Table "public.vector_store_test"
  Column   |    Type     | Collation | Nullable | Default | Storage  | Compression | Stats target | Description
-----------+-------------+-----------+----------+---------+----------+-------------+--------------+-------------
 id        | text        |           | not null |         | extended |             |              |
 document  | jsonb       |           |          |         | extended |             |              |
 embedding | vector(384) |           |          |         | external |             |              |
Indexes:
    "vector_store_test_pkey" PRIMARY KEY, btree (id)
Access method: heap

rag_blueprint=# SELECT COUNT(*) FROM vector_store_test;
 count
-------
   154
(1 row)