# superset/Dockerfile
FROM apache/superset:3.1.0

# Switch to root to install packages
USER root

# Install database drivers and file format support
RUN pip install \
    trino[sqlalchemy] \
    sqlalchemy-trino \
    pyhive[trino] \
    psycopg2-binary \
    pymongo \
    redis \
    s3fs \
    pandas \
    pyarrow \
    boto3

# Switch back to superset user
USER superset