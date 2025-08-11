# Dockerfile for Hive Metastore
FROM apache/hive:4.0.0

USER root

# Download and install the PostgreSQL JDBC driver
ADD https://jdbc.postgresql.org/download/postgresql-42.7.7.jar /opt/hive/lib/
RUN apt-get update && apt-get install -y postgresql-client

COPY entrypoint.sh /
RUN chmod 644 /opt/hive/lib/postgresql-42.7.7.jar && chmod +x /entrypoint.sh

USER hive

ENTRYPOINT [ "/entrypoint.sh" ]