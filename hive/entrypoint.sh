#!/bin/bash
set -e

POSTGRES_JAR=/opt/hive/lib/postgresql-42.7.7.jar
export HADOOP_CLASSPATH=$HADOOP_CLASSPATH:$POSTGRES_JAR

echo "PostgreSQL JDBC driver present:"
ls -l "$POSTGRES_JAR"

echo "Checking if Hive metastore schema exists..."
# Check if one of the core tables exists to decide if schema init is needed:
echo "TABLE_EXISTS is: '$TABLE_EXISTS'"
TABLE_EXISTS=$(psql "postgresql://superset:superset@superset-postgres:5432/superset" -tAc "SELECT to_regclass('public.\"VERSION\"');")

if [ "$TABLE_EXISTS" = "version" ]; then
  echo "Hive metastore schema already exists, skipping initialization."
else
  echo "Hive metastore schema not found. Initializing..."
  /opt/hive/bin/schematool -initSchema -dbType postgres
fi

echo "Starting Hive Metastore..."
/opt/hive/bin/hive --service metastore
