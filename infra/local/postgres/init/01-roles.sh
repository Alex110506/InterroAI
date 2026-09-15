#!/bin/bash
# Runs once, when Postgres starts on an empty data volume.
#
# Creates the login the Cloud API and the worker connect as. It is deliberately
# not the superuser: superusers bypass row-level security, which would quietly
# turn the tenant-isolation policies into decoration. Migrations run as the
# owner (POSTGRES_USER) and grant this role what it needs.
#
# Also creates the database the integration tests migrate and truncate, so a
# test run never touches development data.
set -euo pipefail

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" \
     -v app_password="$INTERROAI_APP_DB_PASSWORD" <<'SQL'
CREATE ROLE interroai_app LOGIN PASSWORD :'app_password';
CREATE DATABASE interroai_test;
SQL
