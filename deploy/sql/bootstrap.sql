-- Run once by a privileged user (superuser or the managed equivalent).
-- Passwords are placeholders: set real ones with ALTER ROLE or pass them via psql variables.
-- Usage: psql -v app_password="'...'" -v admin_password="'...'" -f bootstrap.sql
\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'footnote_admin') THEN
    CREATE ROLE footnote_admin LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'footnote_app') THEN
    CREATE ROLE footnote_app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
  END IF;
END
$$;

ALTER ROLE footnote_admin PASSWORD :admin_password;
ALTER ROLE footnote_app PASSWORD :app_password;

DO $$
BEGIN
  EXECUTE format('GRANT CREATE, CONNECT ON DATABASE %I TO footnote_admin', current_database());
  EXECUTE format('GRANT CONNECT ON DATABASE %I TO footnote_app', current_database());
END
$$;

CREATE SCHEMA IF NOT EXISTS footnote AUTHORIZATION footnote_admin;
GRANT USAGE ON SCHEMA footnote TO footnote_app;
GRANT USAGE ON SCHEMA public TO footnote_admin, footnote_app;
