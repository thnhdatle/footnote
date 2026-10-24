# SPDX-License-Identifier: Apache-2.0

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TENANT_TABLES = (
    "api_keys",
    "collections",
    "documents",
    "document_versions",
    "document_blobs",
    "chunks",
    "version_chunks",
    "chunk_vectors",
    "jobs",
    "webhook_deliveries",
    "tenant_usage",
)

APP_DML_TABLES = tuple(t for t in TENANT_TABLES if t != "api_keys")

TABLES = """
CREATE TABLE footnote.tenants (
  id uuid PRIMARY KEY,
  name text NOT NULL CHECK (length(name) BETWEEN 1 AND 256),
  quotas jsonb NOT NULL DEFAULT '{}'::jsonb,
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deleting')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE footnote.api_keys (
  tenant_id uuid NOT NULL REFERENCES footnote.tenants (id) ON DELETE CASCADE,
  id uuid NOT NULL,
  key_hash bytea NOT NULL CHECK (length(key_hash) = 32),
  prefix text NOT NULL,
  name text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  last_used_at timestamptz,
  revoked_at timestamptz,
  expires_at timestamptz,
  PRIMARY KEY (tenant_id, id),
  CONSTRAINT api_keys_key_hash_key UNIQUE (key_hash)
);

CREATE TABLE footnote.embedding_spaces (
  id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name text NOT NULL CONSTRAINT embedding_spaces_name_key UNIQUE
    CHECK (name ~ '^[a-z0-9][a-z0-9._-]{0,62}$'),
  provider text NOT NULL,
  model text NOT NULL,
  dimensions integer NOT NULL CHECK (dimensions BETWEEN 1 AND 4000),
  send_dimensions boolean NOT NULL DEFAULT false,
  normalise boolean NOT NULL DEFAULT true,
  index_type text NOT NULL CHECK (index_type IN ('vector', 'halfvec')),
  calibrated_threshold double precision
    CHECK (calibrated_threshold IS NULL OR calibrated_threshold BETWEEN 0 AND 1),
  created_at timestamptz NOT NULL DEFAULT now(),
  CHECK ((index_type = 'vector') = (dimensions <= 2000))
);

CREATE TABLE footnote.collections (
  tenant_id uuid NOT NULL REFERENCES footnote.tenants (id) ON DELETE CASCADE,
  id uuid NOT NULL,
  name text NOT NULL CHECK (length(name) BETWEEN 1 AND 256),
  settings jsonb NOT NULL DEFAULT '{}'::jsonb,
  language text NOT NULL DEFAULT 'simple' CHECK (language IN ('simple', 'english')),
  active_space_id integer NOT NULL REFERENCES footnote.embedding_spaces (id),
  target_space_id integer REFERENCES footnote.embedding_spaces (id),
  chunk_count bigint NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
  reembed jsonb,
  webhook_url text,
  webhook_secret text,
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deleting')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, id),
  CONSTRAINT collections_tenant_name_key UNIQUE (tenant_id, name)
);

CREATE TABLE footnote.documents (
  tenant_id uuid NOT NULL,
  id uuid NOT NULL,
  collection_id uuid NOT NULL,
  external_id text NOT NULL CHECK (length(external_id) BETWEEN 1 AND 512),
  title text NOT NULL DEFAULT '',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
  status text NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued', 'parsing', 'embedding', 'ready', 'failed')),
  error jsonb,
  current_version_id uuid,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, id),
  CONSTRAINT documents_tenant_collection_external_key
    UNIQUE (tenant_id, collection_id, external_id),
  FOREIGN KEY (tenant_id, collection_id)
    REFERENCES footnote.collections (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE footnote.document_versions (
  tenant_id uuid NOT NULL,
  id uuid NOT NULL,
  document_id uuid NOT NULL,
  version integer NOT NULL CHECK (version >= 1),
  source_sha256 bytea NOT NULL CHECK (length(source_sha256) = 32),
  media_type text NOT NULL,
  bytes bigint NOT NULL CHECK (bytes >= 0),
  pages integer,
  indexed_title text NOT NULL DEFAULT '',
  chunking_hash text NOT NULL,
  elements bytea,
  status text NOT NULL DEFAULT 'processing'
    CHECK (status IN ('processing', 'ready', 'failed', 'superseded')),
  error jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, id),
  CONSTRAINT document_versions_tenant_document_version_key
    UNIQUE (tenant_id, document_id, version),
  FOREIGN KEY (tenant_id, document_id)
    REFERENCES footnote.documents (tenant_id, id) ON DELETE CASCADE
);

ALTER TABLE footnote.documents
  ADD CONSTRAINT documents_current_version_fkey
  FOREIGN KEY (tenant_id, current_version_id)
  REFERENCES footnote.document_versions (tenant_id, id)
  DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE footnote.document_blobs (
  tenant_id uuid NOT NULL,
  version_id uuid NOT NULL,
  data bytea NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, version_id),
  FOREIGN KEY (tenant_id, version_id)
    REFERENCES footnote.document_versions (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE footnote.chunks (
  tenant_id uuid NOT NULL,
  id uuid NOT NULL,
  collection_id uuid NOT NULL,
  document_id uuid NOT NULL,
  content_hash text NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
  text text NOT NULL,
  heading_path text[] NOT NULL DEFAULT '{}',
  title text NOT NULL DEFAULT '',
  embedded_text text NOT NULL,
  token_count integer NOT NULL CHECK (token_count >= 0),
  fts_config regconfig NOT NULL,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector(fts_config, embedded_text)) STORED,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, id),
  CONSTRAINT chunks_tenant_document_hash_key UNIQUE (tenant_id, document_id, content_hash),
  FOREIGN KEY (tenant_id, collection_id)
    REFERENCES footnote.collections (tenant_id, id) ON DELETE CASCADE,
  FOREIGN KEY (tenant_id, document_id)
    REFERENCES footnote.documents (tenant_id, id) ON DELETE CASCADE
);
CREATE INDEX chunks_tenant_collection_idx ON footnote.chunks (tenant_id, collection_id);
CREATE INDEX chunks_tsv_idx ON footnote.chunks USING gin (tsv);

CREATE TABLE footnote.version_chunks (
  tenant_id uuid NOT NULL,
  version_id uuid NOT NULL,
  ordinal integer NOT NULL CHECK (ordinal >= 0),
  chunk_id uuid NOT NULL,
  page_from integer,
  page_to integer,
  PRIMARY KEY (tenant_id, version_id, ordinal),
  FOREIGN KEY (tenant_id, version_id)
    REFERENCES footnote.document_versions (tenant_id, id) ON DELETE CASCADE,
  -- Deferred: in a cascade (document/collection/tenant delete) the chunk rows can be
  -- removed before the memberships that reference them.
  FOREIGN KEY (tenant_id, chunk_id) REFERENCES footnote.chunks (tenant_id, id)
    DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX version_chunks_tenant_chunk_idx ON footnote.version_chunks (tenant_id, chunk_id);

CREATE TABLE footnote.chunk_vectors (
  tenant_id uuid NOT NULL,
  chunk_id uuid NOT NULL,
  space_id integer NOT NULL REFERENCES footnote.embedding_spaces (id) ON DELETE CASCADE,
  collection_id uuid NOT NULL,
  embedding vector NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, chunk_id, space_id),
  FOREIGN KEY (tenant_id, chunk_id)
    REFERENCES footnote.chunks (tenant_id, id) ON DELETE CASCADE,
  FOREIGN KEY (tenant_id, collection_id)
    REFERENCES footnote.collections (tenant_id, id) ON DELETE CASCADE
);
CREATE INDEX chunk_vectors_tenant_collection_space_idx
  ON footnote.chunk_vectors (tenant_id, collection_id, space_id);

CREATE TABLE footnote.jobs (
  tenant_id uuid NOT NULL,
  id uuid NOT NULL,
  kind text NOT NULL CHECK (kind IN ('ingest', 'reindex', 'fts_reconfig', 'reembed', 'gc',
                                     'webhook', 'collection_delete', 'tenant_delete')),
  args jsonb NOT NULL DEFAULT '{}'::jsonb,
  status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
  attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
  last_error text,
  progress jsonb,
  run_at timestamptz NOT NULL DEFAULT now(),
  lease_owner text,
  lease_until timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, id)
);
CREATE INDEX jobs_status_run_at_idx ON footnote.jobs (status, run_at);

CREATE TABLE footnote.webhook_deliveries (
  tenant_id uuid NOT NULL,
  id uuid NOT NULL,
  collection_id uuid NOT NULL,
  event text NOT NULL,
  url text NOT NULL,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'delivered', 'failed')),
  attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
  response_code integer,
  error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  delivered_at timestamptz,
  PRIMARY KEY (tenant_id, id),
  FOREIGN KEY (tenant_id, collection_id)
    REFERENCES footnote.collections (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE footnote.tenant_usage (
  tenant_id uuid NOT NULL REFERENCES footnote.tenants (id) ON DELETE CASCADE,
  day date NOT NULL,
  upload_bytes bigint NOT NULL DEFAULT 0 CHECK (upload_bytes >= 0),
  PRIMARY KEY (tenant_id, day)
);
"""

POLICY_TEMPLATE = """
ALTER TABLE footnote.{table} ENABLE ROW LEVEL SECURITY;
ALTER TABLE footnote.{table} FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON footnote.{table} TO footnote_app
  USING (tenant_id = NULLIF(current_setting('footnote.tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = NULLIF(current_setting('footnote.tenant_id', true), '')::uuid);
CREATE POLICY admin_all ON footnote.{table} TO footnote_admin USING (true) WITH CHECK (true);
"""

TENANTS_POLICY = """
ALTER TABLE footnote.tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE footnote.tenants FORCE ROW LEVEL SECURITY;
CREATE POLICY admin_all ON footnote.tenants TO footnote_admin USING (true) WITH CHECK (true);
"""

RESOLVE_KEY = """
CREATE FUNCTION footnote.footnote_resolve_key(key_hash bytea)
RETURNS TABLE (tenant_id uuid, key_id uuid)
LANGUAGE sql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $fn$
  WITH found AS (
    SELECT k.tenant_id, k.id, k.last_used_at
    FROM footnote.api_keys AS k
    JOIN footnote.tenants AS t ON t.id = k.tenant_id
    WHERE k.key_hash = footnote_resolve_key.key_hash
      AND k.revoked_at IS NULL
      AND (k.expires_at IS NULL OR k.expires_at > pg_catalog.now())
      AND t.status = 'active'
  ),
  touched AS (
    UPDATE footnote.api_keys AS u
    SET last_used_at = pg_catalog.now()
    WHERE (u.tenant_id, u.id) IN (
      SELECT l.tenant_id, l.id
      FROM footnote.api_keys AS l
      JOIN found AS f ON f.tenant_id = l.tenant_id AND f.id = l.id
      WHERE l.last_used_at IS NULL
         OR l.last_used_at < pg_catalog.now() - interval '1 minute'
      FOR UPDATE OF l SKIP LOCKED
    )
    RETURNING u.id
  )
  SELECT f.tenant_id, f.id FROM found AS f
$fn$;
REVOKE ALL ON FUNCTION footnote.footnote_resolve_key(bytea) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION footnote.footnote_resolve_key(bytea) TO footnote_app;
"""

REGISTER_SPACE = """
CREATE FUNCTION footnote.footnote_register_space(
  name text,
  provider text,
  model text,
  dimensions integer,
  send_dimensions boolean,
  normalise boolean,
  calibrated_threshold double precision,
  hnsw_m integer DEFAULT 16,
  hnsw_ef_construction integer DEFAULT 64
)
RETURNS integer
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $fn$
DECLARE
  v_id integer;
  v_dims integer;
  v_type text;
  v_ext_schema name;
BEGIN
  IF name IS NULL OR name !~ '^[a-z0-9][a-z0-9._-]{0,62}$' THEN
    RAISE EXCEPTION 'validation_failed: name must match ^[a-z0-9][a-z0-9._-]{0,62}$'
      USING ERRCODE = '22023';
  END IF;
  IF dimensions IS NULL OR dimensions < 1 THEN
    RAISE EXCEPTION 'validation_failed: dimensions must be >= 1' USING ERRCODE = '22023';
  END IF;
  IF dimensions > 4000 THEN
    RAISE EXCEPTION 'embedding_dimension_unsupported: % exceeds 4000', dimensions
      USING ERRCODE = '22023';
  END IF;
  IF hnsw_m IS NULL OR hnsw_m NOT BETWEEN 2 AND 100
     OR hnsw_ef_construction IS NULL OR hnsw_ef_construction NOT BETWEEN 4 AND 1000 THEN
    RAISE EXCEPTION 'validation_failed: hnsw parameters out of range' USING ERRCODE = '22023';
  END IF;

  -- Serialise registrations (several replicas register the configured spaces at startup).
  PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('footnote_register_space'));

  SELECT s.id, s.dimensions, s.index_type INTO v_id, v_dims, v_type
  FROM footnote.embedding_spaces AS s
  WHERE s.name = footnote_register_space.name;

  IF v_id IS NULL THEN
    v_type := CASE WHEN dimensions <= 2000 THEN 'vector' ELSE 'halfvec' END;
    v_dims := dimensions;
    INSERT INTO footnote.embedding_spaces
      (name, provider, model, dimensions, send_dimensions, normalise, index_type,
       calibrated_threshold)
    VALUES
      (footnote_register_space.name, footnote_register_space.provider,
       footnote_register_space.model, footnote_register_space.dimensions,
       footnote_register_space.send_dimensions, footnote_register_space.normalise,
       v_type, footnote_register_space.calibrated_threshold)
    RETURNING id INTO v_id;
  ELSIF calibrated_threshold IS NOT NULL THEN
    UPDATE footnote.embedding_spaces AS s
    SET calibrated_threshold = footnote_register_space.calibrated_threshold
    WHERE s.id = v_id;
  END IF;

  SELECT n.nspname INTO v_ext_schema
  FROM pg_catalog.pg_extension AS e
  JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
  WHERE e.extname = 'vector';

  EXECUTE pg_catalog.format(
    'CREATE INDEX IF NOT EXISTS %I ON footnote.chunk_vectors USING hnsw '
    '((embedding::%I.%I(%s)) %I.%I) WITH (m = %s, ef_construction = %s) WHERE space_id = %s',
    'chunk_vectors_hnsw_s' || v_id, v_ext_schema, v_type, v_dims,
    v_ext_schema, v_type || '_cosine_ops', hnsw_m, hnsw_ef_construction, v_id);
  RETURN v_id;
END
$fn$;
REVOKE ALL ON FUNCTION footnote.footnote_register_space(
  text, text, text, integer, boolean, boolean, double precision, integer, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION footnote.footnote_register_space(
  text, text, text, integer, boolean, boolean, double precision, integer, integer)
  TO footnote_admin;
"""

CLAIM_JOBS = """
CREATE FUNCTION footnote.footnote_claim_jobs(worker text, n integer, lease interval)
RETURNS TABLE (job_id uuid, tenant_id uuid, kind text)
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $fn$
BEGIN
  IF worker IS NULL OR length(worker) NOT BETWEEN 1 AND 200 THEN
    RAISE EXCEPTION 'validation_failed: worker must be 1-200 characters' USING ERRCODE = '22023';
  END IF;
  IF n IS NULL OR n NOT BETWEEN 1 AND 1000 THEN
    RAISE EXCEPTION 'validation_failed: n must be in [1, 1000]' USING ERRCODE = '22023';
  END IF;
  IF lease IS NULL OR lease <= interval '0' THEN
    RAISE EXCEPTION 'validation_failed: lease must be positive' USING ERRCODE = '22023';
  END IF;
  RETURN QUERY
  WITH picked AS (
    SELECT j.tenant_id, j.id
    FROM footnote.jobs AS j
    WHERE (j.status = 'queued' AND j.run_at <= pg_catalog.now())
       OR (j.status = 'running' AND j.lease_until < pg_catalog.now())
    ORDER BY j.run_at, j.id
    LIMIT n
    FOR UPDATE OF j SKIP LOCKED
  )
  UPDATE footnote.jobs AS u
  SET status = 'running',
      lease_owner = worker,
      lease_until = pg_catalog.now() + lease,
      attempts = u.attempts + 1,
      started_at = COALESCE(u.started_at, pg_catalog.now()),
      updated_at = pg_catalog.now()
  FROM picked AS p
  WHERE u.tenant_id = p.tenant_id AND u.id = p.id
  RETURNING u.id, u.tenant_id, u.kind;
END
$fn$;
REVOKE ALL ON FUNCTION footnote.footnote_claim_jobs(text, integer, interval) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION footnote.footnote_claim_jobs(text, integer, interval) TO footnote_app;
"""


def _grants() -> str:
    dml = ", ".join(f"footnote.{t}" for t in APP_DML_TABLES)
    return f"""
REVOKE ALL ON footnote.tenants, footnote.api_keys FROM footnote_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON {dml} TO footnote_app;
GRANT SELECT ON footnote.embedding_spaces, footnote.alembic_version TO footnote_app;
"""


def _execute(script: str) -> None:
    op.get_bind().connection.driver_connection.execute(script)


def upgrade() -> None:
    _execute(TABLES)
    _execute(TENANTS_POLICY)
    for table in TENANT_TABLES:
        _execute(POLICY_TEMPLATE.format(table=table))
    _execute(RESOLVE_KEY)
    _execute(REGISTER_SPACE)
    _execute(CLAIM_JOBS)
    _execute(_grants())
