# Multi-tenancy

Footnote serves many tenants from one deployment and one database. This guide explains
the isolation model and how to manage tenants, keys and quotas. The admin routes are in
[api.md](api.md); the database roles are in [operations.md](operations.md).

## 1. The model

- A tenant owns collections, documents, chunks, vectors, jobs and webhook settings.
- A tenant API key (`fn_...`) resolves to exactly one tenant. Every tenant request runs
  with that tenant fixed for the transaction.
- Embedding spaces are global: they are registered by an operator and shared by all tenants
  ([models-and-data.md](models-and-data.md)).
- The admin key is separate. It manages tenants, keys, quotas and spaces, and can export a tenant's
  data. It cannot call tenant routes.

## 2. How isolation is enforced

Isolation is enforced in the database, not only in application code.

- Every tenant table has row-level security enabled and forced. The policy for the
  application role `footnote_app` allows only rows whose `tenant_id` equals the tenant set
  for the current transaction. If no tenant is set, the role sees no rows.
- `footnote_app` is not a superuser, does not have `BYPASSRLS`, owns no objects and is not a
  member of `footnote_admin`. It cannot read the `tenants` and `api_keys` tables at all.
- Keys are stored as SHA-256 hashes and looked up by a `SECURITY DEFINER` function that
  ignores revoked and expired keys.
- Foreign keys include `tenant_id`, so a row cannot reference another tenant's row.
- Repositories also filter by `tenant_id` in SQL, as a second layer.
- `/readyz` fails with an `app_role` problem if the application role is a superuser, has
  `BYPASSRLS`, owns objects in schema `footnote` or is a member of `footnote_admin`.

Consequences for clients:

- An id that belongs to another tenant is `404`, never `403`, so ids cannot be probed.
- A tenant key on `/v1/admin` and the admin key on tenant routes are both `401`.

What this does not cover: SQL injection executed as `footnote_app` could set the tenant
itself, so Footnote uses parameterised SQL only. A leaked admin key or admin database URL
exposes every tenant. Keep both secret ([configuration.md](configuration.md)).

## 3. Create a tenant

With the admin key:

```sh
curl -s -X POST $FOOTNOTE_URL/v1/admin/tenants \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name": "acme", "quotas": {"max_documents": 5000}}'
```

or on the host with database access:

```sh
footnote tenants create acme
footnote tenants list --limit 50
```

CLI commands read `FOOTNOTE_ADMIN_DATABASE_URL` and print one JSON object per line. Omitted
quotas take their value from `tenancy.default_quotas`.

## 4. Quotas

| Quota | Enforced |
|---|---|
| `max_collections` | on `POST /v1/collections`: `429 quota_exceeded` |
| `max_documents` | on upload of a new document: `429 quota_exceeded` |
| `max_upload_bytes_per_day` | on upload, summed per day: `429 quota_exceeded` |
| `max_chunks` | while indexing: the document fails with `quota_exceeded` |

`0` means unlimited, which is the default. Change quotas at any time:

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/admin/tenants/$TENANT \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"quotas": {"max_documents": 1000}}'
```

`GET /v1/admin/tenants/{id}/usage` returns `collections`, `documents`, `chunks`, `vectors`
and `stored_bytes`. Metrics carry no tenant or collection labels, so use this endpoint for
per-tenant accounting.

## 5. API keys

```sh
footnote keys create $TENANT --name ci
footnote keys create $TENANT --name temp --expires-at 2027-01-01T00:00:00+00:00
footnote keys list $TENANT
footnote keys revoke $TENANT $KEY_ID
footnote keys rotate $TENANT $KEY_ID
```

- The plaintext key is printed once, when it is created. Only its hash and a short
  `prefix` (for recognising it in lists) are stored.
- `revoke` takes effect at once.
- `rotate` issues a replacement. The old key keeps working for `auth.key_rotation_grace`
  (default `24h`), so you can deploy the new key without downtime. Revoke the old key early
  if it leaked.
- `expires_at` must be in the future and carry a time zone.
- `last_used_at` is updated at most once a minute.

The same operations exist under `/v1/admin/tenants/{id}/keys`.

## 6. Export and deletion

```sh
footnote tenants export $TENANT -o acme.ndjson
footnote tenants export $TENANT --include-vectors --include-originals -o acme-full.ndjson
```

The export is NDJSON with one `{"type", "data"}` record per line: collections, documents,
versions, elements, chunks and memberships. Webhook secrets are never exported. Vectors and
originals (base64) are optional and make the file much larger.

```sh
footnote tenants delete $TENANT
```

Deletion revokes every key of the tenant immediately and queues a `tenant_delete` job that
removes its data, keys and the tenant row. It returns a `job_id`. The job runs in a worker
over the admin database connection. Deletion cannot be undone; export first if you need a
copy.

## 7. Tenant deployment patterns

- One Footnote deployment with many tenants is the supported model. Tenants share the same
  Postgres instance, embedding spaces and worker capacity.
- Per-tenant throttling is not built in. Put rate limiting in front of the data plane if
  tenants share it.
- For stronger separation, such as separate encryption or backup policy per customer, run
  separate databases and separate Footnote deployments.
