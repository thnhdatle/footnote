# Multi-tenancy

Manage tenants, keys and quotas on one shared deployment.

## 1. The model

- A tenant owns its collections, documents, jobs and webhooks.
- A tenant key (`fn_...`) belongs to one tenant.
- Embedding spaces are shared ([models-and-data.md](models-and-data.md)).
- The admin key manages tenants, keys, quotas and spaces. It cannot call tenant routes.

The database enforces isolation. Another tenant's id returns `404`.

A leaked admin key or admin database URL exposes every tenant. Keep both secret.

## 2. Create a tenant

```sh
curl -s -X POST $FOOTNOTE_URL/v1/admin/tenants \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name": "acme", "quotas": {"max_documents": 5000}}'    # over the API

footnote tenants create acme          # or on the host
footnote tenants list --limit 50
```

CLI commands use `FOOTNOTE_ADMIN_DATABASE_URL`.

## 3. Quotas

| Quota | When exceeded |
|---|---|
| `max_collections` | `429 quota_exceeded` on create |
| `max_documents` | `429 quota_exceeded` on upload |
| `max_upload_bytes_per_day` | `429 quota_exceeded` on upload |
| `max_chunks` | the document fails with `quota_exceeded` |

`0` means unlimited, the default (`tenancy.default_quotas`).

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/admin/tenants/$TENANT \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"quotas": {"max_documents": 1000}}'          # change a quota

curl -s -H "Authorization: Bearer $ADMIN_KEY" \
  $FOOTNOTE_URL/v1/admin/tenants/$TENANT/usage      # usage per tenant
```

## 4. API keys

```sh
footnote keys create $TENANT --name ci                                        # new key
footnote keys create $TENANT --name temp --expires-at 2027-01-01T00:00:00+00:00
footnote keys list $TENANT
footnote keys revoke $TENANT $KEY_ID                                          # at once
footnote keys rotate $TENANT $KEY_ID                                          # replacement
```

The key is printed once. After `rotate` the old key works for `auth.key_rotation_grace`
(default `24h`). Revoke it early if it leaked.

## 5. Export and deletion

```sh
footnote tenants export $TENANT -o acme.ndjson                                # data
footnote tenants export $TENANT --include-vectors --include-originals -o acme-full.ndjson
footnote tenants delete $TENANT                                               # cannot be undone
```

Deletion revokes every key at once and deletes the data in a job. Export first if you need
a copy.

## 6. Deployment patterns

- Tenants share the database, spaces and workers.
- There is no per-tenant rate limiting. Add one in front of the API.
- For separate encryption or backups per customer, run separate deployments.
