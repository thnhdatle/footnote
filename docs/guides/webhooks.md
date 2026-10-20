# Webhooks

Get notified when a document finishes indexing or fails.

## 1. Events

| Event | Sent when |
|---|---|
| `document.ready` | a new version is searchable |
| `document.failed` | ingestion failed after retries ([ingestion.md](ingestion.md)) |

## 2. Configure a webhook

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/collections/$COLLECTION \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"webhook": {"url": "https://example.com/hooks/footnote",
                   "events": ["document.ready", "document.failed"]}}'
```

The response shows `webhook_secret` once. Store it. Both events are sent by default.

| Body | Effect |
|---|---|
| `{"webhook": {"rotate_secret": true}}` | new secret; the old one stops at once |
| `{"webhook": {"url": null}}` | remove the webhook |

A changed URL also gets a new secret.

## 3. The delivery

```json
{"event": "document.ready", "collection_id": "7a3f...", "document_id": "c91b...",
 "version": 2, "status": "ready", "error": null, "occurred_at": "2026-10-20T09:00:00Z"}
```

On failure, `error` holds a `code` ([errors.md](errors.md)) and a `message`.

| Header | Value |
|---|---|
| `X-Footnote-Timestamp` | Unix time in seconds |
| `X-Footnote-Signature` | `sha256=` + hex HMAC-SHA256 of `"<timestamp>.<body>"` |
| `User-Agent` | `footnote-webhook/1` |

Reply `2xx` within `webhooks.timeout` (default `5s`). Redirects are not followed.

## 4. Verify the signature

Verify the raw body before parsing it. Reject old timestamps.

```python
import hashlib
import hmac
import time


def verify(secret: bytes, timestamp: str, body: bytes, header: str) -> bool:
    if abs(time.time() - int(timestamp)) > 300:
        return False
    mac = hmac.new(secret, timestamp.encode() + b"." + body, hashlib.sha256)
    return hmac.compare_digest("sha256=" + mac.hexdigest(), header)
```

## 5. Retries and delivery log

```sh
curl -s -H "Authorization: Bearer $KEY" \
  "$FOOTNOTE_URL/v1/collections/$COLLECTION/webhook-deliveries?limit=20"    # newest first
```

Failed attempts retry up to `webhooks.max_attempts` (default 5). Delivery is at least
once: deduplicate on `event`, `document_id` and `version`.

## 6. Network policy

- URLs must be `https`, unless `webhooks.allow_http` is true.
- Addresses must be public, unless `webhooks.allow_private` is true.

Other URLs return `422 webhook_url_rejected`. Allow private or HTTP targets only on a
trusted network: every tenant can then reach internal addresses.

See [troubleshooting.md](troubleshooting.md) if deliveries stay `pending` or fail.
