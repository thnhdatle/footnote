# Webhooks

A collection can notify your service when a document finishes indexing or fails. This guide
covers setup, the payload, signature verification, retries and the network policy. The
fields of the collection `webhook` setting are in [api.md](api.md).

## 1. Events

| Event | Sent when |
|---|---|
| `document.ready` | a version of a document became the current version and is searchable |
| `document.failed` | ingestion failed for good, after any retries ([ingestion.md](ingestion.md)) |

A version that was superseded by a newer one before it finished does not send
`document.ready`. By default a webhook gets both events; set `events` to subscribe to one.

## 2. Configure a webhook

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/collections/$COLLECTION \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"webhook": {"url": "https://example.com/hooks/footnote",
                   "events": ["document.ready", "document.failed"]}}'
```

The response contains `webhook_secret`. It is shown once, in the response that created or
replaced it. Store it. Later reads show only `webhook.has_secret`.

- A changed URL gets a new secret.
- `{"webhook": {"rotate_secret": true}}` replaces the secret and returns the new one. The
  collection must already have a URL. The old secret stops working at once, so update your
  receiver first or accept both for a short time.
- `{"webhook": {"url": null}}` removes the webhook.

## 3. The delivery

Footnote sends a `POST` with `Content-Type: application/json` and this body:

```json
{"event": "document.ready", "collection_id": "7a3f...", "document_id": "c91b...",
 "version": 2, "status": "ready", "error": null, "occurred_at": "2026-10-20T09:00:00Z"}
```

For `document.failed`, `status` is `failed` and `error` is `{"code": "parse_failed",
"message": "..."}` with a code from [errors.md](errors.md). The payload does not contain
document text or metadata; fetch the document with the API if you need more.

Headers:

| Header | Value |
|---|---|
| `X-Footnote-Timestamp` | Unix time in seconds when the attempt was made |
| `X-Footnote-Signature` | `sha256=` followed by the hex HMAC-SHA256 of `"<timestamp>.<body>"` with your secret |
| `User-Agent` | `footnote-webhook/1` |

Answer with any `2xx` status within `webhooks.timeout` (default `5s`). Redirects are not
followed. Footnote reads at most 1 KiB of your response.

## 4. Verify the signature

Verify the signature over the raw body bytes before parsing the JSON, and reject old
timestamps to stop replays.

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

To check a signature by hand:

```sh
printf '%s.%s' "$TIMESTAMP" "$BODY" | openssl dgst -sha256 -hmac "$SECRET"
```

## 5. Retries and delivery log

Deliveries are background jobs. A failed attempt (connection error, timeout, a non-2xx
status) is retried with jittered exponential backoff, up to `webhooks.max_attempts` (default
5) attempts. A policy failure (section 6) is not retried.

Delivery is at least once. Use `event`, `document_id` and `version` as an idempotency key in
your receiver.

Every attempt is recorded:

```sh
curl -s -H "Authorization: Bearer $KEY" \
  "$FOOTNOTE_URL/v1/collections/$COLLECTION/webhook-deliveries?limit=20"
```

Each record has `event`, `url`, `status` (`pending` while retries remain, `delivered`,
`failed`), `attempts`, `response_code`, `error` and timestamps, newest first. The metric
`footnote_webhook_deliveries_total{status}` counts attempts by status.

If the collection is deleted or its webhook removed before a queued delivery runs, the
delivery is dropped.

## 6. Network policy

Webhooks are an outbound request to a URL a tenant chooses, so Footnote restricts them. The
policy is checked when the URL is set and again at every delivery.

- The scheme must be `https`, unless `webhooks.allow_http` is true.
- The URL must have a host, no credentials and no fragment, and be at most 2048 characters.
- The host is resolved, and every address must be public. Loopback, private (RFC 1918),
  CGNAT (`100.64.0.0/10`), link-local (including the cloud metadata address
  `169.254.169.254`), unique local IPv6, multicast and unspecified addresses, and IPv4-mapped
  forms of these, are rejected unless `webhooks.allow_private` is true.
- The connection goes to the address that was checked, with the host name sent as `Host`
  and TLS server name, so DNS cannot change between the check and the connection.

A rejected URL is `422 webhook_url_rejected` when you set it. For a receiver on a private
network, such as another container, set `webhooks.allow_private: true` and, without TLS,
`webhooks.allow_http: true`. Do both only on a trusted network, because every tenant can
then point webhooks at internal addresses.

## 7. Testing a receiver

1. Run a receiver on a public HTTPS URL, or use `allow_http` and `allow_private` locally.
2. Set the webhook on a collection and upload a small document.
3. Look at `webhook-deliveries` for the status and `response_code`.

See [troubleshooting.md](troubleshooting.md) if deliveries stay `pending` or end `failed`.
