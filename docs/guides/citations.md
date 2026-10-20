# Answers and citations

Get cited answers, and control when Footnote abstains.

Request and response fields are in [api.md](api.md).

## 1. How an answer is made

Footnote searches the collection and checks the evidence. Weak evidence means an
abstention, with no model call. Otherwise the model answers with markers like `[1]`, and
Footnote checks each one.

Answers need `llm.base_url`. Without it, `/answers` returns `503 llm_not_configured`.

## 2. The evidence gate

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/collections/$COLLECTION \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"answer": {"evidence_threshold_rerank": 0.5}}'    # this collection's threshold
```

With reranking the gate uses `evidence_threshold_rerank`; without it,
`evidence_threshold_vector`. The first value set wins:

1. the collection setting;
2. the configuration (`answer.*`);
3. the calibrated value of the reranker or embedding space.

With none, the request returns `409 evidence_threshold_uncalibrated`. The shipped models
are calibrated ([evaluation.md](evaluation.md)). Tune the threshold on your own questions:
higher abstains more often. `evidence_score` shows the score used.

## 3. Abstention

An abstention has `answer: null` and `abstained: true`.

| `abstention_reason` | Meaning |
|---|---|
| `insufficient_evidence` | score below the threshold |
| `unsupported` | no valid citation, with `answer.unsupported_as_abstain: true` |

## 4. Reading a citation

```json
{"n": 1, "chunk_id": "4f0a...", "document_id": "c91b...", "title": "handbook.pdf",
 "page_from": 14, "page_to": 14, "quote_start": 0, "quote_end": 54}
```

```sh
curl -s -H "Authorization: Bearer $KEY" \
  $FOOTNOTE_URL/v1/collections/$COLLECTION/chunks/4f0a... | jq -r .text    # full chunk
```

- `quote_start` and `quote_end` are character offsets into the chunk text.
- Pages are null except for PDF.
- Every citation points to a retrieved chunk. Footnote does not check that it proves the
  sentence.
- Old chunks can return `404 chunk_not_found` after a new version. Store the quote if you
  need it later.

## 5. Unsupported answers

An answer without a valid marker has `unsupported: true` and no citations. Show it as
unverified, or drop it.

## 6. Streaming

Show `delta` text only as a preview. Replace it with `citations.answer` when that event
arrives ([api.md](api.md)).

## 7. Tuning answers

| Goal | Change |
|---|---|
| fewer wrong answers | raise the threshold or `k` |
| fewer abstentions | lower the threshold; check retrieval with `debug: true` |
| consistent text | keep `llm.temperature` at `0` (default) |
| no reasoning tags | turn reasoning off in `llm.extra_body` |
| another prompt | shipped templates only (`answer.prompt_template`, today `answer_v1`) |

Prompts are logged only with `log.capture_content: true`.
