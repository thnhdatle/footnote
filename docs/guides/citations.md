# Answers and citations

This guide explains how `POST /v1/collections/{id}/answers` produces an answer, how the
citations are verified, and when Footnote refuses to answer. The request and response
fields are in [api.md](api.md).

## 1. How an answer is made

1. Footnote searches the collection with the same pipeline as
   [retrieval.md](retrieval.md) and takes the top `k` chunks.
2. The evidence gate compares the best score of those chunks with a threshold. Below the
   threshold the answer is an abstention and the language model is never called.
3. Otherwise the chunks are numbered `1..k` in the prompt, each with its title and heading path, and the model is asked to cite
   them as `[1]`, `[2][3]`. The prompt template is `answer_v1`
   (`answer.prompt_template`).
4. The citation verifier cleans the model's text and builds the citation list.

A configured language model is required: with an empty `llm.base_url` the endpoint is
`503 llm_not_configured`. Search and the other endpoints work without one.

## 2. The evidence gate

The gate score depends on whether the reranker ran for the search:

- With reranking: the sigmoid of the best reranker logit, a value between 0 and 1.
- Without reranking (`rerank: false`, or `rerank.provider: noop`): the best cosine
  similarity among the returned chunks.

The threshold is the first value that is set, in this order:

1. the collection's `answer.evidence_threshold_rerank` or `evidence_threshold_vector`;
2. the configuration `answer.evidence_threshold_rerank` or `evidence_threshold_vector`;
3. the calibrated value stored with the reranker file in `models/manifest.yaml`
   (reranker gate) or with the embedding space (vector gate).

If none exists the request is `409 evidence_threshold_uncalibrated`. The shipped models
carry calibrated values in `models/manifest.yaml`, set on the SciFact train split for a
target answer rate of 0.85 ([evaluation.md](evaluation.md)). Treat them as a starting
point: measure on your own questions and set the collection value.

```sh
curl -s -X PATCH $FOOTNOTE_URL/v1/collections/$COLLECTION \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"answer": {"evidence_threshold_rerank": 0.5}}'
```

A higher threshold abstains more often and answers fewer weakly supported questions.
`evidence_score` in the response is the score the gate used.

## 3. Abstention

An abstention has `answer: null`, `abstained: true` and an `abstention_reason`:

| Reason | Meaning |
|---|---|
| `insufficient_evidence` | the gate score is below the threshold; no model call was made |
| `unsupported` | only with `answer.unsupported_as_abstain: true`: the model answered without a valid citation |

The prompt also tells the model to reply "I cannot find the answer in the provided
documents." when the passages do not contain the answer. That reply has no citation, so
it is reported as `unsupported`.

## 4. What the verifier does

The model's text is processed in this order.

1. A leading `<think>...</think>` block is removed.
2. Markers are read: `[n]` and `[n, m]`, where `n` is a passage number. A marker followed
   by `(`, such as a Markdown link `[1](url)`, and markers inside code spans or code blocks
   are ignored.
3. Markers that name a passage that does not exist (`[0]`, `[9]` with 8 passages) are
   removed. A marker group is rewritten to its valid numbers, repeated numbers collapse, and
   a doubled bracket such as `[[1]]` becomes `[1]`.
4. One citation is built for every distinct valid number, in order of first use.
5. For each citation, the verifier picks the quote: it splits the cited chunk into
   sentences and chooses the one with the highest word-overlap F1 score with the answer
   sentence that contains the marker. `quote_start` and `quote_end` are character offsets
   into the chunk text.

Every citation therefore points to a chunk retrieved for that answer, from the queried
collection's current document versions. Footnote verifies that the number is valid and picks
the closest passage text; it does not check that the passage entails the sentence.

## 5. Reading a citation

```json
{"n": 1, "chunk_id": "4f0a...", "document_id": "c91b...", "title": "handbook.pdf",
 "page_from": 14, "page_to": 14, "quote_start": 0, "quote_end": 54}
```

Fetch the full chunk and cut the quote:

```sh
curl -s -H "Authorization: Bearer $KEY" \
  $FOOTNOTE_URL/v1/collections/$COLLECTION/chunks/4f0a... | jq -r .text
```

`page_from` and `page_to` are null for formats without pages (everything except PDF). The
chunk endpoint also returns `heading_path` and the document title for display. Offsets
count Unicode characters in the chunk's `text`.

If the document has a newer version by the time you read the chunk, the old chunk can be
`404 chunk_not_found`. Citations are a record of one answer, so store the quote text with
the answer when you need it later.

## 6. Unsupported answers

If the model produced text with no valid marker, `unsupported` is `true`, `citations` is
empty and `answer` still holds the text (unless `unsupported_as_abstain` is set). Show it
as unverified or discard it. The metric `footnote_answers_total{outcome="unsupported"}`
tracks how often this happens; a rise usually means a model or prompt problem.

## 7. Streaming

With `"stream": true` the events are `retrieval`, `delta`, `citations`, `done` (or `error`).
Markers can be split across `delta` events, so show `delta` text only as a preview and
replace it with `citations.answer` when that event arrives. The event shapes are in
[api.md](api.md).

## 8. Tuning answers

| Goal | Change |
|---|---|
| fewer wrong answers | raise the evidence threshold; raise `k` so the model sees more evidence |
| fewer abstentions | lower the threshold; check retrieval first with `debug: true` |
| more consistent text | keep `llm.temperature` at `0` (the default) |
| a model that wraps text in reasoning tags | use `llm.extra_body` to switch reasoning off; leading `<think>` blocks are removed anyway |
| a different prompt | only templates shipped in the package can be selected (`answer.prompt_template`; today `answer_v1`); a new one is a code change in `src/footnote/app/prompts/` |

Prompt and completion text is logged only with `log.capture_content: true`.
