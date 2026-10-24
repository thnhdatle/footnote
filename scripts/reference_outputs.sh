#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Generates the ONNX parity reference files in tests/reference/.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

manifest=models/manifest.yaml
export HF_HOME="${HOME}/.cache/rayld/footnote/hf"
mkdir -p "$HF_HOME" tests/reference
EMB_REVISION="$(yq '.models[] | select(.id == "bge-small-en-v1.5") | .revision' "$manifest")"
RER_REVISION="$(yq '.models[] | select(.id == "ms-marco-MiniLM-L6-v2") | .revision' "$manifest")"
export EMB_REVISION RER_REVISION
export REF_DIR="tests/reference"

uv run --no-project --isolated --python 3.12 \
  --with sentence-transformers --with torch --with pyyaml \
  python - <<'PY'
import json
import os
from importlib.metadata import version

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

out = os.environ["REF_DIR"]
generator = f"sentence-transformers {version('sentence-transformers')}, torch {version('torch')}"


def rounded(values):
    return [round(float(v), 7) for v in values]


long_text = " ".join(
    f"Sentence {i} explains how retrieval augmented generation grounds answers in cited evidence."
    for i in range(1, 51)
)
long_text = " ".join((long_text + " ") * 3 for _ in range(1)).split(" ")
long_text = " ".join(long_text[:600])

texts = [
    "Footnote answers questions with citations.",
    "Hi",
    "What is the refund policy for annual plans?",
    "Represent this sentence for searching relevant passages: how do I rotate an API key?",
    "The quick brown fox jumps over the lazy dog.",
    "Rotate keys every 90 days and revoke the old key after the grace period ends.",
    "Café résumé naïve façade — Zürich, São Paulo, Kraków.",
    "日本語のテキストも埋め込めますか？これはテストです。",
    "Привет, мир! Это проверка многоязычного текста.",
    "emoji test \U0001F680 \U0001F4DA done",
    "SELECT * FROM documents WHERE tenant_id = 'a' AND deleted_at IS NULL;",
    "def add(a: int, b: int) -> int:\n    return a + b",
    "A  line\twith   odd   whitespace\nand a newline.",
    "Invoices are issued on the first business day of each month and are payable within 30 days.",
    long_text,
    "Who wrote the report?",
]

emb_rev = os.environ["EMB_REVISION"]
emb = SentenceTransformer("BAAI/bge-small-en-v1.5", revision=emb_rev, device="cpu")
pooling = emb[1].get_config_dict()["pooling_mode"]
assert pooling == "cls", pooling
assert emb.max_seq_length == 512, emb.max_seq_length
vectors = emb.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
with open(f"{out}/bge-small-en-v1.5.json", "w", encoding="utf-8") as fh:
    json.dump(
        {
            "model_id": "bge-small-en-v1.5",
            "repo": "BAAI/bge-small-en-v1.5",
            "revision": emb_rev,
            "texts": texts,
            "embeddings": [rounded(v) for v in vectors],
            "normalised": True,
            "pooling": pooling,
            "max_tokens": 512,
            "generator": generator,
        },
        fh,
        ensure_ascii=False,
    )
    fh.write("\n")

pairs = [
    ["how many people live in berlin", "Berlin had a population of 3,520,031 registered inhabitants in 2019."],
    ["how many people live in berlin", "Berlin is well known for its museums and nightlife."],
    ["what is the refund policy", "Annual plans can be refunded in full within 14 days of purchase."],
    ["what is the refund policy", "Our office is closed on public holidays."],
    ["how do I rotate an API key", "Rotate keys every 90 days and revoke the old key after the grace period ends."],
    ["how do I rotate an API key", "The quick brown fox jumps over the lazy dog."],
    ["capital of france", "Paris is the capital and most populous city of France."],
    ["capital of france", "Lyon is a city in France known for its cuisine."],
    ["capital of france", "Python is a programming language."],
    ["résumé tips", "Café résumé naïve façade — Zürich, São Paulo, Kraków."],
    ["日本の首都は", "東京は日本の首都であり、世界最大の都市圏の一つです。"],
    ["tenant isolation", "SELECT * FROM documents WHERE tenant_id = 'a' AND deleted_at IS NULL;"],
    ["when are invoices due", "Invoices are issued on the first business day of each month and are payable within 30 days."],
    ["when are invoices due", "Shipping takes three to five business days."],
    ["retrieval augmented generation", long_text],
    ["x", "y"],
]
rer_rev = os.environ["RER_REVISION"]
ce = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    revision=rer_rev,
    max_length=256,
    device="cpu",
    activation_fn=torch.nn.Identity(),
)
ce.model.double()
logits = ce.predict([tuple(p) for p in pairs], convert_to_numpy=True)
with open(f"{out}/ms-marco-MiniLM-L6-v2.json", "w", encoding="utf-8") as fh:
    json.dump(
        {
            "model_id": "ms-marco-MiniLM-L6-v2",
            "repo": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "revision": rer_rev,
            "pairs": pairs,
            "logits": rounded(logits.reshape(-1)),
            "max_tokens": 256,
            "reference_dtype": "float64",
            "generator": generator,
        },
        fh,
        ensure_ascii=False,
    )
    fh.write("\n")
print("pooling:", pooling, "|", generator)
PY
