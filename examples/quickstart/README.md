# Quickstart: offline smoke test

Brings up Postgres, Footnote and `fakellm`, uploads `handbook.pdf`, asks a question and
checks that the answer carries a verified citation. The LLM is a stub that returns a canned
answer, so this checks the pipeline, not answer quality.

Needs Docker Compose v2, `curl` and `jq`.

```sh
make docker
examples/quickstart/run.sh     # add RAYLD_FAKELLM_IMAGE=thnhdatle/fakellm:dev if you built it locally
make down
```

Expected output (abridged):

```text
== start the stack
== create a tenant and a key
== create a collection (shipped thresholds)
== upload the sample PDF
job <id>: done
== ask: How many vacation days do full-time employees get per year?
{"answer": "Offline smoke test (canned answer) citing [1].", "abstained": false, "citations": [{"n": 1, ...}]}
```

Files: `config.yaml` (points `llm.base_url` at fakellm and sets the `X-Fake-Answer` header),
`compose.yaml` (mounts it), `make_sample_pdf.py` (regenerates `handbook.pdf`), `check.sh`
(offline check run by `make examples`).
