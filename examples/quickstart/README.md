# Quickstart: offline smoke test

Check the full pipeline offline: upload `handbook.pdf`, ask a question, get a verified
citation. The LLM is a stub with a canned answer, so this checks the pipeline, not answer
quality.

Needs Docker Compose v2, `curl` and `jq`.

```sh
make docker                    # build the image
examples/quickstart/run.sh     # add RAYLD_FAKELLM_IMAGE=thnhdatle/fakellm:dev if you built it locally
make down                      # stop the stack
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

| File | Purpose |
|---|---|
| `config.yaml` | sets the `X-Fake-Answer` header for fakellm |
| `compose.yaml` | mounts the config |
| `make_sample_pdf.py` | regenerates `handbook.pdf` |
| `check.sh` | offline check run by `make examples` |
