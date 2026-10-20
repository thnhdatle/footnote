# Contributing to Footnote

Thanks for helping. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 and the locked dependencies)
- Docker with Compose (Postgres with pgvector, the fake LLM and integration tests)
- `make`, `git`
- for `make lint` and `make security`: `shellcheck`, `hadolint`, `gitleaks`, `kubeconform`
  and `trivy` (the versions CI uses are pinned in `scripts/ci_install_tools.sh`)

## Setup

```bash
git clone https://github.com/thnhdatle/footnote.git
cd footnote
make bootstrap
```

`make bootstrap` syncs the environment and installs the repository's git hooks.

## Makefile targets

Every target is idempotent and prints a one-line PASS/FAIL summary.

| Target | Does |
|---|---|
| `bootstrap` | install toolchain, dependencies and hooks path |
| `generate` | code generation |
| `fmt` | format all code |
| `lint` | ruff, mypy, import-linter, hadolint, shellcheck, gitleaks, kubeconform |
| `test` | unit tests |
| `test-integration` | integration tests against real dependencies (Docker) |
| `coverage` | merged coverage and threshold check |
| `mutation` | mutation testing (mutmut) with threshold |
| `acceptance` | acceptance runner; `CI=1` uses CI thresholds |
| `bench` | benchmarks, writes `bench/RESULTS.md` |
| `fuzz` | fuzz targets for `FUZZTIME` (default 30s) |
| `examples` | runs every example and checks its output |
| `quickstart-test` | builds the image and runs the README offline quickstart through the acceptance runner |
| `security` | pip-audit, trivy on the local image, license check |
| `build` | build wheel and sdist |
| `docker` | local image for the host architecture |
| `up` / `down` | compose stack up / down -v |
| `ci` | `lint test test-integration coverage examples security` |
| `release-check` | pre-release checks (run before tagging) |
| `release-dry` | `uv build` plus multi-arch `docker buildx build` |

## Running tests

- Unit: `make test` (fast, no network, no Docker).
- Integration: `make test-integration` (needs Docker; uses Testcontainers).
- Acceptance: the [acceptance suite](acceptance/README.md) checks Footnote end to end;
  `make acceptance`, a subset with `make acceptance ONLY=FNT-04,FNT-05`, and
  `make acceptance CI=1` for the CI thresholds. Dataset-heavy criteria run when
  `FOOTNOTE_NIGHTLY=1` is set or `CI` is unset.

## Commit messages

One line only: no body, no footer, no trailers.

```text
<type>(<scope>): <summary>
```

- `type` is one of `feat, fix, perf, refactor, test, docs, build, ci, chore`.
- `scope` is a package or area (`api`, `chunking`, `retrieval`, `deps`, ...).
- Breaking change: `feat(api)!: rename field`.
- `summary`: imperative, lowercase first letter, no trailing period, whole line
  at most 72 characters.

Validating regex:

```text
^(feat|fix|perf|refactor|test|docs|build|ci|chore)(\([a-z0-9-]+\))?!?: [a-z0-9].{0,70}[^.]$
```

with total length at most 72. Example: `fix(ingest): skip empty paragraphs in html parser`.

One logical change per commit; every commit passes `make lint test`.

## Adding a dependency

Record this checklist in the PR:

- [ ] Maintained: a release or commit in the last 12 months.
- [ ] License is one of Apache-2.0, MIT, MIT-CMU, BSD-2/3, ISC, MPL-2.0, 0BSD,
      Zlib, PSF-2.0, Unlicense, CC0-1.0 (listed exception: `psycopg` 3, LGPL-3.0, in `licenses/exceptions.yaml`).
- [ ] No transitive GPL or AGPL dependency (`pip-licenses --allow-only`).
- [ ] `make security` is clean.

Model weights and datasets are not dependencies: pin each in
`models/manifest.yaml` (revision and SHA-256, license, provenance) and never
commit them. Vendored front-end assets follow the same license list and are
recorded in `NOTICE`.

## Proposing a design change

Open an issue describing the problem and the proposed change before opening the PR,
and reference the issue in your PR.

## Pull request checklist

- [ ] Tests added or updated
- [ ] Docs updated
- [ ] `CHANGELOG.md` `[Unreleased]` entry
- [ ] `make lint test test-integration` passes
- [ ] No `TODO` without `(roadmap)` and a matching `ROADMAP.md` entry
- [ ] Every Python file starts with an SPDX header

## Questions

Use GitHub Discussions. For security issues see [SECURITY.md](SECURITY.md).
