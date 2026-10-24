# Contributing to Footnote

Set up a development environment and get a change merged. Read the
[Code of Conduct](CODE_OF_CONDUCT.md) first.

## Prerequisites

- [uv](https://docs.astral.sh/uv/): installs Python 3.12 and the locked dependencies
- Docker with Compose: Postgres, the fake LLM and integration tests
- make and git
- for `make lint` and `make security`: shellcheck, hadolint, gitleaks, kubeconform and trivy
  (CI versions are pinned in `scripts/ci_install_tools.sh`)

## Setup

```bash
git clone https://github.com/thnhdatle/footnote.git
cd footnote
make bootstrap    # sync the environment and install the git hooks
```

## Makefile targets

Every target is idempotent and prints a one-line PASS/FAIL summary.

| Target | Does |
|---|---|
| `bootstrap` | toolchain, dependencies, hooks |
| `generate` | code generation |
| `fmt` | format all code |
| `lint` | ruff, mypy, import-linter, hadolint, shellcheck, gitleaks, kubeconform |
| `test` | unit tests |
| `test-integration` | integration tests (Docker) |
| `coverage` | merged coverage, threshold check |
| `fuzz` | fuzz targets for `FUZZTIME` (default 30s) |
| `examples` | run and check every example |
| `quickstart-test` | build the image, run the offline quickstart |
| `security` | pip-audit, trivy, license check |
| `build` | wheel and sdist |
| `docker` | local image, host architecture |
| `up` / `down` | compose stack up / down -v |
| `ci` | `lint test test-integration coverage examples security` |
| `release-check` | pre-release checks, before tagging |
| `release-dry` | `uv build` plus multi-arch `docker buildx build` |
| `eval` | small eval runs, about 10 minutes (answers need LM Studio) |

## Running tests

```sh
make test                                # unit: fast, no network, no Docker
make test-integration                    # integration: Docker, Testcontainers
```

## Commit messages

The commit-msg hook enforces one line only: no body, no footer, no trailers.

```text
<type>(<scope>): <summary>
```

- type: one of `feat, fix, perf, refactor, test, docs, build, ci, chore`
- scope: a package or area (`api`, `chunking`, `retrieval`, `deps`, ...)
- breaking change: `feat(api)!: rename field`
- summary: imperative, lowercase first letter, no trailing period
- whole line at most 72 characters

Validating regex:

```text
^(feat|fix|perf|refactor|test|docs|build|ci|chore)(\([a-z0-9-]+\))?!?: [a-z0-9].{0,70}[^.]$
```

Example: `fix(ingest): skip empty paragraphs in html parser`.

One logical change per commit. Every commit passes `make lint test`.

## Adding a dependency

Record this checklist in the PR:

- [ ] Maintained: a release or commit in the last 12 months.
- [ ] License is one of Apache-2.0, MIT, MIT-CMU, BSD-2/3, ISC, MPL-2.0, 0BSD,
      Zlib, PSF-2.0, Unlicense, CC0-1.0 (listed exception: psycopg 3, LGPL-3.0, in `licenses/exceptions.yaml`).
- [ ] No transitive GPL or AGPL dependency (`pip-licenses --allow-only`).
- [ ] `make security` is clean.

Model weights and datasets are not dependencies. Pin each in `models/manifest.yaml`
(revision, SHA-256, license, provenance). Never commit them. Vendored front-end assets follow
the same license list and are recorded in `NOTICE`.

## Proposing a design change

Open an issue first. Reference it in your PR.

## Pull request checklist

- [ ] Tests added or updated
- [ ] Docs updated
- [ ] `CHANGELOG.md` `[Unreleased]` entry
- [ ] `make lint test test-integration` passes
- [ ] No `TODO` without `(roadmap)` and a matching `ROADMAP.md` entry
- [ ] Every Python file starts with an SPDX header

## Questions

Use GitHub Discussions. For security issues see [SECURITY.md](SECURITY.md).
