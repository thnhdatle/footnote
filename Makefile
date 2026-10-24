# SPDX-License-Identifier: Apache-2.0
# Footnote Makefile. Every target prints a one-line PASS/FAIL summary.
SHELL := /usr/bin/env bash
STEP := scripts/step.sh
UV := uv run --frozen
# Tests import the optional `eval` extra (bm25s), so they run with every extra installed.
UV_TEST := uv run --frozen --all-extras
FUZZTIME ?= 30
IMAGE ?= thnhdatle/footnote:dev
JUDGE_MODEL ?= gemma-4-26b-a4b-it-qat-mlx
COMPOSE := docker compose -f deploy/docker-compose.yml -p footnote
export RAYLD_FOOTNOTE_IMAGE ?= $(IMAGE)

.PHONY: bootstrap generate fmt lint test test-integration coverage fuzz \
	examples quickstart-test security build docker up down ci release-check release-dry eval eval-data

bootstrap:
	@$(STEP) bootstrap scripts/bootstrap.sh

generate:
	@$(STEP) generate true

fmt:
	@$(STEP) fmt bash -c '$(UV) ruff check --fix . && $(UV) ruff format .'

lint:
	@$(STEP) lint bash -c '$(UV) ruff format --check . && $(UV) ruff check . && $(UV) mypy \
		&& $(UV) lint-imports && shellcheck scripts/*.sh .githooks/* \
		&& { [ ! -f Dockerfile ] || hadolint Dockerfile; } \
		&& gitleaks detect --no-banner --redact \
		&& { [ ! -d deploy/kubernetes ] || kubeconform -strict -summary deploy/kubernetes; }'

test:
	@$(STEP) test bash -c '$(UV_TEST) pytest tests/unit -n auto -q -p no:cacheprovider \
		-m "not integration"'

test-integration:
	@$(STEP) test-integration bash -c '$(UV_TEST) pytest tests/integration -m integration -q -p no:cacheprovider'

coverage:
	@$(STEP) coverage bash -c 'rm -f .coverage .coverage.*; $(UV_TEST) coverage run --parallel-mode -m pytest tests/unit tests/integration \
		-q -p no:cacheprovider && $(UV) coverage combine -q && scripts/coverage_check.sh'

fuzz:
	@$(STEP) fuzz bash -c 'HYPOTHESIS_PROFILE=fuzz FUZZTIME=$(FUZZTIME) $(UV) pytest tests/unit -m "not integration" -q -k fuzz'

examples:
	@$(STEP) examples scripts/examples.sh

quickstart-test:
	@$(STEP) quickstart-test scripts/quickstart_test.sh

security:
	@$(STEP) security scripts/security.sh

build:
	@$(STEP) build uv build

docker:
	@$(STEP) docker docker build -t $(IMAGE) .

up:
	@$(STEP) up $(COMPOSE) --profile fakellm up -d --wait

down:
	@$(STEP) down $(COMPOSE) --profile fakellm down -v --remove-orphans

ci:
	@$(MAKE) --no-print-directory lint test test-integration coverage security

release-check:
	@$(STEP) release-check scripts/release_check.sh

release-dry:
	@$(STEP) release-dry scripts/release_dry.sh

eval:
	@$(STEP) eval bash -c '$(UV) footnote eval retrieval --dataset scifact --quick \
		&& $(UV) footnote eval abstention --dataset scifact \
		&& $(UV) footnote eval answers --dataset scifact --limit 10 --judge-model $(JUDGE_MODEL)'

eval-data:
	@$(STEP) eval-data scripts/fetch_eval_data.sh
