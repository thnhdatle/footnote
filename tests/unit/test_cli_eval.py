# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from footnote.__main__ import app
from footnote.app.evaluation.cache import CachingEmbedder, mark_warm

runner = CliRunner()


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.startswith("FOOTNOTE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("FOOTNOTE_EVAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FOOTNOTE_EVAL_DATA_DIR", str(tmp_path / "data"))


def test_status_cold_exits_one_only_when_required(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "cache", "status"])
    assert result.exit_code == 0
    assert "scifact\tcold\tdata absent" in result.stdout
    required = runner.invoke(app, ["eval", "cache", "status", "--require", "scifact"])
    assert required.exit_code == 1


def test_status_warm_with_data(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    CachingEmbedder(_NoInner(), cache, "k").close()  # type: ignore[arg-type]
    mark_warm(cache, "scifact", "k", 1)
    (tmp_path / "data" / "scifact").mkdir(parents=True)
    (tmp_path / "data" / "scifact" / "corpus.jsonl").write_text("")
    result = runner.invoke(app, ["eval", "cache", "status", "--require", "scifact"])
    assert result.exit_code == 0
    assert "scifact\twarm\tdata present" in result.stdout
    assert "trec-covid\tcold\tdata absent" in result.stdout


class _NoInner:
    pass


def test_run_id_is_utc_stamp_and_commit() -> None:
    from datetime import UTC, datetime  # noqa: PLC0415

    from footnote.cli_eval import new_run_id, short_commit  # noqa: PLC0415

    assert new_run_id("abc1234", datetime(2026, 10, 20, 8, 41, 0, tzinfo=UTC)) == (
        "20261020T084100Z-abc1234"
    )
    assert short_commit()


def test_retrieval_without_eval_extra_exits_two(monkeypatch: pytest.MonkeyPatch) -> None:
    from footnote.app.evaluation import bm25  # noqa: PLC0415

    def missing() -> None:
        raise bm25.MissingExtraError(f"bm25s baseline unavailable: {bm25.EXTRA_HINT}")

    monkeypatch.setattr(bm25, "require_extra", missing)
    result = runner.invoke(app, ["eval", "retrieval", "--dataset", "scifact"])
    assert result.exit_code == 2
    assert "eval extra" in result.stderr


def test_retrieval_with_missing_dataset_exits_one(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "retrieval", "--dataset", "nope"])
    assert result.exit_code == 1
    assert "dataset 'nope' not found" in result.stderr
