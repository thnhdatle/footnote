# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from footnote.app import prompts_loader
from footnote.app.prompts_loader import available_templates, load_template
from footnote.domain.documents import ChunkId, DocumentId
from footnote.domain.errors import ValidationError
from footnote.domain.retrieval import Hit


def _hit(n: int, title: str, path: tuple[str, ...], text: str) -> Hit:
    return Hit(ChunkId(UUID(int=n)), DocumentId(UUID(int=n)), title, path, text, text, None, None)


def test_available_templates_contains_answer_v1() -> None:
    assert "answer_v1" in available_templates()
    assert available_templates() == sorted(available_templates())


def test_load_template_has_system_and_user_sections() -> None:
    template = load_template("answer_v1")
    assert template.name == "answer_v1"
    assert "numbered context passages" in template.system
    assert "$context" in template.user
    assert "$question" in template.user


def test_render_numbers_hits_with_title_and_heading_path() -> None:
    hits = [
        _hit(1, "Guide", ("A", "B"), "first text"),
        _hit(2, "Notes", (), "second text"),
    ]
    messages = load_template("answer_v1").render("What is it?", hits)
    assert [m.role for m in messages] == ["system", "user"]
    user = messages[1].content
    assert "[1] Guide — A > B\nfirst text" in user
    assert "[2] Notes\nsecond text" in user
    assert user.index("[1]") < user.index("[2]")
    assert user.rstrip().endswith("Question: What is it?")


def test_render_does_not_expand_placeholders_inside_question_or_text() -> None:
    hits = [_hit(1, "T", (), "cost is $question")]
    user = load_template("answer_v1").render("$context?", hits)[1].content
    assert "cost is $question" in user
    assert "Question: $context?" in user


def test_load_template_unknown_name_rejected() -> None:
    with pytest.raises(ValidationError) as info:
        load_template("nope")
    assert info.value.field == "answer.prompt_template"


def test_load_template_missing_section_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "broken.md").write_text("## system\nonly a system part\n", encoding="utf-8")
    monkeypatch.setattr(prompts_loader.resources, "files", lambda _pkg: tmp_path)
    assert available_templates() == ["broken"]
    with pytest.raises(ValidationError, match="needs system and user sections"):
        load_template("broken")
