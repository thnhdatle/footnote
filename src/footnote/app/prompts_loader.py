# SPDX-License-Identifier: Apache-2.0
"""Loads and renders the versioned prompt templates."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from string import Template

from footnote.domain.errors import ValidationError
from footnote.domain.ports import ChatMessage
from footnote.domain.retrieval import Hit

_SECTION = re.compile(r"^## (system|user)\s*$", re.MULTILINE)
_IMAGE_PROMPT = "describe_image_v1.txt"


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    system: str
    user: str

    def render(self, question: str, hits: Sequence[Hit]) -> list[ChatMessage]:
        blocks = []
        for n, hit in enumerate(hits, start=1):
            path = " > ".join(hit.heading_path)
            header = f"[{n}] {hit.title}" + (f" — {path}" if path else "")
            blocks.append(f"{header}\n{hit.text}")
        user = Template(self.user).safe_substitute(context="\n\n".join(blocks), question=question)
        return [ChatMessage("system", self.system), ChatMessage("user", user)]


def available_templates() -> list[str]:
    root = resources.files("footnote.app.prompts")
    return sorted(p.name[:-3] for p in root.iterdir() if p.name.endswith(".md"))


def load_image_prompt() -> str:
    """Loads the prompt that asks the vision model for an image description."""
    return (
        resources.files("footnote.app.prompts").joinpath(_IMAGE_PROMPT).read_text("utf-8").strip()
    )


def load_template(name: str) -> PromptTemplate:
    """Loads a shipped template by name."""
    if name not in available_templates():
        raise ValidationError("answer.prompt_template", f"unknown template {name!r}")
    text = resources.files("footnote.app.prompts").joinpath(f"{name}.md").read_text("utf-8")
    parts = _SECTION.split(text)
    sections = dict(zip(parts[1::2], (p.strip() for p in parts[2::2]), strict=True))
    if set(sections) != {"system", "user"}:
        raise ValidationError("answer.prompt_template", f"{name} needs system and user sections")
    return PromptTemplate(name, sections["system"], sections["user"])
