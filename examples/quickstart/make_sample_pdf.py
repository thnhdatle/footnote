# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGE_BREAK_BEFORE = 2
SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Welcome to Acme Robotics",
        [
            "Acme Robotics builds small warehouse robots. This handbook explains how we work "
            "together: working hours, time off, expenses and security.",
            "Questions about this handbook go to the People team at people@acme.example.",
        ],
    ),
    (
        "Vacation policy",
        [
            "Full-time employees receive 25 paid vacation days per calendar year. The days are "
            "granted on 1 January. Employees who join during the year receive two vacation "
            "days for every full month of employment remaining in that year.",
            "Up to 5 unused vacation days carry over to the next year and expire on 31 March. "
            "Vacation requests are approved by the team lead at least two weeks before the "
            "first day of leave.",
        ],
    ),
    (
        "Working hours and remote work",
        [
            "Core hours are 10:00 to 15:00 local time. Employees may work remotely up to three "
            "days per week after agreeing the schedule with their team lead.",
        ],
    ),
    (
        "Expenses",
        [
            "Submit expense reports within 30 days of the purchase. Travel must be booked "
            "through the company portal. Receipts are required for every item above 25 euros.",
        ],
    ),
    (
        "Information security",
        [
            "Lock your laptop when you leave your desk. Use the company password manager and "
            "enable two-factor authentication on every work account. Report lost devices to "
            "security@acme.example within one hour.",
        ],
    ),
]


def build(path: Path) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title="Acme Robotics Employee Handbook",
        author="Footnote quickstart",
        invariant=1,
    )
    story: list[object] = [Paragraph("Acme Robotics Employee Handbook", styles["Title"])]
    for i, (heading, paragraphs) in enumerate(SECTIONS):
        if i == PAGE_BREAK_BEFORE:
            story.append(PageBreak())
        story.append(Paragraph(heading, styles["Heading1"]))
        story.extend(Paragraph(text, styles["BodyText"]) for text in paragraphs)
        story.append(Spacer(1, 12))
    doc.build(story)  # type: ignore[arg-type]


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("handbook.pdf")
    build(target)
    print(target)
