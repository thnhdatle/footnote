# SPDX-License-Identifier: Apache-2.0
"""`UploadInspector` adapter: content sniffing plus the synchronous upload checks.

Runs `sniff` and `precheck` (pypdf page count / encryption, DOCX archive limits) in a
worker thread so a large PDF does not block the event loop.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from footnote.adapters.parsers.limits import ParseLimits, check_docx, check_pdf
from footnote.adapters.parsers.media import DOCX, PDF, sniff
from footnote.domain.documents import InspectedFile


class UploadChecks:
    """Sniffs the media type from content and enforces page and archive limits."""

    def __init__(self, limits: ParseLimits) -> None:
        self._limits = limits

    def check(self, data: bytes, filename: str) -> InspectedFile:
        """Synchronous form of `inspect`."""
        media_type = sniff(data, filename)
        pages = None
        if media_type == PDF:
            pages = check_pdf(data, self._limits)
        elif media_type == DOCX:
            check_docx(data, self._limits)
        return InspectedFile(media_type, pages)

    async def inspect(self, data: bytes, filename: str) -> InspectedFile:
        """Media type and page count; raises ParseError with the upload error codes."""
        return await asyncio.to_thread(self.check, data, filename)


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_uploadinspector() -> ports.UploadInspector:
        return UploadChecks(ParseLimits())
