# SPDX-License-Identifier: Apache-2.0
"""Upload inspection by content sniffing and limit checks."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from footnote.adapters.parsers.limits import ParseLimits, check_docx, check_pdf
from footnote.adapters.parsers.media import DOCX, PDF, sniff
from footnote.domain.documents import InspectedFile


class UploadChecks:
    """Detects the media type of an upload and enforces its limits."""

    def __init__(self, limits: ParseLimits) -> None:
        self._limits = limits

    def check(self, data: bytes, filename: str) -> InspectedFile:
        media_type = sniff(data, filename)
        pages = None
        if media_type == PDF:
            pages = check_pdf(data, self._limits)
        elif media_type == DOCX:
            check_docx(data, self._limits)
        return InspectedFile(media_type, pages)

    async def inspect(self, data: bytes, filename: str) -> InspectedFile:
        """Returns the media type and page count of an upload."""
        return await asyncio.to_thread(self.check, data, filename)


if TYPE_CHECKING:
    from footnote.domain import ports

    def _conforms_uploadinspector() -> ports.UploadInspector:
        return UploadChecks(ParseLimits())
