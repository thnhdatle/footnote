# SPDX-License-Identifier: Apache-2.0
"""Document parsers (pdf, docx, html, md, txt) producing domain `Element`s.

`SandboxedParser` implements `domain.ports.Parser`; `parse_bytes` is the in-process
dispatch used inside the sandbox child and by tests::

    parser = SandboxedParser(ParseLimits(max_pages=500))
    elements = await parser.parse(sniff(data, "report.pdf"), data)
"""

from footnote.adapters.parsers.errors import ParseError
from footnote.adapters.parsers.inspect import UploadChecks
from footnote.adapters.parsers.limits import ParseLimits, precheck
from footnote.adapters.parsers.media import sniff
from footnote.adapters.parsers.parse import parse_bytes
from footnote.adapters.parsers.sandbox import SandboxedParser, run_sandboxed

__all__ = [
    "ParseError",
    "ParseLimits",
    "SandboxedParser",
    "UploadChecks",
    "parse_bytes",
    "precheck",
    "run_sandboxed",
    "sniff",
]
