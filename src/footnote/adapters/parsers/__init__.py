# SPDX-License-Identifier: Apache-2.0
"""Document parsers that turn uploads into domain elements."""

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
