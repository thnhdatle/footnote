# SPDX-License-Identifier: Apache-2.0
"""Run a parser in a spawned child process that is killed after `parse_timeout`.

The child gets no inherited state (spawn), no core dumps, no file creation, a CPU-time
limit and, on Linux, an address-space limit. Elements come back over a pipe.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import sys
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from footnote.adapters.parsers.errors import PARSE_FAILED, PARSE_TIMEOUT, ParseError
from footnote.adapters.parsers.limits import ParseLimits
from footnote.adapters.parsers.parse import parse_bytes
from footnote.domain.documents import Element

_MIB = 1024 * 1024
_CPU_GRACE_SECONDS = 5
_ADDRESS_SPACE = 4096 * _MIB

Runner = Callable[[str, bytes, ParseLimits], list[Element]]


def _apply_resource_limits(limits: ParseLimits) -> None:
    try:
        import resource  # noqa: PLC0415 (POSIX only)
    except ImportError:  # pragma: no cover
        return
    cpu = int(limits.parse_timeout) + _CPU_GRACE_SECONDS
    settings = [
        (resource.RLIMIT_CPU, cpu),
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_FSIZE, 0),
    ]
    if sys.platform == "linux":
        settings.append((resource.RLIMIT_AS, _ADDRESS_SPACE))
    for which, value in settings:
        _, hard = resource.getrlimit(which)
        cap = value if hard == resource.RLIM_INFINITY else min(value, hard)
        try:
            resource.setrlimit(which, (cap, cap))
        except (ValueError, OSError):  # pragma: no cover (platform refuses the limit)
            continue


def _child(
    conn: Connection, runner: Runner, media_type: str, data: bytes, limits: ParseLimits
) -> None:
    """Child entry point: parse and send `("ok", dicts)` or `("err", code, message)`."""
    try:
        _apply_resource_limits(limits)
        elements = runner(media_type, data, limits)
        conn.send(("ok", [e.to_dict() for e in elements]))
    except ParseError as exc:
        conn.send(("err", exc.code, exc.message))
    except BaseException as exc:
        conn.send(("err", PARSE_FAILED, f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def run_sandboxed(
    media_type: str,
    data: bytes,
    limits: ParseLimits,
    runner: Runner = parse_bytes,
) -> list[Element]:
    """Blocking sandboxed parse; raises `ParseError` (`parse_timeout`, `parse_failed`, ...)."""
    ctx = mp.get_context("spawn")
    parent, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_child, args=(child_conn, runner, media_type, data, limits), daemon=True
    )
    proc.start()
    child_conn.close()
    try:
        message = _receive(parent, limits.parse_timeout)
    finally:
        parent.close()
        if proc.is_alive():
            proc.kill()
        proc.join()
    if message is None:
        raise ParseError(f"parsing exceeded {limits.parse_timeout:g}s", code=PARSE_TIMEOUT)
    if message[0] == "err":
        raise ParseError(message[2], code=message[1])
    return [Element.from_dict(d) for d in message[1]]


def _receive(conn: Connection, timeout: float) -> tuple[Any, ...] | None:
    """The child's message, or None on timeout; a crashed child is `parse_failed`."""
    if not conn.poll(timeout):
        return None
    try:
        message: tuple[Any, ...] = conn.recv()
    except (EOFError, OSError) as exc:
        raise ParseError("parser process died unexpectedly") from exc
    return message


class SandboxedParser:
    """`domain.ports.Parser` adapter: every parse runs in a killable child process."""

    def __init__(self, limits: ParseLimits | None = None, runner: Runner = parse_bytes) -> None:
        self._limits = limits or ParseLimits()
        self._runner = runner

    async def parse(self, media_type: str, data: bytes) -> list[Element]:
        """Parse `data` in the sandbox without blocking the event loop."""
        return await asyncio.to_thread(run_sandboxed, media_type, data, self._limits, self._runner)
