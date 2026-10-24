# SPDX-License-Identifier: Apache-2.0
"""Embedded PDF images, rendered as PNG for the vision model."""

from __future__ import annotations

import io
from typing import Any

PNG = "image/png"
_MAX_SIDE_PX = 1568
_POINTS_PER_INCH = 72

type Box = tuple[float, float, float, float]


def page_images(page: Any, limit: int, min_px: int) -> list[tuple[float, bytes]]:
    """Renders up to `limit` images of a page that are at least `min_px` on each side."""
    out: list[tuple[float, bytes]] = []
    for image in sorted(page.images, key=lambda i: float(i["top"])):
        if len(out) >= limit:
            break
        try:
            rendered = _render(page, image, min_px)
        except Exception:  # noqa: S112 - a broken image must not lose the text of the page
            continue
        if rendered is not None:
            out.append(rendered)
    return out


def _render(page: Any, image: dict[str, Any], min_px: int) -> tuple[float, bytes] | None:
    width, height = (int(n) for n in image["srcsize"])
    box = _clip(image, page.bbox)
    if min(width, height) < min_px or box is None:
        return None
    native = width / float(image["width"])
    scale = min(native, _MAX_SIDE_PX / max(box[2] - box[0], box[3] - box[1]))
    picture = page.crop(box).to_image(resolution=_POINTS_PER_INCH * scale).original
    buf = io.BytesIO()
    picture.save(buf, format="PNG")
    return box[1], buf.getvalue()


def _clip(image: dict[str, Any], bbox: Box) -> Box | None:
    x0 = max(float(image["x0"]), bbox[0])
    top = max(float(image["top"]), bbox[1])
    x1 = min(float(image["x1"]), bbox[2])
    bottom = min(float(image["bottom"]), bbox[3])
    return (x0, top, x1, bottom) if x0 < x1 and top < bottom else None
