# SPDX-License-Identifier: Apache-2.0
"""Image descriptions from an OpenAI-compatible chat endpoint with image input."""

from __future__ import annotations

import base64

from footnote.adapters.llm.client import OpenAICompatLLM
from footnote.domain.documents import PageImage
from footnote.domain.errors import UpstreamError


class OpenAICompatVision:
    """Describes images with a chat model that accepts `image_url` parts."""

    def __init__(self, llm: OpenAICompatLLM) -> None:
        self._llm = llm

    async def describe(self, image: PageImage, prompt: str) -> str:
        encoded = base64.b64encode(image.data).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image.media_type};base64,{encoded}"},
            },
        ]
        result = await self._llm.complete([{"role": "user", "content": content}])
        text = result.text.strip()
        if not text:
            raise UpstreamError("vision model returned an empty description")
        return text
