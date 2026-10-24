# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from footnote.adapters.llm import LLMConfig, OpenAICompatLLM, OpenAICompatVision
from footnote.domain.documents import PageImage
from footnote.domain.errors import UpstreamError

BASE_URL = "http://vision.test/v1"
IMAGE = PageImage(0, 14, "image/png", b"\x89PNG chart")


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
async def vision() -> AsyncIterator[OpenAICompatVision]:
    config = LLMConfig(base_url=BASE_URL, model="vl-model", api_key="k", max_tokens=300)
    async with httpx.AsyncClient() as client:
        yield OpenAICompatVision(OpenAICompatLLM(config, client, sleep=_no_sleep))


def _reply(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"model": "vl-model", "choices": [{"message": {"content": content}}]}
    )


@respx.mock
async def test_describe_sends_the_image_as_a_data_url(vision: OpenAICompatVision) -> None:
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=_reply("  A bar chart of revenue by year.\n")
    )

    text = await vision.describe(IMAGE, "Describe the image.")

    assert text == "A bar chart of revenue by year."
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer k"
    body = json.loads(request.content)
    assert (body["model"], body["max_tokens"]) == ("vl-model", 300)
    [message] = body["messages"]
    assert message["role"] == "user"
    assert message["content"] == [
        {"type": "text", "text": "Describe the image."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(IMAGE.data).decode()},
        },
    ]


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(httpx.Response(500), id="server-error"),
        pytest.param(httpx.ReadTimeout("slow"), id="timeout"),
        pytest.param(_reply("   "), id="empty"),
    ],
)
@respx.mock
async def test_describe_failures_are_upstream_errors(
    vision: OpenAICompatVision, outcome: httpx.Response | Exception
) -> None:
    respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=[outcome] * 3)

    with pytest.raises(UpstreamError):
        await vision.describe(IMAGE, "Describe the image.")
