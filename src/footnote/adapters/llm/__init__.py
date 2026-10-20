# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat client implementing `domain.ports.LLM`.

::

    llm = OpenAICompatLLM(LLMConfig(base_url="http://localhost:1234/v1", model="m"), client)
    result = await llm.chat([ChatMessage("user", "hi")])
"""

from footnote.adapters.llm.client import LLMConfig, OpenAICompatLLM

__all__ = ["LLMConfig", "OpenAICompatLLM"]
