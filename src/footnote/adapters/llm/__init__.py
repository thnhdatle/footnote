# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat and vision clients."""

from footnote.adapters.llm.client import LLMConfig, OpenAICompatLLM
from footnote.adapters.llm.vision import OpenAICompatVision

__all__ = ["LLMConfig", "OpenAICompatLLM", "OpenAICompatVision"]
