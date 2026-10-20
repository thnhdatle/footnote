# SPDX-License-Identifier: Apache-2.0
"""Embedders implementing `domain.ports.Embedder`: OpenAI-compatible HTTP and local ONNX.

Usage::

    embedder = OpenAICompatEmbedder(OpenAICompatSettings("http://lm:1234/v1", "m", 1024))
    vectors = await embedder.embed_documents(["a", "b"])
"""

from footnote.adapters.embedders.factory import DefaultEmbedderFactory, OpenAICompatDefaults
from footnote.adapters.embedders.onnx_local import OnnxEmbedder
from footnote.adapters.embedders.openai_compat import OpenAICompatEmbedder, OpenAICompatSettings

__all__ = [
    "DefaultEmbedderFactory",
    "OnnxEmbedder",
    "OpenAICompatDefaults",
    "OpenAICompatEmbedder",
    "OpenAICompatSettings",
]
