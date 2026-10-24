# SPDX-License-Identifier: Apache-2.0
"""Embedders for OpenAI-compatible HTTP providers and local ONNX models."""

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
