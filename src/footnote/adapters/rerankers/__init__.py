# SPDX-License-Identifier: Apache-2.0
"""Rerankers implementing `domain.ports.Reranker`: ONNX cross-encoder and noop.

Usage::

    reranker = OnnxReranker.from_manifest(models_dir, entry, variant="int8")
    logits = await reranker.score("query", ["passage one", "passage two"])
"""

from footnote.adapters.rerankers.noop import NoopReranker
from footnote.adapters.rerankers.onnx_cross_encoder import LazyOnnxReranker, OnnxReranker

__all__ = ["LazyOnnxReranker", "NoopReranker", "OnnxReranker"]
