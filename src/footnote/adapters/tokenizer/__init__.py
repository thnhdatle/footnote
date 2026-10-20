# SPDX-License-Identifier: Apache-2.0
"""Token counting with a Hugging Face `tokenizer.json` (implements `ports.TokenCounter`)
and the manifest-backed `TokenCounterFactory`.

Usage::

    counter = HfTokenCounter.from_file(Path("tokenizer.json"))
    n = counter.count("Footnote answers questions.")
    tokenizers = ManifestTokenizers(manifest, models_dir)
    tokenizers.for_tokenizer("bge-small-en-v1.5").count("text")
"""

from footnote.adapters.tokenizer.counter import HfTokenCounter
from footnote.adapters.tokenizer.factory import ManifestTokenizers, tokenizer_ids

__all__ = ["HfTokenCounter", "ManifestTokenizers", "tokenizer_ids"]
