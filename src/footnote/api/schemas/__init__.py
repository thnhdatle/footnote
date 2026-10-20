# SPDX-License-Identifier: Apache-2.0
"""Request and response models of the HTTP API (pydantic v2).

Request models forbid unknown fields. Setting bounds are enforced by the domain
(`footnote.domain.settings`), so the API and the CLI validate identically; the models here
only fix types and shapes.
"""
