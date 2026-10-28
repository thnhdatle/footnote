# SPDX-License-Identifier: Apache-2.0
# Footnote image: uv-managed python-build-standalone 3.12 and the
# project venv copied into distroless/cc (the distroless Python image ships 3.11).
# Runs as 65532 with a read-only root filesystem; models live on the /models volume.

FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM debian:12-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
# hadolint ignore=DL3008
RUN uv python install 3.12
# Dependencies first so the layer is cached until the lockfile changes.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --python 3.12
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --python 3.12 \
    && mkdir -p /models \
    && chown 65532:65532 /models
# distroless/cc has no zlib; numpy and Pillow (pdfplumber) link libz.so.1 from the system.
RUN triplet="$(uname -m)-linux-gnu" \
    && mkdir -p "/syslibs/usr/lib/${triplet}" \
    && cp -a /usr/lib/"${triplet}"/libz.so.1* "/syslibs/usr/lib/${triplet}/"

FROM gcr.io/distroless/cc-debian12:nonroot@sha256:9dac0a79194e45a7da0158a9c6da57b217585af0786db3845d1f0ec1a0dd182f
COPY --from=build /syslibs/ /
COPY --from=build /opt/python /opt/python
COPY --from=build /app/.venv /app/.venv
COPY --from=build --chown=65532:65532 /models /models
WORKDIR /app
COPY alembic.ini ./
COPY migrations ./migrations
COPY models/manifest.yaml ./models/manifest.yaml
# Build metadata for /v1/version and footnote_build_info (`--build-arg FOOTNOTE_BUILD_COMMIT=…`).
ARG FOOTNOTE_BUILD_COMMIT=unknown
ARG FOOTNOTE_BUILD_DATE=unknown
ENV PATH=/app/.venv/bin:$PATH \
    FOOTNOTE_BUILD_COMMIT=$FOOTNOTE_BUILD_COMMIT \
    FOOTNOTE_BUILD_DATE=$FOOTNOTE_BUILD_DATE \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FOOTNOTE_MODELS_DIR=/models
VOLUME ["/models"]
EXPOSE 8000 9094
USER 65532:65532
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD ["footnote", "healthcheck"]
ENTRYPOINT ["footnote"]
CMD ["serve"]
