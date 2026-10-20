# SPDX-License-Identifier: Apache-2.0
"""HTTP API (FastAPI): data plane, admin API and the admin listener.

Contract with the composition root (`bootstrap.py`): build one `ApiDeps` (use cases,
`JobQueue`, `KeyResolver`, `Metrics`, logger, `ApiSettings`, readiness checks), then serve
`create_app(deps)` on `server.listen` and `create_admin_app(deps)` on `admin.listen`.
Neither factory opens connections or starts tasks. `api` never imports `adapters`
(import-linter); adapter behaviour reaches the API only through ports and use cases.

    app = create_app(deps)
    admin = create_admin_app(deps)
"""

from footnote.api.admin_listener import create_admin_app
from footnote.api.app import create_app
from footnote.api.deps import ApiDeps, ApiSettings, ReadinessCheck, VersionInfo

__all__ = [
    "ApiDeps",
    "ApiSettings",
    "ReadinessCheck",
    "VersionInfo",
    "create_admin_app",
    "create_app",
]
