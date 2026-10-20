# SPDX-License-Identifier: Apache-2.0
"""Job worker loop over `PgJobs`: claim, lease renewal, retry/backoff, shutdown.

::

    worker = Worker(PgJobs(db), {"gc": run_gc}, WorkerConfig(worker_id="w1"))
    await worker.run(stop_event)
"""

from footnote.adapters.queue.worker import JobContext, Worker, WorkerConfig
from footnote.domain.jobs import PermanentJobError

__all__ = ["JobContext", "PermanentJobError", "Worker", "WorkerConfig"]
