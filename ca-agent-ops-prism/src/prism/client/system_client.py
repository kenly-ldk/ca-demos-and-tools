# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""System Client implementation."""

from fast_depends import Depends
from fast_depends import inject
from prism.client import dependencies
from prism.server.config import settings
from prism.server.services.worker import WorkerProcessManager


class SystemClient:
  """System Client for background operations."""

  @inject
  def start_worker_pool(
      self,
      num_workers: int | None = None,
      max_concurrent_runs: int | None = None,
      service: WorkerProcessManager = Depends(
          dependencies.get_worker_pool_service
      ),
  ) -> None:
    """Starts the background worker manager.

    Args:
      num_workers: Optional override for the global concurrent-trial ceiling (T).
        Defaults to settings.worker_max_concurrent_trials.
      max_concurrent_runs: Optional override for the concurrent-run limit (R).
        Defaults to settings.worker_max_concurrent_runs.
      service: The (singleton) worker manager to start.
    """
    service.max_concurrent_trials = (
        num_workers
        if num_workers is not None
        else settings.worker_max_concurrent_trials
    )
    service.max_concurrent_runs = (
        max_concurrent_runs
        if max_concurrent_runs is not None
        else settings.worker_max_concurrent_runs
    )
    service.start()
