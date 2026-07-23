"""Tests for WorkerProcessManager multi-run concurrency + round-robin allocation.

The worker admits up to `max_concurrent_runs` (R) runs into RUNNING and then
distributes up to `max_concurrent_trials` (T) in-flight trial slots *globally*
across those runs round-robin, each run still bounded by its own concurrency.
"""

import datetime
import itertools
from unittest import mock
from prism.common.schemas.execution import RunStatus
from prism.server.models.run import Run
from prism.server.models.run import Trial
from prism.server.services.worker import WorkerProcessManager
import pytest


# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=unused-argument


@pytest.fixture
def mock_session_factory():
  """Fixture for a mocked session factory."""
  mock_session = mock.MagicMock()
  factory = mock.MagicMock(return_value=mock_session)
  factory.return_value.__enter__.return_value = mock_session
  return factory


def _make_manager(mock_session_factory, max_trials, max_runs):
  """Builds a fresh (singleton-reset) manager with explicit limits."""
  WorkerProcessManager._instance = None
  return WorkerProcessManager(
      max_concurrent_trials=max_trials,
      max_concurrent_runs=max_runs,
      session_factory=mock_session_factory,
  )


def _run(run_id, status=RunStatus.RUNNING, concurrency=10, day=1):
  return Run(
      id=run_id,
      status=status,
      created_at=datetime.datetime(2023, 1, day),
      concurrency=concurrency,
  )


def test_promotes_up_to_max_concurrent_runs(mock_session_factory):
  """The worker admits PENDING runs until R runs are RUNNING."""
  manager = _make_manager(mock_session_factory, max_trials=4, max_runs=2)

  with (
      mock.patch(
          "prism.server.repositories.run_repository.RunRepository.count_running"
      ) as mock_count,
      mock.patch(
          "prism.server.repositories.run_repository.RunRepository.promote_next_run"
      ) as mock_promote,
      mock.patch(
          "prism.server.repositories.run_repository.RunRepository.list_all"
      ) as mock_list_all,
      mock.patch(
          "prism.server.repositories.trial_repository.TrialRepository.list_by_status"
      ) as mock_list_trials,
      mock.patch(
          "prism.server.repositories.trial_repository.TrialRepository.pick_next_pending_trial"
      ) as mock_pick,
  ):
    # count goes 0 -> 1 -> 2; the loop stops once it reaches R=2.
    mock_count.side_effect = [0, 1, 2]
    mock_promote.side_effect = [_run(1), _run(2)]
    mock_list_all.return_value = [_run(1), _run(2)]
    mock_list_trials.return_value = []
    mock_pick.return_value = None  # nothing pending -> no spawns

    manager._start_new_trials()

    assert mock_promote.call_count == 2
    mock_promote.assert_called_with(max_running=2)


def test_promotion_stops_when_nothing_pending(mock_session_factory):
  """Promotion halts early when there is no PENDING run left to admit."""
  manager = _make_manager(mock_session_factory, max_trials=4, max_runs=5)

  with (
      mock.patch(
          "prism.server.repositories.run_repository.RunRepository.count_running"
      ) as mock_count,
      mock.patch(
          "prism.server.repositories.run_repository.RunRepository.promote_next_run"
      ) as mock_promote,
      mock.patch(
          "prism.server.repositories.run_repository.RunRepository.list_all"
      ) as mock_list_all,
      mock.patch(
          "prism.server.repositories.trial_repository.TrialRepository.list_by_status"
      ) as mock_list_trials,
      mock.patch(
          "prism.server.repositories.trial_repository.TrialRepository.pick_next_pending_trial"
      ) as mock_pick,
  ):
    mock_count.side_effect = [0, 1]
    mock_promote.side_effect = [_run(1), None]  # 2nd call: nothing pending
    mock_list_all.return_value = [_run(1)]
    mock_list_trials.return_value = []
    mock_pick.return_value = None

    manager._start_new_trials()

    assert mock_promote.call_count == 2  # tried once more, got None, stopped


def test_respects_per_run_concurrency(mock_session_factory):
  """A run already at its own concurrency gets no new trials."""
  manager = _make_manager(mock_session_factory, max_trials=4, max_runs=1)

  active_run = _run(1, concurrency=1)
  trial1 = Trial(id=10, run_id=1, status=RunStatus.RUNNING)

  with (
      mock.patch(
          "prism.server.repositories.run_repository.RunRepository.count_running"
      ) as mock_count,
      mock.patch(
          "prism.server.repositories.run_repository.RunRepository.list_all"
      ) as mock_list_all,
      mock.patch(
          "prism.server.repositories.trial_repository.TrialRepository.list_by_status"
      ) as mock_list_trials,
      mock.patch(
          "prism.server.repositories.trial_repository.TrialRepository.pick_next_pending_trial"
      ) as mock_pick,
  ):
    mock_count.return_value = 1  # already at R, no promotion
    mock_list_all.return_value = [active_run]
    mock_list_trials.return_value = [trial1]  # 1 in-flight; headroom 1-1=0

    manager._start_new_trials()

    mock_pick.assert_not_called()


def test_respects_global_trial_cap_and_round_robin(mock_session_factory):
  """Total spawns are capped by T and distributed round-robin across runs."""
  manager = _make_manager(mock_session_factory, max_trials=4, max_runs=2)

  run1 = _run(1, concurrency=10)
  run2 = _run(2, concurrency=10)

  # Unlimited pending trials; return a fresh Trial tagged with the run_id asked.
  ids = itertools.count(100)

  def _pick(run_id=None):
    return Trial(id=next(ids), run_id=run_id, status=RunStatus.PENDING)

  with (
      mock.patch(
          "prism.server.repositories.run_repository.RunRepository.count_running"
      ) as mock_count,
      mock.patch(
          "prism.server.repositories.run_repository.RunRepository.list_all"
      ) as mock_list_all,
      mock.patch(
          "prism.server.repositories.trial_repository.TrialRepository.list_by_status"
      ) as mock_list_trials,
      mock.patch(
          "prism.server.repositories.trial_repository.TrialRepository.pick_next_pending_trial"
      ) as mock_pick,
      mock.patch("multiprocessing.get_context") as mock_ctx,
  ):
    mock_count.return_value = 2  # at R, no promotion
    mock_list_all.return_value = [run1, run2]
    mock_list_trials.return_value = []  # no in-flight trials
    mock_pick.side_effect = _pick

    mock_proc = mock.MagicMock()
    mock_ctx.return_value.Process.return_value = mock_proc

    manager._start_new_trials()

    # Global cap T=4 honoured, not 2*concurrency=20.
    assert mock_proc.start.call_count == 4
    # Round-robin: run1, run2, run1, run2 (not run1 x4).
    picked_run_ids = [c.kwargs["run_id"] for c in mock_pick.call_args_list]
    assert picked_run_ids == [1, 2, 1, 2]


def test_aggregate_run_statuses_completes_run(mock_session_factory):
  """A run is marked COMPLETED when all its trials are terminal."""
  mock_session = mock_session_factory.return_value.__enter__.return_value
  manager = _make_manager(mock_session_factory, max_trials=4, max_runs=2)

  trial1 = Trial(id=1, status=RunStatus.COMPLETED)
  trial2 = Trial(id=2, status=RunStatus.FAILED)
  run = Run(id=1, status=RunStatus.RUNNING, trials=[trial1, trial2])

  with mock.patch(
      "prism.server.repositories.run_repository.RunRepository.list_active"
  ) as mock_list_active:
    mock_list_active.return_value = [run]

    manager._aggregate_run_statuses()

    assert run.status == RunStatus.COMPLETED
    assert run.completed_at is not None
    mock_session.commit.assert_called()
