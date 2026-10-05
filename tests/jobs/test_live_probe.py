from __future__ import annotations

import pytest
from importlib import import_module, util


def _live_probe_api():
    name = "gods_mlops.jobs.live_probe"
    assert util.find_spec(name) is not None, "the committed live probe harness is missing"
    return import_module(name)


def test_remote_database_url_only_redirects_the_host_through_the_ssh_tunnel() -> None:
    assert _live_probe_api().remote_database_url(
        "postgresql://operator:p%40ss@127.0.0.1:15437/task7?sslmode=disable",
        54123,
    ) == "postgresql://operator:p%40ss@127.0.0.1:54123/task7?sslmode=disable"


def test_remote_database_url_rejects_non_postgresql_schemes() -> None:
    with pytest.raises(ValueError, match="PostgreSQL"):
        _live_probe_api().remote_database_url("sqlite:///tmp/task7.db", 54123)


def test_proc_start_ticks_handles_spaces_and_parentheses_in_the_process_name() -> None:
    stat = "123 (python worker (cuda)) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 987654"
    assert _live_probe_api().proc_start_ticks(stat) == 987654


def test_proc_start_ticks_rejects_incomplete_process_records() -> None:
    with pytest.raises(ValueError, match="start time"):
        _live_probe_api().proc_start_ticks("123 (python) S 1 2")
