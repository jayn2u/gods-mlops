from __future__ import annotations

from gods_mlops import cli
from gods_mlops.web import app as web_app


def test_operator_ui_cli_starts_the_local_web_entrypoint(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(web_app, "main", lambda: calls.append("started"))

    result = cli.main(["operator-ui"])

    assert result == 0
    assert calls == ["started"]
