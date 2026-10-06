from __future__ import annotations

import http.cookiejar
import socket
import threading
import time
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from fastapi import Depends, FastAPI
import uvicorn

from gods_mlops.web.auth import AUTH_COOKIE, CSRF_COOKIE, OperatorAuth, OperatorAuthSettings


OPERATOR_PASSWORD = "test-only-operator-password-42"
SESSION_SECRET = "test-only-session-secret-with-more-than-32-bytes"


class LocalClient:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.cookies = http.cookiejar.CookieJar()
        self.listen_socket = socket.socket()
        self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listen_socket.bind(("127.0.0.1", 0))
        self.listen_socket.listen(128)
        self.listen_socket.setblocking(False)
        self.port = self.listen_socket.getsockname()[1]
        self.server = uvicorn.Server(
            uvicorn.Config(app, log_level="critical", access_log=False, lifespan="on")
        )
        self.opener = urllib_request.build_opener(
            urllib_request.HTTPCookieProcessor(self.cookies),
            _NoRedirect(),
        )
        self.thread = threading.Thread(
            target=self.server.run,
            kwargs={"sockets": [self.listen_socket]},
            daemon=True,
        )
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.025)
        assert self.server.started, "operator test app failed to start on loopback"

    def close(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.listen_socket.close()

    def request(self, path: str, *, method: str = "GET", data: dict[str, str] | None = None, headers=None):
        encoded = urllib_parse.urlencode(data).encode("utf-8") if data is not None else None
        request = urllib_request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=encoded,
            headers=headers or {},
            method=method,
        )
        try:
            response = self.opener.open(request, timeout=10)
        except urllib_error.HTTPError as error:
            response = error
        return LocalResponse(response)

    def cookie(self, name: str) -> str | None:
        return next((cookie.value for cookie in self.cookies if cookie.name == name), None)

    def __enter__(self) -> LocalClient:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class _NoRedirect(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


class LocalResponse:
    def __init__(self, response) -> None:
        self._response = response
        self.status_code = response.status if hasattr(response, "status") else response.code
        self.headers = response.headers

    def read(self) -> bytes:
        return self._response.read()

    def geturl(self) -> str:
        return self._response.geturl()


def _client(*, password: str = OPERATOR_PASSWORD) -> LocalClient:
    auth = OperatorAuth(
        OperatorAuthSettings(
            username="operator",
            password=password,
            session_secret=SESSION_SECRET,
            secure_cookie=False,
        )
    )
    app = FastAPI()
    app.include_router(auth.router)
    app.state.operator_auth = auth
    app.state.write_calls = 0

    @app.post(
        "/protected-write",
        dependencies=[Depends(auth.require_operator), Depends(auth.require_csrf)],
    )
    async def protected_write():
        app.state.write_calls += 1
        return {"written": True}

    return LocalClient(app)


def _login(client: LocalClient) -> None:
    page = client.request("/login")
    csrf = client.cookie(CSRF_COOKIE)
    assert page.status_code == 200
    assert csrf
    response = client.request(
        "/login",
        method="POST",
        data={"username": "operator", "password": OPERATOR_PASSWORD},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 303


def test_session_identity_returns_only_the_verified_nonce_and_expiry() -> None:
    auth = OperatorAuth(
        OperatorAuthSettings(
            username="operator",
            password=OPERATOR_PASSWORD,
            session_secret=SESSION_SECRET,
            secure_cookie=False,
        )
    )
    session_token = auth._issue_session()

    identity = auth.session_identity(session_token)

    assert identity is not None
    nonce, expires_at = identity
    assert nonce
    assert expires_at > int(time.time())
    tampered = ("A" if session_token[0] != "A" else "B") + session_token[1:]
    assert auth.session_identity(tampered) is None


def test_unauthenticated_write_is_rejected_even_with_a_forged_csrf_header() -> None:
    client = _client()
    try:
        response = client.request("/protected-write", method="POST", headers={"X-CSRF-Token": "forged"})
        assert response.status_code == 401
        assert client.app.state.write_calls == 0
    finally:
        client.close()


def test_authenticated_write_requires_a_matching_csrf_cookie_and_header() -> None:
    client = _client()
    try:
        _login(client)
        csrf = client.cookie(CSRF_COOKIE)
        assert csrf

        missing = client.request("/protected-write", method="POST")
        invalid = client.request("/protected-write", method="POST", headers={"X-CSRF-Token": "wrong"})
        valid = client.request("/protected-write", method="POST", headers={"X-CSRF-Token": csrf})

        assert missing.status_code == 403
        assert invalid.status_code == 403
        assert valid.status_code == 200
        assert client.app.state.write_calls == 1
    finally:
        client.close()


def test_login_and_logout_require_csrf_without_exposing_credentials_or_tokens(caplog) -> None:
    client = _client()
    try:
        login_page = client.request("/login")
        csrf = client.cookie(CSRF_COOKIE)
        assert csrf
        login_html = login_page.read().decode("utf-8")
        assert OPERATOR_PASSWORD not in login_html
        assert csrf not in login_html

        response = client.request(
            "/login",
            method="POST",
            data={"username": "operator", "password": OPERATOR_PASSWORD},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/samples"
        session = client.cookie(AUTH_COOKIE)
        assert session
        response_html = response.read().decode("utf-8")
        assert OPERATOR_PASSWORD not in response_html
        assert session not in response_html
        assert csrf not in response_html
        assert OPERATOR_PASSWORD not in response.headers["location"]
        assert session not in response.headers["location"]

        missing_csrf = client.request("/logout", method="POST")
        assert missing_csrf.status_code == 403
        assert client.cookie(AUTH_COOKIE) == session

        logout = client.request(
            "/logout",
            method="POST",
            headers={"X-CSRF-Token": client.cookie(CSRF_COOKIE)},
        )
        assert logout.status_code == 303
        assert client.cookie(AUTH_COOKIE) is None
        assert OPERATOR_PASSWORD not in caplog.text
        assert session not in caplog.text
    finally:
        client.close()


def test_invalid_login_does_not_echo_or_log_password(caplog) -> None:
    client = _client()
    try:
        client.request("/login")
        csrf = client.cookie(CSRF_COOKIE)
        assert csrf
        supplied_password = "wrong-private-password-should-not-leak"

        response = client.request(
            "/login",
            method="POST",
            data={"username": "operator", "password": supplied_password},
            headers={"X-CSRF-Token": csrf},
        )

        assert response.status_code == 401
        assert supplied_password not in response.read().decode("utf-8")
        assert supplied_password not in response.geturl()
        assert supplied_password not in caplog.text
    finally:
        client.close()
