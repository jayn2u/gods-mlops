"""Single-operator cookie authentication and CSRF protection for the local UI."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import hmac
import html
import json
import os
from pathlib import Path
import secrets
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse

AUTH_COOKIE = "gods_mlops_operator_session"
CSRF_COOKIE = "gods_mlops_operator_csrf"
CSRF_HEADER = "X-CSRF-Token"
DEFAULT_SESSION_SECONDS = 8 * 60 * 60
_LOGIN_TEMPLATE = Path(__file__).with_name("templates") / "login.html"


@dataclass(frozen=True, slots=True)
class OperatorAuthSettings:
    """Protected process configuration for the one local operator account."""

    username: str
    password: str = field(repr=False)
    session_secret: str = field(repr=False)
    secure_cookie: bool = False
    session_seconds: int = DEFAULT_SESSION_SECONDS

    @classmethod
    def from_environment(cls) -> OperatorAuthSettings:
        """Load credentials injected by the protected runtime environment."""
        return cls(
            username=_required_environment("GODS_MLOPS_OPERATOR_USERNAME"),
            password=_required_environment("GODS_MLOPS_OPERATOR_PASSWORD"),
            session_secret=_required_environment("GODS_MLOPS_SESSION_SECRET"),
            secure_cookie=os.environ.get("GODS_MLOPS_OPERATOR_COOKIE_SECURE", "false").lower() == "true",
        )

    def __post_init__(self) -> None:
        if not self.username or len(self.username) > 255:
            raise ValueError("operator username must contain 1 to 255 characters")
        if len(self.password) < 16:
            raise ValueError("operator password must contain at least 16 characters")
        if len(self.session_secret.encode("utf-8")) < 32:
            raise ValueError("operator session secret must contain at least 32 bytes")
        if not 60 <= self.session_seconds <= 7 * 24 * 60 * 60:
            raise ValueError("operator session lifetime must be between one minute and seven days")


class OperatorAuth:
    """Build the login/logout routes and dependencies shared by UI routes."""

    def __init__(self, settings: OperatorAuthSettings) -> None:
        self.settings = settings
        self.router = APIRouter(tags=["operator"])
        self._register_routes()

    async def require_operator(self, request: Request) -> str:
        token = request.cookies.get(AUTH_COOKIE)
        if token is None or self._session_expiry(token) is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="operator authentication required",
            )
        return self.settings.username

    async def require_csrf(self, request: Request) -> None:
        cookie_token = request.cookies.get(CSRF_COOKIE)
        header_token = request.headers.get(CSRF_HEADER)
        if (
            cookie_token is None
            or header_token is None
            or not secrets.compare_digest(cookie_token.encode("utf-8"), header_token.encode("utf-8"))
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF verification failed",
            )

    def _register_routes(self) -> None:
        @self.router.get("/login", response_class=HTMLResponse)
        async def login_page() -> HTMLResponse:
            response = HTMLResponse(_login_html())
            self._set_no_store_headers(response)
            self._set_csrf_cookie(response, secrets.token_urlsafe(32))
            return response

        @self.router.post("/login", dependencies=[Depends(self.require_csrf)])
        async def login(
            username: Annotated[str, Form()],
            password: Annotated[str, Form()],
        ) -> Response:
            username_matches = secrets.compare_digest(
                username.encode("utf-8"), self.settings.username.encode("utf-8")
            )
            password_matches = secrets.compare_digest(
                password.encode("utf-8"), self.settings.password.encode("utf-8")
            )
            if not (username_matches & password_matches):
                response = HTMLResponse(_login_html(error="Invalid operator credentials."), status_code=401)
                self._set_no_store_headers(response)
                return response

            response = RedirectResponse("/samples", status_code=status.HTTP_303_SEE_OTHER)
            self._set_no_store_headers(response)
            self._set_session_cookie(response, self._issue_session())
            self._set_csrf_cookie(response, secrets.token_urlsafe(32))
            return response

        @self.router.post(
            "/logout",
            dependencies=[Depends(self.require_operator), Depends(self.require_csrf)],
        )
        async def logout() -> RedirectResponse:
            response = RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
            self._set_no_store_headers(response)
            response.delete_cookie(AUTH_COOKIE, path="/", secure=self.settings.secure_cookie, httponly=True, samesite="strict")
            response.delete_cookie(CSRF_COOKIE, path="/", secure=self.settings.secure_cookie, httponly=False, samesite="strict")
            return response

    def _issue_session(self) -> str:
        payload = json.dumps(
            {"exp": int(time.time()) + self.settings.session_seconds, "nonce": secrets.token_urlsafe(24)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        signature = hmac.new(self.settings.session_secret.encode("utf-8"), payload, hashlib.sha256).digest()
        return f"{_base64url(payload)}.{_base64url(signature)}"

    def _session_expiry(self, token: str) -> int | None:
        try:
            payload_part, signature_part = token.split(".", maxsplit=1)
            payload = _unbase64url(payload_part)
            signature = _unbase64url(signature_part)
            expected = hmac.new(self.settings.session_secret.encode("utf-8"), payload, hashlib.sha256).digest()
            if not hmac.compare_digest(signature, expected):
                return None
            decoded: Any = json.loads(payload)
            expiry = decoded.get("exp") if isinstance(decoded, dict) else None
            if not isinstance(expiry, int) or expiry <= time.time():
                return None
            return expiry
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    def session_identity(self, token: str) -> tuple[str, int] | None:
        """Return the verified nonce and expiry from an existing operator session."""
        expiry = self._session_expiry(token)
        if expiry is None:
            return None
        try:
            payload_part, _signature_part = token.split(".", maxsplit=1)
            decoded: Any = json.loads(_unbase64url(payload_part))
        except (ValueError, TypeError, json.JSONDecodeError):
            return None
        nonce = decoded.get("nonce") if isinstance(decoded, dict) else None
        if not isinstance(nonce, str) or not nonce:
            return None
        return nonce, expiry

    def _set_session_cookie(self, response: Response, token: str) -> None:
        response.set_cookie(
            AUTH_COOKIE,
            token,
            max_age=self.settings.session_seconds,
            path="/",
            secure=self.settings.secure_cookie,
            httponly=True,
            samesite="strict",
        )

    def _set_csrf_cookie(self, response: Response, token: str) -> None:
        response.set_cookie(
            CSRF_COOKIE,
            token,
            max_age=self.settings.session_seconds,
            path="/",
            secure=self.settings.secure_cookie,
            httponly=False,
            samesite="strict",
        )

    @staticmethod
    def _set_no_store_headers(response: Response) -> None:
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "same-origin"


def _login_html(*, error: str = "") -> str:
    template = _LOGIN_TEMPLATE.read_text(encoding="utf-8")
    return template.replace("{{error}}", html.escape(error))


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        raise ValueError(f"{name} is required to start the local MLOps UI")
    return value


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unbase64url(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
