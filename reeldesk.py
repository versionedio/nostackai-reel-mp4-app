"""Reeldesk's app API, as the render user: the same calls nostackai-reels/app/src/api/client.ts makes.

Sign-in is the app's two steps (POST /user/login, then GET /token). The token authorizes API calls;
the same /token response also sets the CDN's signed cookies, which this session keeps in memory for
the run, exactly as a browser does, because file downloads come from the CDN. Nothing is written to
disk and nothing secret is ever printed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests

TIMEOUT = 60
UPLOAD_TIMEOUT = 600
# Record and content ids (e.g. cvimunlmh4p8, c-71de): anything else is refused before it reaches a URL.
RECORD_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ReeldeskError(Exception):
    """A failed call, with a message safe to show in public logs (no URLs, tokens or bodies)."""


@dataclass(frozen=True)
class Connection:
    base_url: str
    org_code: str
    api_key: str
    client_id: str

    @property
    def app_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/app/{self.org_code}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_id(value: str) -> str:
    if not RECORD_ID.match(value or ""):
        raise ReeldeskError("Not a valid record id.")
    return value


def refusal(res: requests.Response) -> str:
    """Why a call was refused, safe for public logs: the status, the gateway's error type and short
    message (e.g. "ForbiddenException: Forbidden", "Missing Authentication Token") and the request id
    to look up in the platform's logs. Never the request, headers or a long body."""
    parts = [str(res.status_code)]
    kind = res.headers.get("x-amzn-errortype", "").split(":")[0]
    try:
        body = res.json()
        message = body.get("message") or body.get("error") if isinstance(body, dict) else None
    except ValueError:
        message = None
    detail = ": ".join(x for x in (kind, str(message)[:120] if message else "") if x)
    if detail:
        parts.append(detail)
    request_id = res.headers.get("x-amzn-requestid") or res.headers.get("x-amz-apigw-id")
    if request_id:
        parts.append(f"request id {request_id}")
    return ", ".join(parts)


class Reeldesk:
    def __init__(self, conn: Connection, session: requests.Session | None = None):
        self.conn = conn
        self.http = session or requests.Session()
        self.token: str | None = None
        self.email: str | None = None

    # ── sign-in ──────────────────────────────────────────────────────────────────────────────

    def sign_in(self, email: str, password: str) -> None:
        login = self.http.post(f"{self.conn.app_url}/user/login", json={"email": email, "pwd": password},
                               headers={"x-api-key": self.conn.api_key}, timeout=TIMEOUT)
        oauth = login.headers.get("oauthtoken")
        if login.status_code != 200 or not oauth:
            raise ReeldeskError(f"Sign-in refused ({refusal(login)}). A 403 here comes from the gateway: check "
                                "REELDESK_APP_URL (just the host, e.g. https://dd1use2.nostackai.com) and REELDESK_API_KEY; "
                                "a 401 means the render user's email or password.")
        token_res = self.http.get(f"{self.conn.app_url}/token", timeout=TIMEOUT, headers={
            "oauthtoken": oauth, "x-api-key": self.conn.api_key, "client_id": self.conn.client_id,
        })
        token = token_res.headers.get("x-nostackai-token")
        if token_res.status_code != 200 or not token:
            raise ReeldeskError(f"Token refused ({refusal(token_res)}): check REELDESK_CLIENT_ID and REELDESK_API_KEY.")
        self.token, self.email = token, email

    # ── records ──────────────────────────────────────────────────────────────────────────────

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        if not self.token:
            raise ReeldeskError("Not signed in.")
        res = self.http.request(method, f"{self.conn.app_url}{path}", json=body, timeout=TIMEOUT,
                                headers={"authorization": self.token, "Content-Type": "application/json"})
        if res.status_code == 401:
            raise ReeldeskError("Session expired.")
        route = "/".join(path.split("?")[0].split("/")[:3])  # e.g. /update/videos, never the id
        if res.status_code == 403:
            raise ReeldeskError(f"The render user's access policy does not allow {method} {route}.")
        if not 200 <= res.status_code < 300:
            raise ReeldeskError(f"{method} {route} failed ({res.status_code}).")
        return res.json() if res.content else {}

    def get_video(self, video_id: str) -> dict:
        return self._call("GET", f"/item/videos/{check_id(video_id)}")

    def list_videos(self, **filters: str) -> list[dict]:
        query = urlencode({"by": "all", "pagesize": 500, **filters})
        return self._call("GET", f"/list/videos?{query}").get("data", [])

    def update_video(self, video_id: str, fields: dict) -> dict:
        return self._call("PUT", f"/update/videos/{check_id(video_id)}", {"last_touched_at": now_iso(), **fields})

    def list_files(self, video_id: str) -> list[dict]:
        query = urlencode({"by": "all", "pagesize": 500, "video_id": check_id(video_id)})
        return self._call("GET", f"/list/video_files?{query}").get("data", [])

    def add_file(self, row: dict) -> dict:
        return self._call("POST", "/create/video_files", {**row, "created_at": now_iso()})

    def update_file(self, file_id: str, row: dict) -> dict:
        return self._call("PUT", f"/update/video_files/{check_id(file_id)}", row)

    def delete_content(self, content_id: str) -> None:
        """Removes a replaced upload (the platform lets the uploader delete its own). Best effort:
        a leftover old file must never fail a render that already succeeded."""
        try:
            self._call("DELETE", f"/content/{check_id(content_id)}")
        except ReeldeskError:
            pass

    def add_note(self, video_id: str, text: str) -> dict:
        return self._call("POST", "/create/video_notes", {
            "video_id": check_id(video_id), "author": self.email or "render", "text": text, "created_at": now_iso(),
        })

    # ── files ────────────────────────────────────────────────────────────────────────────────

    def download(self, file_row: dict, dest: Path) -> Path:
        """A video_files row's bytes, from the CDN (authorized by the cookies /token set)."""
        res = self.http.get(file_row["cdn_url"], timeout=UPLOAD_TIMEOUT, stream=True)
        if res.status_code != 200:
            raise ReeldeskError(f"Download of {file_row.get('kind', 'file')} '{file_row.get('name', '?')}' failed ({res.status_code}).")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            for chunk in res.iter_content(1 << 20):
                out.write(chunk)
        return dest

    def upload(self, path: Path, name: str, content_type: str, context: str) -> dict:
        """Uploads a file as a content item; returns the presign result (contentId, cdnUrl)."""
        query = urlencode({"name": name, "contentType": content_type, "context": context})
        presign = self._call("GET", f"/content/uploadurl?{query}")
        with path.open("rb") as body:
            put = self.http.put(presign["uploadUrl"], data=body, headers={"Content-Type": content_type}, timeout=UPLOAD_TIMEOUT)
        if not 200 <= put.status_code < 300:
            raise ReeldeskError(f"Upload of '{name}' failed ({put.status_code}).")
        return presign
