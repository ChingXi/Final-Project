
import os
from typing import Any, Protocol

import requests

from akg.config import Settings

API = "https://api.telegram.org"


class TelegramError(Exception):
    """Transport failure. Never a valid observation about a message."""


class Transport(Protocol):

    def send_message(self, chat_id: int, text: str) -> None: ...

    def send_action(self, chat_id: int, action: str = "typing") -> None: ...

    def download(self, file_id: str, dest: str) -> str: ...


class HttpTransport:
    def __init__(self, settings: Settings):
        settings.require_telegram_token()
        self._token = settings.telegram_bot_token
        self._timeout = 60

    def _safe(self, message: str) -> str:
        return message.replace(self._token, "<token>") if self._token else message

    def _call(self, method: str, **params: Any) -> Any:
        try:
            resp = requests.post(
                f"{API}/bot{self._token}/{method}",
                json=params,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as e:
            raise TelegramError(self._safe(f"{method} failed: {e}")) from e
        if not body.get("ok"):
            raise TelegramError(f"{method} rejected: {body.get('description')}")
        return body.get("result")

    def get_updates(self, offset: int, timeout: int = 30) -> list[dict]:
        try:
            resp = requests.post(
                f"{API}/bot{self._token}/getUpdates",
                json={"offset": offset, "timeout": timeout},
                timeout=timeout + 15,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as e:
            raise TelegramError(self._safe(f"getUpdates failed: {e}")) from e
        if not body.get("ok"):
            raise TelegramError(f"getUpdates rejected: {body.get('description')}")
        return body.get("result") or []

    def send_message(self, chat_id: int, text: str) -> None:
        for part in _split(text):
            self._call("sendMessage", chat_id=chat_id, text=part)

    def send_action(self, chat_id: int, action: str = "typing") -> None:
        try:
            self._call("sendChatAction", chat_id=chat_id, action=action)
        except TelegramError:
            pass

    def download(self, file_id: str, dest: str) -> str:
        path = self._call("getFile", file_id=file_id).get("file_path")
        if not path:
            raise TelegramError(f"no file_path for {file_id}")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        try:
            resp = requests.get(
                f"{API}/file/bot{self._token}/{path}", timeout=self._timeout
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise TelegramError(self._safe(f"download failed: {e}")) from e
        with open(dest, "wb") as f:
            f.write(resp.content)
        return dest


LIMIT = 4096


def _split(text: str, limit: int = LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            parts.append(current)
            current = ""
        current += line
    if current:
        parts.append(current)
    return parts
