from __future__ import annotations

import httpx
import time
import threading


class SecHttp:
    _MIN_INTERVAL = 0.12  # ~8 req/s, safely under the 10/s limit

    def __init__(self, user_agent: str) -> None:
        self._client = httpx.Client(headers={"User-Agent": user_agent})
        self._lock = threading.Lock()
        self._last_request = 0.0

    def _throttle(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_request
            wait = self._MIN_INTERVAL - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def _request(self, url: str) -> httpx.Response:
        self._throttle()
        r: httpx.Response | None = None
        for attempt in range(3):
            r = self._client.get(url)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r
        assert r is not None
        r.raise_for_status()
        return r

    def get_text(self, url: str) -> str:
        return self._request(url).text

    def get_json(self, url: str) -> dict:
        return self._request(url).json()

    def close(self) -> None:
        self._client.close()