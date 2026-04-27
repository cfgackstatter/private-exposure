from __future__ import annotations
import logging
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

logger = logging.getLogger(__name__)

_retry_decorator = retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class SecHttp:
    def __init__(self, user_agent: str, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=httpx.Timeout(timeout, connect=10.0),
            follow_redirects=True,
        )

    @_retry_decorator
    def get_json(self, url: str) -> dict:
        r = self._client.get(url)
        r.raise_for_status()
        return r.json()

    @_retry_decorator
    def get_text(self, url: str) -> str:
        r = self._client.get(url)
        r.raise_for_status()
        return r.text

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SecHttp":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()