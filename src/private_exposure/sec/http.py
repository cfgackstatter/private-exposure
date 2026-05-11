from __future__ import annotations
import logging
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential, before_sleep_log

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


_retry_decorator = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=10, max=60),
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