from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

RetryEvent = dict[str, Any]
RetryCallback = Callable[[RetryEvent], None]


@dataclass(frozen=True)
class LLMRetryPolicy:
    """Bounded retry settings for LLM calls."""

    max_attempts: int = 3
    per_attempt_timeout_s: float | None = 30.0
    initial_backoff_s: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_s: float = 8.0

    def normalized(self) -> "LLMRetryPolicy":
        timeout = self.per_attempt_timeout_s
        return LLMRetryPolicy(
            max_attempts=max(1, self.max_attempts),
            per_attempt_timeout_s=timeout if timeout is None or timeout > 0 else None,
            initial_backoff_s=max(0.0, self.initial_backoff_s),
            backoff_multiplier=max(1.0, self.backoff_multiplier),
            max_backoff_s=max(0.0, self.max_backoff_s),
        )

    def backoff_for_failure(self, failure_index: int) -> float:
        """Return sleep seconds after a failed attempt, where first failure is 1."""
        if self.initial_backoff_s <= 0:
            return 0.0
        backoff = self.initial_backoff_s * (
            self.backoff_multiplier ** max(0, failure_index - 1)
        )
        return min(backoff, self.max_backoff_s)


class LLMRetryError(RuntimeError):
    """Raised when a bounded LLM retry policy is exhausted."""

    def __init__(
        self,
        *,
        operation: str,
        attempts: int,
        last_error: BaseException,
    ) -> None:
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{operation} failed after {attempts} attempt(s): {last_error!r}"
        )


def is_retryable_llm_error(error: BaseException) -> bool:
    """Best-effort detection for transient model/provider failures."""
    if isinstance(error, TimeoutError):
        return True

    status_code = _extract_status_code(error)
    if status_code is not None:
        return status_code in {408, 409, 425, 429} or status_code >= 500

    text = " ".join(
        str(part)
        for part in (
            type(error).__name__,
            str(error),
            repr(error),
            getattr(error, "code", ""),
        )
        if part
    ).lower()
    retry_markers = (
        "timeout",
        "timed out",
        "temporarily",
        "temporary",
        "rate limit",
        "ratelimit",
        "too many requests",
        "throttle",
        "throttling",
        "connection",
        "connect",
        "reset by peer",
        "unavailable",
        "overloaded",
        "server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
    )
    return any(marker in text for marker in retry_markers)


def invoke_with_retry(
    invoke: Callable[[], T],
    *,
    retry_policy: LLMRetryPolicy | None = None,
    operation: str = "llm.invoke",
    on_failure: RetryCallback | None = None,
) -> T:
    """Run a synchronous LLM call with bounded timeout and retry."""
    policy = (retry_policy or LLMRetryPolicy()).normalized()
    last_error: BaseException | None = None
    attempts_made = 0

    for attempt in range(1, policy.max_attempts + 1):
        attempts_made = attempt
        start = time.perf_counter()
        try:
            return _invoke_sync_with_timeout(invoke, policy.per_attempt_timeout_s)
        except Exception as error:
            last_error = error
            retryable = is_retryable_llm_error(error)
            will_retry = retryable and attempt < policy.max_attempts
            sleep_s = policy.backoff_for_failure(attempt) if will_retry else 0.0
            _emit_failure(
                on_failure,
                operation=operation,
                attempt=attempt,
                max_attempts=policy.max_attempts,
                retryable=retryable,
                will_retry=will_retry,
                sleep_s=sleep_s,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=error,
            )
            if not will_retry:
                break
            time.sleep(sleep_s)

    raise LLMRetryError(
        operation=operation,
        attempts=attempts_made,
        last_error=last_error or RuntimeError("unknown LLM error"),
    ) from last_error


async def ainvoke_with_retry(
    invoke: Callable[[], Awaitable[T]],
    *,
    retry_policy: LLMRetryPolicy | None = None,
    operation: str = "llm.ainvoke",
    on_failure: RetryCallback | None = None,
) -> T:
    """Run an async LLM call with bounded timeout and retry."""
    policy = (retry_policy or LLMRetryPolicy()).normalized()
    last_error: BaseException | None = None
    attempts_made = 0

    for attempt in range(1, policy.max_attempts + 1):
        attempts_made = attempt
        start = time.perf_counter()
        try:
            awaitable = invoke()
            if policy.per_attempt_timeout_s is None:
                return await awaitable
            return await asyncio.wait_for(
                awaitable,
                timeout=policy.per_attempt_timeout_s,
            )
        except Exception as error:
            last_error = error
            retryable = is_retryable_llm_error(error)
            will_retry = retryable and attempt < policy.max_attempts
            sleep_s = policy.backoff_for_failure(attempt) if will_retry else 0.0
            _emit_failure(
                on_failure,
                operation=operation,
                attempt=attempt,
                max_attempts=policy.max_attempts,
                retryable=retryable,
                will_retry=will_retry,
                sleep_s=sleep_s,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=error,
            )
            if not will_retry:
                break
            await asyncio.sleep(sleep_s)

    raise LLMRetryError(
        operation=operation,
        attempts=attempts_made,
        last_error=last_error or RuntimeError("unknown LLM error"),
    ) from last_error


def _invoke_sync_with_timeout(
    invoke: Callable[[], T],
    timeout_s: float | None,
) -> T:
    if timeout_s is None:
        return invoke()

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(invoke)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeoutError as error:
        future.cancel()
        raise TimeoutError(f"LLM call exceeded {timeout_s:.2f}s") from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _emit_failure(
    on_failure: RetryCallback | None,
    *,
    operation: str,
    attempt: int,
    max_attempts: int,
    retryable: bool,
    will_retry: bool,
    sleep_s: float,
    elapsed_ms: float,
    error: BaseException,
) -> None:
    if on_failure is None:
        return
    on_failure(
        {
            "operation": operation,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "retryable": retryable,
            "will_retry": will_retry,
            "sleep_s": sleep_s,
            "elapsed_ms": elapsed_ms,
            "error_type": type(error).__name__,
            "error": repr(error),
        }
    )


def _extract_status_code(error: BaseException) -> int | None:
    for candidate in (
        getattr(error, "status_code", None),
        getattr(error, "status", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            continue
    return None
