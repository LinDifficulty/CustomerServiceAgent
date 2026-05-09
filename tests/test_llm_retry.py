from __future__ import annotations

import asyncio
import time
import unittest

from rag_server.llm_retry import (
    LLMRetryError,
    LLMRetryPolicy,
    ainvoke_with_retry,
    invoke_with_retry,
    is_retryable_llm_error,
)


class LLMRetryPolicyTests(unittest.TestCase):
    def test_defaults(self) -> None:
        policy = LLMRetryPolicy()
        self.assertEqual(policy.max_attempts, 3)
        self.assertEqual(policy.per_attempt_timeout_s, 30.0)
        self.assertEqual(policy.initial_backoff_s, 1.0)

    def test_normalized_clamps_values(self) -> None:
        policy = LLMRetryPolicy(
            max_attempts=-1,
            per_attempt_timeout_s=-5,
            initial_backoff_s=-1,
            backoff_multiplier=0.5,
        )
        normalized = policy.normalized()
        self.assertEqual(normalized.max_attempts, 1)
        self.assertIsNone(normalized.per_attempt_timeout_s)
        self.assertEqual(normalized.initial_backoff_s, 0.0)
        self.assertEqual(normalized.backoff_multiplier, 1.0)

    def test_backoff_for_failure(self) -> None:
        policy = LLMRetryPolicy(initial_backoff_s=1.0, backoff_multiplier=2.0)
        self.assertAlmostEqual(policy.backoff_for_failure(1), 1.0)
        self.assertAlmostEqual(policy.backoff_for_failure(2), 2.0)
        self.assertAlmostEqual(policy.backoff_for_failure(3), 4.0)

    def test_backoff_respects_max(self) -> None:
        policy = LLMRetryPolicy(
            initial_backoff_s=1.0,
            backoff_multiplier=10.0,
            max_backoff_s=5.0,
        )
        self.assertAlmostEqual(policy.backoff_for_failure(3), 5.0)


class IsRetryableTests(unittest.TestCase):
    def test_timeout_is_retryable(self) -> None:
        self.assertTrue(is_retryable_llm_error(TimeoutError("timed out")))

    def test_rate_limit_is_retryable(self) -> None:
        error = RuntimeError("rate limit exceeded")
        self.assertTrue(is_retryable_llm_error(error))

    def test_connection_error_is_retryable(self) -> None:
        error = ConnectionError("connection reset by peer")
        self.assertTrue(is_retryable_llm_error(error))

    def test_status_429_is_retryable(self) -> None:
        error = RuntimeError("too many requests")
        error.status_code = 429
        self.assertTrue(is_retryable_llm_error(error))

    def test_status_500_is_retryable(self) -> None:
        error = RuntimeError("server error")
        error.status_code = 500
        self.assertTrue(is_retryable_llm_error(error))

    def test_status_400_is_not_retryable(self) -> None:
        error = RuntimeError("bad request")
        error.status_code = 400
        self.assertFalse(is_retryable_llm_error(error))

    def test_value_error_is_not_retryable(self) -> None:
        self.assertFalse(is_retryable_llm_error(ValueError("wrong type")))


class InvokeWithRetryTests(unittest.TestCase):
    def test_success_on_first_attempt(self) -> None:
        result = invoke_with_retry(lambda: 42)
        self.assertEqual(result, 42)

    def test_retries_on_transient_error(self) -> None:
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("slow")
            return "ok"

        policy = LLMRetryPolicy(max_attempts=3, initial_backoff_s=0.01)
        result = invoke_with_retry(flaky, retry_policy=policy)
        self.assertEqual(result, "ok")
        self.assertEqual(call_count, 2)

    def test_raises_after_exhaustion(self) -> None:
        policy = LLMRetryPolicy(max_attempts=2, initial_backoff_s=0.01)
        with self.assertRaises(LLMRetryError) as ctx:
            invoke_with_retry(
                lambda: (_ for _ in ()).throw(TimeoutError("always slow")),
                retry_policy=policy,
                operation="test_op",
            )
        self.assertEqual(ctx.exception.operation, "test_op")
        self.assertEqual(ctx.exception.attempts, 2)

    def test_non_retryable_fails_immediately(self) -> None:
        call_count = 0

        def bad():
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent error")

        policy = LLMRetryPolicy(max_attempts=3, initial_backoff_s=0.01)
        with self.assertRaises(LLMRetryError):
            invoke_with_retry(bad, retry_policy=policy)
        self.assertEqual(call_count, 1)

    def test_on_failure_callback(self) -> None:
        events: list[dict] = []

        def callback(event):
            events.append(event)

        policy = LLMRetryPolicy(max_attempts=2, initial_backoff_s=0.01)
        with self.assertRaises(LLMRetryError):
            invoke_with_retry(
                lambda: (_ for _ in ()).throw(TimeoutError("slow")),
                retry_policy=policy,
                on_failure=callback,
            )
        self.assertEqual(len(events), 2)
        self.assertTrue(events[0]["will_retry"])
        self.assertFalse(events[1]["will_retry"])

    def test_timeout_enforcement(self) -> None:
        def slow():
            time.sleep(5)
            return "done"

        policy = LLMRetryPolicy(max_attempts=1, per_attempt_timeout_s=0.1)
        with self.assertRaises(LLMRetryError):
            invoke_with_retry(slow, retry_policy=policy)


class AsyncInvokeWithRetryTests(unittest.TestCase):
    def test_async_success(self) -> None:
        async def go():
            async def success():
                return 42

            return await ainvoke_with_retry(success)

        result = asyncio.run(go())
        self.assertEqual(result, 42)

    def test_async_retries(self) -> None:
        call_count = 0

        async def go():
            nonlocal call_count

            async def flaky():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise TimeoutError("slow")
                return "ok"

            policy = LLMRetryPolicy(max_attempts=3, initial_backoff_s=0.01)
            return await ainvoke_with_retry(flaky, retry_policy=policy)

        result = asyncio.run(go())
        self.assertEqual(result, "ok")
        self.assertEqual(call_count, 2)


if __name__ == "__main__":
    unittest.main()
