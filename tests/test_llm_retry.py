"""测试 LLM 调用的重试策略。

包含以下测试场景：
- LLMRetryPolicy 的默认值、归一化和退避时间计算
- is_retryable_llm_error 对各种错误的判断（超时/限流/连接错误可重试，参数错误不可重试）
- invoke_with_retry 同步重试：首次成功、瞬态错误重试、重试耗尽、不可重试错误立即失败、回调、超时
- ainvoke_with_retry 异步重试：异步成功、异步重试
"""

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
    """测试 LLMRetryPolicy 的默认值、归一化和退避策略。"""

    def test_defaults(self) -> None:
        """验证默认重试策略参数：最多 3 次尝试，单次超时 30 秒，初始退避 1 秒。"""
        policy = LLMRetryPolicy()
        self.assertEqual(policy.max_attempts, 3)
        self.assertEqual(policy.per_attempt_timeout_s, 30.0)
        self.assertEqual(policy.initial_backoff_s, 1.0)

    def test_normalized_clamps_values(self) -> None:
        """验证归一化方法对非法参数（负数）的钳位处理。"""
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
        """验证指数退避时间计算：第 1 次失败等待 1 秒，第 2 次 2 秒，第 3 次 4 秒。"""
        policy = LLMRetryPolicy(initial_backoff_s=1.0, backoff_multiplier=2.0)
        self.assertAlmostEqual(policy.backoff_for_failure(1), 1.0)
        self.assertAlmostEqual(policy.backoff_for_failure(2), 2.0)
        self.assertAlmostEqual(policy.backoff_for_failure(3), 4.0)

    def test_backoff_respects_max(self) -> None:
        """验证退避时间不会超过 max_backoff_s 设定的上限。"""
        policy = LLMRetryPolicy(
            initial_backoff_s=1.0,
            backoff_multiplier=10.0,
            max_backoff_s=5.0,
        )
        self.assertAlmostEqual(policy.backoff_for_failure(3), 5.0)


class IsRetryableTests(unittest.TestCase):
    """测试 is_retryable_llm_error 函数对各种错误类型的判断。"""

    def test_timeout_is_retryable(self) -> None:
        """超时错误应被判定为可重试。"""
        self.assertTrue(is_retryable_llm_error(TimeoutError("timed out")))

    def test_rate_limit_is_retryable(self) -> None:
        """限流错误（rate limit）应被判定为可重试。"""
        error = RuntimeError("rate limit exceeded")
        self.assertTrue(is_retryable_llm_error(error))

    def test_connection_error_is_retryable(self) -> None:
        """连接错误应被判定为可重试。"""
        error = ConnectionError("connection reset by peer")
        self.assertTrue(is_retryable_llm_error(error))

    def test_status_429_is_retryable(self) -> None:
        """HTTP 429 状态码（请求过多）应被判定为可重试。"""
        error = RuntimeError("too many requests")
        error.status_code = 429
        self.assertTrue(is_retryable_llm_error(error))

    def test_status_500_is_retryable(self) -> None:
        """HTTP 500 状态码（服务器内部错误）应被判定为可重试。"""
        error = RuntimeError("server error")
        error.status_code = 500
        self.assertTrue(is_retryable_llm_error(error))

    def test_status_400_is_not_retryable(self) -> None:
        """HTTP 400 状态码（客户端请求错误）不应被判定为可重试。"""
        error = RuntimeError("bad request")
        error.status_code = 400
        self.assertFalse(is_retryable_llm_error(error))

    def test_value_error_is_not_retryable(self) -> None:
        """参数错误（ValueError）不应被判定为可重试。"""
        self.assertFalse(is_retryable_llm_error(ValueError("wrong type")))


class InvokeWithRetryTests(unittest.TestCase):
    """测试同步重试函数 invoke_with_retry 的各种场景。"""

    def test_success_on_first_attempt(self) -> None:
        """首次调用即成功，直接返回结果，不触发重试。"""
        result = invoke_with_retry(lambda: 42)
        self.assertEqual(result, 42)

    def test_retries_on_transient_error(self) -> None:
        """瞬态错误（超时）后重试成功，验证实际调用次数为 2 次。"""
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
        """重试次数耗尽后抛出 LLMRetryError，错误中包含操作名和尝试次数。"""
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
        """不可重试的错误（ValueError）立即失败，不进行重试，调用次数为 1。"""
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
        """验证失败回调被触发，且最后一次事件的 will_retry 为 False。"""
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
        """验证单次调用超时生效：耗时超过 per_attempt_timeout_s 时抛出 LLMRetryError。"""
        def slow():
            time.sleep(5)
            return "done"

        policy = LLMRetryPolicy(max_attempts=1, per_attempt_timeout_s=0.1)
        with self.assertRaises(LLMRetryError):
            invoke_with_retry(slow, retry_policy=policy)


class AsyncInvokeWithRetryTests(unittest.TestCase):
    """测试异步重试函数 ainvoke_with_retry 的各种场景。"""

    def test_async_success(self) -> None:
        """异步调用首次成功，直接返回结果。"""
        async def go():
            async def success():
                return 42

            return await ainvoke_with_retry(success)

        result = asyncio.run(go())
        self.assertEqual(result, 42)

    def test_async_retries(self) -> None:
        """异步调用在瞬态错误后重试成功，验证实际调用次数为 2 次。"""
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
