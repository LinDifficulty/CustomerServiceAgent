from __future__ import annotations

import asyncio
import time
# ThreadPoolExecutor: 用于在同步调用中实现超时控制
# TimeoutError as FutureTimeoutError: 区分线程池超时和 asyncio 超时
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass  # 用于定义不可变配置数据类
from typing import Any, Awaitable, Callable, TypeVar

# 泛型变量，用于保持重试调用返回值类型的一致性
T = TypeVar("T")

# 重试事件的类型别名：一个包含重试信息的字典
RetryEvent = dict[str, Any]
# 失败回调类型：接收重试事件字典，无返回值
RetryCallback = Callable[[RetryEvent], None]


@dataclass(frozen=True)
class LLMRetryPolicy:
    """LLM 调用的有界重试策略配置。"""

    max_attempts: int = 3
    """最大尝试次数（含首次调用）。"""
    per_attempt_timeout_s: float | None = 30.0
    """单次调用的超时秒数。None 表示不设超时。"""
    initial_backoff_s: float = 1.0
    """第一次失败后的退避等待秒数。"""
    backoff_multiplier: float = 2.0
    """退避时间的倍增因子，每次重试后等待时间乘以该值。"""
    max_backoff_s: float = 8.0
    """单次退避等待的最大秒数上限。"""

    def normalized(self) -> "LLMRetryPolicy":
        """返回规范化的重试策略——将所有参数钳制到合理范围内。"""
        timeout = self.per_attempt_timeout_s
        return LLMRetryPolicy(
            # 最大尝试次数至少为 1
            max_attempts=max(1, self.max_attempts),
            # 超时只保留正数，None 或非正数统一为 None（不设超时）
            per_attempt_timeout_s=timeout if timeout is None or timeout > 0 else None,
            # 初始退避时间不能为负
            initial_backoff_s=max(0.0, self.initial_backoff_s),
            # 倍增因子至少为 1.0，避免退避时间递减
            backoff_multiplier=max(1.0, self.backoff_multiplier),
            # 最大退避时间不能为负
            max_backoff_s=max(0.0, self.max_backoff_s),
        )

    def backoff_for_failure(self, failure_index: int) -> float:
        """计算第 N 次失败后应等待的退避秒数（指数退避）。
        failure_index=1 对应第一次失败。
        """
        # 如果初始退避为 0，则不等待，直接返回 0
        if self.initial_backoff_s <= 0:
            return 0.0
        # 指数退避公式: initial_backoff * (multiplier ^ (failure_index - 1))
        backoff = self.initial_backoff_s * (
            self.backoff_multiplier ** max(0, failure_index - 1)
        )
        # 退避时间不超过上限
        return min(backoff, self.max_backoff_s)


class LLMRetryError(RuntimeError):
    """当有界 LLM 重试策略耗尽所有尝试次数后抛出的异常。"""

    def __init__(
        self,
        *,
        operation: str,         # 操作名称（如 "llm.invoke"）
        attempts: int,          # 已尝试的次数
        last_error: BaseException,  # 最后一次捕获的异常
    ) -> None:
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{operation} failed after {attempts} attempt(s): {last_error!r}"
        )


def is_retryable_llm_error(error: BaseException) -> bool:
    """尽力检测是否为可重试的瞬时性模型/提供商错误。
    判断依据：HTTP 状态码 + 异常文本中的关键词匹配。
    """
    # 超时错误通常是瞬时的，可重试
    if isinstance(error, TimeoutError):
        return True

    # 从异常中提取 HTTP 状态码，判断是否为可重试的服务端错误
    status_code = _extract_status_code(error)
    if status_code is not None:
        # 408 请求超时 | 409 冲突 | 425 过早 | 429 限流 | >=500 服务端错误
        return status_code in {408, 409, 425, 429} or status_code >= 500

    # 如果无法提取状态码，通过异常文本匹配可重试关键词
    text = " ".join(
        str(part)
        for part in (
            type(error).__name__,
            str(error),
            getattr(error, "code", ""),
        )
        if part
    ).lower()

    # 匹配常见瞬时性错误关键词
    retry_markers = (
        "timeout",           # 超时
        "timed out",         # 超时（短语）
        "temporarily",       # 临时
        "temporary",         # 临时
        "rate limit",        # 频率限制
        "ratelimit",         # 频率限制（连写）
        "too many requests", # 请求过多
        "throttle",          # 限流
        "throttling",        # 限流中
        "connection",        # 连接错误
        "connect",           # 连接错误（变体）
        "reset by peer",     # 对端重置连接
        "unavailable",       # 服务不可用
        "overloaded",        # 过载
        "server error",      # 服务器错误
        "bad gateway",       # 网关错误
        "service unavailable",  # 服务不可用
        "gateway timeout",   # 网关超时
    )
    # 只要异常文本中包含任一关键词，就认为是可重试的
    return any(marker in text for marker in retry_markers)


def _classify_attempt_error(
    error: Exception,
    attempt: int,
    policy: LLMRetryPolicy,
    operation: str,
    on_failure: RetryCallback | None,
    start: float,
) -> tuple[bool, float]:
    """Classify a retry attempt error: determine retryability, emit failure callback.

    Returns:
        (will_retry, sleep_s) — whether to retry and how long to sleep.
    """
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
    return will_retry, sleep_s


def invoke_with_retry(
    invoke: Callable[[], T],
    *,
    retry_policy: LLMRetryPolicy | None = None,
    operation: str = "llm.invoke",
    on_failure: RetryCallback | None = None,
) -> T:
    """执行同步 LLM 调用，带有超时和重试机制。
    invoke: 要执行的同步可调用对象
    retry_policy: 重试策略，None 则使用默认策略
    operation: 操作名称，用于日志和错误信息
    on_failure: 每次失败时的回调，接收重试事件字典
    """
    # 规范化重试策略（钳制参数到合理范围）
    policy = (retry_policy or LLMRetryPolicy()).normalized()
    last_error: BaseException | None = None  # 记录最后一次捕获的异常
    attempts_made = 0  # 记录实际已完成的尝试次数

    # 重试循环：从第 1 次到 max_attempts 次
    for attempt in range(1, policy.max_attempts + 1):
        attempts_made = attempt
        start = time.perf_counter()  # 记录本次尝试的开始时间（高精度）

        try:
            # 用线程池执行同步调用，实现超时控制
            return _invoke_sync_with_timeout(invoke, policy.per_attempt_timeout_s)
        except Exception as error:
            last_error = error
            will_retry, sleep_s = _classify_attempt_error(
                error, attempt, policy, operation, on_failure, start
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
    """执行异步 LLM 调用，带有超时和重试机制。
    与 invoke_with_retry 逻辑相同，但适用于 async/await 场景。
    invoke: 返回 awaitable 的可调用对象
    """
    # 规范化重试策略
    policy = (retry_policy or LLMRetryPolicy()).normalized()
    last_error: BaseException | None = None
    attempts_made = 0

    # 重试循环
    for attempt in range(1, policy.max_attempts + 1):
        attempts_made = attempt
        start = time.perf_counter()

        try:
            # 调用 invoke 获取 awaitable 对象
            awaitable = invoke()
            # 如果没有设置超时，直接 await 执行
            if policy.per_attempt_timeout_s is None:
                return await awaitable
            # 使用 asyncio.wait_for 实现超时控制
            return await asyncio.wait_for(
                awaitable,
                timeout=policy.per_attempt_timeout_s,
            )
        except Exception as error:
            last_error = error
            will_retry, sleep_s = _classify_attempt_error(
                error, attempt, policy, operation, on_failure, start
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
    """在独立线程中执行同步调用，实现超时控制。
    使用 ThreadPoolExecutor 提交任务并通过 future.result(timeout=...) 限时等待。
    """
    # 未设置超时则直接调用，避免线程池开销
    if timeout_s is None:
        return invoke()

    # 创建单线程的线程池
    executor = ThreadPoolExecutor(max_workers=1)
    # 提交调用任务到线程池
    future = executor.submit(invoke)
    try:
        # 等待结果，超时则抛出 FutureTimeoutError
        return future.result(timeout=timeout_s)
    except FutureTimeoutError as error:
        # 超时后取消 future，防止后台线程继续占用资源
        future.cancel()
        # 抛出标准的 TimeoutError，携带超时信息
        raise TimeoutError(f"LLM call exceeded {timeout_s:.2f}s") from error
    finally:
        # 无论成功与否，关闭线程池（不等待正在运行的任务）
        # cancel_futures=True 会取消所有待处理的 future
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
    """如果设置了失败回调 on_failure，则触发回调并传入重试事件的详细信息。"""
    if on_failure is None:
        return
    # 构建重试事件字典，包含操作名、尝试次数、是否可重试、错误类型等信息
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
    """从异常对象中尽力提取 HTTP 状态码。
    尝试从 status_code、status、response.status_code 等常见属性中提取。
    """
    # 依次尝试多个可能的属性路径
    for candidate in (
        getattr(error, "status_code", None),       # 直接属性
        getattr(error, "status", None),             # 简写形式
        getattr(getattr(error, "response", None), "status_code", None),  # response 子对象
    ):
        try:
            if candidate is not None:
                return int(candidate)  # 确保转为整数
        except (TypeError, ValueError):
            # 如果转换失败（如返回了非数字字符串），跳过该候选
            continue
    return None
