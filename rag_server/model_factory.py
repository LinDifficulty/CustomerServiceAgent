# 启用延迟注解求值（PEP 563）
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from typing import Any, Callable

# ── 默认模型提供商和模型名称 ──────────────────────────────────
# 对话模型：默认使用阿里云通义千问
DEFAULT_CHAT_PROVIDER = "tongyi"
DEFAULT_CHAT_MODEL = "deepseek-v4-flash"
# 嵌入模型：默认使用阿里云 DashScope
DEFAULT_EMBEDDING_PROVIDER = "dashscope"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
# 重排序模型：默认使用 BGE CrossEncoder
DEFAULT_RERANKER_PROVIDER = "cross_encoder"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class ModelProviderError(ValueError):
    """Raised when a configured model provider cannot be constructed.

    模型提供商错误：当配置的模型提供商无法实例化时抛出。
    """


def create_chat_model(
    provider: str = DEFAULT_CHAT_PROVIDER,
    model_name: str = DEFAULT_CHAT_MODEL,
    **model_kwargs: Any,
) -> Any:
    """Create a chat model from a named provider or Python import path.

    创建对话模型实例。支持：
    - 内置提供商标识（如 "tongyi"、"openai"）
    - 自定义 Python 导入路径（如 "package.module:Factory"）
    """
    # 规范化提供商标识为小写+下划线格式
    normalized = _normalize_provider(provider)
    # 清理关键字参数，去除值为 None 的项
    kwargs = _clean_kwargs(model_kwargs)

    # 内置支持：阿里云通义千问对话模型
    if normalized in {"tongyi", "dashscope", "chat_tongyi"}:
        from langchain_community.chat_models import ChatTongyi

        kwargs.setdefault("max_retries", 0)
        return ChatTongyi(model=model_name, **kwargs)

    # 内置支持：OpenAI 对话模型
    if normalized in {"openai", "chat_openai"}:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, **kwargs)

    # 自定义提供商：通过 Python 导入路径动态加载工厂函数
    factory = import_provider(provider)
    kwargs.setdefault("max_retries", 0)
    return instantiate_model_provider(factory, model_name, kwargs)


def create_embeddings(
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    **model_kwargs: Any,
) -> Any:
    """Create an embedding model from a named provider or Python import path.

    创建嵌入模型实例。支持：
    - 内置提供商标识（如 "dashscope"、"openai"）
    - 自定义 Python 导入路径
    """
    normalized = _normalize_provider(provider)
    kwargs = _clean_kwargs(model_kwargs)

    # 内置支持：阿里云 DashScope 嵌入模型
    if normalized in {"dashscope", "dashscope_embeddings"}:
        from langchain_community.embeddings import DashScopeEmbeddings

        return DashScopeEmbeddings(model=model_name, **kwargs)

    # 内置支持：OpenAI 嵌入模型
    if normalized in {"openai", "openai_embeddings"}:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_name, **kwargs)

    # 自定义提供商：动态导入
    factory = import_provider(provider)
    return instantiate_model_provider(factory, model_name, kwargs)


def create_reranker(
    provider: str = DEFAULT_RERANKER_PROVIDER,
    model_name: str = DEFAULT_RERANKER_MODEL,
    *,
    device: str | None = None,
    **model_kwargs: Any,
) -> Any:
    """Create a reranker object from a named provider or Python import path.

    创建重排序模型实例。支持：
    - 内置 SentenceTransformer CrossEncoder
    - 自定义 Python 导入路径
    - 可指定设备（cpu/cuda/mps 等）
    """
    normalized = _normalize_provider(provider)
    kwargs = _clean_kwargs(model_kwargs)

    # 内置支持：SentenceTransformers CrossEncoder（本地模型）
    if normalized in {
        "cross_encoder",
        "sentence_transformers",
        "sentence_transformers_cross_encoder",
    }:
        from sentence_transformers import CrossEncoder

        # trust_remote_code 允许加载使用自定义代码的模型
        kwargs.setdefault("trust_remote_code", True)
        if device is not None:
            kwargs["device"] = device
        return CrossEncoder(model_name, **kwargs)

    # 自定义提供商：动态导入
    factory = import_provider(provider)
    if device is not None:
        kwargs.setdefault("device", device)
    # 重排序模型可能使用不同的参数名（model_name_or_path 等）
    return instantiate_model_provider(
        factory,
        model_name,
        kwargs,
        model_parameter_names=(
            "model",
            "model_name",
            "model_id",
            "model_name_or_path",
        ),
    )


def import_provider(provider: str) -> Callable[..., Any]:
    """Import a provider written as ``package.module:Factory`` or ``package.Factory``.

    从 Python 导入路径动态加载模型提供商的工厂函数或类。
    支持两种格式：
    - package.module:Factory — 冒号分隔模块和属性
    - package.module.Factory — 点号分隔（自动识别最后一段为属性）
    """
    provider = provider.strip()
    # 使用冒号分隔符：明确区模块路径和属性名
    if ":" in provider:
        module_name, attribute = provider.split(":", 1)
    else:
        # 使用点号分隔符：rpartition 从右侧分割，最后一段作为属性名
        module_name, separator, attribute = provider.rpartition(".")
        if not separator:
            raise ModelProviderError(
                f"Unknown model provider {provider!r}; "
                "use a built-in provider or an import path"
            )

    # 模块名和属性名都不能为空
    if not module_name or not attribute:
        raise ModelProviderError(f"Invalid model provider import path: {provider!r}")

    # 动态导入模块
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ModelProviderError(
            f"Cannot import model provider module {module_name!r}"
        ) from error

    # 从模块中逐级获取属性（支持嵌套属性如 module.submodule.Class）
    try:
        value = module
        for part in attribute.split("."):
            value = getattr(value, part)
    except AttributeError as error:
        raise ModelProviderError(
            f"Cannot find model provider attribute {attribute!r} in {module_name!r}"
        ) from error

    # 最终获取的值必须是可调用的（函数或类）
    if not callable(value):
        raise ModelProviderError(f"Model provider {provider!r} is not callable")
    return value


def instantiate_model_provider(
    factory: Callable[..., Any],
    model_name: str,
    model_kwargs: dict[str, Any] | None = None,
    *,
    model_parameter_names: tuple[str, ...] = ("model", "model_name", "model_id"),
) -> Any:
    """Instantiate a custom provider while adapting common model parameter names.

    实例化一个自定义的模型提供商，自动适配常见的模型参数名。
    流程：
    1. 检查工厂函数的签名
    2. 过滤出该函数接受的参数
    3. 找到模型名称对应的参数名并传入
    4. 调用工厂函数创建实例
    """
    kwargs = _clean_kwargs(model_kwargs or {})

    # 检查工厂函数是否接受 **kwargs
    accepts_kwargs = False
    parameters: dict[str, inspect.Parameter] = {}
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        # 无法获取签名（如部分内建函数），直接跳过签名分析
        signature = None

    if signature is not None:
        parameters = dict(signature.parameters)
        # 检查是否有 VAR_KEYWORD（**kwargs）参数
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    # 如果无法获取签名，直接传入 model_name 作为位置参数
    if signature is None:
        return factory(model_name, **kwargs)

    # 过滤 kwargs：只保留工厂函数签名中存在的参数（或 accepts_kwargs=True 时全部保留）
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if accepts_kwargs or key in parameters
    }

    # 在候选模型参数名中，找到第一个被工厂函数接受的参数名
    model_parameter = next(
        (
            name
            for name in model_parameter_names
            if accepts_kwargs or name in parameters
        ),
        None,
    )
    if model_parameter is not None:
        # 用 found 的参数名设置模型名称，其余参数通过 **kwargs 传入
        filtered_kwargs.setdefault(model_parameter, model_name)
        return factory(**filtered_kwargs)

    # 如果没有匹配到任何模型参数名，直接将 model_name 作为第一个位置参数
    return factory(model_name, **filtered_kwargs)


def model_config_fingerprint(
    provider: str,
    model_name: str,
    model_kwargs: dict[str, Any] | None = None,
) -> str:
    """Stable fingerprint for deciding whether persisted embeddings are stale.

    生成模型配置的稳定指纹（SHA-256 哈希），用于判断持久化的嵌入是否需要重建。
    当 provider、model_name 或 model_kwargs 变化时，指纹也会变化。
    """
    payload = {
        "provider": provider,
        "model": model_name,
        "kwargs": model_kwargs or {},
    }
    # 通过 JSON 序列化确保一致性：排序键、不使用 ASCII 转义、default=str 处理非标类型
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_provider(provider: str) -> str:
    """规范化提供商标识：去除空白、转小写、将连字符替换为下划线。

    确保 "TongYi"、"tong-yi"、"tongyi" 都被视为同一个提供商。
    """
    text = str(provider).strip()
    if not text:
        raise ModelProviderError("model provider must not be empty")
    return text.lower().replace("-", "_")


def _clean_kwargs(model_kwargs: dict[str, Any]) -> dict[str, Any]:
    """清理关键字参数：键转为字符串，过滤掉值为 None 的项。

    去除 None 值的目的是避免覆盖工厂函数自身的默认值。
    """
    return {
        str(key): value
        for key, value in model_kwargs.items()
        if value is not None
    }
