# 启用延迟注解求值（PEP 563）
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Callable
from typing import Any

# ── 默认模型提供商和模型名称 ──────────────────────────────────
# 对话模型：默认使用 DeepSeekV4（通过 Tongyi 网关）
DEFAULT_CHAT_PROVIDER = "tongyi"
DEFAULT_CHAT_MODEL = "deepseek-v4-flash"
# 嵌入模型：默认使用阿里云 DashScope
DEFAULT_EMBEDDING_PROVIDER = "dashscope"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
# 重排序模型：默认使用 BGE CrossEncoder
DEFAULT_RERANKER_PROVIDER = "cross_encoder"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class ModelProviderError(ValueError):
    """模型提供商错误：当配置的模型提供商无法实例化时抛出。"""


# ═══════════════════════════════════════════════════════════════
# 内置 Provider 注册表
# 每个 entry: (module_path, class_name)
# 协议说明：
#   - openai 协议: ChatOpenAI / OpenAIEmbeddings，兼容所有 OpenAI 风格 API
#   - anthropic 协议: ChatAnthropic
#   - native 协议: 原生 SDK（ChatTongyi / DashScopeEmbeddings / CrossEncoder）
# ═══════════════════════════════════════════════════════════════

_BUILTIN_CHAT: dict[str, tuple[str, str]] = {
    "tongyi": ("langchain_community.chat_models", "ChatTongyi"),
    "dashscope": ("langchain_community.chat_models", "ChatTongyi"),
    "chat_tongyi": ("langchain_community.chat_models", "ChatTongyi"),
    "openai": ("langchain_openai", "ChatOpenAI"),
    "chat_openai": ("langchain_openai", "ChatOpenAI"),
    "anthropic": ("langchain_community.chat_models", "ChatAnthropic"),
}

_BUILTIN_EMBEDDINGS: dict[str, tuple[str, str]] = {
    "dashscope": ("langchain_community.embeddings", "DashScopeEmbeddings"),
    "dashscope_embeddings": ("langchain_community.embeddings", "DashScopeEmbeddings"),
    "openai": ("langchain_openai", "OpenAIEmbeddings"),
    "openai_embeddings": ("langchain_openai", "OpenAIEmbeddings"),
}

_BUILTIN_RERANKER: dict[str, tuple[str, str]] = {
    "cross_encoder": ("sentence_transformers", "CrossEncoder"),
    "sentence_transformers": ("sentence_transformers", "CrossEncoder"),
    "sentence_transformers_cross_encoder": ("sentence_transformers", "CrossEncoder"),
}


# ═══════════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════════


def create_chat_model(
    provider: str = DEFAULT_CHAT_PROVIDER,
    model_name: str = DEFAULT_CHAT_MODEL,
    **model_kwargs: Any,
) -> Any:
    """创建对话模型实例。

    支持三种接入方式：
    1. 内置 provider 名称 —— "tongyi"、"openai"、"anthropic" 等
       - openai 协议兼容所有 OpenAI 风格 API，可通过 openai_api_base 指定自定义端点
    2. 自定义 Python 导入路径 —— "package.module:ClassName"

    Examples:
        # 内置 Tongyi
        create_chat_model("tongyi", "qwen3-max")

        # OpenAI 兼容协议（可接入任何兼容 API）
        create_chat_model("openai", "gpt-4o")

        # 通过 OpenAI 协议接入通义 DashScope 兼容端点
        create_chat_model("openai", "qwen-plus",
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            openai_api_key=os.environ["DASHSCOPE_API_KEY"])

        # Anthropic 协议
        create_chat_model("anthropic", "claude-sonnet-4-6")

        # 自定义导入
        create_chat_model("my_package.models:MyChatClass", "my-model")
    """
    normalized = _normalize_provider(provider)
    kwargs = _clean_kwargs(model_kwargs)

    # 查找内置 provider
    cls = _get_builtin_class(_BUILTIN_CHAT, normalized)
    if cls is not None:
        kwargs.setdefault("max_retries", 0)
        return cls(model=model_name, **kwargs)

    # fallback：自定义导入路径
    factory = import_provider(provider)
    kwargs.setdefault("max_retries", 0)
    return instantiate_model_provider(factory, model_name, kwargs)


def create_embeddings(
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    **model_kwargs: Any,
) -> Any:
    """创建嵌入模型实例。

    支持三种接入方式：
    1. 内置 provider 名称 —— "dashscope"、"openai" 等
       - openai 协议兼容所有 OpenAI 风格 Embeddings API
    2. 自定义 Python 导入路径 —— "package.module:ClassName"

    Examples:
        # 内置 DashScope
        create_embeddings("dashscope", "text-embedding-v4")

        # OpenAI 兼容协议
        create_embeddings("openai", "text-embedding-3-small")

        # 自定义导入
        create_embeddings("my_package.models:MyEmbeddings", "my-model")
    """
    normalized = _normalize_provider(provider)
    kwargs = _clean_kwargs(model_kwargs)

    # 查找内置 provider
    cls = _get_builtin_class(_BUILTIN_EMBEDDINGS, normalized)
    if cls is not None:
        return cls(model=model_name, **kwargs)

    # fallback：自定义导入路径
    factory = import_provider(provider)
    return instantiate_model_provider(factory, model_name, kwargs)


def create_reranker(
    provider: str = DEFAULT_RERANKER_PROVIDER,
    model_name: str = DEFAULT_RERANKER_MODEL,
    *,
    device: str | None = None,
    trust_remote_code: bool = True,
    **model_kwargs: Any,
) -> Any:
    """创建重排序模型实例。

    支持两种接入方式：
    1. 内置 provider —— "cross_encoder"（SentenceTransformers CrossEncoder 本地模型）
    2. 自定义 Python 导入路径 —— "package.module:ClassName"

    Args:
        device: 设备选择（"cuda"、"cpu"、"mps" 等），None 表示自动选择
        trust_remote_code: 是否信任远程代码，仅对 HuggingFace 模型生效
    """
    normalized = _normalize_provider(provider)
    kwargs = _clean_kwargs(model_kwargs)

    # 查找内置 provider
    cls = _get_builtin_class(_BUILTIN_RERANKER, normalized)
    if cls is not None:
        kwargs.setdefault("trust_remote_code", trust_remote_code)
        if device is not None:
            kwargs["device"] = device
        return cls(model_name, **kwargs)

    # fallback：自定义导入路径
    factory = import_provider(provider)
    if device is not None:
        kwargs.setdefault("device", device)
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


# ═══════════════════════════════════════════════════════════════
# 自定义 Provider 导入
# ═══════════════════════════════════════════════════════════════


def import_provider(provider: str) -> Callable[..., Any]:
    """从 Python 导入路径动态加载模型提供商的工厂函数或类。

    支持两种格式：
    - package.module:Factory —— 冒号分隔模块和属性
    - package.module.Factory —— 点号分隔（自动识别最后一段为属性）
    """
    provider = provider.strip()
    if ":" in provider:
        module_name, attribute = provider.split(":", 1)
    else:
        module_name, separator, attribute = provider.rpartition(".")
        if not separator:
            raise ModelProviderError(f"Unknown model provider {provider!r}; use a built-in provider or an import path")

    if not module_name or not attribute:
        raise ModelProviderError(f"Invalid model provider import path: {provider!r}")

    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ModelProviderError(f"Cannot import model provider module {module_name!r}") from error

    try:
        value = module
        for part in attribute.split("."):
            value = getattr(value, part)
    except AttributeError as error:
        raise ModelProviderError(f"Cannot find model provider attribute {attribute!r} in {module_name!r}") from error

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
    """实例化自定义模型提供商，自动适配常见的模型参数名。

    流程：
    1. 检查工厂函数的签名
    2. 过滤出该函数接受的参数
    3. 找到模型名称对应的参数名并传入
    4. 调用工厂函数创建实例
    """
    kwargs = _clean_kwargs(model_kwargs or {})

    accepts_kwargs = False
    parameters: dict[str, inspect.Parameter] = {}
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        parameters = dict(signature.parameters)
        accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())

    if signature is None:
        return factory(model_name, **kwargs)

    filtered_kwargs = {key: value for key, value in kwargs.items() if accepts_kwargs or key in parameters}

    model_parameter = next(
        (name for name in model_parameter_names if accepts_kwargs or name in parameters),
        None,
    )
    if model_parameter is not None:
        filtered_kwargs.setdefault(model_parameter, model_name)
        return factory(**filtered_kwargs)

    return factory(model_name, **filtered_kwargs)


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════


def model_config_fingerprint(
    provider: str,
    model_name: str,
    model_kwargs: dict[str, Any] | None = None,
) -> str:
    """生成模型配置的稳定指纹（SHA-256 哈希），用于判断持久化嵌入是否需要重建。

    当 provider、model_name 或 model_kwargs 变化时，指纹也会变化。
    """
    payload = {
        "provider": provider,
        "model": model_name,
        "kwargs": model_kwargs or {},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_provider(provider: str) -> str:
    """规范化提供商标识：去除空白、转小写、将连字符替换为下划线。"""
    text = str(provider).strip()
    if not text:
        raise ModelProviderError("model provider must not be empty")
    return text.lower().replace("-", "_")


def _clean_kwargs(model_kwargs: dict[str, Any]) -> dict[str, Any]:
    """清理关键字参数：键转为字符串，过滤掉值为 None 的项。"""
    return {str(key): value for key, value in model_kwargs.items() if value is not None}


def _get_builtin_class(registry: dict[str, tuple[str, str]], provider: str) -> type | None:
    """从注册表中查找并导入内置 provider 的类。如果未找到则返回 None。"""
    entry = registry.get(provider)
    if entry is None:
        return None
    module = importlib.import_module(entry[0])
    cls = getattr(module, entry[1])
    if not inspect.isclass(cls):
        return None
    return cls
