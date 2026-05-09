from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from typing import Any, Callable

DEFAULT_CHAT_PROVIDER = "tongyi"
DEFAULT_CHAT_MODEL = "deepseek-v4-flash"
DEFAULT_EMBEDDING_PROVIDER = "dashscope"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_RERANKER_PROVIDER = "cross_encoder"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class ModelProviderError(ValueError):
    """Raised when a configured model provider cannot be constructed."""


def create_chat_model(
    provider: str = DEFAULT_CHAT_PROVIDER,
    model_name: str = DEFAULT_CHAT_MODEL,
    **model_kwargs: Any,
) -> Any:
    """Create a chat model from a named provider or Python import path."""
    normalized = _normalize_provider(provider)
    kwargs = _clean_kwargs(model_kwargs)
    if normalized in {"tongyi", "dashscope", "chat_tongyi"}:
        from langchain_community.chat_models import ChatTongyi

        kwargs.setdefault("max_retries", 0)
        return ChatTongyi(model=model_name, **kwargs)
    if normalized in {"openai", "chat_openai"}:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, **kwargs)

    factory = import_provider(provider)
    kwargs.setdefault("max_retries", 0)
    return instantiate_model_provider(factory, model_name, kwargs)


def create_embeddings(
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    **model_kwargs: Any,
) -> Any:
    """Create an embedding model from a named provider or Python import path."""
    normalized = _normalize_provider(provider)
    kwargs = _clean_kwargs(model_kwargs)
    if normalized in {"dashscope", "dashscope_embeddings"}:
        from langchain_community.embeddings import DashScopeEmbeddings

        return DashScopeEmbeddings(model=model_name, **kwargs)
    if normalized in {"openai", "openai_embeddings"}:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_name, **kwargs)

    factory = import_provider(provider)
    return instantiate_model_provider(factory, model_name, kwargs)


def create_reranker(
    provider: str = DEFAULT_RERANKER_PROVIDER,
    model_name: str = DEFAULT_RERANKER_MODEL,
    *,
    device: str | None = None,
    **model_kwargs: Any,
) -> Any:
    """Create a reranker object from a named provider or Python import path."""
    normalized = _normalize_provider(provider)
    kwargs = _clean_kwargs(model_kwargs)
    if normalized in {
        "cross_encoder",
        "sentence_transformers",
        "sentence_transformers_cross_encoder",
    }:
        from sentence_transformers import CrossEncoder

        kwargs.setdefault("trust_remote_code", True)
        if device is not None:
            kwargs["device"] = device
        return CrossEncoder(model_name, **kwargs)

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


def import_provider(provider: str) -> Callable[..., Any]:
    """Import a provider written as ``package.module:Factory`` or ``package.Factory``."""
    provider = provider.strip()
    if ":" in provider:
        module_name, attribute = provider.split(":", 1)
    else:
        module_name, separator, attribute = provider.rpartition(".")
        if not separator:
            raise ModelProviderError(
                f"Unknown model provider {provider!r}; "
                "use a built-in provider or an import path"
            )

    if not module_name or not attribute:
        raise ModelProviderError(f"Invalid model provider import path: {provider!r}")

    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ModelProviderError(
            f"Cannot import model provider module {module_name!r}"
        ) from error

    try:
        value = module
        for part in attribute.split("."):
            value = getattr(value, part)
    except AttributeError as error:
        raise ModelProviderError(
            f"Cannot find model provider attribute {attribute!r} in {module_name!r}"
        ) from error

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
    """Instantiate a custom provider while adapting common model parameter names."""
    kwargs = _clean_kwargs(model_kwargs or {})
    accepts_kwargs = False
    parameters: dict[str, inspect.Parameter] = {}
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        parameters = dict(signature.parameters)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    if signature is None:
        return factory(model_name, **kwargs)

    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if accepts_kwargs or key in parameters
    }
    model_parameter = next(
        (
            name
            for name in model_parameter_names
            if accepts_kwargs or name in parameters
        ),
        None,
    )
    if model_parameter is not None:
        filtered_kwargs.setdefault(model_parameter, model_name)
        return factory(**filtered_kwargs)

    return factory(model_name, **filtered_kwargs)


def model_config_fingerprint(
    provider: str,
    model_name: str,
    model_kwargs: dict[str, Any] | None = None,
) -> str:
    """Stable fingerprint for deciding whether persisted embeddings are stale."""
    payload = {
        "provider": provider,
        "model": model_name,
        "kwargs": model_kwargs or {},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_provider(provider: str) -> str:
    text = str(provider).strip()
    if not text:
        raise ModelProviderError("model provider must not be empty")
    return text.lower().replace("-", "_")


def _clean_kwargs(model_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in model_kwargs.items()
        if value is not None
    }
