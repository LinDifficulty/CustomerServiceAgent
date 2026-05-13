# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Server is a local, reusable RAG component library for Chinese-language e-commerce knowledge bases. It includes hybrid retrieval (FAISS + BM25 + CrossEncoder rerank), a LangGraph CLI Agent, per-user long-term memory, Anthropic-style Skills, MCP client integration, and retrieval evaluation. This is a library + CLI prototype — not an HTTP service.

## Common Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync

# Run the CLI Agent
uv run rag-cli
uv run rag-cli --query-rewrite on --bm25 on --cross-encoder off --memory on --trace on

# Run all tests
uv run python -m pytest tests/ -v

# Run a single test file
uv run python -m unittest tests/test_rag_service.py

# Run a single test case
uv run python -m unittest tests.test_rag_service.RAGServiceLifecycleTest.test_add_documents_is_idempotent_and_updates_changed_source

# Ingest example documents into the knowledge base
uv run python examples/ingest_documents.py

# Run retrieval evaluation
uv run python -m rag_server.eval_runner \
  --dataset evals/retrieval_eval.jsonl \
  --data-dir data --top-k 3 --cross-encoder off \
  --output evals/latest_report.json
```

## Required Environment Variables

- **`DASHSCOPE_API_KEY`**: Required for DashScope embeddings and Tongyi LLM. All embedding/LLM calls go through Alibaba Cloud DashScope.

## Architecture

### Core Modules (all under `rag_server/`)

- **`rag_service.py`** — `RAGService`: document ingestion (txt/md/pdf), chunking, FAISS vector + BM25 keyword hybrid retrieval, CrossEncoder reranking, idempotent upsert via content hashing and `documents.json` manifest
- **`cli.py`** — LangGraph `StateGraph` Agent with flow: `load_memory → load_skills → agent ⇄ tools → save_memory`. Uses `ChatTongyi(model="qwen3-max-2026-01-23")`
- **`memory_service.py`** — `MemoryService`: SQLite + per-user FAISS indexes, 3-layer memory (profile/episode/procedure), soft-delete with `deleted_at`
- **`skill_service.py`** — `SkillRegistry`: discovers `.claude/skills/<name>/SKILL.md` files with YAML frontmatter, progressive disclosure (metadata first, full content on demand)
- **`query_rewrite.py`** — `LLMQueryRewriter`: single rewrite or multi-query fusion search
- **`mcp_service.py`** — MCP client config loading with `${ENV_VAR}` expansion, supports stdio/http/sse/websocket transports
- **`eval_service.py`** / **`eval_runner.py`** — Retrieval eval (hit_rate, MRR, source/substring match)
- **`trace_service.py`** — `TraceRecorder`: append-only JSONL tracing for RAG, agent, and eval events
- **`cache_service.py`** — `JsonCache` abstract interface with `InMemoryJsonCache` and `RedisJsonCache` implementations; TTL-based key eviction, default enabled via config
- **`llm_retry.py`** — `LLMRetryPolicy`: exponential backoff retry for LLM API calls with configurable max retries, jitter, and trace-aware failure logging
- **`reflection_service.py`** — `ReflectionService`: post-hoc hallucination detection and answer revision using a separate LLM call with supplemental retrieval
- **`config.py`** — `AppConfig`: frozen-dataclass configuration with 4-level priority (defaults < TOML < JSON < env vars), alias normalization, and `config.example.toml` reference

### RAG Search Pipeline

1. Vector recall (FAISS IndexFlatIP, L2-normalized)
2. BM25 keyword recall (jieba tokenization)
3. Weighted hybrid fusion (default 0.7 vector / 0.3 BM25)
4. CrossEncoder rerank (`BAAI/bge-reranker-v2-m3`, lazy-loaded, ~700MB first download)
5. Return top_k results

### Agent Tools

- `search_product_knowledge` — RAG retrieval
- `load_skill` / `read_skill_file` — on-demand Skill loading
- Optional MCP server tools (enabled via `--mcp on --mcp-config ./mcp_servers.json`)

### Data Layout

- `data/` — persisted FAISS index, metadata.json, documents.json (DO NOT delete casually)
- `memory/` — SQLite DB + per-user FAISS indexes (gitignored)
- `traces/` — JSONL trace output (gitignored)
- `.claude/skills/` — project-level Anthropic-style Skills

## Testing Patterns

- **Framework**: `unittest.TestCase` (runs via pytest)
- **Isolation**: each test creates a `tempfile.TemporaryDirectory()`
- **No API calls**: `FakeEmbeddings` provides deterministic keyword-based vectors; `FakeModel`/`FakeMemoryExtractor` stub LLM calls
- **Reranker disabled**: all RAG tests use `default_use_rerank=False` to skip CrossEncoder model download
- CI via GitHub Actions (`.github/workflows/test.yml`); linting/formatting via `ruff`; pre-commit hooks configured

## Code Conventions

- `from __future__ import annotations` in all modules
- Type annotations: `str | None`, `list[dict]`, `dict[str, Any]`
- Frozen dataclasses for value objects (`SkillDefinition`, `MCPConfig`)
- Private methods prefixed with `_`
- Chinese user-facing messages; English internal/API docstrings
- Composition over inheritance; no abstract base classes

## Key Constraints

- Document formats: `.txt`, `.md`, `.pdf` only
- `chunk_overlap` must be < `chunk_size`
- DashScope API required — no offline mode for embeddings/LLM
- Memory FAISS indexes are per-user to prevent cross-user recall
- Skill directory names must match the `name` field in YAML frontmatter (lowercase kebab-case)
