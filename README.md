# RAG Server

本地可复用的中文 RAG / Agent 组件库，当前围绕电商客服知识库场景搭建。它不是 HTTP API 服务，而是一组可直接复用的 Python 包、一个 CLI Agent，以及配套的长期记忆、Skills、MCP、Trace 和评测能力。

如果你后续想接 FastAPI、Flask 或别的应用层，这个仓库更像一个底座，而不是现成的服务进程。

## 适合什么场景

- 想快速验证中文知识库检索效果
- 想在 RAG 上叠加 query rewrite、长期记忆、Skills、MCP tools、trace 和 eval
- 想复用本地 Python 组件，而不是直接接一个封装好的在线 API

## 使用前先知道

- 所有向量化和 LLM 默认依赖 DashScope，必须设置 `DASHSCOPE_API_KEY`
- 当前没有 Web 服务层；对外入口是 Python API 和 CLI
- CrossEncoder 精排默认关闭；首次开启会下载 `BAAI/bge-reranker-v2-m3` 权重，体积约 700MB
- 索引和记忆基于本地文件持久化，更适合单机、单写入者场景
- CLI 目前固定使用 `data/` 和 `memory/` 目录，尚未开放 `--data-dir` / `--memory-dir`
- PDF 解析依赖 `pypdf` 文本抽取，不包含 OCR

## 阅读路径

- 想先跑起来：看下面的“5 分钟上手”
- 想作为库集成：看“作为 Python 包使用”
- 想理解架构和模块边界：看 [项目技术文档](./项目技术文档.md)
- 想扩展 Skills、MCP、评测：看“扩展能力”和 `tests/`

## 5 分钟上手

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

```bash
export DASHSCOPE_API_KEY="你的 DashScope Key"
```

默认 embedding 模型是 DashScope `text-embedding-v4`，默认 Agent / query rewrite / memory extractor 模型是 `qwen3-max-2026-01-23`。

### 3. 把示例文档入库

```bash
uv run python examples/ingest_documents.py
```

这一步会把 `docs/` 下的示例知识库写入 `data/`。

### 4. 启动 CLI Agent

```bash
uv run rag-cli
```

或保持兼容入口：

```bash
uv run python main.py
```

退出时输入 `quit` 或 `exit`。

## 核心组件

- `RAGService`：向量检索、BM25、混合召回、CrossEncoder 精排、文档生命周期管理
- `CLI Agent`：基于 LangGraph 的命令行客服 Agent
- `LLMQueryRewriter`：query 改写和多 query 融合检索
- `MemoryService`：按用户隔离的长期记忆
- `SkillRegistry`：Anthropic-style Skills 发现、按需加载和受控读取
- `MCP Client`：把外部 MCP Server 工具接成 LangChain tools
- `TraceRecorder`：JSONL 格式的运行链路记录
- `LLMRetryPolicy`：统一控制 LLM 调用的超时、有限重试和退避
- `Retrieval Eval`：检索评测与指标输出

## 作为 Python 包使用

### 最小检索示例

```python
from rag_server import RAGService

rag = RAGService(data_dir="data")

rag.add_documents([
    "./docs/尺码推荐.txt",
    "./docs/颜色选择.txt",
    "./docs/洗涤养护.txt",
])

results = rag.search("160cm、95斤适合穿什么尺码？", top_k=3)

for item in results:
    print(item["score"], item["source"])
    print(item["content"])
```

### 常用检索入口

```python
rag.search("查询", top_k=3)
rag.search("查询", use_bm25=False)
rag.search("查询", use_rerank=False)

rag.search_by_vector("查询", top_k=5)
rag.search_by_bm25("查询", top_k=5)
rag.search_by_hybrid("查询", top_k=10, vector_weight=0.7, bm25_weight=0.3)
rag.rerank("查询", candidates, top_k=3)
```

### 文档生命周期接口

```python
rag.list_documents()
rag.update_document("./docs/商品说明.txt")
rag.delete_document("./docs/商品说明.txt")
rag.sync_documents(file_list, remove_missing=True)
```

支持的文档格式：`.txt`、`.md`、`.pdf`。

## CLI Agent

完整参数可通过下面命令查看：

```bash
uv run rag-cli --help
```

### 常用选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--query-rewrite` | `on` | 改写模式：`on` / `off` / `rewrite_only` / `multi_query` |
| `--bm25` | `on` | 是否启用 BM25 关键词召回 |
| `--cross-encoder` | `off` | 是否启用 CrossEncoder 精排 |
| `--memory` | `on` | 是否启用长期记忆 |
| `--skills` | `on` | 是否启用 Anthropic-style Skills |
| `--mcp` | `off` | 是否加载 MCP 工具 |
| `--trace` | `off` | 是否写入 JSONL trace |
| `--user-id` | `default_user` | 用户记忆隔离标识 |
| `--llm-retry-attempts` | `3` | 每次 LLM 调用最多尝试次数 |
| `--llm-timeout` | `30` | 每次 LLM 尝试的超时时间，单位秒；传 `0` 或负数可关闭 |
| `--llm-retry-backoff` | `1` | 首次重试等待时间，单位秒，之后指数退避 |
| `--max-tool-rounds` | `6` | 单轮用户输入最多允许的 Agent 工具调用轮次 |
| `--max-repeated-tool-calls` | `2` | 单轮用户输入中相同工具调用最多连续重复次数 |

### 常见启动方式

```bash
# 只做向量检索
uv run rag-cli --query-rewrite off --bm25 off

# 开启精排和 trace
uv run rag-cli --cross-encoder on --trace on

# 使用指定用户记忆空间
uv run rag-cli --user-id user_001

# LLM 响应较慢时放宽单次超时
uv run rag-cli --llm-timeout 60 --llm-retry-attempts 3

# 对外部工具较多的场景收紧循环保护
uv run rag-cli --mcp on --max-tool-rounds 4 --max-repeated-tool-calls 1
```

### CLI 内置命令

| 命令 | 说明 |
|------|------|
| `/memory` | 查看当前用户的长期记忆 |
| `/remember 内容` | 写入一条长期指令 / 偏好记忆 |
| `/remember-episode 内容` | 写入一条历史事件摘要 |
| `/remember-procedure 内容` | 写入一条可复用流程 |
| `/forget 记忆ID前缀` | 删除一条记忆 |
| `/clear-memory` | 清空当前用户全部记忆 |
| `/sizing-advice 问题` | 显式调用尺码建议 Skill |
| `/care-guidance 问题` | 显式调用洗涤养护 Skill |

### LLM 重试与循环保护

Agent、query rewrite 和 memory extractor 都使用同一套 LLM 重试策略。默认每次 LLM 调用最多尝试 `3` 次，单次尝试超时 `30` 秒，重试之间做指数退避；只对超时、限流、连接异常、5xx 等临时性错误重试，其他错误会直接失败。

为避免死循环，LangGraph 工具调用还有两层保护：

- 单轮用户输入最多执行 `--max-tool-rounds` 轮工具调用
- 如果模型连续发起完全相同的工具调用，最多允许 `--max-repeated-tool-calls` 次

触发保护后，Agent 会停止继续调用工具并返回一条客服回复。query rewrite 如果重试耗尽，会降级为直接使用原始问题检索，避免辅助改写模型阻断整轮对话。

## 检索流程

`RAGService.search()` 默认流程如下：

```text
向量召回 ──┐
           ├→ 加权融合（0.7 / 0.3）→ candidate_top_k → 可选 CrossEncoder 精排 → top_k 结果
BM25 召回 ─┘
```

结果中会保留 `vector_score`、`bm25_score`、`hybrid_score`、`rerank_score` 等字段，便于调试和评测。更完整的实现说明见 [项目技术文档](./项目技术文档.md)。

## 长期记忆

`MemoryService` 与商品知识库独立存储，按 `user_id` 隔离：

```python
from rag_server import MemoryService

memory = MemoryService(data_dir="memory")

memory.add_memory("user_001", "用户偏好通勤风格，喜欢基础色。")
results = memory.search_memory("user_001", "这件适合我平时上班穿吗？")
layered = memory.search_memory_layers("user_001", "这件适合我平时上班穿吗？")
```

记忆分成三层：

- `profile`：画像、偏好、约束、长期指令
- `episode`：有复用价值的历史事件
- `procedure`：需要长期遵循的流程

## 扩展能力

### Skills

项目级 Skills 放在 `.claude/skills/<skill-name>/SKILL.md`。Agent 启动时只发现技能元信息，真正需要时再调用 `load_skill(name)` 读取完整内容。

内置 Skills：

- `sizing-advice`
- `care-guidance`

### MCP

仓库根目录的 [mcp_servers.example.json](./mcp_servers.example.json) 提供了一个可直接修改的模板；默认的 [mcp_servers.json](./mcp_servers.json) 是一个安全的空配置。

准备好配置后再启用 MCP：

```bash
uv run rag-cli --mcp on --mcp-config ./mcp_servers.json
```

支持的 transport：

- `stdio`
- `http`
- `streamable_http`
- `sse`
- `websocket`

字符串字段支持 `${ENV_VAR}` 和 `${ENV_VAR:-default}` 环境变量展开。

### Trace

```bash
uv run rag-cli --trace on --trace-dir traces
```

`TraceRecorder` 会记录 RAG、query rewrite、Agent、Memory、Skills、Eval 等链路事件，输出为 JSONL。开启 trace 后，LLM 重试失败、query rewrite 降级和 Agent 工具循环保护也会被记录。

### 检索评测

```bash
uv run python -m rag_server.eval_runner \
  --dataset evals/retrieval_eval.jsonl \
  --data-dir data \
  --top-k 3 \
  --cross-encoder off \
  --output evals/latest_report.json
```

当前输出的核心指标包括：

- `hit_rate`
- `mrr`
- `source_hit_rate`
- `substring_hit_rate`

## 项目结构

```text
.
├── main.py
├── README.md
├── 项目技术文档.md
├── pyproject.toml
├── mcp_servers.json
├── mcp_servers.example.json
├── rag_server/
├── docs/
├── data/
├── memory/
├── traces/
├── evals/
├── examples/
└── tests/
```

## 开发与测试

运行单元测试：

```bash
uv run python -m unittest discover -s tests -v
```

仓库里这些目录主要是运行期产物，已经被 `.gitignore` 处理：

- `memory/`
- `traces/`
- `evals/*_report.json`

如果你要继续维护这个项目，推荐按下面顺序阅读：

1. [README.md](./README.md)
2. [项目技术文档.md](./项目技术文档.md)
3. `rag_server/rag_service.py`
4. `rag_server/cli.py`
5. `rag_server/memory_service.py`
6. `rag_server/query_rewrite.py`
7. `rag_server/llm_retry.py`
8. `tests/`

## 已知边界

- 当前没有 HTTP / Web 服务层
- 所有模型能力默认依赖 DashScope，离线环境无法完成 embedding 或 LLM 调用
- LLM 重试是有上限的保护机制，不保证模型服务故障时一定成功；重试耗尽后会返回错误或走降级路径
- CrossEncoder 首次启用会下载较大的模型权重
- 本地文件写入没有做多进程并发保护
- CLI 目前没有暴露 `--data-dir`、`--memory-dir`、`--agent-model`
