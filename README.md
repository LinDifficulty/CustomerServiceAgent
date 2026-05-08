# RAG Server

本地可复用的中文 RAG / 智能体组件库，当前围绕电商客服知识库场景搭建。它不是 HTTP 接口服务，而是一组可直接复用的 Python 包、一个 CLI 智能体，以及配套的长期记忆、技能、MCP、追踪和评测能力。

如果你后续想接 FastAPI、Flask 或别的应用层，这个仓库更像一个底座，而不是现成的服务进程。

## 适合什么场景

- 想快速验证中文知识库检索效果
- 想在 RAG 上叠加查询改写、长期记忆、技能、MCP 工具、追踪和评测
- 想复用本地 Python 组件，而不是直接接一个封装好的在线接口

## 使用前先知道

- 所有向量化和 LLM 默认依赖 DashScope，必须设置 `DASHSCOPE_API_KEY`
- 当前没有 Web 服务层；对外入口是 Python 接口和 CLI
- CrossEncoder 精排默认关闭；首次开启会下载 `BAAI/bge-reranker-v2-m3` 权重，体积约 700MB
- 索引和记忆基于本地文件持久化，更适合单机、单写入者场景
- CLI 支持通过配置文件、环境变量和命令行参数覆盖 `data/`、`memory/`、模型与开关项
- PDF 解析依赖 `pypdf` 文本抽取，不包含 OCR

## 阅读路径

- 想先跑起来：看下面的“5 分钟上手”
- 想作为库集成：看“作为 Python 包使用”
- 想理解架构和模块边界：看 [项目技术文档](./项目技术文档.md)
- 想扩展技能、MCP、评测：看“扩展能力”和 `tests/`

## 5 分钟上手

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

```bash
export DASHSCOPE_API_KEY="你的 DashScope 密钥"
```

默认嵌入模型是 DashScope `text-embedding-v4`，默认智能体 / 查询改写 / 记忆抽取器模型是 `qwen3-max-2026-01-23`。

### 3. 把示例文档入库

```bash
uv run python examples/ingest_documents.py
```

这一步会把 `docs/` 下的示例知识库写入 `data/`。

### 4. 启动 CLI 智能体

```bash
uv run rag-cli
```

或保持兼容入口：

```bash
uv run python main.py
```

退出时输入 `quit` 或 `exit`。

## 核心组件

- `RAGService`：多向量检索、BM25、混合召回、CrossEncoder 精排、文档生命周期管理
- `CLI Agent`：基于 LangGraph 的命令行客服智能体
- `LLMQueryRewriter`：查询改写和多查询融合检索
- `MemoryService`：按用户隔离的长期记忆
- `SkillRegistry`：Anthropic 风格技能发现、按需加载和受控读取
- `MCP Client`：把外部 MCP 服务器工具接成 LangChain 工具
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

默认采用父子分块：父块提供返回给模型的上下文，子块用于 FAISS / BM25 召回。每个子块会生成 `summary`、`keyword`、`semantic` 三类嵌入文本，并以多向量方式写入 FAISS；召回后会聚合回原子块，再按父块去重返回。`chunk_size` / `chunk_overlap` 表示子块参数，`parent_chunk_size` / `parent_chunk_overlap` 可在初始化或入库时覆盖。

## CLI 智能体

完整参数可通过下面命令查看：

```bash
uv run rag-cli --help
```

### 常用选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 无 | 读取 `.toml` / `.json` 配置文件；也可用 `RAG_SERVER_CONFIG` |
| `--data-dir` | `data` | RAG 索引与元数据目录 |
| `--memory-dir` | `memory` | 长期记忆目录 |
| `--agent-model` | `qwen3-max-2026-01-23` | 智能体主模型 |
| `--rewrite-model` | 同智能体模型 | 查询改写使用的模型 |
| `--memory-model` | 同智能体模型 | 长期记忆抽取使用的模型 |
| `--reflection` | `on` | 是否启用回答后反思、补检索和修正 |
| `--query-rewrite` | `on` | 改写模式：`on` / `off` / `rewrite_only` / `multi_query` |
| `--bm25` | `on` | 是否启用 BM25 关键词召回 |
| `--cross-encoder` | `off` | 是否启用 CrossEncoder 精排 |
| `--memory` | `on` | 是否启用长期记忆 |
| `--memory-top-k` | `5` | 每层长期记忆召回数量 |
| `--skills` | `on` | 是否启用 Anthropic 风格技能 |
| `--skills-dir` | 未追加 | 额外技能目录；默认仍扫描 `.claude/skills`，可重复传入 |
| `--mcp` | `off` | 是否加载 MCP 工具 |
| `--mcp-config` | `mcp_servers.json` | MCP 服务器 JSON 配置路径 |
| `--trace` | `off` | 是否写入 JSONL 追踪 |
| `--trace-dir` | `traces` | 追踪 JSONL 输出目录 |
| `--live-events` | `on` | 是否在 CLI 实时展示 RAG、记忆、技能、MCP 调用 |
| `--user-id` | `default_user` | 用户记忆隔离标识 |
| `--llm-retry-attempts` | `3` | 每次 LLM 调用最多尝试次数 |
| `--llm-timeout` | `30` | 每次 LLM 尝试的超时时间，单位秒；传 `0` 或负数可关闭 |
| `--llm-retry-backoff` | `1` | 首次重试等待时间，单位秒，之后指数退避 |
| `--max-tool-rounds` | `6` | 单轮用户输入最多允许的智能体工具调用轮次 |
| `--max-repeated-tool-calls` | `2` | 单轮用户输入中相同工具调用最多连续重复次数 |

### 常见启动方式

```bash
# 只做向量检索
uv run rag-cli --query-rewrite off --bm25 off

# 开启精排和追踪
uv run rag-cli --cross-encoder on --trace on

# 使用指定用户记忆空间
uv run rag-cli --user-id user_001

# 使用独立数据目录和记忆目录
uv run rag-cli --data-dir ./runtime/data --memory-dir ./runtime/memory

# LLM 响应较慢时放宽单次超时
uv run rag-cli --llm-timeout 60 --llm-retry-attempts 3

# 对外部工具较多的场景收紧循环保护
uv run rag-cli --mcp on --max-tool-rounds 4 --max-repeated-tool-calls 1

# 关闭 CLI 实时事件，只保留正常对话输出
uv run rag-cli --live-events off
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
| `/sizing-advice 问题` | 显式调用尺码建议技能 |
| `/care-guidance 问题` | 显式调用洗涤养护技能 |

### 反思

反思默认开启。智能体生成不含工具调用的最终回复后，会用同一套 LLM 重试策略审查回答是否包含缺少证据支持的商品事实、政策承诺、尺码建议或售后规则；如果需要，会根据建议查询对 RAG 做一次补检索，再用原证据和补充证据修正回复。需要更少模型调用时可用 `--reflection off` 关闭。

### LLM 重试与循环保护

智能体、查询改写、反思和记忆抽取器都使用同一套 LLM 重试策略。默认每次 LLM 调用最多尝试 `3` 次，单次尝试超时 `30` 秒，重试之间做指数退避；只对超时、限流、连接异常、5xx 等临时性错误重试，其他错误会直接失败。

为避免死循环，LangGraph 工具调用还有两层保护：

- 单轮用户输入最多执行 `--max-tool-rounds` 轮工具调用
- 如果模型连续发起完全相同的工具调用，最多允许 `--max-repeated-tool-calls` 次

触发保护后，智能体会停止继续调用工具并返回一条客服回复。查询改写如果重试耗尽，会降级为直接使用原始问题检索，避免辅助改写模型阻断整轮对话。

## 检索流程

`RAGService.search()` 默认流程如下：

```text
摘要嵌入 ──┐
关键词嵌入 ─┼→ 多向量召回并聚合为 vector_score ──┐
语义嵌入 ──┘                                      ├→ 加权融合（0.7 / 0.3）→ candidate_top_k → 可选 CrossEncoder 精排 → top_k 结果
BM25 召回 ──────────────────────────────────────────────────┘
```

结果中会保留 `vector_score`、`multi_vector_scores`、`best_vector_type`、`bm25_score`、`hybrid_score`、`rerank_score` 等字段，便于调试和评测。更完整的实现说明见 [项目技术文档](./项目技术文档.md)。

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

### 技能

项目级技能放在 `.claude/skills/<skill-name>/SKILL.md`。智能体启动时只发现技能元信息，真正需要时再调用 `load_skill(name)` 读取完整内容。

内置技能：

- `sizing-advice`
- `care-guidance`

### MCP

仓库根目录的 [mcp_servers.example.json](./mcp_servers.example.json) 提供了一个可直接修改的模板；默认的 [mcp_servers.json](./mcp_servers.json) 是一个安全的空配置。

准备好配置后再启用 MCP：

```bash
uv run rag-cli --mcp on --mcp-config ./mcp_servers.json
```

支持的传输方式：

- `stdio`
- `http`
- `streamable_http`
- `sse`
- `websocket`

字符串字段支持 `${ENV_VAR}` 和 `${ENV_VAR:-default}` 环境变量展开。

### 配置系统

CLI 会按下面顺序合并配置，后者覆盖前者：

```text
内置默认值 -> --config / RAG_SERVER_CONFIG 文件 -> RAG_SERVER_* 环境变量 -> CLI 参数
```

示例 TOML：

```toml
[paths]
data_dir = "data"
memory_dir = "memory"
trace_dir = "traces"
mcp_config_path = "mcp_servers.json"

[agent]
model = "qwen3-max-2026-01-23"
user_id = "default_user"
max_tool_rounds = 6
max_repeated_tool_calls = 2
reflection_enabled = true

[retrieval]
query_rewrite = "on"
bm25 = true
cross_encoder = false

[llm]
rewrite_model = "qwen3-max-2026-01-23"
memory_model = "qwen3-max-2026-01-23"
retry_attempts = 3
timeout_s = 30
retry_backoff_s = 1

[memory]
enabled = true
top_k = 5

[skills]
enabled = true
dirs = []

[mcp]
enabled = false

[trace]
enabled = false
live = true
```

常用环境变量与配置字段一一对应，例如 `RAG_SERVER_DATA_DIR`、`RAG_SERVER_MEMORY_DIR`、`RAG_SERVER_AGENT_MODEL`、`RAG_SERVER_REWRITE_MODEL`、`RAG_SERVER_MEMORY_MODEL`、`RAG_SERVER_REFLECTION`、`RAG_SERVER_QUERY_REWRITE`、`RAG_SERVER_BM25`、`RAG_SERVER_CROSS_ENCODER`、`RAG_SERVER_SKILLS_DIRS`、`RAG_SERVER_MCP_CONFIG`、`RAG_SERVER_TRACE` 和 `RAG_SERVER_LIVE_EVENTS`。布尔值支持 `on/off`、`true/false`、`yes/no`、`1/0`；`RAG_SERVER_SKILLS_DIRS` 支持逗号分隔多个目录。

### 追踪

```bash
uv run rag-cli --trace on --trace-dir traces
```

`TraceRecorder` 会记录 RAG、查询改写、反思、智能体、记忆、技能、MCP、评测等链路事件，输出为 JSONL。开启追踪后，启动配置、单轮对话耗时、模型用量元数据、LLM 重试失败、查询改写降级和智能体工具循环保护也会被记录。追踪会自动脱敏常见敏感键名，例如 `api_key`、`authorization`、`password`、`token`、`secret`。

CLI 默认会实时打印 RAG 检索、记忆读取、技能加载/读取和 MCP 工具调用事件；它不要求开启 JSONL 追踪。需要安静输出时可用 `--live-events off`，或设置 `RAG_SERVER_LIVE_EVENTS=off`。

### 检索评测

```bash
uv run python -m rag_server.eval_runner \
  --dataset evals/retrieval_eval.jsonl \
  --data-dir data \
  --top-k 3 \
  --cross-encoder off \
  --output evals/latest_report.json \
  --min-hit-rate 0.8 \
  --min-mrr 0.8
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
- 所有模型能力默认依赖 DashScope，离线环境无法完成嵌入或 LLM 调用
- LLM 重试是有上限的保护机制，不保证模型服务故障时一定成功；重试耗尽后会返回错误或走降级路径
- CrossEncoder 首次启用会下载较大的模型权重
- 本地文件写入没有做多进程并发保护
