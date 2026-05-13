# RAG Server

本地可复用的中文 RAG / 智能体组件库，当前面向电商客服知识库场景。CLI 智能体展示名称为 **Tulip Agent**。

它不是 HTTP 接口服务，而是一组 Python 组件和一个 CLI 智能体：把 `docs/` 下的文档写入 `data/`，用 FAISS 多向量检索、BM25、可选 CrossEncoder 精排完成知识库召回，再按需叠加查询改写、长期记忆、Skills、MCP 工具、Redis 缓存、链路追踪和检索评测。

更完整的实现说明见 [项目技术文档.md](./项目技术文档.md)。

## 快速开始

```bash
uv sync
export DASHSCOPE_API_KEY="你的 DashScope 密钥"

uv run python examples/ingest_documents.py
uv run rag-cli
```

退出 CLI 时输入 `quit` 或 `exit`。

默认模型配置：

- 聊天 / 智能体模型：`tongyi` + `deepseek-v4-flash`
- 向量模型：`dashscope` + `text-embedding-v4`
- 精排模型：`cross_encoder` + `BAAI/bge-reranker-v2-m3`，默认关闭

使用默认配置时必须设置 `DASHSCOPE_API_KEY`。首次开启 CrossEncoder 会下载较大的模型权重。

## 适合什么场景

- 快速验证中文知识库检索效果
- 在 RAG 上加入查询改写、长期记忆、技能和外部工具
- 把本地 Python 组件接进你自己的应用层
- 做检索链路追踪、离线评测和效果回归

当前没有 Web 服务层；如果需要 FastAPI / Flask / WebSocket，需要在这个底座上自行封装。

## 核心能力

- `RAGService`：文档入库、FAISS 多向量召回（summary/keyword/semantic）、BM25、混合检索、可选精排
- `rag-cli`：基于 LangGraph 的命令行客服智能体（Tulip Agent），支持流式输出与思考状态展示
- `LLMQueryRewriter`：查询改写和多查询融合检索
- `MemoryService`：按 `user_id` 隔离的长期记忆（profile/episode/procedure 三层）
- `SkillRegistry`：Anthropic 风格 `SKILL.md` 技能加载
- `MCP Client`：把 MCP 服务器工具接成 LangChain 工具
- `JsonCache`：Redis / 内存缓存，加速重复查询，连接失败自动降级
- `TraceRecorder`：JSONL 链路追踪，支持实时事件打印
- `eval_runner`：检索评测和指标输出
- `LLMRetryPolicy`：LLM 调用统一重试与退避
- `ReflectionService`：回答后事实审校与修正

支持文档格式：`.txt`、`.md`、`.pdf`。PDF 使用 `pypdf` 抽取文本，不包含 OCR。

## Python 用法

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

常用入口：

```python
rag.search("查询", top_k=3)
rag.search_by_vector("查询", top_k=5)
rag.search_by_bm25("查询", top_k=5)
rag.search_by_hybrid("查询", top_k=10)
rag.rerank("查询", candidates, top_k=3)

await rag.asearch("查询", top_k=3)
await rag.asearch_by_hybrid("查询", top_k=10)
await rag.arerank("查询", candidates, top_k=3)
```

文档生命周期：

```python
rag.list_documents()
rag.update_document("./docs/商品说明.txt")
rag.delete_document("./docs/商品说明.txt")
rag.sync_documents(file_list, remove_missing=True)
```

## CLI 用法

查看完整参数：

```bash
uv run rag-cli --help
```

常用启动方式：

```bash
# 只做向量检索
uv run rag-cli --query-rewrite off --bm25 off

# 开启精排和追踪
uv run rag-cli --cross-encoder on --trace on

# 使用独立数据目录和记忆目录
uv run rag-cli --data-dir ./runtime/data --memory-dir ./runtime/memory

# 使用指定用户记忆空间
uv run rag-cli --user-id user_001

# 启用 MCP 工具
uv run rag-cli --mcp on --mcp-config ./mcp_servers.json

# 关闭 Redis 缓存（默认开启）
uv run rag-cli --cache off

# 指定 Redis 地址
uv run rag-cli --redis-url redis://localhost:6379/0
```

常用开关：

- `--query-rewrite on|off|rewrite_only|multi_query`
- `--bm25 on|off`
- `--cross-encoder on|off`
- `--reflection on|off`
- `--memory on|off`
- `--skills on|off`
- `--mcp on|off`
- `--trace on|off`
- `--live-events on|off`
- `--show-config on|off`
- `--stream-output on|off`
- `--cache on|off`

CLI 内置命令：

- `/help`：显示快捷帮助和可用命令列表
- `/clear`：清空当前会话上下文
- `/memory`：查看当前用户长期记忆
- `/remember 内容`：写入长期偏好或指令
- `/remember-episode 内容`：写入历史事件
- `/remember-procedure 内容`：写入可复用流程
- `/forget 记忆ID前缀`：删除一条记忆
- `/clear-memory`：清空当前用户记忆
- `/sizing-advice 问题`：调用尺码建议技能
- `/care-guidance 问题`：调用洗涤养护技能

## 配置

CLI 配置合并顺序：

```text
内置默认值 -> --config / RAG_SERVER_CONFIG -> RAG_SERVER_* 环境变量 -> CLI 参数
```

配置文件支持 `.toml` / `.json`。常用环境变量包括：

- `RAG_SERVER_DATA_DIR`
- `RAG_SERVER_MEMORY_DIR`
- `RAG_SERVER_AGENT_MODEL`
- `RAG_SERVER_QUERY_REWRITE`
- `RAG_SERVER_BM25`
- `RAG_SERVER_CROSS_ENCODER`
- `RAG_SERVER_MCP_CONFIG`
- `RAG_SERVER_TRACE`
- `RAG_SERVER_CACHE`
- `RAG_SERVER_REDIS_URL`

模型 provider 支持内置名称，也支持 `package.module:Factory` 或 `package.module.Factory` 形式的 Python import path。内置 provider：

- 聊天模型：`tongyi`，可选 `openai`
- 嵌入模型：`dashscope`，可选 `openai`
- 重排序模型：`cross_encoder`

使用 `openai` provider 需要额外安装 `langchain-openai`。

CLI 交互式输入支持 Tab 补全（`readline`）或增强的实时补全菜单（`prompt-toolkit`，可选依赖）。

## 扩展能力

### 长期记忆

```python
from rag_server import MemoryService

memory = MemoryService(data_dir="memory")
memory.add_memory("user_001", "用户偏好通勤风格，喜欢基础色。")
results = memory.search_memory("user_001", "这件适合我平时上班穿吗？")
layered = memory.search_memory_layers("user_001", "这件适合我平时上班穿吗？")
```

记忆分为 `profile`、`episode`、`procedure` 三层，并按 `user_id` 隔离。

### 技能

项目级技能放在 `.claude/skills/<skill-name>/SKILL.md`。CLI 默认支持：

- `sizing-advice`
- `care-guidance`

### MCP

复制 [mcp_servers.example.json](./mcp_servers.example.json) 后按需修改，再启动：

```bash
uv run rag-cli --mcp on --mcp-config ./mcp_servers.json
```

默认 [mcp_servers.json](./mcp_servers.json) 是空配置。

### 追踪

```bash
uv run rag-cli --trace on --trace-dir traces
```

追踪输出为 JSONL，会记录 RAG、查询改写、反思、智能体、记忆、技能、MCP 和评测等事件，并自动脱敏常见敏感键名。

CLI 默认开启回答流式输出。需要关闭时使用 `--stream-output off`，或设置 `RAG_SERVER_STREAM_OUTPUT=off`。

### 缓存

Redis 缓存默认开启，会缓存 query rewrite、embedding、检索结果、rerank 结果和记忆检索。Redis 连接失败时自动降级为无缓存模式。

关闭缓存或自定义 Redis 地址：

```bash
uv run rag-cli --cache off
uv run rag-cli --redis-url redis://your-redis:6379/0
```

各 TTL 可通过 `--cache-query-rewrite-ttl`、`--cache-embedding-ttl` 等参数调整。

### 检索评测

```bash
uv run python -m rag_server.eval_runner \
  --dataset evals/retrieval_eval.jsonl \
  --data-dir data \
  --top-k 3 \
  --cross-encoder off \
  --output evals/latest_report.json
```

核心指标包括 `hit_rate`、`mrr`、`source_hit_rate`、`substring_hit_rate`。

## 项目结构

```text
.
├── rag_server/              # 核心组件
│   ├── cli.py               #   LangGraph Agent 编排、工具定义、节点工厂
│   ├── cli_view.py          #   CLI 展示层：样式、实时事件、斜杠命令、输入补全
│   ├── rag_service.py       #   文档入库、FAISS多向量、BM25、混合检索、精排
│   ├── config.py            #   四层优先级配置加载与校验
│   ├── model_factory.py     #   模型 provider 工厂
│   ├── memory_service.py    #   SQLite + FAISS 长期记忆
│   ├── skill_service.py     #   Anthropic 风格 Skills
│   ├── mcp_service.py       #   MCP 客户端配置与工具加载
│   ├── cache_service.py     #   Redis / 内存缓存服务
│   ├── trace_service.py     #   JSONL 链路追踪与事件总线
│   ├── query_rewrite.py     #   LLM 查询改写与多查询融合检索
│   ├── reflection_service.py #  回答后事实审校与修正
│   ├── eval_service.py      #   检索评测核心逻辑
│   ├── eval_runner.py       #   检索评测 CLI
│   ├── llm_retry.py         #   LLM 统一重试与退避
│   ├── utils.py             #   通用工具函数
│   └── __init__.py          #   公开 API 导出
├── docs/                    # 示例知识库文档
├── data/                    # 本地索引与元数据（FAISS、metadata.json、documents.json）
├── evals/                   # 检索评测数据集
├── prompts/                 # 系统提示词模板
├── examples/                # 示例脚本（入库等）
├── tests/                   # 单元测试
├── .github/                 # CI 工作流（GitHub Actions）
├── .claude/                 # Claude Code 项目配置与 Skills
├── CLAUDE.md                # Claude Code 项目指令
├── LICENSE                  # MIT 许可证
├── config.example.toml      # 配置文件模板
├── mcp_servers.json         # MCP 配置（默认空）
├── mcp_servers.example.json # MCP 配置模板
├── pyproject.toml           # 包元信息、依赖、脚本入口
├── uv.lock                  # uv 锁定依赖
├── .pre-commit-config.yaml  # pre-commit 钩子
└── 项目技术文档.md           # 详细技术文档
```

## 开发与测试

```bash
# 运行所有测试
uv run python -m pytest tests/ -v

# 运行单个测试文件
uv run python -m unittest tests/test_rag_service.py

# 运行单个测试用例
uv run python -m unittest tests.test_rag_service.RAGServiceLifecycleTest.test_add_documents_is_idempotent_and_updates_changed_source

# 代码检查
uv run ruff check rag_server/ tests/
```

CI 通过 GitHub Actions 自动运行；提交前通过 `pre-commit` 钩子做格式化和基础检查。

运行期产物主要包括：

- `data/`
- `memory/`
- `traces/`
- `evals/*_report.json`

## 已知边界

- 没有 HTTP / Web 服务层
- 默认模型能力依赖 DashScope / Tongyi
- 本地索引和记忆更适合单机、单写入者场景
- 异步入口主要优化单进程内并发，不提供多进程写入保护
- CrossEncoder 默认关闭，首次开启会下载较大的模型权重
- PDF 不支持 OCR，只做文本抽取
