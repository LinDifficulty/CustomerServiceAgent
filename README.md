# RAG Server

本地可复用的 RAG 组件库，面向中文知识库场景。当前包含：

- **RAGService**：向量 + BM25 混合召回 + CrossEncoder 精排的检索服务
- **CLI Agent**：基于 LangGraph 的电商客服命令行 Agent
- **QueryRewriter**：同模型 query 改写与多 query 融合检索
- **MemoryService**：按用户隔离的长期记忆（profile / episode / procedure 三层）
- **SkillRegistry**：Anthropic-style Skills 发现与按需加载
- **MCP Client**：加载外部 MCP Server 工具并注入 Agent
- **TraceRecorder**：JSONL 格式的全链路 trace
- **Retrieval Eval**：检索评测框架，输出 hit rate、MRR 等指标

这不是 HTTP API 服务。核心能力已经封装为 Python 包，后续可直接用 FastAPI / Flask 等框架包装。

## 快速开始

### 环境要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
uv sync
```

### 环境变量

```bash
export DASHSCOPE_API_KEY="你的 DashScope Key"
```

所有向量化（DashScope `text-embedding-v4`）和 LLM 调用（`ChatTongyi`，模型 `qwen3-max-2026-01-23`）都依赖此 Key。

### 入库示例文档

```bash
uv run python examples/ingest_documents.py
```

会把 `docs/` 下的三份示例文档写入 `data/` 索引。

### 启动 CLI Agent

```bash
uv run python main.py
```

退出：输入 `quit` 或 `exit`。

## 作为 RAG 组件调用

```python
from rag_server import RAGService

rag = RAGService(data_dir="data")

# 入库
rag.add_documents([
    "./docs/尺码推荐.txt",
    "./docs/颜色选择.txt",
    "./docs/洗涤养护.txt",
])

# 检索（默认：向量 + BM25 混合召回 → CrossEncoder 精排）
results = rag.search("160cm、95斤适合穿什么尺码？", top_k=3)

for item in results:
    print(item["score"], item["source"])
    print(item["content"])
```

## CLI Agent 选项

```bash
uv run python main.py [OPTIONS]
```

| 选项 | 可选值 | 默认 | 说明 |
|------|--------|------|------|
| `--query-rewrite` | `on` `off` `rewrite_only` `multi_query` | `on` | query 改写模式。`on` 等价于 `multi_query` |
| `--bm25` | `on` `off` | `on` | BM25 关键词召回 |
| `--cross-encoder` | `on` `off` | `off` | CrossEncoder 精排（首次使用会下载模型） |
| `--memory` | `on` `off` | `on` | 长期记忆 |
| `--memory-model` | 模型名 | `qwen3-max-2026-01-23` | 记忆抽取模型 |
| `--skills` | `on` `off` | `on` | Anthropic-style Skills |
| `--skills-dir` | 目录路径 | `.claude/skills` | 追加 Skills 目录，可多次传入 |
| `--mcp` | `on` `off` | `off` | MCP Client 工具加载 |
| `--mcp-config` | 文件路径 | `mcp_servers.json` | MCP Server 配置文件 |
| `--trace` | `on` `off` | `off` | JSONL trace |
| `--trace-dir` | 目录路径 | `traces` | trace 输出目录 |
| `--user-id` | 字符串 | `default_user` | 用户记忆空间隔离 |
| `--rewrite-model` | 模型名 | `qwen3-max-2026-01-23` | query 改写模型 |

示例：

```bash
# 关闭改写和 BM25，只做向量召回
uv run python main.py --query-rewrite off --bm25 off

# 开启精排和 trace
uv run python main.py --cross-encoder on --trace on

# 加载 MCP 工具
uv run python main.py --mcp on --mcp-config ./mcp_servers.json
```

### CLI 内置命令

**记忆管理：**

| 命令 | 说明 |
|------|------|
| `/memory` | 查看当前用户的长期记忆 |
| `/remember 内容` | 写入一条偏好记忆 |
| `/remember-episode 内容` | 写入一条历史事件摘要 |
| `/remember-procedure 内容` | 写入一条可复用流程 |
| `/forget 记忆ID前缀` | 删除一条记忆 |
| `/clear-memory` | 清空当前用户全部记忆 |

**Skill 调用：**

| 命令 | 说明 |
|------|------|
| `/sizing-advice 问题` | 显式调用尺码建议 Skill |
| `/care-guidance 问题` | 显式调用洗涤养护 Skill |

## 检索流程

`RAGService.search()` 的默认流程：

```
向量召回 ──┐
           ├→ 加权融合（0.7 / 0.3）→ 取 candidate_top_k 候选 → CrossEncoder 精排 → top_k 结果
BM25 召回 ─┘
```

可通过参数关闭部分环节：

```python
rag.search("查询", use_bm25=False)      # 只用向量召回
rag.search("查询", use_rerank=False)     # 跳过精排
```

也可以单独调用各阶段：

```python
rag.search_by_vector("查询", top_k=5)
rag.search_by_bm25("查询", top_k=5)
rag.search_by_hybrid("查询", top_k=10, vector_weight=0.7, bm25_weight=0.3)
rag.rerank("查询", candidates, top_k=3)
```

### 搜索结果格式

```python
{
    "score": 0.91,            # 最终排序分数
    "vector_score": 0.88,     # 向量检索分数
    "bm25_score": 1.0,        # BM25 分数
    "hybrid_score": 0.916,    # 融合分数
    "rerank_score": 3.42,     # 精排分数（未精排时为 None）
    "content": "命中的文本片段",
    "source": "docs/尺码推荐.txt",
    "metadata": {"chunk_index": 0},
    "retrieval_mode": "hybrid_rerank",  # vector / bm25 / hybrid / hybrid_rerank
}
```

## 文档管理

`add_documents()` 是幂等 upsert——通过内容 hash 跟踪变更，未变化的文档自动跳过：

```python
result = rag.add_documents(["./docs/商品说明.txt", "./docs/售后政策.pdf"])
# {
#     "added_chunks": 12,
#     "deleted_chunks": 0,
#     "added_documents": ["./docs/商品说明.txt"],
#     "updated_documents": ["./docs/售后政策.pdf"],
#     "skipped_documents": [],
# }
```

其他文档生命周期接口：

```python
rag.list_documents()                                    # 查看 manifest
rag.update_document("./docs/商品说明.txt")                # 单文档 upsert
rag.delete_document("./docs/商品说明.txt")                # 按路径或 doc_id 删除
rag.sync_documents(file_list, remove_missing=True)       # 与目录扫描结果同步
```

支持的文档格式：`.txt`、`.md`、`.pdf`。

## 长期记忆

`MemoryService` 管理按用户隔离的长期记忆，与商品知识库完全独立：

```python
from rag_server import MemoryService

memory = MemoryService(data_dir="memory")

# 写入
memory.add_memory("user_001", "用户偏好通勤风格，喜欢基础色。")

# 语义检索
results = memory.search_memory("user_001", "这件适合我平时上班穿吗？")

# 分层检索（profile / episode / procedure）
layered = memory.search_memory_layers("user_001", "这件适合我平时上班穿吗？")

# 删除
memory.forget_memory(results[0]["id"], user_id="user_001")

# 清空
memory.clear_user_memory("user_001")
```

**三层记忆模型：**

| 层级 | 包含类型 | 用途 |
|------|----------|------|
| `profile` | profile / preference / constraint / instruction | 用户偏好、约束、指令 |
| `episode` | episode | 有复用价值的历史事件摘要 |
| `procedure` | procedure | 用户要求长期遵循的流程 |

**存储结构：** SQLite 保存结构化记录，FAISS 按用户单独维护语义索引（避免跨用户召回竞争）。

## Anthropic-style Skills

项目级 Skills 放在 `.claude/skills/<skill-name>/SKILL.md`，使用 YAML frontmatter：

```markdown
---
name: sizing-advice
description: Use when the customer asks about clothing size or fit.
when_to_use: 用户询问尺码相关问题时使用。
allowed-tools:
  - search_product_knowledge
---

# Sizing Advice

这里写该 Skill 的具体指令和工作流程。
```

**Progressive disclosure 机制：**

1. Agent 启动时只暴露 skill 的 `name`、`description`、`when_to_use`
2. 模型判断需要时调用 `load_skill(name)` 读取完整内容
3. 如果 skill 目录有额外文件，可调用 `read_skill_file(name, path)` 读取
4. 用户也可用 `/skill-name` 前缀显式触发

**Skill 名规则：** 小写字母、数字、连字符（`^[a-z0-9][a-z0-9-]{0,63}$`），目录名必须与 `name` 字段一致。

内置 Skills：

- `sizing-advice`：尺码与版型建议
- `care-guidance`：洗涤养护建议

## MCP Client

CLI Agent 通过 `langchain-mcp-adapters` 把外部 MCP Server 工具转为 LangChain tools，和本地 RAG、Skills 工具一起暴露给模型。

```bash
uv run python main.py --mcp on --mcp-config ./mcp_servers.json
```

配置示例（`mcp_servers.json`）：

```json
{
  "tool_name_prefix": true,
  "servers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    },
    "crm": {
      "transport": "http",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer ${CRM_MCP_TOKEN}"
      },
      "timeout": 10
    }
  }
}
```

**配置说明：**

- 支持 `stdio`、`http`、`streamable_http`、`sse`、`websocket` 五种 transport
- 字符串字段支持 `${ENV_VAR}` 和 `${ENV_VAR:-default}` 环境变量替换
- `tool_name_prefix`（默认开启）为工具名加上 server 前缀，避免重名
- 单个 server 设置 `"enabled": false` 可临时禁用
- 也兼容 server 直接放在顶层的扁平格式

## Trace

`TraceRecorder` 以 JSONL 格式记录全链路事件：

```python
from rag_server import RAGService, TraceRecorder

trace = TraceRecorder(trace_dir="traces")
rag = RAGService(data_dir="data", trace_recorder=trace)
rag.search("真丝连衣裙怎么洗？")
```

CLI 启用：

```bash
uv run python main.py --trace on --trace-dir traces
```

记录范围：

- **RAG**：文档 upsert / delete、vector / BM25 / hybrid / rerank / search
- **Query rewrite**：原 query、改写 query、多 query 检索
- **Agent**：memory / skill 注入、模型调用、工具调用、memory 写入
- **Eval**：每条 case 的命中情况和 summary

## 检索评测

评测集支持 JSON 或 JSONL 格式，示例：

```jsonl
{"id":"silk_care","query":"真丝连衣裙怎么洗？","expected_sources":["docs/洗涤养护.txt"],"expected_substrings":["真丝材质"]}
```

运行：

```bash
uv run python -m rag_server.eval_runner \
  --dataset evals/retrieval_eval.jsonl \
  --data-dir data \
  --top-k 3 \
  --candidate-top-k 10 \
  --cross-encoder off \
  --output evals/latest_report.json
```

输出指标：

| 指标 | 说明 |
|------|------|
| `hit_rate` | 同时满足 source 和 substring 匹配的比例 |
| `mrr` | 第一个命中结果的平均倒数排名 |
| `source_hit_rate` | source 或 doc_id 命中率 |
| `substring_hit_rate` | 内容关键词命中率 |

评测默认开启 trace，可用 `--trace off` 关闭。

## RAGService 参数参考

```python
RAGService(
    data_dir="data",                        # 索引和元数据保存目录
    model_name="text-embedding-v4",         # DashScope embedding 模型
    embeddings=None,                        # 自定义 embedding 对象（测试用）
    reranker_model_name="BAAI/bge-reranker-v2-m3",  # CrossEncoder 模型
    reranker=None,                          # 自定义精排器
    reranker_device=None,                   # 精排设备：cpu / cuda
    reranker_batch_size=16,                 # 精排批大小
    default_use_bm25=True,                  # 默认启用 BM25
    default_use_rerank=True,                # 默认启用精排
    default_candidate_top_k=20,             # 精排前保留候选数
    chunk_size=500,                         # 切片大小
    chunk_overlap=100,                      # 切片重叠（必须小于 chunk_size）
    trace_recorder=None,                    # 注入 TraceRecorder
)
```

## 项目结构

```
.
├── main.py                     # CLI 启动入口
├── rag_server/                 # 核心 Python 包
│   ├── rag_service.py          # RAG 检索服务
│   ├── cli.py                  # LangGraph Agent + CLI REPL
│   ├── query_rewrite.py        # query 改写与多 query 融合
│   ├── memory_service.py       # 用户长期记忆
│   ├── skill_service.py        # Skills 发现与加载
│   ├── mcp_service.py          # MCP Client 配置与工具加载
│   ├── eval_service.py         # 检索评测逻辑
│   ├── eval_runner.py          # 评测 CLI 入口
│   └── trace_service.py        # JSONL trace
├── .claude/skills/             # 项目级 Skills
├── docs/                       # 示例知识库文档
├── data/                       # 持久化索引（faiss.index / metadata.json / documents.json）
├── memory/                     # 用户记忆数据（gitignored）
├── traces/                     # trace 输出（gitignored）
├── evals/                      # 评测数据集
├── examples/                   # 示例脚本
└── pyproject.toml              # 项目依赖（uv）
```

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 语言 | Python 3.12+ |
| 包管理 | uv |
| LLM | ChatTongyi（qwen3-max-2026-01-23）via DashScope |
| Embedding | DashScope text-embedding-v4 |
| 向量检索 | FAISS（IndexFlatIP，L2 归一化） |
| 关键词检索 | jieba 分词 + BM25Plus |
| 精排 | sentence-transformers CrossEncoder（BAAI/bge-reranker-v2-m3） |
| Agent 编排 | LangGraph StateGraph |
| 文档解析 | pypdf（PDF）、原生读取（txt / md） |
| 分片 | LangChain RecursiveCharacterTextSplitter（中文分隔符） |
| MCP | langchain-mcp-adapters |

## 已知边界

- 当前没有 Web 服务层
- 依赖 DashScope，离线环境无法做向量化或 LLM 调用
- CrossEncoder 精排模型首次加载会下载权重（~700MB），部署前建议预热
- 仅支持 `.txt`、`.md`、`.pdf` 格式的文档
