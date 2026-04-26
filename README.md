# RAG Server

这是一个本地可复用的 RAG 组件项目，当前包含两部分能力：

- 一个面向中文知识库的 `RAGService`
- 一个基于 LangGraph 的电商客服 CLI Agent

它现在还不是 HTTP API 服务。更准确地说，这个仓库已经把“知识入库、检索、精排、Agent 调用”这层能力搭好了，后续如果要接 `FastAPI`、`Flask` 或其他服务层，可以直接在现有代码上继续封装。

## 当前能力

- 支持 `txt`、`md`、`pdf` 文档入库
- 使用 `DashScopeEmbeddings` 做向量化
- 使用 `FAISS` 做向量检索
- 使用 `jieba + BM25Plus` 做关键词检索
- 支持向量检索与 BM25 混合召回
- 支持 `CrossEncoder` 精排
- 索引和切片元数据持久化到本地磁盘
- 提供一个可直接运行的电商客服命令行 Agent

## 项目结构

```text
.
├── rag_service.py          # RAG 核心实现
├── ecommerce_agent_cli.py  # 电商客服 CLI Agent
├── main.py                 # CLI 启动入口
├── docs/                   # 示例知识库文档
├── data/
│   ├── faiss.index         # 向量索引
│   └── metadata.json       # 切片内容与元数据
├── pyproject.toml          # 项目依赖
└── README.md
```

## 技术栈

- Python 3.12+
- LangChain
- LangGraph
- DashScope
- FAISS
- rank-bm25
- sentence-transformers

## 安装依赖

项目使用 `uv` 管理依赖：

```bash
uv sync
```

## 环境变量

运行前至少需要配置 DashScope Key：

```bash
export DASHSCOPE_API_KEY="你的 DashScope Key"
```

说明：

- `RAGService` 在做向量化时会调用 DashScope embedding
- CLI Agent 使用 `ChatTongyi(model="qwen3-max-2026-01-23")`
- 如果首次执行精排，`CrossEncoder` 模型会下载到本地，首轮会比后续更慢

## 快速开始

### 1. 作为 RAG 组件直接调用

```python
from rag_service import RAGService

rag = RAGService(data_dir="data")

rag.add_documents(
    [
        "./docs/尺码推荐.txt",
        "./docs/颜色选择.txt",
        "./docs/洗涤养护.txt",
    ]
)

results = rag.search("160cm、95斤适合穿什么尺码？", top_k=3)

for item in results:
    print(item["score"], item["source"])
    print(item["content"])
    print("-" * 40)
```

### 2. 启动电商客服 CLI

```bash
uv run python main.py
```

退出方式：

- 输入 `quit`
- 输入 `exit`

## 现有示例知识库

仓库里的 `docs/` 目前放的是一组服饰电商知识文档，包含：

- 尺码推荐
- 颜色选择建议
- 洗涤与养护建议

CLI Agent 不会直接把 `docs/` 目录当作硬编码答案读取，而是通过 `RAGService.search()` 检索相关片段后再组织回复。

## RAG 检索流程

默认 `search()` 的执行流程如下：

1. 向量召回
2. BM25 关键词召回
3. 两路结果按权重融合
4. 取前 `candidate_top_k` 个候选
5. 使用 `CrossEncoder` 精排
6. 返回最终 `top_k` 结果

如果不希望精排，可以传入：

```python
rag.search("查询文案", use_rerank=False)
```

## 核心类说明

### `RAGService`

初始化签名：

```python
RAGService(
    data_dir: str = "data",
    model_name: str = "text-embedding-v4",
    embeddings: Any | None = None,
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
    reranker: Any | None = None,
    reranker_device: str | None = None,
    reranker_batch_size: int = 16,
    default_use_rerank: bool = True,
    default_candidate_top_k: int = 20,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
)
```

常用参数：

- `data_dir`：索引和元数据保存目录
- `model_name`：DashScope embedding 模型名
- `chunk_size`：默认切片大小
- `chunk_overlap`：默认切片重叠大小
- `default_use_rerank`：默认是否启用精排
- `default_candidate_top_k`：精排前保留多少候选

进阶参数：

- `embeddings`：注入自定义 embedding 对象，方便测试或替换模型
- `reranker`：注入自定义精排器
- `reranker_device`：指定精排设备，例如 `cpu` 或 `cuda`
- `reranker_batch_size`：精排批大小

## 常用接口

### `add_documents(file_paths, chunk_size=None, chunk_overlap=None)`

作用：

- 读取文档
- 切片
- 生成向量
- 写入 FAISS
- 重建 BM25
- 持久化索引和元数据

示例：

```python
rag.add_documents(
    ["./docs/商品说明.txt", "./docs/售后政策.pdf"],
    chunk_size=800,
    chunk_overlap=150,
)
```

返回格式：

```python
{
    "added_chunks": 12,
    "sources": ["./docs/商品说明.txt", "./docs/售后政策.pdf"]
}
```

注意：

- 仅支持 `txt`、`md`、`pdf`
- `chunk_overlap` 必须小于 `chunk_size`

### `search_by_vector(query, top_k=3)`

只使用向量召回。

### `search_by_bm25(query, top_k=3)`

只使用 BM25 关键词检索。

### `search_by_hybrid(query, top_k=10, vector_weight=0.7, bm25_weight=0.3)`

混合召回，不做精排。

### `rerank(query, candidates, top_k=None)`

对候选结果执行精排。

### `search(query, top_k=3, vector_weight=0.7, bm25_weight=0.3, use_rerank=None, candidate_top_k=None)`

默认搜索入口，适合业务代码直接调用。

### `reset()`

清空当前知识库的 FAISS、BM25 和元数据。

## 返回结果格式

搜索结果为 `list[dict]`，单条数据结构类似：

```python
{
    "score": 0.91,
    "vector_score": 0.88,
    "bm25_score": 1.0,
    "hybrid_score": 0.916,
    "rerank_score": 3.42,
    "content": "命中的文本片段",
    "source": "docs/尺码推荐.txt",
    "metadata": {"chunk_index": 0},
    "retrieval_mode": "hybrid_rerank",
}
```

字段说明：

- `score`：当前排序最终分数
- `vector_score`：向量检索分数
- `bm25_score`：BM25 分数
- `hybrid_score`：融合后的召回分数
- `rerank_score`：精排分数；未精排时为 `None`
- `content`：命中的文本片段
- `source`：来源文档
- `metadata`：当前包含 `chunk_index`
- `retrieval_mode`：结果来源阶段，如 `vector`、`bm25`、`hybrid`、`hybrid_rerank`

## Agent 说明

CLI Agent 的实现位于 `ecommerce_agent_cli.py`，当前特性如下：

- 使用 `ChatTongyi`
- 使用 LangGraph 编排调用流程
- 暴露一个检索工具 `search_product_knowledge`
- 当问题涉及尺码、材质、颜色、洗护、售后等事实信息时，优先检索知识库
- 如果知识库没有足够信息，会明确表示无法确认，而不是编造答案

这部分很适合作为后续 Web 客服、企业微信机器人或 API 服务的原型。

## 重新构建知识库

如果你修改了 `docs/` 内容，或者想替换成自己的语料，建议显式重建索引。

示例：

```python
from rag_service import RAGService

rag = RAGService(data_dir="data")
rag.reset()
rag.add_documents(
    [
        "./docs/尺码推荐.txt",
        "./docs/颜色选择.txt",
        "./docs/洗涤养护.txt",
    ]
)
```

这里有一个很重要的行为需要注意：

- `data/faiss.index` 和 `data/metadata.json` 是持久化数据
- 删除 `docs/` 原文件，不会自动把已入库片段从索引里删掉
- 如果语料变更较大，最稳妥的做法是先 `reset()` 再重新 `add_documents()`

## 适合怎么继续扩展

这个仓库当前最适合继续往下面几个方向发展：

- 加一层 `FastAPI`，把 `search()` 和 Agent 能力暴露成 HTTP 接口
- 增加文档去重、删除、更新能力
- 给 `data/metadata.json` 增加更丰富的业务元数据
- 增加评测集，验证召回与精排效果
- 为 Agent 增加更多工具，例如订单状态、优惠信息、库存查询

## 一些已知边界

- 当前没有 Web 服务层
- 当前没有“按文档删除已入库数据”的接口，只有整体 `reset()`
- 当前依赖 DashScope，离线环境下不能直接做向量化或调用 Tongyi
- 精排模型首次加载会下载权重，部署前最好先预热一次

## 启动命令汇总

安装依赖：

```bash
uv sync
```

启动 CLI：

```bash
uv run python main.py
```

如果后续要把它集成进你自己的服务，核心入口通常就是：

```python
from rag_service import RAGService
```
