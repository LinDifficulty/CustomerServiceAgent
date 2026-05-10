# RAG Server 的 CLI 入口文件
# 从 rag_server.cli 模块导入 main 函数并执行
# 运行方式：python main.py [参数]
# 常用参数示例：
#   python main.py --query-rewrite on --bm25 on --cross-encoder off --memory on --trace on
from rag_server.cli import main


if __name__ == "__main__":
    # 当直接执行本文件时，启动 CLI Agent 主流程
    main()
