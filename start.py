"""
校园答疑智能客服 - 启动脚本
极速优化版
"""
import os
import sys
import socket
import time

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_port(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result != 0


def kill_port_process(port):
    import subprocess
    try:
        result = subprocess.run(f'netstat -ano | findstr :{port}', shell=True, capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'LISTENING' in line:
                pid = line.strip().split()[-1]
                subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                return True
    except:
        pass
    return False


def main():
    print("=" * 60)
    print("       校园答疑智能客服 - 极速版")
    print("=" * 60)

    api_port = 8000

    # 清理端口
    if not check_port(api_port):
        kill_port_process(api_port)
        time.sleep(0.3)

    # 预加载模型
    print("\n[1/2] 加载Embedding模型...")
    from src.core.cache import get_embedding_model
    get_embedding_model()
    print("      ✓ 完成")

    # 条件预加载 Reranker（避免首问冷启动）
    try:
        from src.core.config import get_config
        from src.core.cache import get_reranker_model
        retriever_cfg = get_config().retriever
        if retriever_cfg.get("enable_rerank", True):
            print("\n[1.5/2] 加载Reranker模型...")
            rerank_cfg = retriever_cfg.get("rerank", {})
            get_reranker_model(rerank_cfg.get("model_name", "BAAI/bge-reranker-base"))
            print("        ✓ 完成")
    except Exception as e:
        print(f"\n[1.5/2] Reranker 加载失败（重排序将回退）: {e}")

    # 启动FastAPI
    print("\n[2/2] 启动FastAPI...")
    import uvicorn
    from src.main import app

    print("\n" + "=" * 60)
    print("  启动完成！")
    print(f"  API:  http://localhost:{api_port}/docs")
    print("=" * 60)
    print("  响应时间: 首Token约3秒，流式输出\n")

    uvicorn.run(app, host="127.0.0.1", port=api_port, log_level="warning")


if __name__ == "__main__":
    main()
