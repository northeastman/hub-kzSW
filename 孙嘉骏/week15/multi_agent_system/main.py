# main.py
import sys
sys.path.append('.')  # 确保模块可导入

from layers.memory import Memory
from agents.main_agent import MainAgent
from tools import register_all_tools
from utils.config import DEEPSEEK_API_KEY

def main():
    # 检查API Key
    if DEEPSEEK_API_KEY == "your-api-key-here":
        print("请先设置环境变量 DEEPSEEK_API_KEY 或在 utils/config.py 中填写你的 API Key")
        return

    # 注册工具
    register_all_tools()

    # 创建记忆
    memory = Memory()

    # 创建主Agent
    main_agent = MainAgent(memory=memory)

    # 示例任务
    user_query = input("请输入任务描述：")
    if not user_query:
        user_query = "分析人工智能在医疗领域的应用，并给出三个具体案例。"

    print(f"\n主Agent开始处理：{user_query}\n")
    result = main_agent.run(user_query)
    print(f"\n最终答案：\n{result}")

if __name__ == "__main__":
    main()