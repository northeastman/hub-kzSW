# utils/config.py
import os

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"  # 或 deepseek-reasoner

# 系统参数
MAX_MAIN_AGENT_STEPS = 15          # 主Agent最大ReAct步数
MAX_SUB_AGENT_STEPS = 5            # 子Agent最大ReAct步数
MAX_SUBAGENT_DEPTH = 2             # 子Agent嵌套深度限制
MAX_PARALLEL_SUBAGENTS = 5         # 同时运行的子Agent数量上限
DEFAULT_SUBAGENT_TOOLS = []        # 子Agent默认工具列表（空=仅内置基础工具）

# 记忆设置
LONG_TERM_MEMORY_TOP_K = 3         # 长期记忆检索条数