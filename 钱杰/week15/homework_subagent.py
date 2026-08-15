import re
import json
import time
import urllib.request
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 🔑 API Key 配置区 (请在这里填入你的真实 Key)
# ==========================================
# 推荐使用 DeepSeek API (便宜、速度快、支持良好)
# 注册地址: https://platform.deepseek.com/
LLM_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
LLM_BASE_URL = "https://api.deepseek.com/chat/completions"
LLM_MODEL = "deepseek-chat"

# Tavily 搜索 API (专为大模型优化的搜索引擎)
# 注册地址: https://tavily.com/ (免费额度完全够交作业)
TAVILY_API_KEY = "tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


# ==========================================
# 2. 底层网络请求封装 (LLM & 搜索引擎)
# ==========================================
def call_llm(system_prompt, user_content):
    """真实的大模型调用接口"""
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,  # Agent 推理需要低温度保证确定性
        "stop": ["Observation:"] # ReAct 核心：让模型在需要外部工具输入的地方停下
    }
    
    req = urllib.request.Request(LLM_BASE_URL, data=json.dumps(data).encode("utf-8"), headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    })
    
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        return f"LLM HTTP Error {e.code}: {error_msg}"
    except Exception as e:
        return f"LLM System Error: {str(e)}"

def real_web_search(query):
    """真实的 Tavily 联网搜索工具"""
    print(f"\n      [网络请求] 正在搜索: {query}...")
    url = "https://api.tavily.com/search"
    data = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "max_results": 3
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={
        "Content-Type": "application/json"
    })
    
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode("utf-8"))
            # Tavily 会直接返回一个针对 query 总结好的 answer
            answer = res.get("answer", "")
            # 同时提取前几条搜索结果的摘要
            results_text = "\n".join([f"- {r['title']}: {r['content'][:150]}..." for r in res.get("results", [])])
            return f"【搜索总结】\n{answer}\n\n【相关资料】\n{results_text}"
    except Exception as e:
        return f"搜索失败: {str(e)}"


# ==========================================
# 3. ReAct 循环引擎 (主从 Agent 通用架构)
# ==========================================
class ReActAgent:
    def __init__(self, name, tools, system_prompt, max_steps=5):
        self.name = name
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(self, question):
        print(f"\n[{self.name}] 🚀 启动任务: {question}")
        history = f"Question: {question}\n\n"
        
        for step in range(self.max_steps):
            llm_response = call_llm(self.system_prompt, history)
            
            # 异常处理兜底
            if "Error" in llm_response:
                print(f"[{self.name}] ❌ API报错: {llm_response}")
                return "API 调用失败"

            thought, action, action_input = "", "", ""
            
            # 解析 Thought (思考)
            m_thought = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", llm_response, re.S)
            if m_thought: thought = m_thought.group(1).strip()
                
            # 解析 Final Answer (最终答案)
            m_final = re.search(r"Final Answer:\s*(.*)", llm_response, re.S)
            if m_final:
                print(f"[{self.name}] ✅ 最终得出结论！")
                return m_final.group(1).strip()
                
            # 解析 Action (动作) & Action Input (动作参数)
            m_action = re.search(r"Action:\s*(.*)", llm_response)
            m_input = re.search(r"Action Input:\s*(.*)", llm_response)
            if m_action: action = m_action.group(1).strip()
            if m_input: action_input = m_input.group(1).strip()

            print(f"[{self.name}] 🧠 思考: {thought}")
            print(f"[{self.name}] 🛠️  行动: {action}({action_input})")

            # 路由到具体工具
            if action in self.tools:
                observation = self.tools[action](action_input)
            else:
                # 兜底机制：大模型幻觉输出了不存在的工具
                observation = f"工具 '{action}' 不存在，请使用: {list(self.tools.keys())}"
                
            print(f"[{self.name}] 👁️  观察(截断): {observation[:80].replace(chr(10), ' ')}...")
            
            # 将本轮轨迹拼接入历史，继续下轮循环
            history += llm_response + f"\nObservation: {observation}\n"
            
        return f"[{self.name}] 达到最大步数 {self.max_steps}，未能得出最终结论。"


# ==========================================
# 4. 主 Agent 核心工具: 并行下发调度器
# ==========================================
def dispatch_subagents(action_input):
    """主 Agent 的专属工具：下发多个 Subagent 联网执行"""
    subtasks = [s.strip() for s in action_input.split("|") if s.strip()]
    print(f"\n>>> [调度器] 收到主 Agent 拆分指令，准备【并行下发】 {len(subtasks)} 个子任务...")
    
    # 强制子 Agent 只能用 web_search 工具，且只负责当前细分领域
    SUB_SYSTEM = """你是专业的信息检索研究员。
可用工具:
- web_search: 联网搜索最新信息。参数为查询关键词。

严格按以下格式输出（遇到Observation前停下）：
Thought: 思考我需要搜索什么
Action: web_search
Action Input: 搜索关键词

拿到Observation后，整理输出：
Thought: 收集完毕
Final Answer: 详细的调研结果报告"""
    
    results = []
    start_time = time.time()
    
    # 【并发加速核心】
    with ThreadPoolExecutor(max_workers=len(subtasks)) as pool:
        futures = {}
        for i, task in enumerate(subtasks):
            sub_agent = ReActAgent(
                name=f"SubAgent-{i+1}", 
                tools={"web_search": real_web_search}, 
                system_prompt=SUB_SYSTEM,
                max_steps=4
            )
            futures[pool.submit(sub_agent.run, task)] = task
            
        for fut in as_completed(futures):
            task = futures[fut]
            res = fut.result()
            results.append(f"【子课题: {task}】\n{res}\n")
            
    wall_clock = time.time() - start_time
    print(f"\n>>> [调度器] 所有子智能体已回归！并行真实耗时: {wall_clock:.2f}秒\n")
    return "\n".join(results)


# ==========================================
# 5. 主程序启动入口
# ==========================================
if __name__ == "__main__":
    # 在运行前，确保你已经替换了开头的 API Key
    if "sk-xxxxx" in LLM_API_KEY or "tvly-xxxxx" in TAVILY_API_KEY:
        print("⚠️ 警告：请先在代码开头填入你的 DeepSeek API Key 和 Tavily API Key！")
        exit(1)
        
    MAIN_SYSTEM = """你是高级市场分析总监。
可用工具：
- web_search: 简单单次搜索（参数: 搜索词）。
- dispatch_subagents: 派发多个子智能体并行深度调研（参数: 用 '|' 分隔的多个子课题，如 '销量数据 | 竞争格局 | 政策'）。

决策规则：
1. 遇到复杂、多维度的调研任务，必须使用 dispatch_subagents 拆解并下发。
2. 收集完子智能体的汇报后，综合撰写最终报告。

严格按以下格式输出（遇到Observation前停下）：
Thought: 分析需要哪个工具
Action: 工具名称
Action Input: 工具参数

最后综合输出：
Thought: 我已拿到所有维度的调研结果
Final Answer: 综合结构化报告"""

    # 组装主 Agent
    main_agent = ReActAgent(
        name="MainAgent(总监)", 
        tools={
            "web_search": real_web_search,
            "dispatch_subagents": dispatch_subagents
        },
        system_prompt=MAIN_SYSTEM,
        max_steps=6
    )
    
    question = "调研一下2024年中国咖啡市场：目前的市场规模增速、瑞幸与库迪的竞争现状、以及下沉市场(三四线城市)的消费趋势。"
    print("========== 🌟 真实网络多智能体并发系统 🌟 ==========")
    final_report = main_agent.run(question)
    print("\n========== 📄 最终调研报告 📄 ==========")
    print(final_report)
