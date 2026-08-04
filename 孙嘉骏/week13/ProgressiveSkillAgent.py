import json
from typing import Dict, Any, Callable

# ==========================================
# 1. 定义 Skill 数据结构
# ==========================================
class Skill:
    def __init__(self, name: str, description: str, shallow_prompt: str, deep_prompt: str, execute_func: Callable):
        self.name = name
        self.description = description
        self.shallow_prompt = shallow_prompt  # 浅层描述：用于系统Prompt
        self.deep_prompt = deep_prompt        # 深层描述：使用时才加载的详细Prompt
        self.execute_func = execute_func      # 实际执行该技能的函数

# ==========================================
# 2. 定义具体的 Skills
# ==========================================
def execute_web_search(params: Dict[str, Any]) -> str:
    query = params.get("query", "")
    # 这里应该是真实的搜索API调用，此处模拟返回
    return f"搜索结果：关于'{query}'的最新技术文章有3篇..."

def execute_code_runner(params: Dict[str, Any]) -> str:
    code = params.get("code", "")
    # 这里应该是安全的沙箱执行，此处模拟返回
    return f"代码执行成功，输出：42"

# 注册 Skills
skill_web_search = Skill(
    name="web_search",
    description="当你需要获取最新信息、事实或新闻时使用。",
    shallow_prompt="- web_search: 获取最新信息、事实或新闻。参数: {\"query\": \"string\"}",
    deep_prompt="你正在使用web_search技能。请严格按照以下步骤执行：\n1. 提取用户的搜索意图\n2. 构造精准的搜索关键词\n3. 返回JSON格式: {\"action\": \"web_search\", \"params\": {\"query\": \"...\"}}",
    execute_func=execute_web_search
)

skill_code_runner = Skill(
    name="code_runner",
    description="当你需要执行Python代码进行数学计算或逻辑处理时使用。",
    shallow_prompt="- code_runner: 执行Python代码进行计算。参数: {\"code\": \"string\"}",
    deep_prompt="你正在使用code_runner技能。请严格按照以下步骤执行：\n1. 分析用户的计算需求\n2. 编写简洁的Python代码\n3. 返回JSON格式: {\"action\": \"code_runner\", \"params\": {\"code\": \"...\"}}",
    execute_func=execute_code_runner
)

# ==========================================
# 3. Agent 核心：渐进式加载管理器
# ==========================================
class ProgressiveSkillAgent:
    def __init__(self, skills: list[Skill], llm_api_call: Callable):
        self.skills: Dict[str, Skill] = {s.name: s for s in skills}
        self.llm_api_call = llm_api_call
        self.loaded_skills = set()  # 记录当前会话中已经加载过深层Prompt的Skills
        
        # 构建初始的浅层系统Prompt（只包含名称和简短描述）
        self.base_system_prompt = "你是一个智能助手。你可以使用以下工具：\n"
        for skill in skills:
            self.base_system_prompt += f"{skill.shallow_prompt}\n"
        self.base_system_prompt += "\n请先思考，然后以JSON格式输出你需要调用的工具。如果不需要工具，直接回复用户。"

    def _build_messages(self, user_input: str, history: list) -> list:
        """构建发送给LLM的消息列表，动态注入深层Prompt"""
        # 基础系统Prompt
        current_system_prompt = self.base_system_prompt
        
        # 渐进式加载：如果之前已经决定使用某个Skill，将深层Prompt追加到系统Prompt中
        # 这样LLM在下一轮就能看到详细的执行规范
        for skill_name in self.loaded_skills:
            skill = self.skills[skill_name]
            current_system_prompt += f"\n\n--- {skill_name} 详细指引 ---\n{skill.deep_prompt}"

        messages = [{"role": "system", "content": current_system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def run(self, user_input: str, max_turns: int = 5):
        print(f"\n🧑 用户: {user_input}")
        
        history = []
        
        for turn in range(max_turns):
            messages = self._build_messages(user_input, history)
            
            # 调用 LLM API
            llm_response = self.llm_api_call(messages)
            print(f"🤖 LLM 思考/输出: {llm_response}")
            
            # 尝试解析LLM的输出，看是否需要调用Skill
            try:
                # 假设LLM输出的是JSON格式的Action
                action_data = json.loads(llm_response)
                action_name = action_data.get("action")
                action_params = action_data.get("params", {})
                
                if action_name in self.skills:
                    # === 渐进式加载的核心步骤 ===
                    # 1. 标记该Skill已被加载，下轮对话会注入深层Prompt
                    self.loaded_skills.add(action_name)
                    
                    # 2. 执行Skill的实际代码
                    skill = self.skills[action_name]
                    observation = skill.execute_func(action_params)
                    print(f"⚙️ 系统执行 [{action_name}] 返回: {observation}")
                    
                    # 3. 将执行结果作为Observation追加到历史记录，进入下一轮
                    history.append({"role": "assistant", "content": llm_response})
                    user_input = f"Observation: {observation}\n请根据以上执行结果回复用户，或继续调用其他工具。"
                    continue
                else:
                    break # 无效的Action，退出
                    
            except json.JSONDecodeError:
                # 不是JSON格式，说明LLM直接给出了最终文本回复
                break
        
        print(f"💡 最终回复: {llm_response}")
        return llm_response

# ==========================================
# 4. 模拟 LLM API 调用
# ==========================================
def mock_llm_api_call(messages: list) -> str:
    """模拟大模型API，实际应用中替换为 openai.ChatCompletion.create 等"""
    last_user_msg = messages[-1]["content"]
    
    # 模拟第一轮：LLM看到浅层Prompt，决定使用搜索
    if "最新" in last_user_msg and "web_search" not in messages[0]["content"]:
        return json.dumps({"action": "web_search", "params": {"query": "2023年AI Agent最新进展"}})
    
    # 模拟第二轮：LLM看到了深层Prompt，并拿到了搜索结果，决定写代码计算
    elif "Observation" in last_user_msg and "搜索结果" in last_user_msg:
        return json.dumps({"action": "code_runner", "params": {"code": "print(3+4)"}})
    
    # 模拟第三轮：LLM拿到了代码执行结果，生成最终回复
    elif "Observation" in last_user_msg and "42" in last_user_msg:
        return "根据搜索和计算，AI Agent正在快速发展，另外3加4等于7。"
    
    return "我无法理解您的问题。"

# ==========================================
# 5. 运行测试
# ==========================================
if __name__ == "__main__":
    # 初始化Agent，传入注册的Skills
    agent = ProgressiveSkillAgent(
        skills=[skill_web_search, skill_code_runner],
        llm_api_call=mock_llm_api_call
    )
    
    # 测试渐进式加载
    agent.run("帮我查一下AI Agent的最新进展，并计算3+4的结果。")
