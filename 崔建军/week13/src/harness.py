"""
Skill Harness 主框架 — 通用技能执行器

核心设计：
  1. 零代码扩展：新 skill 复制到 skills 目录即可执行，无需修改 harness 代码
  2. LLM 驱动的参数提取和技能执行
  3. 工具调用协议：LLM 输出 JSON 指令，harness 解析并执行通用操作

工具调用协议（JSON 指令格式）：
{
  "action": "<动作名称>",
  "args": {
    "<参数名>": "<参数值>"
  }
}

支持的工具：
  - save_file: 保存文件
  - run_script: 运行脚本
  - open_browser: 打开浏览器
  - read_file: 读取文件
  - mkdir: 创建目录
"""

import os
import sys
import json
import logging
import subprocess
import webbrowser
import re
import shlex
from pathlib import Path
from typing import List, Dict, Optional, Any

sys.path.insert(0, str(Path(__file__).parent))

from skill_loader import SkillLoader, Skill
from progressive_disclosure import ProgressiveDisclosure, generate_skill_disclosure
from llm_config import get_chat_client, current_model_info, call_llm


class Harness:
    """Skill Harness 主框架"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.skills_dir = Path(self.config.get("skills_dir", "src/skills"))
        self.output_dir = Path(self.config.get("output_dir", "outputs"))
        self.log_level = self.config.get("log_level", "INFO")
        
        self._setup_logging()
        self.skill_loader = SkillLoader(str(self.skills_dir))
        self.skill_loader.load_all_skills()
        self.disclosure = ProgressiveDisclosure()
        
        # LLM 客户端（延迟初始化）
        self._llm_client = None
        self._llm_model = None
    
    def _setup_logging(self):
        """设置日志（使用独立 logger，避免全局配置冲突）"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
        
        self.logger = logging.getLogger(f"harness_{id(self)}")
        self.logger.setLevel(level_map.get(self.log_level, logging.INFO))
        self.logger.propagate = False
        
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler(self.output_dir / "harness.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)
    
    def _get_llm_client(self):
        """获取 LLM 客户端（延迟初始化）"""
        if self._llm_client is None:
            try:
                self._llm_client, self._llm_model = get_chat_client()
                self.logger.info(f"LLM 客户端已初始化: {current_model_info()['display']}")
            except EnvironmentError as e:
                self.logger.warning(f"LLM 配置未设置，将使用规则匹配模式：{e}")
        return self._llm_client, self._llm_model
    
    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """使用已缓存的客户端调用 LLM（复用连接，避免每次新建）"""
        client, model = self._get_llm_client()
        if not client:
            raise EnvironmentError("LLM 客户端未初始化，请检查环境变量配置")
        
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        return response.choices[0].message.content or ""
    
    def _sanitize_path(self, path: str, skill: Skill) -> Path:
        """路径安全校验，确保在技能目录内（防止路径遍历攻击）"""
        path = path.replace("{baseDir}", str(skill.base_dir))
        abs_path = Path(path).resolve()
        skill_base = skill.base_dir.resolve()
        
        if not abs_path.is_relative_to(skill_base):
            raise ValueError(f"路径越界: {path}")
        
        return abs_path
    
    def load_skills(self) -> List[Skill]:
        """加载所有技能"""
        skills = self.skill_loader.load_all_skills()
        self.logger.info(f"加载了 {len(skills)} 个技能")
        return skills
    
    def find_skill(self, user_input: str) -> Optional[Skill]:
        """查找匹配的技能（优先 LLM，降级规则）
        
        使用渐进式披露 Level 0（基础信息）进行匹配，避免注入过多上下文。
        """
        client, model = self._get_llm_client()
        
        if client:
            # 渐进式匹配：仅请求 Level 0（基础信息），不加载完整内容
            skills_summary = []
            for skill in self.skill_loader.skills.values():
                levels = generate_skill_disclosure(skill, max_level=0)
                level0_content = levels[0].content if levels else ""
                skills_summary.append(f"## {skill.name}\n{level0_content}")
            
            prompt = f"""从以下技能列表中选择最匹配用户输入的技能：

可用技能：
{'\n\n'.join(skills_summary)}

用户输入：{user_input}

请输出 JSON 格式：{{"skill": "<技能名称>", "reason": "<匹配理由>"}}。无匹配输出 {{"skill": null}}。"""
            
            try:
                response = self._call_llm([{"role": "system", "content": prompt}], temperature=0.3)
                result = json.loads(response)
                if result.get("skill"):
                    return self.skill_loader.get_skill(result["skill"])
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"LLM 匹配失败，回退到规则匹配: {e}")
                pass
        
        # 降级到规则匹配
        return self.skill_loader.find_matching_skill(user_input)
    
    def _extract_parameters(self, user_input: str, skill: Skill) -> Dict[str, str]:
        """LLM 驱动的参数提取（通用，不依赖技能名称）"""
        client, model = self._get_llm_client()
        
        if client:
            prompt = f"""从用户输入中提取技能所需的参数。

技能名称：{skill.name}
技能描述：{skill.description}

用户输入：{user_input}

请分析技能需要哪些参数，并从用户输入中提取。输出 JSON 格式：{{"参数名": "参数值"}}。
如果没有找到参数，输出 {{}}。"""
            
            try:
                response = self._call_llm([{"role": "system", "content": prompt}], temperature=0.3)
                result = json.loads(response)
                return result
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"LLM 参数提取失败，使用降级模式: {e}")
                pass
        
        # 降级：简单提取英文单词（适用于 flash-card 等技能）
        kwargs = {}
        word_match = re.search(r'([a-zA-Z]+)', user_input)
        if word_match:
            kwargs["word"] = word_match.group(1)
        
        # 如果没有其他参数，将用户输入作为描述
        if not kwargs:
            kwargs["user_input"] = user_input
        
        return kwargs
    
    def execute_skill(self, skill: Skill, **kwargs) -> Dict[str, Any]:
        """通用技能执行器
        
        优化策略：一次性规划模式（Single-shot Planning）+ 渐进式披露
        - 让 LLM 一次性输出所有工具调用序列
        - harness 按顺序执行，无需每步都调用 LLM
        - 根据披露层级动态注入技能信息（Level 0/1/2）
        - 失败时自动回退到逐轮模式
        
        执行流程：
        1. 按需加载完整 SKILL.md（渐进式加载第二阶段）
        2. 根据当前披露层级获取技能信息
        3. LLM 一次性规划所有工具调用
        4. harness 顺序执行工具调用
        5. 失败时回退到逐轮模式
        """
        # 渐进式加载第二阶段：匹配成功后，按需加载完整内容
        self.skill_loader.load_full_content(skill)
        
        client, model = self._get_llm_client()
        
        if not client:
            return {
                "success": True,
                "response": f"技能 {skill.name} 执行完成（无 LLM）",
                "steps": [{"step": i+1, "title": a.title, "status": "completed"} for i, a in enumerate(skill.actions)]
            }
        
        # 获取当前披露层级的技能内容
        current_level = self.disclosure.get_current_level()
        print(f"[执行] 当前披露层级: Level {current_level}")
        
        # 先尝试一次性规划模式（优化路径）
        result = self._execute_single_shot(skill, kwargs, current_level)
        if result:
            return result
        
        # 回退到逐轮模式（兼容路径）
        print("[回退] 一次性规划失败，切换到逐轮模式")
        return self._execute_round_by_round(skill, kwargs, current_level)
    
    def _execute_single_shot(self, skill: Skill, kwargs: Dict[str, Any], current_level: int = 2) -> Optional[Dict[str, Any]]:
        """一次性规划模式：LLM 输出完整工具调用序列，harness 顺序执行
        
        Args:
            current_level: 当前披露层级，决定注入到 prompt 中的信息量
                - Level 0: 仅基础信息（name, description, keywords）
                - Level 1: + 执行流程（actions）
                - Level 2: + 完整内容（raw_content）
        """
        # 使用渐进式披露获取当前层级的技能内容
        levels = generate_skill_disclosure(skill, max_level=current_level)
        skill_content = ""
        for level in levels:
            skill_content += f"\n【Level {level.level}】{level.name}\n{level.content}\n"
        
        system_prompt = f"""你正在执行 {skill.name} 技能。

技能详情：
{skill_content}

用户输入参数：{json.dumps(kwargs, ensure_ascii=False)}

请分析技能步骤，一次性输出所有需要执行的工具调用序列。

输出格式（JSON 数组）：
[
  {{
    "action": "<工具名称>",
    "args": {{
      "<参数名>": "<参数值>"
    }}
  }},
  ...
]

可用工具：
1. save_file - 保存文件
   参数：path（文件路径，可用 {{baseDir}} 表示技能目录）, content（文件内容）
   
2. run_script - 运行脚本
   参数：command（命令）, timeout（超时时间，默认60秒）
   
3. open_browser - 打开浏览器
   参数：url（URL或文件路径）
   
4. read_file - 读取文件
   参数：path（文件路径）
   
5. mkdir - 创建目录
   参数：path（目录路径）

重要：
- {{baseDir}} 会自动替换为技能目录的绝对路径
- 只输出 JSON 数组，不要输出其他内容
- 根据技能逻辑决定需要哪些工具调用，不需要的步骤可以跳过"""
        
        try:
            print("[一次性规划] LLM 正在规划所有工具调用...")
            response = self._call_llm([{"role": "system", "content": system_prompt}], temperature=0.5)
            
            # 提取 JSON 数组
            first_bracket = response.find('[')
            last_bracket = response.rfind(']')
            
            if first_bracket == -1 or last_bracket == -1 or last_bracket < first_bracket:
                return None  # 无法解析，回退
            
            actions_str = response[first_bracket:last_bracket+1]
            actions = json.loads(actions_str)
            
            if not isinstance(actions, list):
                return None
            
            print(f"[一次性规划] 规划了 {len(actions)} 个工具调用")
            
            # 顺序执行所有工具调用
            steps = []
            success = True
            final_response = ""
            
            for idx, action_data in enumerate(actions):
                action = action_data.get("action", "")
                action_args = action_data.get("args", {})
                
                if not action:
                    continue
                
                result = self._execute_tool(action, action_args, skill)
                
                # 判断是否跳过
                skipped = result.get("skipped", False)
                status = "skipped" if skipped else ("completed" if result["success"] else "failed")
                
                steps.append({
                    "step": idx + 1,
                    "title": f"执行工具: {action}",
                    "status": status,
                    "output": result.get("output", "")
                })
                print(f"[步骤 {idx + 1}] 工具: {action}, 结果: {'跳过' if skipped else ('成功' if result['success'] else '失败')}")
                
                if not result["success"]:
                    success = False
                    final_response = f"执行失败: {result.get('output', '')}"
                    break
            
            if success:
                final_response = f"技能 {skill.name} 执行完成！共执行了 {len(steps)} 个步骤。"
            
            return {
                "success": success,
                "response": final_response,
                "steps": steps,
                "output": {"llm_response": response, "mode": "single_shot"},
                "mode": "single_shot"
            }
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"[一次性规划失败] {e}")
            return None
    
    def _execute_round_by_round(self, skill: Skill, kwargs: Dict[str, Any], current_level: int = 2) -> Dict[str, Any]:
        """逐轮模式：每轮 LLM 输出一个工具调用或 DONE 信号（兼容路径）
        
        Args:
            current_level: 当前披露层级，决定注入到 prompt 中的信息量
        """
        messages = []
        
        # 使用渐进式披露获取当前层级的技能内容
        levels = generate_skill_disclosure(skill, max_level=current_level)
        skill_content = ""
        for level in levels:
            skill_content += f"\n【Level {level.level}】{level.name}\n{level.content}\n"
        
        system_prompt = f"""你正在执行 {skill.name} 技能。

技能详情：
{skill_content}

用户输入参数：{json.dumps(kwargs, ensure_ascii=False)}

你可以调用以下工具来执行操作：

工具调用格式（每轮只输出一个）：
TOOL: {{
  "action": "<动作名称>",
  "args": {{
    "<参数名>": "<参数值>"
  }}
}}

可用工具：
1. save_file - 保存文件
   参数：path（文件路径，可用 {{baseDir}} 表示技能目录）, content（文件内容）
   
2. run_script - 运行脚本
   参数：command（命令）, timeout（超时时间，默认60秒）
   
3. open_browser - 打开浏览器
   参数：url（URL或文件路径）
   
4. read_file - 读取文件
   参数：path（文件路径）
   
5. mkdir - 创建目录
   参数：path（目录路径）

完成所有步骤后，输出：
DONE: <执行结果总结>

重要：
- {{baseDir}} 会自动替换为技能目录的绝对路径
- 每轮只能输出一个工具调用
- 执行完一个工具后，我会告诉你结果，然后你决定下一步"""
        
        messages.append({"role": "system", "content": system_prompt})
        
        steps = []
        success = True
        current_step = 0
        max_rounds = 10
        final_response = ""
        
        for round_num in range(max_rounds):
            try:
                print(f"[第 {round_num + 1} 轮] LLM 思考中...")
                
                response = self._call_llm(messages, temperature=0.7)
                messages.append({"role": "assistant", "content": response})
                
                # 检查是否输出 DONE
                done_match = re.search(r'DONE:\s*(.+)', response, re.DOTALL)
                if done_match:
                    final_response = done_match.group(1).strip()
                    current_step += 1
                    steps.append({
                        "step": current_step,
                        "title": "执行完成",
                        "status": "completed",
                        "output": final_response
                    })
                    print(f"[步骤 {current_step}] 完成")
                    break
                
                # 检查是否输出工具调用
                first_brace = response.find('{')
                last_brace = response.rfind('}')
                
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    action_str = response[first_brace:last_brace+1]
                    try:
                        action_str = action_str.replace('\n', ' ').replace('\r', '')
                        action_data = json.loads(action_str)
                        action = action_data.get("action", "")
                        action_args = action_data.get("args", {})
                        
                        current_step += 1
                        result = self._execute_tool(action, action_args, skill)
                        
                        steps.append({
                            "step": current_step,
                            "title": f"执行工具: {action}",
                            "status": "completed" if result["success"] else "failed",
                            "output": result.get("output", "")
                        })
                        print(f"[步骤 {current_step}] 工具: {action}, 结果: {'成功' if result['success'] else '失败'}")
                        
                        messages.append({
                            "role": "user",
                            "content": f"工具 {action} 执行结果：{'成功' if result['success'] else '失败'}。输出：{result.get('output', '')}"
                        })
                        
                        if not result["success"]:
                            success = False
                            final_response = f"执行失败: {result.get('output', '')}"
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"[错误] JSON 解析失败: {e}")
                        messages.append({
                            "role": "user",
                            "content": f"工具调用格式错误，请重新输出。错误：{e}"
                        })
                        continue
                
                else:
                    final_response = response
                    current_step += 1
                    steps.append({
                        "step": current_step,
                        "title": "LLM 响应",
                        "status": "completed",
                        "output": response[:200] + "..." if len(response) > 200 else response
                    })
                    print(f"[步骤 {current_step}] LLM 响应")
                    break
                    
            except Exception as e:
                steps.append({
                    "step": current_step + 1,
                    "title": "执行失败",
                    "status": "failed",
                    "output": str(e)
                })
                print(f"[错误] {e}")
                success = False
                final_response = str(e)
                break
        
        return {
            "success": success,
            "response": final_response,
            "steps": steps,
            "output": {"llm_response": final_response, "mode": "round_by_round"},
            "mode": "round_by_round"
        }
    
    ALLOWED_COMMANDS = {"python", "python3"}

    def _execute_tool(self, action: str, args: Dict[str, Any], skill: Skill) -> Dict[str, Any]:
        """执行工具调用"""
        action = action.lower().strip()
        
        try:
            if action == "save_file":
                path = args.get("path", "")
                content = args.get("content", "")
                
                file_path = self._sanitize_path(path, skill)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
                
                return {"success": True, "output": f"已保存: {file_path}"}
            
            elif action == "run_script":
                command = args.get("command", "")
                timeout = args.get("timeout", 60)
                
                command = command.replace("{baseDir}", str(skill.base_dir))
                
                cmd_parts = []
                current_part = ""
                in_quotes = False
                for char in command:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ' ' and not in_quotes:
                        if current_part:
                            cmd_parts.append(current_part)
                            current_part = ""
                    else:
                        current_part += char
                if current_part:
                    cmd_parts.append(current_part)
                
                if not cmd_parts:
                    return {"success": False, "output": "命令为空"}
                
                executable = cmd_parts[0].lower()
                if executable not in self.ALLOWED_COMMANDS:
                    return {"success": False, "output": f"不允许的命令: {executable}，仅支持: {', '.join(self.ALLOWED_COMMANDS)}"}
                
                result = subprocess.run(cmd_parts, shell=False, capture_output=True, text=True, timeout=timeout)
                
                if result.returncode == 0:
                    return {"success": True, "output": result.stdout[:200] + ("..." if len(result.stdout) > 200 else "")}
                else:
                    return {"success": False, "output": f"执行失败: {result.stderr}"}
            
            elif action == "open_browser":
                url = args.get("url", "")
                
                url = url.replace("{baseDir}", str(skill.base_dir))
                
                webbrowser.open(url)
                return {"success": True, "output": f"已打开: {url}"}
            
            elif action == "read_file":
                path = args.get("path", "")
                
                file_path = self._sanitize_path(path, skill)
                if file_path.exists():
                    content = file_path.read_text(encoding="utf-8")
                    max_len = 2000
                    truncated = content[:max_len] + ("..." if len(content) > max_len else "")
                    return {"success": True, "output": truncated, "original_size": len(content)}
                else:
                    return {"success": False, "output": f"文件不存在: {file_path}"}
            
            elif action == "mkdir":
                path = args.get("path", "")
                
                dir_path = self._sanitize_path(path, skill)
                if dir_path.exists():
                    return {"success": True, "output": f"目录已存在，跳过创建: {dir_path}", "skipped": True}
                dir_path.mkdir(parents=True, exist_ok=True)
                return {"success": True, "output": f"已创建目录: {dir_path}"}
            
            else:
                return {"success": False, "output": f"未知工具: {action}"}
        
        except ValueError as e:
            return {"success": False, "output": f"安全错误: {e}"}
        except Exception as e:
            return {"success": False, "output": str(e)}
    
    def chat(self, user_input: str) -> str:
        """处理用户输入并返回回复
        
        使用渐进式披露机制：
        - 根据对话历史自动调整披露层级
        - 用户追问"详细"、"更多"等关键词时自动升级层级
        """
        skill = self.find_skill(user_input)
        
        if skill:
            print(f"\n[匹配到技能] {skill.name}")
            kwargs = self._extract_parameters(user_input, skill)
            print(f"[参数] {kwargs}")
            
            # 执行前：确保披露层级至少为 Level 1（包含执行流程），默认提升到 Level 2（完整内容）
            # 因为技能执行需要完整的执行流程和详细信息
            current_level = self.disclosure.get_current_level()
            if current_level < 2:
                self.disclosure.context.current_level = 2
                print(f"[披露] 自动提升层级到 Level 2（技能执行需要完整信息）")
            
            result = self.execute_skill(skill, **kwargs)
            
            # 构建回复
            response = f"{result['response'][:300]}\n\n" if result["response"] else ""
            
            # 检查是否有工具执行结果
            has_tool_results = any("工具" in step.get("title", "") for step in result.get("steps", []))
            
            if has_tool_results:
                response = f"技能 {skill.name} 执行完成！\n\n"
                for step in result.get("steps", []):
                    status = "✓" if step["status"] == "completed" else ("✗" if step["status"] == "failed" else "~")
                    response += f"{status} {step['title']}\n"
                    if step.get("output"):
                        response += f"   输出: {step['output']}\n"
            
            # 更新披露上下文
            self.disclosure.update_context(user_input, response)
            
            return response
        else:
            # 无匹配技能，LLM 直接回答
            client, model = self._get_llm_client()
            if client:
                llm_response = self._call_llm([{"role": "system", "content": "你是一个友好的助手。"}, {"role": "user", "content": user_input}])
                # 更新披露上下文
                self.disclosure.update_context(user_input, llm_response)
                return llm_response
            
            skills = self.skill_loader.skills.values()
            return f"我不太确定你需要什么。目前支持：{', '.join([s.name for s in skills])}"
    
    def run_cli(self):
        """运行命令行接口"""
        print("="*60)
        print("  Skill Harness — CLI")
        print("="*60)
        
        client, model = self._get_llm_client()
        if client:
            print(f"当前模型：{current_model_info()['display']}")
        
        print("命令: /list, /skill <name>, /execute <name>, /reset, /exit")
        print("="*60)
        
        print(f"已加载 {len(self.skill_loader.skills)} 个技能")
        print()
        
        while True:
            user_input = input("你: ").strip()
            if not user_input:
                continue
            
            if user_input.startswith("/"):
                self._handle_command(user_input)
            else:
                response = self.chat(user_input)
                print(f"\nAgent: {response}\n")
    
    def _handle_command(self, command: str):
        """处理命令"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/exit":
            print("再见！")
            sys.exit(0)
        
        elif cmd == "/list":
            for name, skill in self.skill_loader.skills.items():
                print(f"  • {name}: {skill.description}")
            print()
        
        elif cmd == "/skill":
            if len(parts) < 2:
                print("用法: /skill <技能名称>")
                return
            skill = self.skill_loader.get_skill(parts[1])
            if skill:
                self.skill_loader.load_full_content(skill)
                levels = generate_skill_disclosure(skill, max_level=2)
                self.disclosure.context.current_level = 2
                print(self.disclosure.format_disclosure(skill.name, levels))
            else:
                print(f"未找到技能: {parts[1]}")
            print()
        
        elif cmd == "/execute":
            if len(parts) < 2:
                print("用法: /execute <技能名称> [参数]")
                return
            skill = self.skill_loader.get_skill(parts[1])
            if skill:
                kwargs = {k: v for k, v in [p.split("=", 1) for p in parts[2:] if "=" in p]}
                result = self.execute_skill(skill, **kwargs)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"未找到技能: {parts[1]}")
            print()
        
        elif cmd == "/reset":
            self.disclosure = ProgressiveDisclosure()
            print("已重置披露状态\n")
        
        else:
            print(f"未知命令: {cmd}\n")
    
    def as_api(self, host: str = "0.0.0.0", port: int = 8000):
        """启动 API 服务"""
        try:
            import uvicorn
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            
            app = FastAPI(title="Skill Harness API", version="1.0")
            
            class ChatRequest(BaseModel):
                message: str
            
            @app.get("/skills")
            def list_skills():
                return {"count": len(self.skill_loader.skills), "skills": [{"name": s.name, "description": s.description} for s in self.skill_loader.skills.values()]}
            
            @app.post("/chat")
            def chat(request: ChatRequest):
                return {"response": self.chat(request.message)}
            
            @app.post("/execute/{skill_name}")
            def execute(skill_name: str, request: ChatRequest):
                skill = self.skill_loader.get_skill(skill_name)
                if not skill:
                    raise HTTPException(status_code=404, detail="Skill not found")
                kwargs = self._extract_parameters(request.message, skill)
                return self.execute_skill(skill, **kwargs)
            
            print(f"\n启动 API 服务: http://{host}:{port}")
            uvicorn.run(app, host=host, port=port)
            
        except ImportError:
            print("需要安装 fastapi 和 uvicorn")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    config = {
        "skills_dir": str(project_root / "src" / "skills"),
        "output_dir": str(project_root / "outputs"),
        "log_level": "INFO"
    }
    
    Harness(config).run_cli()