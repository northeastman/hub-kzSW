import json
from pathlib import Path
from openai import OpenAI
from skills import SkillRegistry
from tools import create_tools

class Harness:
    def __init__(self, skills_dir: Path, api_key: str = None, base_url: str = None, model: str = "gpt-4o", max_iterations: int = 5):
        self.registry = SkillRegistry(skills_dir)
        self.state = {'loaded_skills': {}}
        self.tools = create_tools(self.registry, self.state)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.messages = []
        self.max_iterations = max_iterations

    @staticmethod
    def _clean_text(s: str) -> str:
        if not isinstance(s, str):
            return s
        return s.encode('utf-8', errors='replace').decode('utf-8')

    def _build_system_prompt(self) -> str:
        parts = []
        parts.append("You are an AI assistant with access to specialized skills.")
        parts.append(self.registry.get_descriptions())

        if self.state['loaded_skills']:
            parts.append("\n## Loaded Skills (Full Instructions)")
            for name, body in self.state['loaded_skills'].items():
                parts.append(body)

        parts.append("\n## Instructions")
        parts.append("1. Check if the user's request matches any available skill.")
        parts.append("2. If yes, call `load_skill(skill_name)` to load it.")
        parts.append("3. After loading, if the skill refers to external resources (e.g., `references/style-guide.md`), call `read_skill_resource` **once** to get the content.")
        parts.append("4. Once you have the skill instructions and any referenced resources, **immediately** perform the task and output the final answer in a clear format.")
        parts.append("5. **Do not call any more tools** after you have all necessary information. Do not repeat tool calls.")
        parts.append("6. If a tool returns an error, report it and ask the user for guidance, but do not retry the same tool call automatically.")
        parts.append("\n## Output Format")
        parts.append("After loading the skill and reading any referenced resources, you MUST output a single, complete, and final answer in one message.")
        parts.append("Do not produce incremental output. Do not show your step-by-step reasoning unless the skill explicitly requires it.")
        parts.append("Your final answer should be well-structured, actionable, and ready for the user to use.")
        parts.append("If you have all necessary information, stop calling tools and produce the final answer immediately.")

        return self._clean_text("\n".join(parts))

    def _execute_tool_calls(self, tool_calls: list) -> list:
        results = []
        for tc in tool_calls:
            func_name = tc.function.name
            args = json.loads(tc.function.arguments)
            tool = next((t for t in self.tools if t.name == func_name), None)
            result = tool.func(**args) if tool else f"Error: Unknown tool '{func_name}'"
            cleaned_result = self._clean_text(result)
            results.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "content": cleaned_result
            })
        return results

    def chat(self, user_input: str) -> str:
        user_input = self._clean_text(user_input)
        self.messages.append({"role": "user", "content": user_input})
        tool_schemas = [t.to_schema() for t in self.tools]
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            system_prompt = self._build_system_prompt()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}] + self.messages,
                tools=tool_schemas,
                tool_choice="auto"
            )

            assistant_msg = response.choices[0].message
            assistant_dict = assistant_msg.model_dump()
            if assistant_dict.get('content') is not None:
                assistant_dict['content'] = self._clean_text(assistant_dict['content'])
            if assistant_dict.get('tool_calls'):
                for tc in assistant_dict['tool_calls']:
                    if 'function' in tc and 'arguments' in tc['function']:
                        tc['function']['arguments'] = self._clean_text(tc['function']['arguments'])
            self.messages.append(assistant_dict)

            if not assistant_msg.tool_calls:
                return assistant_dict.get('content', '')

            # 执行工具
            tool_results = self._execute_tool_calls(assistant_msg.tool_calls)
            for result in tool_results:
                self.messages.append(result)

            # 如果已经加载了技能，并且已经执行了至少2轮工具调用（load + read），强制进入最终答案
            if self.state['loaded_skills'] and iteration >= 2:
                force_prompt = "\n\n现在你已经有了所有必要信息（技能指令和参考资源）。请立即输出最终的完整审查报告，不要再调用任何工具。"
                self.messages.append({"role": "user", "content": self._clean_text(force_prompt)})
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": self._build_system_prompt()}] + self.messages,
                    tools=[],  # 禁用工具调用
                    tool_choice="none"
                )
                final_msg = final_response.choices[0].message
                final_content = final_msg.content or "（无法生成最终回答）"
                self.messages.append({"role": "assistant", "content": self._clean_text(final_content)})
                return self._clean_text(final_content)

        # 超时强制结束
        force_prompt = "\n\n你已调用工具超过最大次数。现在请直接根据已有信息给出最终答案，不要再调用任何工具。"
        self.messages.append({"role": "user", "content": self._clean_text(force_prompt)})
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self._build_system_prompt()}] + self.messages,
            tools=[],
            tool_choice="none"
        )
        final_msg = final_response.choices[0].message
        final_content = final_msg.content or "（超时未生成回复）"
        self.messages.append({"role": "assistant", "content": self._clean_text(final_content)})
        return self._clean_text(final_content)

    def reset(self):
        self.messages = []
        self.state = {'loaded_skills': {}}