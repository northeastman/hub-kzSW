"""
weather_client.py — 使用 MCP 协议 + DeepSeek API
完全遵循 MCP 规范，LLM 自主调用工具
"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("DEEPSEEK_API")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

MAX_TURN = 10

async def chat_with_weather():
    """主函数：使用 MCP 连接 Weather Server，并通过 DeepSeek 调用"""

    # 1. 配置 MCP Server 参数
    server_params = StdioServerParameters(
        command="python",
        args=["weather_server.py"]  # 根据你的实际路径调整
    )

    print("🌤️ 正在连接 Weather MCP Server...", file=sys.stderr)

    # 2. 建立 MCP 连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 3. 初始化 MCP Session
            await session.initialize()
            print("✅ MCP Server 连接成功", file=sys.stderr)

            # 4. 获取 MCP 提供的工具列表
            tools_result = await session.list_tools()
            mcp_tools = tools_result.tools
            print(f"📦 加载了 {len(mcp_tools)} 个工具:",
                  [t.name for t in mcp_tools], file=sys.stderr)

            # 5. 将 MCP 工具转换为 OpenAI/DeepSeek 格式
            tool_defs = []
            for tool in mcp_tools:
                # 确保参数 schema 是正确格式
                parameters = tool.inputSchema or {"type": "object", "properties": {}}
                tool_defs.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or f"调用 {tool.name} 工具",
                        "parameters": parameters
                    }
                })

            # 6. 对话历史
            messages = []

            # 7. 多轮对话示例
            user_questions = [
                "北京今天天气怎么样？",
                "那明天呢？适合出去玩吗？",
                "上海和深圳哪个更暖和？"
            ]

            for question in user_questions:
                print(f"\n{'=' * 50}", file=sys.stderr)
                print(f"👤 用户: {question}", file=sys.stderr)

                # 添加用户消息
                messages.append({
                    "role": "user",
                    "content": question
                })

                # 8. 调用 DeepSeek API（可能多轮工具调用）
                response = await process_with_tools(session, messages, tool_defs)

                # 9. 输出最终回答
                print(f"\n🤖 DeepSeek: {response}")

                # 等待用户继续
                input("\n按 Enter 继续下一个问题...")


async def process_with_tools(session: ClientSession, messages: list, tool_defs: list, max_turns: int = 5):
    """
    处理带工具调用的对话

    Args:
        session: MCP Session
        messages: 对话历史
        tool_defs: 工具定义列表
        max_turns: 最大工具调用轮次

    Returns:
        str: DeepSeek 的最终回答
    """

    def extract_tool_content(result) -> str:
        """把 MCP 返回值整理成 LLM 容易读取的纯文本。"""
        if not hasattr(result, "content"):
            return str(result)

        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools=tool_defs,
            tool_choice="auto"  # 让 DeepSeek 自主决定
        )

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))  # 保存到对话历史

        # 没有工具调用，返回文本回答
        if not message.tool_calls:
            return message.content

        # 没有工具调用，返回文本回答
        # 有工具调用，执行 MCP 工具
        print(f"\n🔧 DeepSeek 决定调用工具:", file=sys.stderr)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"   📞 {tool_name}({tool_args})", file=sys.stderr)

            try:
                # 调用 MCP Server 的工具
                result = await session.call_tool(
                    tool_name,
                    arguments=tool_args
                )

                result_content = extract_tool_content(result)

                print(f"   ✅ 返回: {result_content[:100]}...", file=sys.stderr)

                # 将工具结果添加到对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content
                })

            except Exception as e:
                error_msg = f"工具调用失败: {str(e)}"
                print(f"   ❌ {error_msg}", file=sys.stderr)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": error_msg
                })
        # 本轮工具结果已写入 messages，下一轮会发回 DeepSeek 继续推理。

    return "工具调用轮次过多，已停止。请检查工具描述或模型是否反复请求同一个工具。"




if __name__ == "__main__":
    # 运行主程序
    asyncio.run(chat_with_weather())


