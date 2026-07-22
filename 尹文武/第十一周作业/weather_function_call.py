"""
weather_function_call.py — 基于 Ollama OpenAI 兼容接口的天气 Function Call 示例

暴露给模型的函数：
    1. get_geo_coding(city_name)
    2. get_weather(lat, lon)
    3. format_weather_report(loc_resp, weather_resp)

核心特点：
    - 不在 Python 中硬编码 get_geo_coding → get_weather → format_weather_report
    - 使用多轮 Event Loop，让模型根据当前上下文自行决定下一次工具调用
    - 当模型不再返回 tool_calls 时，循环自动结束并输出最终答案

依赖：
    pip install -U openai httpx

运行示例：
    python weather_function_call.py --provider qwen --query "查询重庆未来三天的天气"
    python weather_function_call.py --provider deepseek
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from openai import OpenAI

# 严格复用你现有 weather_backend.py 中的三个业务函数
from src.weather_backend import (
    format_weather_report,
    get_geo_coding,
    get_weather,
)


PROVIDERS = {
    "deepseek": {
        "api_key": "deepseek",
        "base_url": "http://localhost:11434",
        "model": "deepseek-r1:14b",
    },
    "qwen": {
        "api_key": "qwen2.5:7b",
        "base_url": "http://localhost:11434",
        "model": "qwen2.5:7b",
    },
}


# -----------------------------------------------------------------------------
# 1. 暴露给模型的 Function Schema
# -----------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_geo_coding",
            "description": (
                "根据中文或英文城市名称查询地理位置，返回纬度 lat、经度 lon、"
                "城市名 city_name、国家 country 和省/州 admin1。"
                "当用户只提供城市名、尚不知道经纬度时，应先调用此函数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "需要查询的城市名称，例如：重庆、宁德、北京。",
                    }
                },
                "required": ["city_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "根据确定的纬度和经度查询当前天气及未来3天天气预报。"
                "如果用户只给出了城市名称，不要猜测坐标，应先调用 get_geo_coding。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "纬度，必须来自可靠的地理编码结果。",
                    },
                    "lon": {
                        "type": "number",
                        "description": "经度，必须来自可靠的地理编码结果。",
                    },
                },
                "required": ["lat", "lon"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "format_weather_report",
            "description": (
                "把 get_geo_coding 返回的位置信息和 get_weather 返回的天气数据"
                "格式化为完整中文天气报告。调用时应使用前面工具返回的真实数据，"
                "不要自行编造或修改数据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "loc_resp": {
                        "type": "object",
                        "description": "get_geo_coding 返回的地点字典。",
                        "properties": {
                            "lat": {"type": "number"},
                            "lon": {"type": "number"},
                            "city_name": {"type": "string"},
                            "country": {"type": "string"},
                            "admin1": {"type": "string"},
                        },
                        "required": ["lat", "lon", "city_name", "country", "admin1"],
                        "additionalProperties": True,
                    },
                    "weather_resp": {
                        "type": "object",
                        "description": (
                            "get_weather 返回的天气字典。至少应保留 current 和 daily 字段。"
                        ),
                        "properties": {
                            "current": {
                                "type": "object",
                                "properties": {
                                    "temperature_2m": {"type": "number"},
                                    "relative_humidity_2m": {"type": "number"},
                                    "wind_speed_10m": {"type": "number"},
                                    "weather_code": {"type": "integer"},
                                },
                                "required": [
                                    "temperature_2m",
                                    "relative_humidity_2m",
                                    "wind_speed_10m",
                                    "weather_code",
                                ],
                                "additionalProperties": True,
                            },
                            "daily": {
                                "type": "object",
                                "properties": {
                                    "time": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "temperature_2m_max": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                    },
                                    "temperature_2m_min": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                    },
                                    "precipitation_sum": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                    },
                                    "weather_code": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
                                },
                                "required": [
                                    "time",
                                    "temperature_2m_max",
                                    "temperature_2m_min",
                                    "precipitation_sum",
                                    "weather_code",
                                ],
                                "additionalProperties": True,
                            },
                        },
                        "required": ["current", "daily"],
                        "additionalProperties": True,
                    },
                },
                "required": ["loc_resp", "weather_resp"],
                "additionalProperties": False,
            },
        },
    },
]


# 工具名到真实 Python 函数的映射。
# 这里只负责“按模型给出的函数名执行”，不负责规定函数调用顺序。
AVAILABLE_FUNCTIONS: dict[str, Callable[..., dict | str]] = {
    "get_geo_coding": get_geo_coding,
    "get_weather": get_weather,
    "format_weather_report": format_weather_report,
}


SYSTEM_PROMPT = """
你是一个可以自主调用工具的天气查询助手。

你处在一个多轮工具调用循环中：每次工具执行后，工具结果都会返回给你；
你需要重新判断是否还要调用其他工具。只有在信息已经足够时，才直接回答用户。

工具之间的数据依赖如下：
- get_geo_coding：把城市名转换为真实经纬度和地点信息。
- get_weather：需要真实的 lat 和 lon，返回当前天气与未来3天预报。
- format_weather_report：需要前两个工具的真实返回值，生成最终中文报告。

规则：
1. 用户只提供城市名称时，不要猜测经纬度。
2. 不要伪造工具结果，也不要用常识替代工具查询。
3. 完整天气报告应使用 format_weather_report 生成。
4. 如果某个工具返回错误字符串，停止调用依赖该结果的后续工具，并向用户说明错误。
5. 每一轮都根据已有消息和工具结果自行决定下一步；不要一次性假设尚未获得的数据。
6. format_weather_report 成功返回后，应以它的报告为主要答案，不要篡改其中的数值。
""".strip()


def normalize_openai_base_url(base_url: str) -> str:
    """把 http://localhost:11434 规范化为 OpenAI 兼容地址 /v1。"""
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized += "/v1"
    return normalized


def create_client(provider_name: str) -> tuple[OpenAI, str]:
    """根据 PROVIDERS 配置创建 OpenAI 兼容客户端。"""
    if provider_name not in PROVIDERS:
        available = ", ".join(PROVIDERS)
        raise ValueError(f"未知 provider：{provider_name}，可选值：{available}")

    provider = PROVIDERS[provider_name]
    client = OpenAI(
        api_key=provider["api_key"],
        base_url=normalize_openai_base_url(provider["base_url"]),
        timeout=120.0,
        max_retries=1,
    )
    return client, provider["model"]


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """兼容 arguments 为 JSON 字符串或字典的情况。"""
    if isinstance(raw_arguments, dict):
        return raw_arguments

    if not isinstance(raw_arguments, str):
        raise TypeError(
            f"工具参数应为 JSON 字符串或字典，实际类型：{type(raw_arguments).__name__}"
        )

    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型返回的工具参数不是合法 JSON：{raw_arguments}") from exc

    if not isinstance(parsed, dict):
        raise TypeError("工具参数的 JSON 顶层必须是对象。")

    return parsed


def serialize_tool_result(result: Any) -> str:
    """把工具结果转换为适合放入 role=tool 消息的字符串。"""
    if isinstance(result, str):
        return result

    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """执行模型指定的函数；该函数不推断也不修改调用顺序。"""
    function = AVAILABLE_FUNCTIONS.get(tool_name)
    if function is None:
        return f"工具调用失败：不存在名为 {tool_name} 的函数"

    try:
        return function(**arguments)
    except TypeError as exc:
        return f"工具调用失败：{tool_name} 参数不匹配：{exc}"
    except Exception as exc:
        # 业务函数内部本身已经捕获多数异常；这里是 Event Loop 的最后一道保护。
        return f"工具调用失败：执行 {tool_name} 时发生异常：{exc}"


def run_weather_agent(
    user_query: str,
    provider_name: str = "qwen",
    *,
    max_rounds: int = 10,
    verbose: bool = True,
) -> str:
    """
    运行由模型驱动的多轮 Function Call Event Loop。

    循环终止条件：
        - 模型本轮没有返回 tool_calls；或
        - 达到 max_rounds，防止异常模型无限调用工具。
    """
    client, model = create_client(provider_name)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for round_index in range(1, max_rounds + 1):
        if verbose:
            print(f"\n[LLM ROUND {round_index}] model={model}")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        # 必须把 assistant 的 tool_calls 原样加入上下文，下一轮模型才能知道
        # 自己刚才请求了哪些函数。
        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_message.content or "",
        }
        if tool_calls:
            assistant_entry["tool_calls"] = [
                tool_call.model_dump(exclude_none=True) for tool_call in tool_calls
            ]
        messages.append(assistant_entry)

        # 没有工具调用，说明模型认为信息已经足够，Event Loop 结束。
        if not tool_calls:
            final_answer = (assistant_message.content or "").strip()
            if not final_answer:
                return "模型没有返回工具调用，也没有生成最终答案。"
            return final_answer

        # 模型一轮可能返回一个或多个工具调用，因此逐个执行并回填结果。
        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = parse_tool_arguments(tool_call.function.arguments)
                if verbose:
                    print(
                        f"[TOOL CALL] {tool_name}"
                        f"({json.dumps(arguments, ensure_ascii=False)})"
                    )
                result = execute_tool(tool_name, arguments)
            except Exception as exc:
                result = f"工具调用失败：无法解析或执行 {tool_name}：{exc}"

            result_text = serialize_tool_result(result)

            if verbose:
                preview = result_text
                if len(preview) > 500:
                    preview = preview[:500] + "..."
                print(f"[TOOL RESULT] {preview}")

            # OpenAI 兼容格式使用 tool_call_id 将结果对应到具体工具调用。
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                }
            )

    return f"已达到最大工具调用轮数 {max_rounds}，为避免无限循环已终止。"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用本地 Ollama 模型运行天气 Function Call Agent Loop"
    )
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default="qwen",
        help="选择本地模型提供方配置，默认 qwen",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="单次查询内容；不提供时进入交互模式",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="最大 Function Call 轮数，默认 10",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只显示最终答案，不打印工具调用过程",
    )
    args = parser.parse_args()

    if args.query:
        answer = run_weather_agent(
            args.query,
            provider_name=args.provider,
            max_rounds=args.max_rounds,
            verbose=not args.quiet,
        )
        print("\n" + "=" * 60)
        print(answer)
        return

    print(f"天气 Function Call 交互模式，provider={args.provider}")
    print("输入 exit、quit 或 q 退出。")

    while True:
        try:
            user_query = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break

        answer = run_weather_agent(
            user_query,
            provider_name=args.provider,
            max_rounds=args.max_rounds,
            verbose=not args.quiet,
        )
        print(f"\n助手：\n{answer}")


if __name__ == "__main__":
    main()
