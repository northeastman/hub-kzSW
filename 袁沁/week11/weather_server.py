"""
weather_server.py — 天气查询 MCP Server（方式二：MCP）


使用方式（由 run_mcp.py 作为子进程启动，stdio 通信）：
  python mode_mcp/servers/weather_server.py

依赖：
  pip install mcp httpx
"""
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 用 as 别名避免同名 tool 函数遮蔽后端函数导致递归
from weather_backend import get_weather as _get_weather  # noqa: E402
from weather_backend import get_geocode as _get_geocode


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


mcp = FastMCP("weather-server")


@mcp.tool()
def get_weather(lat: str, lon: str) -> str:
    """
    根据经纬度查询天气。
    如果用户只提供城市名，请先调用 get_geocode 获取经纬度，再调用此工具。
    Args:
        lat: 纬度
        lon: 经度
    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    return _get_weather(lat,lon)


@mcp.tool()
def get_geocode(city: str) -> str:
    """
    查询指定城市的参数，包括城市名、国家、经纬度等

    Args:
        city: 城市中文名，如 '宁德'、'北京'。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

    Returns:
        包括城市名、国家、经纬度等。
    """
    return _get_geocode(city)


if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")
