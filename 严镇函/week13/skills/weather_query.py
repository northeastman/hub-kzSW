"""
天气查询技能 - 示例技能

演示如何定义技能元数据和 execute 函数
"""

SKILL_METADATA = {
    "name": "weather_query",
    "display_name": "天气查询",
    "description": "查询指定城市的天气情况",
    "version": "1.0.0",
    "author": "Agent Memory System",
    "tags": ["天气", "查询", "工具"],
    "triggers": ["天气", "weather", "气温", "下雨"],
    "requires_memory": False,
}


def execute(context: dict) -> dict:
    """执行天气查询技能"""
    city = context.get("city", "北京")
    
    # 模拟天气数据（实际项目中可接入天气 API）
    weather_data = {
        "北京": {"temp": "25°C", "condition": "晴", "humidity": "45%"},
        "上海": {"temp": "28°C", "condition": "多云", "humidity": "60%"},
        "广州": {"temp": "32°C", "condition": "雷阵雨", "humidity": "80%"},
    }
    
    if city in weather_data:
        info = weather_data[city]
        return {
            "city": city,
            "temperature": info["temp"],
            "condition": info["condition"],
            "humidity": info["humidity"],
            "message": f"{city}今天{info['condition']}，气温{info['temp']}，湿度{info['humidity']}"
        }
    else:
        return {
            "city": city,
            "message": f"暂不支持查询 {city} 的天气"
        }