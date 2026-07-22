"""
weather_backend.py — 天气查询后端

教学重点：
    1. 同样是"纯业务逻辑"，与 rag_backend 平级，被三种方式复用
    2. 内部两次 HTTP 请求：Geocoding（城市名→经纬度）+ 天气查询
    3. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费

使用方式（作为模块）：
    from src.weather_backend import get_weather
    print(get_weather("宁德"))

依赖：
    pip install httpx
    Open-Meteo API 完全免费，无需注册
"""
import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo 天气代码 → 中文描述映射
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}

def get_geo_coding(city_name: str) -> dict | str:
    """
    查询指定城市的位置信息

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        dict: 包含经纬度、城市名、国家、省/州级行政区等信息
    """
    try:
        response = httpx.get(GEOCODING_URL, params={
            "name": city_name, "count": 10, "language": "zh", "format": "json",
        })
        response.raise_for_status()
        data = response.json()

        if "results" not in data or len(data["results"]) == 0:
            return f"未找到城市：{city_name}"

        # 在候选里优先取行政级别更高的（feature_code 含 A = 某级政府驻地），
        # 其次取有人口数据的，避免落到同名小村庄
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(data["results"], key=_rank)
        return {
            "lat": loc["latitude"],
            "lon": loc["longitude"],
            "city_name": loc.get("name", city_name),
            "country": loc.get("country", ""),
            "admin1": loc.get("admin1", ""), # 省/州级行政区

        }
    except Exception as e:
        return f"获取经纬度失败：{e}"

def get_weather(lat, lon) -> dict | str:
    """
    根据经纬度获取天气

    Args:
        lat: 纬度
        lon: 经度
    Returns:
        dict: 包含当前天气和未来3天预报的字典
    """
    try:
        response = httpx.get(WEATHER_URL, params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "Asia/Shanghai",
            "forecast_days": 3,
        })
        response.raise_for_status()
        data = response.json()

        return data
    except Exception as e:
        return f"获取天气失败：{e}"

# 格式化输出天气
def format_weather_report(loc_resp: dict, weather_resp: dict) -> str:
    """
    格式化天气报告

    Args:
        loc_resp: get_geo_coding 返回的字典
        weather_resp: get_weather 返回的字典
    Returns:
        str: 格式化后的天气报告字符串
    """
    location_str = f"{loc_resp.get('city_name', '')}, {loc_resp.get('admin1', '')}, {loc_resp.get('country', '')}".strip() if loc_resp else "未知位置"
    if isinstance(weather_resp, dict):
        cur = weather_resp["current"]
        daily = weather_resp["daily"]
        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
        lat, lon = loc_resp["lat"], loc_resp["lon"]
        lines = [
            f"【{location_str}】天气报告",
            f"坐标：{lat:.2f}°N, {lon:.2f}°E",
            "",
            f"当前天气：{weather_desc}",
            f"  温度：{cur['temperature_2m']}°C",
            f"  相对湿度：{cur['relative_humidity_2m']}%",
            f"  风速：{cur['wind_speed_10m']} km/h",
            "",
            "未来3天预报：",
        ]

        for i in range(3):
            day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
            lines.append(
                f"  {daily['time'][i]}：{day_desc}，"
                f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]} mm"
            )

        return "\n".join(lines)

if __name__ == "__main__":
    # 测试
    loc = get_geo_coding("重庆")

    if isinstance(loc, dict):
        location_str = f"{loc.get('city_name', '')}, {loc.get('admin1', '')}, {loc.get('country', '')}".strip()

        lat, lon = loc["lat"], loc["lon"]

        print(location_str)
        print("="*50)
        weather_resp = get_weather(lat, lon)


        if isinstance(weather_resp, dict):
            print(format_weather_report(loc, weather_resp))
        else:
            print(weather_resp)
