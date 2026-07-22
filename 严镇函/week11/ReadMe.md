# 天气查询系统（循环调用版）

一个支持多轮对话的天气查询助手，基于大模型工具调用能力实现。

## 功能特性

- ✅ **多轮对话**：持续接收用户输入，保持完整对话历史
- ✅ **实时天气查询**：调用 Open-Meteo API 获取真实天气数据
- ✅ **上下文理解**：支持连续追问（如"北京天气如何？"→"那上海呢？"）
- ✅ **命令支持**：`exit/quit` 退出、`clear` 清空对话历史
- ✅ **双模型支持**：DeepSeek / DashScope（Qwen）

## 快速开始

### 1. 安装依赖

```bash
cd "天气系统(循环调用)"
pip install openai httpx
```

### 2. 配置 API Key

Windows（PowerShell）：
```powershell
$env:DEEPSEEK_API_KEY = "sk-xxx"
# 或使用 DashScope
$env:DASHSCOPE_API_KEY = "sk-xxx"
```

Windows（CMD）：
```cmd
set DEEPSEEK_API_KEY=sk-xxx
set DASHSCOPE_API_KEY=sk-xxx
```

### 3. 运行

```bash
# 启动交互式循环
python run_weather_loop.py

# 带初始问题启动
python run_weather_loop.py --question "北京天气如何？"

# 使用 DashScope 模型
python run_weather_loop.py --provider dashscope
```

## 使用示例

```
============================================================
    天气查询助手（deepseek-chat）
============================================================
输入问题查询天气
命令：exit/quit（退出） | clear（清空历史）
============================================================

用户：北京天气如何？
  → [工具调用] get_weather({'city': '北京'})
    ↩ 【中国 北京市】天气报告 坐标：39.90°N, 116.41°E 当前天气：晴天...
  → [回答]（耗时 2.1s）
助手：北京当前天气为晴天，温度28°C，相对湿度45%，风速12 km/h。未来3天预报：...

用户：那上海呢？
  → [工具调用] get_weather({'city': '上海'})
    ↩ 【中国 上海市】天气报告 坐标：31.23°N, 121.47°E 当前天气：多云...
  → [回答]（耗时 1.9s）
助手：上海当前天气为多云，温度30°C，相对湿度65%，风速8 km/h。未来3天预报：...

用户：clear
对话历史已清空

用户：exit
再见！
```

## 文件结构

```
天气系统(循环调用)/
├── weather_backend.py    # 天气查询核心逻辑
├── run_weather_loop.py   # 主循环调用脚本
└── README.md             # 使用说明
```

## 核心组件

### weather_backend.py

- **Geocoding**：城市名 → 经纬度转换
- **天气查询**：调用 Open-Meteo API
- **结果格式化**：生成人类可读的天气报告

### run_weather_loop.py

- **LLM 集成**：支持 DeepSeek / DashScope
- **工具调用**：自动调用 get_weather 获取实时数据
- **多轮对话**：消息历史持续累积，支持上下文理解

## API Key 获取

- DeepSeek：https://platform.deepseek.com/
- DashScope：https://dashscope.aliyun.com/

## 注意事项

1. 需要联网才能获取实时天气数据
2. Open-Meteo API 完全免费，无需注册
3. 建议使用 DeepSeek 作为默认模型，响应速度较快
