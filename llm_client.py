"""统一 LLM 客户端，供各 notebook 的 Agent 演示使用。

两种模式：
- 真实 API：调用 OpenAI 兼容的 chat completions 端点。端点、key、模型从环境变量读取，
  未设置时回退到 DeepSeek（https://api.deepseek.com）与 ANTHROPIC_API_KEY。
- Mock 模式：设置环境变量 AGENT_LLM_MOCK=1，或 API key 不可用且传入 force_mock=True 时，
  返回确定性的模拟输出。Mock 输出由消息内容哈希驱动，同一输入总是得到同一输出，
  保证 notebook 在没有 key 的环境里也能完整执行。

用法（notebook 内就近 import）：
    from llm_client import get_llm
    client = get_llm()
    reply = client.chat([{"role": "user", "content": "请解释 ReAct 循环"}])
"""

import hashlib
import os
import re
import requests

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

# Mock 模式下的候选回复池。按消息哈希取模轮转，保证确定性。
_MOCK_REPLIES = [
    "模拟回复：这是 mock 模式下的占位输出。配置真实 API key 后，这里会得到模型的完整推理。",
    "Thought: 我需要先检索相关背景。\nAction: search(\"self-improving agents\")",
    "Thought: 检索结果提供了背景信息。\nFinal Answer: 在 mock 模式下，我们用一个占位答案演示循环结构。",
    "思考：这个问题需要分成两步处理。\n行动：第一步分析输入，第二步给出结论。\n结论：演示输出。",
    "模拟回复：Agent 循环正常执行到这一轮。中间结果如下：状态已经推进，没有发现异常。",
]


class LLMClient:
    """OpenAI 兼容 chat completions 客户端。

    chat() 返回回复文本。参数：
    - base_url / api_key / model：显式指定时优先于环境变量
    - temperature：采样温度，None 时用构造时的默认值
    """

    def __init__(self, base_url=None, api_key=None, model=None, temperature=0.7):
        self.base_url = (base_url or os.environ.get("AGENT_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("AGENT_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("AGENT_LLM_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.is_mock = False

    def chat(self, messages, temperature=None, max_tokens=1024):
        """messages：形如 [{"role": "user", "content": "..."}] 的消息列表。返回回复文本。"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=180,
        )
        if resp.status_code == 401:
            raise RuntimeError(
                "API key 无效或缺失。请在环境中设置 AGENT_LLM_API_KEY（或 ANTHROPIC_API_KEY），"
                "或用 get_llm(force_mock=True) 切换到 mock 模式。"
            )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class MockLLM:
    """确定性的模拟 LLM，用于无 key 环境下跑通 notebook。

    输出由最后一条用户消息决定：能识别简单的算术问题（形如"X 加 Y"、"+/-"表达式）时直接计算；
    消息里包含 ReAct 风格的 Thought/Action 标记时返回脚本化轨迹；其余情况按哈希轮转回复池。
    """

    def __init__(self):
        self.is_mock = True

    def chat(self, messages, temperature=None, max_tokens=1024):
        """返回与 LLMClient.chat 相同格式的回复文本，输出完全确定。"""
        content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                content = msg["content"]
                break
        if not content:
            content = str(messages[-1].get("content", ""))

        arithmetic = self._try_arithmetic(content)
        if arithmetic is not None:
            return f"计算结果：{arithmetic}。"

        if "Action" in content or "Thought" in content:
            return ("Thought: 这是 mock 模式下的脚本化推理，先搜索再总结。\n"
                    "Action: search(\"CS329A self-improving agents\")\n"
                    "Thought: 搜索结果显示课程大纲与相关论文。\n"
                    "Final Answer: mock 模式返回一个占位结论，真实 API 下会输出完整推理。")

        digest = int(hashlib.md5(content.encode("utf-8")).hexdigest(), 16)
        return _MOCK_REPLIES[digest % len(_MOCK_REPLIES)]

    @staticmethod
    def _try_arithmetic(content):
        """识别文本里的简单整数算术。识别不到返回 None。

        支持的形态：
        - "15 加 27 等于几"
        - "计算 12 + 8"
        - "difference between 20 and 5"
        """
        m = re.search(r"(\d+)\s*(?:加|加上|plus|和|与|\+|minus|减|减去)\s*(\d+)", content)
        if not m:
            m = re.search(r"(\d+)\s*[\+\-]\s*(\d+)", content)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        if "减" in content or "minus" in content:
            return a - b
        return a + b


def get_llm(force_mock=False, **kwargs):
    """返回一个 LLM 客户端实例。

    force_mock=True 或设置环境变量 AGENT_LLM_MOCK=1 时返回 MockLLM；
    否则返回 LLMClient。notebook 里统一用这个工厂函数创建客户端。
    """
    if force_mock or os.environ.get("AGENT_LLM_MOCK") or os.environ.get("LLM_MOCK"):
        return MockLLM()
    return LLMClient(**kwargs)
