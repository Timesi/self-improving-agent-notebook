"""统一的真实 LLM 客户端，供各 notebook 的 Agent 演示使用。

客户端调用 OpenAI 兼容的 chat completions 端点。端点、key、模型从环境变量读取。
没有有效 API key 时直接报错，不返回占位答案。
"""

import os
import requests

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

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
                "API key 无效或缺失。请在环境中设置 AGENT_LLM_API_KEY。"
            )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
def get_llm(**kwargs):
    """创建真实 LLM 客户端；缺少 API key 时由首次请求明确报错。"""
    return LLMClient(**kwargs)
