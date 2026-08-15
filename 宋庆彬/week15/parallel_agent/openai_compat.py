import asyncio
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMError(RuntimeError):
    """Raised when an OpenAI-compatible completion cannot be returned."""


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None,
        request_timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout

    async def complete(self, messages, *, temperature=0.2) -> str:
        return await asyncio.to_thread(
            self._complete_sync, messages, temperature
        )

    def _complete_sync(self, messages, temperature) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        try:
            request = Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            with urlopen(request, timeout=self.request_timeout) as response:
                body = json.load(response)
        except (HTTPError, URLError, json.JSONDecodeError, TypeError, ValueError, OSError):
            raise LLMError("OpenAI-compatible request failed") from None

        return _extract_content(body)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _extract_content(body) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError("OpenAI-compatible response has invalid content") from None

    if not isinstance(content, str) or not content:
        raise LLMError("OpenAI-compatible response has invalid content")
    return content
