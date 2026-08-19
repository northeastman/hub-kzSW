"""
私有大模型
"""
import logging

import requests
import uuid

from prompt_optimizer_paper.config import BASE_URL, APP_ID, AK, SK
from prompt_optimizer_paper.core.exception.biz_exception import BizException


llm = None

def init_llm():
    global llm
    if llm is None:
        llm = CustomModel()
    return llm

class Response:
    def __init__(self, content):
        self.content = content

class CustomModel:
    def __init__(self):
        self.domain = BASE_URL
        self.appId = APP_ID
        self.token = self.token()

    def _get_token_api(self):
        return "/openapi/auth/v1/api-key/token"

    def _get_chat_api(self):
        return "/openapi/engine/chat/v1/completions"

    def token(self):
        data = {
            "ak": AK,
            "sk": SK
        }
        response = requests.post(url=self.domain + self._get_token_api(), json=data)
        if response.status_code == 200:
            result = response.json()
            if result["code"] != '10000':
                print(f"获取token失败 {response}")
                return ""
                # raise Exception(result)
            return result["data"]["token"]
        else:
            print(f"获取token失败 {response}")
            return ""
            # raise Exception(response.__str__())

    def chat(self, token: str, prompt: str, content: str):
        headers = {
            "token": token
        }
        data = {
            "chatId": str(uuid.uuid4()),
            "appId": self.appId,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "variables": {
                "prompt": prompt
            }
        }
        response = requests.post(url=self.domain + self._get_chat_api(), headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            if "code" in result and result["code"] != '200':
                raise BizException(result["code"], result["message"])
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(response.__str__())

    def invoke(self, prompt, content):
        try:
            result = self.chat(token=self.token, prompt=prompt, content=content)
            return Response(result)
        except BizException as e:
            print(f"大模型调用异常: {e}")
            if "token已失效" in e.err_msg:
                result = self.chat(token=self.token(), prompt=prompt, content=content)
                return Response(result)
        except Exception as e:
            print(f"大模型调用异常: {e}")
