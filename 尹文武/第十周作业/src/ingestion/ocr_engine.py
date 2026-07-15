import base64
import time
from pathlib import Path
from typing import Optional

import requests

from config.settings import (
    BAIDU_OCR_API_KEY,
    BAIDU_OCR_SECRET_KEY,
    BAIDU_OCR_TOKEN_URL,
    BAIDU_UNLIMITED_OCR_TASK_URL,
    BAIDU_UNLIMITED_OCR_QUERY_URL,
    OCR_POLL_INTERVAL,
    OCR_MAX_WAIT_SECONDS,
)


class BaiduUnlimitedOCREngine:
    def __init__(self):
        if not BAIDU_OCR_API_KEY or not BAIDU_OCR_SECRET_KEY:
            raise ValueError(
                "缺少百度 OCR 密钥。请在 .env 中配置：\n"
                "BAIDU_OCR_API_KEY=你的API_KEY\n"
                "BAIDU_OCR_SECRET_KEY=你的SECRET_KEY"
            )

        self.access_token: Optional[str] = None

    def get_access_token(self) -> str:
        if self.access_token:
            return self.access_token

        params = {
            "grant_type": "client_credentials",
            "client_id": BAIDU_OCR_API_KEY,
            "client_secret": BAIDU_OCR_SECRET_KEY,
        }

        resp = requests.get(
            BAIDU_OCR_TOKEN_URL,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()

        data = resp.json()

        if "access_token" not in data:
            raise RuntimeError(f"获取 access_token 失败：{data}")

        self.access_token = data["access_token"]
        return self.access_token

    def create_task(self, file_path: Path) -> str:
        access_token = self.get_access_token()

        url = f"{BAIDU_UNLIMITED_OCR_TASK_URL}?access_token={access_token}"

        with open(file_path, "rb") as f:
            file_data = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "file_data": file_data,
            "file_name": file_path.name,
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        resp = requests.post(
            url,
            headers=headers,
            data=payload,
            timeout=120,
        )
        resp.raise_for_status()

        data = resp.json()

        if data.get("error_code") != 0:
            raise RuntimeError(f"OCR 创建任务失败：{data}")

        task_id = data.get("result", {}).get("task_id")

        if not task_id:
            raise RuntimeError(f"OCR 创建任务未返回 task_id：{data}")

        return task_id

    def query_task(self, task_id: str) -> dict:
        access_token = self.get_access_token()

        url = f"{BAIDU_UNLIMITED_OCR_QUERY_URL}?access_token={access_token}"

        payload = {
            "task_id": task_id,
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        resp = requests.post(
            url,
            headers=headers,
            data=payload,
            timeout=60,
        )
        resp.raise_for_status()

        data = resp.json()

        if data.get("error_code") != 0:
            raise RuntimeError(f"OCR 查询任务失败：{data}")

        return data

    def wait_for_result(self, task_id: str) -> dict:
        start_time = time.time()

        while True:
            data = self.query_task(task_id)
            result = data.get("result", {})
            status = result.get("status")

            print(f"[OCR] task_id={task_id}, status={status}")

            if status == "success":
                return result

            if status == "failed":
                raise RuntimeError(f"OCR 任务失败：{result}")

            if time.time() - start_time > OCR_MAX_WAIT_SECONDS:
                raise TimeoutError(f"OCR 等待超时：task_id={task_id}")

            time.sleep(OCR_POLL_INTERVAL)

    def download_markdown(self, markdown_url: str) -> str:
        resp = requests.get(markdown_url, timeout=120)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return resp.text

    def run_ocr(self, pdf_path: Path, output_txt_path: Path) -> str:
        pdf_path = Path(pdf_path)
        output_txt_path = Path(output_txt_path)
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[OCR] submit: {pdf_path}")

        task_id = self.create_task(pdf_path)
        result = self.wait_for_result(task_id)

        markdown_url = result.get("markdown_url")

        if not markdown_url:
            raise RuntimeError(f"OCR 成功但没有 markdown_url：{result}")

        markdown_text = self.download_markdown(markdown_url)

        output_txt_path.write_text(
            markdown_text,
            encoding="utf-8",
            errors="ignore",
        )

        meta_path = output_txt_path.with_suffix(".ocr_meta.json")
        meta_path.write_text(
            str(result),
            encoding="utf-8",
            errors="ignore",
        )

        print(f"[OCR] saved: {output_txt_path}")

        return markdown_text
