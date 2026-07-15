from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
OCR_OUTPUT_DIR = DATA_DIR / "ocr_output"

OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TEXT_LENGTH = 80

BAIDU_OCR_API_KEY = os.getenv("BAIDU_OCR_API_KEY", "")
BAIDU_OCR_SECRET_KEY = os.getenv("BAIDU_OCR_SECRET_KEY", "")

BAIDU_OCR_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
BAIDU_UNLIMITED_OCR_TASK_URL = (
    "https://aip.baidubce.com/rest/2.0/brain/online/v2/unlimited-ocr-parser/task"
)
BAIDU_UNLIMITED_OCR_QUERY_URL = (
    "https://aip.baidubce.com/rest/2.0/brain/online/v2/unlimited-ocr-parser/task/query"
)

OCR_POLL_INTERVAL = 8
OCR_MAX_WAIT_SECONDS = 600
