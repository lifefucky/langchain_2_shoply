import logging
import json
from datetime import datetime
from pathlib import Path


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if isinstance(record.msg, dict):
            log_entry.update(record.msg)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logger(name: str, log_file: str = "app.jsonl"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    Path("logs").mkdir(exist_ok=True)
    handler = logging.FileHandler(f"logs/{log_file}", encoding="utf-8")
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    return logger
