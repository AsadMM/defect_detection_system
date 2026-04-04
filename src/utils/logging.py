import logging
from logging.config import dictConfig
from pathlib import Path


class ApiOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith("api")


class NonApiFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("api")


def _build_logging_config(logs_dir: Path) -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "api_only": {
                "()": "src.utils.logging.ApiOnlyFilter",
            },
            "non_api_only": {
                "()": "src.utils.logging.NonApiFilter",
            },
        },
        "formatters": {
            "api": {
                "format": "%(asctime)s | API | %(levelname)s | %(name)s | %(message)s",
            },
            "app": {
                "format": "%(asctime)s | APP | %(levelname)s | %(name)s | %(message)s",
            },
        },
        "handlers": {
            "api_console": {
                "class": "logging.StreamHandler",
                "formatter": "api",
                "filters": ["api_only"],
                "level": "INFO",
            },
            "app_console": {
                "class": "logging.StreamHandler",
                "formatter": "app",
                "filters": ["non_api_only"],
                "level": "INFO",
            },
            "api_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": str(logs_dir / "api.log"),
                "when": "midnight",
                "interval": 1,
                "backupCount": 30,
                "encoding": "utf-8",
                "formatter": "api",
                "filters": ["api_only"],
                "level": "INFO",
            },
            "app_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": str(logs_dir / "app.log"),
                "when": "midnight",
                "interval": 1,
                "backupCount": 30,
                "encoding": "utf-8",
                "formatter": "app",
                "filters": ["non_api_only"],
                "level": "INFO",
            },
        },
        "loggers": {
            "api": {
                "handlers": ["api_console", "api_file"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["app_console", "app_file"],
            "level": "INFO",
        },
    }


def setup_logging():
    logs_dir = Path(__file__).resolve().parents[2] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    dictConfig(_build_logging_config(logs_dir))
