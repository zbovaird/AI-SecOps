"""
Logging infrastructure for the red team framework.

Provides structured logging with context tracking for experiments.
"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager
import traceback


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for machine-readable logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
            
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[0] else None,
            }
            
        # Add source location
        log_data["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        return json.dumps(log_data, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Human-readable log formatter for console output."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        level = record.levelname
        
        if self.use_colors:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"
            
        message = record.getMessage()
        
        # Add extra data if present
        extra = ""
        if hasattr(record, "extra_data") and record.extra_data:
            extra = f" | {json.dumps(record.extra_data)}"
        
        return f"{timestamp} {level_str} [{record.name}] {message}{extra}"


class ExperimentLogger(logging.LoggerAdapter):
    """Logger adapter that adds experiment context to all log messages."""
    
    def __init__(self, logger: logging.Logger, run_id: str = "", experiment: str = ""):
        super().__init__(logger, {})
        self.run_id = run_id
        self.experiment = experiment
        
    def process(self, msg, kwargs):
        # Add context to extra
        extra = kwargs.get("extra", {})
        extra["run_id"] = self.run_id
        extra["experiment"] = self.experiment
        kwargs["extra"] = extra
        return msg, kwargs
    
    def log_metric(self, name: str, value: Any, **extra):
        """Log a metric value with structured data."""
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0,
            f"metric: {name}={value}", (), None
        )
        record.extra_data = {"metric": name, "value": value, **extra}
        self.logger.handle(record)
    
    def log_event(self, event: str, **data):
        """Log a structured event."""
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0,
            f"event: {event}", (), None
        )
        record.extra_data = {"event": event, **data}
        self.logger.handle(record)
    
    def log_error(self, message: str, exception: Optional[Exception] = None, **data):
        """Log an error with optional exception details."""
        if exception:
            self.logger.error(message, exc_info=exception, extra={"extra_data": data})
        else:
            record = self.logger.makeRecord(
                self.logger.name, logging.ERROR, "", 0,
                message, (), None
            )
            record.extra_data = data
            self.logger.handle(record)


# Global logger registry
_loggers: Dict[str, ExperimentLogger] = {}
_initialized = False


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    run_id: str = "",
    structured_console: bool = False,
    file_logging: bool = True,
) -> None:
    """
    Initialize logging for the framework.
    
    Args:
        level: Logging level (default INFO)
        log_dir: Directory for log files (default: ./runs/logs/)
        run_id: Run identifier for log file naming
        structured_console: Use JSON format for console (default: human-readable)
        file_logging: Enable file logging (default: True)
    """
    global _initialized
    
    # Get root logger for framework
    root_logger = logging.getLogger("redteam")
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    if structured_console:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(HumanFormatter(use_colors=sys.stdout.isatty()))
    root_logger.addHandler(console_handler)
    
    # File handler (structured JSON)
    if file_logging and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{run_id}.jsonl" if run_id else f"{timestamp}.jsonl"
        
        file_handler = logging.FileHandler(log_path / filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
    
    _initialized = True


def get_logger(
    name: str = "redteam",
    run_id: str = "",
    experiment: str = "",
) -> ExperimentLogger:
    """
    Get or create a logger for the specified context.
    
    Args:
        name: Logger name (typically module name)
        run_id: Run identifier for context
        experiment: Experiment name for context
        
    Returns:
        ExperimentLogger with context attached
    """
    global _initialized
    
    # Auto-initialize with defaults if not done
    if not _initialized:
        setup_logging()
    
    # Create unique key for this logger context
    key = f"{name}:{run_id}:{experiment}"
    
    if key not in _loggers:
        base_logger = logging.getLogger(f"redteam.{name}")
        _loggers[key] = ExperimentLogger(base_logger, run_id, experiment)
    
    return _loggers[key]


@contextmanager
def log_context(logger: ExperimentLogger, operation: str, **context):
    """
    Context manager for logging operation start/end with timing.
    
    Usage:
        with log_context(logger, "model_inference", prompt_id="p1"):
            result = model.generate(...)
    """
    import time
    start = time.time()
    
    logger.log_event(f"{operation}_start", **context)
    
    try:
        yield
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.log_error(
            f"{operation} failed after {elapsed_ms:.1f}ms",
            exception=e,
            **context
        )
        raise
    else:
        elapsed_ms = (time.time() - start) * 1000
        logger.log_event(f"{operation}_complete", elapsed_ms=elapsed_ms, **context)





