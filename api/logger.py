"""
api/logger.py
ELK-style structured JSON-Lines logger.
Ensures fixed-format PII is masked before any log entry reaches disk.
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from crew.guardrails import mask_fixed_pii

LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "nykaa_agent_audit.jsonl")

class StructuredJsonLogger:
    """Thread-safe JSON-Lines file logger with automatic PII sanitization."""

    def __init__(self, log_path: str = LOG_FILE_PATH):
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        
        # Internal standard logger for console debugging
        self._console_logger = logging.getLogger("nykaa_structured_logger")
        if not self._console_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
            handler.setFormatter(formatter)
            self._console_logger.addHandler(handler)
            self._console_logger.setLevel(logging.INFO)

    def _sanitize(self, value: Any) -> Any:
        """Recursively traverses payloads and masks fixed-format PII in string values."""
        if isinstance(value, str):
            return mask_fixed_pii(value)
        elif isinstance(value, dict):
            return {k: self._sanitize(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._sanitize(v) for v in value]
        return value

    def log_event(
        self,
        event_type: str,
        trace_id: str,
        request_text: Optional[str] = None,
        response_text: Optional[str] = None,
        latency_ms: Optional[float] = None,
        status_code: int = 200,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Formats and writes a single structured JSON-Lines record.
        Ensures raw unmasked PII never reaches disk.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Build clean audit document
        record: Dict[str, Any] = {
            "timestamp": timestamp,
            "trace_id": trace_id,
            "event_type": event_type,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        }

        # Apply masking to prompt and response bodies
        if request_text is not None:
            record["request_text_masked"] = mask_fixed_pii(request_text)

        if response_text is not None:
            record["response_text_masked"] = mask_fixed_pii(response_text)

        if extra:
            record["extra"] = self._sanitize(extra)

        # Write single JSON line (ELK / Logstash compatible)
        json_line = json.dumps(record, ensure_ascii=False)
        with open(self.log_path, mode="a", encoding="utf-8") as f:
            f.write(json_line + "\n")

        self._console_logger.info(f"[{trace_id}] {event_type} - {latency_ms or 0:.1f}ms")
        return record


# Global singleton instance
structured_logger = StructuredJsonLogger()


class RequestTimer:
    """Context manager utility to easily compute and log execution latency."""

    def __init__(
        self,
        event_type: str,
        trace_id: str,
        request_text: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ):
        self.event_type = event_type
        self.trace_id = trace_id
        self.request_text = request_text
        self.extra = extra or {}
        self.start_time: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.perf_counter() - self.start_time) * 1000.0
        status_code = 500 if exc_type else 200

        structured_logger.log_event(
            event_type=self.event_type,
            trace_id=self.trace_id,
            request_text=self.request_text,
            latency_ms=latency_ms,
            status_code=status_code,
            extra=self.extra
        )


if __name__ == "__main__":
    import uuid

    sample_trace = str(uuid.uuid4())
    # Demonstration: Fixed PII (phone number & card last-4) gets masked automatically
    raw_query = "My phone is 9876543210 and card ending in 4321, check order NYK-1002"

    with RequestTimer("test_latency_measurement", trace_id=sample_trace, request_text=raw_query):
        time.sleep(0.05)  # Simulate processing delay

    print(f"Audit log written to {LOG_FILE_PATH}. Check contents to verify PII masking.")