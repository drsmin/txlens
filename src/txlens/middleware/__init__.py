"""가드레일·캐시 미들웨어 (설계 §3.0).

Agent 의 행동 제약을 프롬프트가 아니라 **코드로** 둔다. "개인정보를 보내지 마세요"라고 적는 것과
개인정보가 보이면 예외를 던지는 것은 다른 물건이고, 규제 산업에서는 후자만 근거가 된다.
"""

from __future__ import annotations

from txlens.middleware.cache import ModelCallCache
from txlens.middleware.guardrails import (
    CUSTOMER_ID_PATTERN,
    DEFAULT_MODEL_CALL_LIMIT,
    DEFAULT_TOOL_CALL_LIMIT,
    build_guardrails,
)

__all__ = [
    "CUSTOMER_ID_PATTERN",
    "DEFAULT_MODEL_CALL_LIMIT",
    "DEFAULT_TOOL_CALL_LIMIT",
    "ModelCallCache",
    "build_guardrails",
]
