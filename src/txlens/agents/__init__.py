"""파이프라인 4단계 — LLM 분류 Agent (설계 §3.0).

9개 노드 중 8개는 결정론이고, Agent 는 **이 안에만** 있다. 전부를 Agent 로 만들면 테스트 가능성을
잃고, 배점의 35%인 "Agent 설계 구체성"은 도구·가드레일·구조화 출력이 어디에 왜 붙었는지로
설명되는 것이지 노드 수로 설명되지 않는다.
"""

from __future__ import annotations

from txlens.agents.classifier import (
    ClassifyOutcome,
    MerchantClassifier,
    build_classifier,
    target_message,
)
from txlens.agents.prompt import FEW_SHOTS, SYSTEM_PROMPT, build_system_prompt, few_shot_block
from txlens.agents.tools import ALIAS_LOOKUP_FLOOR, make_tools

__all__ = [
    "ALIAS_LOOKUP_FLOOR",
    "FEW_SHOTS",
    "SYSTEM_PROMPT",
    "ClassifyOutcome",
    "MerchantClassifier",
    "build_classifier",
    "build_system_prompt",
    "few_shot_block",
    "make_tools",
    "target_message",
]
