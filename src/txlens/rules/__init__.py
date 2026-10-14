"""파이프라인 3단계 — 결정론 룰 매칭 + 사전 (설계 §3.3 ②).

사전 원본은 코드가 아니라 `rules/v{n}.yaml` 이다. 이유는 두 가지다.
① 사전은 사람이 고치는 물건이고(HITL 승격 루프의 산출물), 코드에 박으면 사람이 못 고친다.
② 버전이 곧 실험 조건이다 — v1 로 낸 지표와 v2 로 낸 지표를 나란히 놓아야 승격 효과가 보인다.

`match_stage` 를 거래마다 기록하는 것이 이 층의 계약이다. 어느 단계가 얼마나 일하는지 모르면
"LLM 이 정말 필요한가"에 답할 수 없다.
"""

from __future__ import annotations

from txlens.rules.dictionary import (
    DEFAULT_DICTIONARY_PATH,
    RuleDictionary,
    RuleDictionaryError,
    RuleEntry,
    load_dictionary,
)
from txlens.rules.match import (
    FUZZY_CONFIDENCE_CAP,
    FUZZY_SCORER,
    PG_MASKED_CANONICAL,
    STAGE_CONFIDENCE,
    RuleMatcher,
)

__all__ = [
    "DEFAULT_DICTIONARY_PATH",
    "FUZZY_CONFIDENCE_CAP",
    "FUZZY_SCORER",
    "PG_MASKED_CANONICAL",
    "STAGE_CONFIDENCE",
    "RuleDictionary",
    "RuleDictionaryError",
    "RuleEntry",
    "RuleMatcher",
    "load_dictionary",
]
