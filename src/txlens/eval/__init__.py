"""평가 하네스 — 베이스라인을 같은 자로 재는 층 (blueprint §5.1).

지표를 노트북·화면에 흩어 계산하지 않고 여기 모은다. 흩어 놓으면 발표 슬라이드의 숫자와
화면의 숫자가 달라지고, 그 순간 모든 수치의 신뢰가 사라진다.
"""

from __future__ import annotations

from txlens.eval.harness import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_TRUTH_PATH,
    load_normalized,
    load_truth,
    run_rules_baseline,
)
from txlens.eval.metrics import (
    RESOLVED_STAGES,
    DistinctSlice,
    EvalResult,
    Slice,
    evaluate,
)

__all__ = [
    "DEFAULT_BUNDLE_DIR",
    "DEFAULT_TRUTH_PATH",
    "RESOLVED_STAGES",
    "DistinctSlice",
    "EvalResult",
    "Slice",
    "evaluate",
    "load_normalized",
    "load_truth",
    "run_rules_baseline",
]
