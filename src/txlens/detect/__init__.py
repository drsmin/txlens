"""2부 — 패턴 탐지 (설계 §3.6). 구독 · 이상지출 · 이상거래패턴.

## 왜 그래프 노드가 아니라 함수인가

설계 §3.2 의 그림은 `finalize` 뒤에 `detect_patterns` 를 둔다. 지금 노드로 만들지 않는
이유는 `TxState` 의 규율과 같다 — **소비자가 없는 필드는 파이프라인이 하는 일을 과장한다.**
탐지 결과를 상태에 실어도 읽는 곳이 아직 없다(화면은 5주차다).

더 결정적인 것은 §5.2 다. 전이 실험은 **같은 탐지기를 서로 다른 분류 결과 위에서** 돌려 비교
한다. 탐지가 분류 실행 안에 묶여 있으면 "분류만 바꾼 비교"를 만들 수 없다. 분류 결과를 받아
탐지 결과를 내는 순수 함수라야 A·C·정답분류를 같은 자로 잰다.

## LLM 을 호출하지 않는다

셋 다 결정론 통계 규칙이다. 같은 입력이면 같은 출력이고, 근거 문장이 규칙을 그대로 옮긴
문장이라 고객에게 설명할 수 있다 — 금융 앱에서 설명 가능성이 정확도보다 중요할 때가 있다.
"""

from __future__ import annotations

from datetime import datetime

from txlens.detect.common import SpendRow, spend_rows
from txlens.detect.patterns import detect_tx_anomalies
from txlens.detect.schemas import Detections, SpendingAlert, Subscription, TxAlert
from txlens.detect.spending import detect_spending_anomalies
from txlens.detect.subscription import detect_subscriptions, monthly_fixed_cost
from txlens.schemas import ClassifiedTx

__all__ = [
    "Detections",
    "SpendRow",
    "SpendingAlert",
    "Subscription",
    "TxAlert",
    "detect",
    "detect_from_rows",
    "detect_spending_anomalies",
    "detect_subscriptions",
    "detect_tx_anomalies",
    "monthly_fixed_cost",
    "spend_rows",
]


def detect(
    classified: list[ClassifiedTx], *, period_end: datetime | None = None
) -> Detections:
    """분류 결과 위에서 탐지 3종을 돌린다. **입력이 바뀌면 출력이 바뀐다** — 그게 §5.2 다."""
    return detect_from_rows(spend_rows(classified), period_end=period_end)


def detect_from_rows(
    rows: list[SpendRow], *, period_end: datetime | None = None
) -> Detections:
    return Detections(
        subscriptions=detect_subscriptions(rows, period_end=period_end),
        spending_alerts=detect_spending_anomalies(rows),
        tx_alerts=detect_tx_anomalies(rows),
    )
