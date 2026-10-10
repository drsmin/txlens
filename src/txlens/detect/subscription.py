"""① 구독 탐지 (blueprint §3.6 ①).

## 규칙은 세 개고, 셋 다 있어야 한다

**주기** — 결제 간격의 중앙값이 주/월/연 창 안에 든다. 평균이 아니라 중앙값인 이유는 한 번의
결제 누락이 평균 간격을 두 배로 만들기 때문이다.
**금액 안정성** — 변동계수 CV 가 낮다. 이 조건 하나가 오탐 함정("매달 같은 가맹점이지만 금액이
들쭉날쭉")을 걸러낸다. 이게 없으면 단골 식당이 전부 구독으로 잡힌다.
**최소 횟수** — 3회. 2회로는 간격이 하나뿐이라 "주기"라고 부를 근거가 없다.

## 최소 3회는 탐지 가능성의 경계를 만든다

조회 기간이 6개월이면 **연 구독은 1회만 나타나 원리상 찾을 수 없다.** 정답셋은 이걸 알고
`detectable=False` 로 표시해 둔다(`ground_truth.py`). 규칙을 느슨하게 해서 잡는 대신, 못 잡는
것을 못 잡는다고 보고한다 — 2회짜리를 구독이라 부르면 단발 결제가 전부 구독이 된다.

## 그룹 키는 `merchant_canonical` 이다 — 그래서 1부가 그대로 전이된다

같은 넷플릭스가 `NETFLIX` · `넷플릭스` · `NAVERPAY*넷플릭스` 로 흩어져 있으면 각 그룹이 3회를
못 채워 **구독이 통째로 사라진다.** 분류를 고쳐야 탐지가 되는 이유이고, §5.2 가 재는 것이다.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import pairwise

from txlens.detect.common import SpendRow, coefficient_of_variation
from txlens.detect.schemas import PeriodKind, Subscription

#: 최소 발생 횟수. 간격이 2개는 있어야 "주기"를 말할 수 있다.
MIN_OCCURRENCES = 3
#: 금액 변동계수 상한. 가격 인상(계단식 18%)은 통과하고 함정(±배수 요동)은 걸리는 자리다.
MAX_AMOUNT_CV = 0.15
#: 주기 창 — (이름, 하한일, 상한일, 월 환산 배수).
PERIOD_WINDOWS: list[tuple[PeriodKind, int, int, float]] = [
    ("weekly", 6, 8, 30 / 7),
    ("monthly", 27, 33, 1.0),
    ("yearly", 355, 375, 1 / 12),
]
#: 마지막 결제 후 이 주기 수 이상 지났으면 "안 쓰는 구독" 후보로 표시한다.
#: blueprint 는 "3개월간 미사용"이라 썼지만 그건 월 구독에만 맞는 표현이라 **주기 배수**로 옮겼다 —
#: 주 구독에 3개월을 그대로 적용하면 12주를 놓쳐야 뜬다.
UNUSED_PERIODS = 2
#: 계단식 가격 인상으로 볼 상승률. 이 아래는 부가세·환율 잔돈 변동으로 본다.
PRICE_STEP_RATIO = 1.05


def _period_kind(median_gap: float) -> tuple[PeriodKind, float] | None:
    for kind, lo, hi, monthly in PERIOD_WINDOWS:
        if lo <= median_gap <= hi:
            return kind, monthly
    return None


def _price_increased(amounts: list[int]) -> bool:
    """계단식 상승인가 — **오르기만 했고** 끝이 시작보다 뚜렷이 높은가.

    한 번이라도 내렸다면 요동이지 인상이 아니다. 단조 조건을 빼면 금액이 흔들리는 가맹점이
    전부 "가격 인상"으로 보고된다.
    """
    if len(amounts) < 2 or amounts[0] <= 0:
        return False
    monotone = all(b >= a for a, b in pairwise(amounts))
    return monotone and amounts[-1] >= amounts[0] * PRICE_STEP_RATIO


def detect_subscriptions(
    rows: list[SpendRow], *, period_end: datetime | None = None
) -> list[Subscription]:
    """(고객 × 대표명) 으로 묶어 정기 결제를 찾는다.

    `period_end` 는 "안 쓰는 구독"의 기준 시각이다. 주지 않으면 데이터의 마지막 거래 시각을
    쓴다 — 오늘 날짜를 쓰면 몇 달 전에 만든 데이터셋이 전부 미사용으로 뜬다.
    """
    if not rows:
        return []
    as_of = period_end or max(r.at for r in rows)

    groups: dict[tuple[str, str], list[SpendRow]] = defaultdict(list)
    for row in rows:
        groups[(row.customer_id, row.merchant_canonical)].append(row)

    out: list[Subscription] = []
    for (customer_id, canonical), items in sorted(groups.items()):
        if len(items) < MIN_OCCURRENCES:
            continue
        items = sorted(items, key=lambda r: r.at)
        gaps = [(b.at - a.at).total_seconds() / 86400.0 for a, b in pairwise(items)]
        median_gap = statistics.median(gaps)
        window = _period_kind(median_gap)
        if window is None:
            continue
        kind, monthly_factor = window

        amounts = [r.amount for r in items]
        cv = coefficient_of_variation([float(a) for a in amounts])
        if cv >= MAX_AMOUNT_CV:
            continue

        median_amount = int(statistics.median(amounts))
        last = items[-1]
        next_expected = last.at + timedelta(days=median_gap)
        elapsed_periods = (as_of - last.at).total_seconds() / 86400.0 / median_gap
        unused = elapsed_periods >= UNUSED_PERIODS
        raised = _price_increased(amounts)

        parts = [
            f"{len(items)}회 결제 · 간격 중앙값 {median_gap:.0f}일 · 금액 변동 {cv:.1%}",
            f"다음 결제 예상 {next_expected:%Y-%m-%d}",
        ]
        if raised:
            parts.append(f"금액이 {amounts[0]:,}원 → {amounts[-1]:,}원으로 올랐다")
        if unused:
            parts.append(f"마지막 결제 {last.at:%Y-%m-%d} 이후 {elapsed_periods:.1f}주기째 결제 없음")

        out.append(
            Subscription(
                customer_id=customer_id,
                merchant_canonical=canonical,
                category_l1=items[0].category_l1,
                period_kind=kind,
                period_days=round(median_gap),
                occurrences=len(items),
                median_amount=median_amount,
                amount_cv=round(cv, 4),
                first_at=items[0].at,
                last_at=last.at,
                next_expected=next_expected,
                unused=unused,
                price_increased=raised,
                monthly_cost=round(median_amount * monthly_factor),
                approval_nos=[r.approval_no for r in items],
                evidence=" · ".join(parts),
            )
        )
    return out


def monthly_fixed_cost(subs: list[Subscription]) -> dict[str, int]:
    """고객별 월 고정지출 합계. 미사용 의심 건도 포함한다 — 아직 해지되지 않았다면 나간다."""
    total: dict[str, int] = defaultdict(int)
    for s in subs:
        total[s.customer_id] += s.monthly_cost
    return dict(total)
