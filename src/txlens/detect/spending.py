"""② 이상지출 탐지 — 카테고리 기반 (설계 §3.6 ②).

## 정답셋이 이 탐지기를 온전히 채점하지 못한다

두 방향 모두에서 그렇다. **아래로** — 생성기가 심은 급증 중 여럿은 싼 브랜드의 **건수**만 늘려
카테고리 월 합계를 움직이지 않는다(한 건은 평소보다 오히려 적은 달이다). 어떤 문턱으로도 찾을 수
없다. **위로** — 생성기는 우연히 만들어진 진짜 급증에 라벨을 붙이지 않으므로, 그걸 찾아낸 알림이
오탐으로 세어진다. 그래서 여기 정밀도는 **하한**으로 읽어야 한다(`spending_precision_floor`).

문턱을 정답셋에 맞춰 구부리는 대신 **제품 관점으로** 정했다 — 평소의 1.5배 이상이고 증가액이
10만원 이상일 때만 알린다. 생성기 수정은 6주차 몫이다. 지금 데이터를 다시 만들면 발표에 실린
1부 실측이 전부 바뀐다.

## 이 탐지기는 1부에 완전히 의존한다

기준선을 카테고리별로 세우므로, 카테고리가 틀리면 **기준선 자체가 틀린다.** `shopping` 을
`food` 로 분류하면 식비가 없던 급증으로 뜨고 실제 쇼핑 급증은 묻힌다. 출력이 조금 나빠지는 게
아니라 방향이 틀린다 — §5.2 가 이 탐지기를 주 소재로 쓰는 이유다.

## 미분류는 기준선을 세우지 않는다

`unclassified` 는 "여러 업종이 섞인 자루"라 그 안의 월별 변동은 소비 패턴이 아니라 **분류
실패의 변동**이다. 여기에 급증 알림을 띄우면 고객에게는 "미분류가 늘었습니다"라는 말이 되어
아무 뜻이 없다. 그래서 모수에서 뺀다.

## 단발 고액의 기준선은 카테고리가 아니라 가맹점이다 (설계 수정)

설계 §3.6 ② 는 "해당 카테고리 P95 초과"라고 썼는데, **재보니 그 기준선이 서지 않았다.**
`transport` 한 칸에 1,400원 교통비와 15만원 주유가 같이 들어 있어서, 카테고리 안의 금액 분포는
"이 사람의 평소"가 아니라 "어느 업종이 섞였나"를 반영한다. 실측: 카테고리 P95 + robust z 조합은
알림 300건에 적중 14건(정밀도 4.7%)이었다 — 고객에게는 소음이다.

기준선을 **같은 가맹점의 통상 결제 금액**으로 옮기면 "이 가게에서 이 금액은 이상하다"가 되어
설명 가능한 문장이 나온다. 급증(월 합계)은 카테고리 그대로 둔다 — 그건 원래 카테고리 단위 질문이다.

## MAD 가 촘촘하면 z 하나로는 부족하다

교통비처럼 금액이 거의 고정된 가맹점은 MAD 가 작아 **2배만 돼도 z 가 폭발한다.** 5,600원 지하철
결제가 15만원 주유와 같은 등급의 알림이 된다. 그래서 z 와 함께 **중앙값 대비 배수** 하한을 건다 —
두 조건은 서로 다른 실패를 막으므로 둘 다 있어야 한다.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from txlens.detect.common import SpendRow, robust_z
from txlens.detect.schemas import SpendingAlert
from txlens.schemas import CategoryL1

#: 월 합계 급증으로 볼 robust z 문턱 (설계 §3.6 ②).
SURGE_Z = 3.0
#: 평소 대비 배수 하한. 20% 더 쓴 달을 "급증"이라 부르면 고객이 알림을 끈다.
SURGE_RATIO = 1.5
#: 절대 증가액 하한(원). 평소 3만원 쓰던 카테고리가 6만원이 된 것은 배수로는 2배지만
#: 고객에게 알릴 일이 아니다. 배수와 절대액은 서로 다른 실패를 막으므로 둘 다 건다.
SURGE_MIN_EXCESS = 100_000
#: 급증 판정에 필요한 최소 관측 개월. 2개월로는 중앙값이 두 점의 평균이라 기준선이 아니다.
MIN_MONTHS = 3
#: 단발 고액의 z 문턱. 월 합계보다 높게 두는 이유는 거래 단위 분포가 훨씬 두껍기 때문이다.
LARGE_Z = 5.0
#: 중앙값 대비 배수 하한. 금액이 거의 고정된 가맹점에서 z 가 폭발하는 것을 막는다.
LARGE_RATIO = 3.0
#: 기준선을 말할 수 있는 최소 거래 수. 표본이 적으면 중앙값도 MAD 도 그 가맹점의 통상 금액이
#: 아니라 우연이다. 못 재는 가맹점은 **알림을 내지 않는다** — 모르는 것을 이상이라 하지 않는다.
MIN_TX_FOR_BASELINE = 20


def detect_spending_anomalies(rows: list[SpendRow]) -> list[SpendingAlert]:
    """카테고리 급증 · 단발 고액을 찾는다.

    기준선은 **고객별·카테고리별**로 세운다. 전체 평균과 비교하면 원래 많이 쓰는 사람이 매달
    이상으로 뜨고, 적게 쓰는 사람의 실제 급증은 묻힌다. "이 사람의 평소"가 기준이어야 한다.
    """
    by_customer_category: dict[tuple[str, CategoryL1], list[SpendRow]] = defaultdict(list)
    for row in rows:
        if row.category_l1 is CategoryL1.UNCLASSIFIED:
            continue
        by_customer_category[(row.customer_id, row.category_l1)].append(row)

    by_merchant: dict[str, list[SpendRow]] = defaultdict(list)
    for row in rows:
        if row.category_l1 is not CategoryL1.UNCLASSIFIED:
            by_merchant[row.merchant_canonical].append(row)

    out: list[SpendingAlert] = []
    for (customer_id, category), items in sorted(by_customer_category.items(), key=_key):
        out.extend(_category_surges(customer_id, category, items))
    out.extend(_single_large(by_merchant))
    out.sort(key=lambda a: (a.customer_id, a.month, a.kind))
    return out


def _key(entry: tuple[tuple[str, CategoryL1], list[SpendRow]]) -> tuple[str, str]:
    (customer_id, category), _items = entry
    return customer_id, category.value


def _category_surges(
    customer_id: str, category: CategoryL1, items: list[SpendRow]
) -> list[SpendingAlert]:
    monthly: dict[str, list[SpendRow]] = defaultdict(list)
    for row in items:
        monthly[row.month].append(row)
    if len(monthly) < MIN_MONTHS:
        return []

    totals = {month: sum(r.amount for r in group) for month, group in monthly.items()}
    sample = [float(v) for v in totals.values()]
    median = statistics.median(sample)

    alerts: list[SpendingAlert] = []
    for month, total in sorted(totals.items()):
        if median <= 0:
            continue
        ratio = total / median
        if ratio < SURGE_RATIO or total - median < SURGE_MIN_EXCESS:
            continue
        z = robust_z(float(total), sample)
        if z < SURGE_Z:
            continue
        alerts.append(
            SpendingAlert(
                customer_id=customer_id,
                kind="category_surge",
                category_l1=category,
                month=month,
                amount=total,
                robust_z=round(z, 2),
                approval_nos=sorted(r.approval_no for r in monthly[month]),
                evidence=(
                    f"{month} {category.value} 지출 {total:,}원 — 평소 월 {median:,.0f}원의 "
                    f"{ratio:.1f}배 (robust z={z:.1f}, {len(monthly)}개월 기준)"
                ),
            )
        )
    return alerts


def _single_large(by_merchant: dict[str, list[SpendRow]]) -> list[SpendingAlert]:
    """같은 가맹점의 통상 금액에서 크게 벗어난 단발 결제.

    기준선을 고객별로 더 쪼개지 않는 이유: 한 사람이 한 가맹점에서 6개월간 쓰는 횟수는 대개
    한 자릿수라 중앙값도 MAD 도 서지 않는다. 가맹점의 통상 금액대는 고객이 달라도 공유되는
    성질이라 여기서 표본을 모은다.
    """
    alerts: list[SpendingAlert] = []
    for canonical, items in sorted(by_merchant.items()):
        if len(items) < MIN_TX_FOR_BASELINE:
            continue
        sample = [float(r.amount) for r in items]
        median = statistics.median(sample)
        for row in items:
            if median <= 0 or row.amount < median * LARGE_RATIO:
                continue
            z = robust_z(float(row.amount), sample)
            if z < LARGE_Z:
                continue
            alerts.append(
                SpendingAlert(
                    customer_id=row.customer_id,
                    kind="single_large",
                    category_l1=row.category_l1,
                    month=row.month,
                    amount=row.amount,
                    robust_z=round(z, 2),
                    approval_nos=[row.approval_no],
                    evidence=(
                        f"{row.at:%Y-%m-%d} {canonical} {row.amount:,}원 — 이 가맹점의 통상 "
                        f"{median:,.0f}원의 {row.amount / median:.1f}배 (robust z={z:.1f}, "
                        f"{len(items)}건 기준)"
                    ),
                )
            )
    return alerts
