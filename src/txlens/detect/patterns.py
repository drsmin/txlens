"""③ 이상거래패턴 탐지 — 거래 형태 기반 (blueprint §3.6 ③).

## 카테고리를 보지 않는다

중복결제와 소액 다발은 **금액·시각·가맹점**만으로 선다. 그래서 분류가 나빠져도 앞의 두 탐지기
만큼 무너지지 않는다 — 다만 가맹점을 못 묶으면 "같은 가맹점에 두 번"이라는 전제가 깨지므로
영향을 아주 안 받지는 않는다. §5.2 에서 세 탐지기의 낙폭이 다르게 나오는 이유다.

## FDS 가 아니다

출력은 차단이 아니라 "확인해 보세요"다(blueprint §3.6). 그래서 재현율을 밀어붙여 조금이라도
의심스러운 것을 다 띄우는 대신, **근거 문장으로 설명되는 것만** 띄운다.

## 두 규칙이 서로를 침범하지 않게 한다

소액 다발은 같은 가맹점에 같은 금액이 짧은 간격으로 반복되는 형태라, 손대지 않으면 중복결제
규칙에도 그대로 걸려 한 사건이 두 알림이 된다. 그래서 중복결제는 소액 문턱 **위**의 금액만
본다. 데이터의 일상 결제 하한이 1,400원이라 1,000원 경계는 정상 소비를 자르지 않는다.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from txlens.detect.common import SpendRow
from txlens.detect.schemas import TxAlert

#: 중복결제로 볼 재결제 간격 상한.
DUPLICATE_WINDOW = timedelta(minutes=10)
#: 소액 다발의 1건 금액 상한 (카드 유효성 테스트 흔적).
MICRO_MAX_AMOUNT = 1000
#: 소액 다발로 볼 최소 건수와 창. 3건 미만은 잔돈 결제와 구분되지 않는다.
MICRO_MIN_COUNT = 3
MICRO_WINDOW = timedelta(minutes=30)


def detect_tx_anomalies(rows: list[SpendRow]) -> list[TxAlert]:
    """중복결제 의심 · 소액 다발을 찾는다."""
    groups: dict[tuple[str, str], list[SpendRow]] = defaultdict(list)
    for row in rows:
        groups[(row.customer_id, row.merchant_canonical)].append(row)

    out: list[TxAlert] = []
    for (customer_id, canonical), items in sorted(groups.items()):
        ordered = sorted(items, key=lambda r: r.at)
        out.extend(_duplicates(customer_id, canonical, ordered))
        out.extend(_micro_bursts(customer_id, canonical, ordered))
    out.sort(key=lambda a: (a.customer_id, a.at, a.kind))
    return out


def _duplicates(customer_id: str, canonical: str, ordered: list[SpendRow]) -> list[TxAlert]:
    """동일 금액 + 창 안 재결제. 취소 페어는 입구(`spend_rows`)에서 이미 빠져 있다."""
    by_amount: dict[int, list[SpendRow]] = defaultdict(list)
    for row in ordered:
        if row.amount > MICRO_MAX_AMOUNT:
            by_amount[row.amount].append(row)

    alerts: list[TxAlert] = []
    for amount, rows in by_amount.items():
        run: list[SpendRow] = []
        for row in rows:
            if run and row.at - run[-1].at > DUPLICATE_WINDOW:
                alerts.extend(_duplicate_alert(customer_id, canonical, amount, run))
                run = []
            run.append(row)
        alerts.extend(_duplicate_alert(customer_id, canonical, amount, run))
    return alerts


def _duplicate_alert(
    customer_id: str, canonical: str, amount: int, run: list[SpendRow]
) -> list[TxAlert]:
    if len(run) < 2:
        return []
    gap_minutes = (run[-1].at - run[0].at).total_seconds() / 60.0
    return [
        TxAlert(
            customer_id=customer_id,
            kind="duplicate_charge",
            merchant_canonical=canonical,
            amount=amount,
            at=run[0].at,
            approval_nos=[r.approval_no for r in run],
            evidence=(
                f"{run[0].at:%Y-%m-%d %H:%M} {canonical} {amount:,}원이 "
                f"{gap_minutes:.0f}분 사이 {len(run)}번 결제됐다 (취소 내역 없음)"
            ),
        )
    ]


def _micro_bursts(customer_id: str, canonical: str, ordered: list[SpendRow]) -> list[TxAlert]:
    """소액이 짧은 창 안에 여러 건 — 카드가 살아 있는지 시험한 흔적일 수 있다."""
    micro = [r for r in ordered if r.amount <= MICRO_MAX_AMOUNT]
    alerts: list[TxAlert] = []
    run: list[SpendRow] = []
    for row in micro:
        if run and row.at - run[0].at > MICRO_WINDOW:
            alerts.extend(_micro_alert(customer_id, canonical, run))
            run = []
        run.append(row)
    alerts.extend(_micro_alert(customer_id, canonical, run))
    return alerts


def _micro_alert(customer_id: str, canonical: str, run: list[SpendRow]) -> list[TxAlert]:
    if len(run) < MICRO_MIN_COUNT:
        return []
    span_minutes = (run[-1].at - run[0].at).total_seconds() / 60.0
    total = sum(r.amount for r in run)
    return [
        TxAlert(
            customer_id=customer_id,
            kind="micro_burst",
            merchant_canonical=canonical,
            amount=total,
            at=run[0].at,
            approval_nos=[r.approval_no for r in run],
            evidence=(
                f"{run[0].at:%Y-%m-%d %H:%M} {canonical}에 1,000원 이하 결제가 "
                f"{span_minutes:.0f}분 사이 {len(run)}건 (합계 {total:,}원)"
            ),
        )
    ]
