"""생성 오케스트레이터 — 2층 구조의 Layer 2 (의미 있는 값).

Layer 1(표준API 봉투)은 `emit.py` 가 담당한다. 여기서는 **의미**를 만든다:
현실적인 금액·시각 분포, 구독 주기, 탐지 대상 패턴, 오탐 함정, 그리고 정답 기록.

핵심 원칙 — **심으면서 정답을 적는다.** 사후 라벨링이 아니므로 정답이 정의상 옳고,
심지 않은 패턴은 평가 대상이 아니다(blueprint §3.6).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from txlens.generator.brands import (
    SPENDING_BRANDS,
    SUBSCRIPTION_BRANDS,
    Brand,
)
from txlens.generator.corrupt import corrupt, make_regno
from txlens.generator.profiles import ProviderSet, load_providers
from txlens.generator.rng import child_rng, weighted_pick
from txlens.ground_truth import (
    DecoyTruth,
    GroundTruth,
    SpendingAnomalyTruth,
    SubscriptionTruth,
    TxAnomalyTruth,
    TxLabel,
)
from txlens.schemas import CardType, CategoryL1, CategoryL2, RawTx, TxType

#: 급여일. 이후 며칠간 소비가 늘어난다.
PAYDAY = 25


@dataclass
class Draft:
    """오염·직렬화 전의 거래 초안. 정답을 이미 알고 있다."""

    customer_id: str
    brand: Brand
    when: datetime
    amount: int
    #: 이 거래가 어떤 의도로 심어졌는지 — 정답 기록에 쓴다
    tag: str = "daily"
    install_months: int = 0
    cancelled: bool = False


@dataclass
class _Acc:
    drafts: list[Draft] = field(default_factory=list)
    subs: list[SubscriptionTruth] = field(default_factory=list)
    spend: list[SpendingAnomalyTruth] = field(default_factory=list)
    txanom: list[TxAnomalyTruth] = field(default_factory=list)
    decoys: list[DecoyTruth] = field(default_factory=list)


# ── 시각 분포 ───────────────────────────────────────────────────────────────
def _pick_hour(rng: random.Random, brand: Brand) -> tuple[int, int]:
    if brand.late_night and rng.random() < 0.18:
        hour = rng.choice((0, 1, 2, 3, 23))
    elif brand.l1 is CategoryL1.CAFE:
        hour = weighted_pick(rng, [8, 9, 10, 13, 14, 15, 16], [3, 4, 3, 4, 3, 2, 2])
    elif brand.l2 is CategoryL2.FOOD_RESTAURANT:
        hour = weighted_pick(rng, [11, 12, 13, 18, 19, 20], [2, 5, 3, 4, 5, 3])
    elif brand.l1 is CategoryL1.TRANSPORT:
        hour = weighted_pick(rng, [7, 8, 9, 18, 19, 20], [4, 5, 3, 4, 4, 2])
    else:
        hour = rng.randrange(9, 22)
    return hour, rng.randrange(0, 60)


def _amount(rng: random.Random, brand: Brand) -> int:
    """로그 성향의 금액 — 소액이 많고 고액이 드물게. 100원 단위로 떨어뜨린다."""
    if brand.lo >= brand.hi:
        return brand.lo
    u = rng.random() ** 1.8           # 하한 쪽으로 치우침
    raw = brand.lo + (brand.hi - brand.lo) * u
    return max(brand.lo, int(round(raw / 100) * 100))


def _day_weight(day: int, dow: int, brand: Brand) -> float:
    """급여일 이후·주말 가중."""
    w = 1.0
    if PAYDAY <= day <= PAYDAY + 5 or day <= 3:
        w *= 1.5
    if dow >= 5:  # 토·일
        if brand.l1 in (CategoryL1.FOOD, CategoryL1.CULTURE, CategoryL1.SHOPPING):
            w *= 1.6
        if brand.l1 is CategoryL1.TRANSPORT:
            w *= 0.6
    return w


def _month_days(start: datetime, months: int) -> list[datetime]:
    days: list[datetime] = []
    d = start
    end = _add_months(start, months)
    while d < end:
        days.append(d)
        d += timedelta(days=1)
    return days


def _add_months(d: datetime, n: int) -> datetime:
    y, m = divmod(d.month - 1 + n, 12)
    return d.replace(year=d.year + y, month=m + 1, day=1)


# ── ① 일상 소비 ─────────────────────────────────────────────────────────────
def _daily(rng: random.Random, cid: str, days: list[datetime], acc: _Acc, per_month: int) -> None:
    weights = [b.frequency for b in SPENDING_BRANDS]
    n = int(per_month * len(days) / 30)
    for _ in range(n):
        brand = weighted_pick(rng, SPENDING_BRANDS, weights)
        # 날짜는 가중 재추첨으로 급여일·주말 편중을 만든다
        for _try in range(4):
            day = rng.choice(days)
            if rng.random() < _day_weight(day.day, day.weekday(), brand) / 2.4:
                break
        hour, minute = _pick_hour(rng, brand)
        amount = _amount(rng, brand)
        install = 0
        if amount >= 150_000 and rng.random() < 0.25:
            install = rng.choice((3, 6, 12))
        acc.drafts.append(
            Draft(cid, brand, day.replace(hour=hour, minute=minute), amount,
                  tag="daily", install_months=install,
                  cancelled=rng.random() < 0.02)
        )


# ── ② 구독 (2부 정답) ──────────────────────────────────────────────────────
def _subscriptions(rng: random.Random, cid: str, start: datetime, months: int, acc: _Acc) -> None:
    pool = list(SUBSCRIPTION_BRANDS)
    rng.shuffle(pool)
    n_monthly = rng.randrange(2, 5)
    chosen = pool[:n_monthly]

    for brand in chosen:
        base_amount = _amount(rng, brand)
        anchor_day = rng.randrange(1, 28)
        period = brand.period_days or 30
        occurrences = 0
        approvals: list[str] = []
        price_increased = False
        amount = base_amount
        # 해지 후 미사용: 마지막 1~2개월을 비운다
        unused = rng.choice((0, 0, 0, 1, 2))
        active_months = max(3, months - unused) if period == 30 else months

        d = start.replace(day=anchor_day)
        step = 0
        while d < _add_months(start, active_months):
            if period == 365 and step > 0:
                break
            # 가격 인상: 중간부터 계단식 상승
            if period == 30 and not price_increased and step >= 3 and rng.random() < 0.35:
                amount = int(round(amount * 1.18 / 100) * 100)
                price_increased = True
            jitter = rng.choice((-1, 0, 0, 0, 1))
            when = d + timedelta(days=jitter, hours=rng.randrange(1, 6))
            acc.drafts.append(Draft(cid, brand, when, amount, tag=f"sub:{brand.brand}"))
            approvals.append("")  # 승인번호는 emit 단계에서 부여 → 사후 연결
            occurrences += 1
            step += 1
            d = _add_months(d, 1) if period == 30 else _add_months(d, 12)

        if occurrences >= 1:
            acc.subs.append(SubscriptionTruth(
                customer_id=cid, canonical=brand.brand, period_days=period,
                amount=base_amount, occurrences=occurrences, approval_nos=[],
                price_increased=price_increased, unused_months=unused,
                # 탐지 규칙의 "최소 3회"를 못 채우면 원리상 찾을 수 없다 (연 구독이 그렇다)
                detectable=occurrences >= 3,
            ))


# ── ③ 이상지출 (2부 정답) ──────────────────────────────────────────────────
def _spending_anomalies(
    rng: random.Random, cid: str, days: list[datetime], acc: _Acc
) -> None:
    # 카테고리 급증: 특정 달에 한 카테고리를 몰아 넣는다
    brand = weighted_pick(rng, SPENDING_BRANDS, [b.frequency for b in SPENDING_BRANDS])
    target_month = rng.choice(days).replace(day=rng.randrange(5, 25))
    burst = rng.randrange(8, 14)
    for _ in range(burst):
        when = target_month + timedelta(days=rng.randrange(0, 6), hours=rng.randrange(9, 22))
        acc.drafts.append(Draft(cid, brand, when, _amount(rng, brand), tag="anom:surge"))
    acc.spend.append(SpendingAnomalyTruth(
        customer_id=cid, kind="category_surge", category_l1=brand.gold_l1,
        month=target_month.strftime("%Y-%m"), approval_nos=[],
    ))

    # 단발 고액: 해당 브랜드 상한의 3배
    big = weighted_pick(rng, SPENDING_BRANDS, [b.frequency for b in SPENDING_BRANDS])
    when = rng.choice(days).replace(hour=rng.randrange(11, 21))
    acc.drafts.append(Draft(cid, big, when, big.hi * 3, tag="anom:large"))
    acc.spend.append(SpendingAnomalyTruth(
        customer_id=cid, kind="single_large", category_l1=big.gold_l1,
        month=when.strftime("%Y-%m"), approval_nos=[],
    ))


# ── ④ 이상거래패턴 (2부 정답) ──────────────────────────────────────────────
def _tx_anomalies(rng: random.Random, cid: str, days: list[datetime], acc: _Acc) -> None:
    # 중복결제: 동일 가맹점·동일 금액·10분 이내, 취소 페어 없음
    if rng.random() < 0.7:
        brand = weighted_pick(rng, SPENDING_BRANDS, [b.frequency for b in SPENDING_BRANDS])
        when = rng.choice(days).replace(hour=rng.randrange(11, 21), minute=rng.randrange(0, 45))
        amt = _amount(rng, brand)
        acc.drafts.append(Draft(cid, brand, when, amt, tag="anom:dup"))
        acc.drafts.append(Draft(cid, brand, when + timedelta(minutes=rng.randrange(2, 9)),
                                amt, tag="anom:dup"))
        acc.txanom.append(TxAnomalyTruth(
            customer_id=cid, kind="duplicate_charge", canonical=brand.brand, approval_nos=[]))

    # 소액 다발: 1,000원 이하 연속 결제 (카드 유효성 테스트 흔적)
    if rng.random() < 0.4:
        brand = weighted_pick(rng, SPENDING_BRANDS, [b.frequency for b in SPENDING_BRANDS])
        base = rng.choice(days).replace(hour=rng.randrange(1, 5))
        for k in range(rng.randrange(4, 7)):
            acc.drafts.append(Draft(cid, brand, base + timedelta(minutes=k * 2),
                                    rng.choice((100, 200, 500, 900)), tag="anom:burst"))
        acc.txanom.append(TxAnomalyTruth(
            customer_id=cid, kind="micro_burst", canonical=brand.brand, approval_nos=[]))


# ── ⑤ 오탐 함정 — 정답이 "이상 아님" ────────────────────────────────────────
def _decoys(rng: random.Random, cid: str, start: datetime, months: int,
            days: list[datetime], acc: _Acc) -> None:
    # 구독처럼 보이는 비구독: 매달 같은 가맹점이지만 금액 편차가 크다 (CV 조건에 걸려야 함)
    brand = weighted_pick(rng, SPENDING_BRANDS, [b.frequency for b in SPENDING_BRANDS])
    anchor = rng.randrange(1, 28)
    d = start.replace(day=anchor)
    while d < _add_months(start, months):
        amt = int(round(_amount(rng, brand) * rng.uniform(0.4, 2.2) / 100) * 100)
        acc.drafts.append(Draft(cid, brand, d + timedelta(hours=rng.randrange(9, 20)),
                                max(1000, amt), tag="decoy:sublike"))
        d = _add_months(d, 1)
    acc.decoys.append(DecoyTruth(
        customer_id=cid, kind="subscription_like", canonical=brand.brand, approval_nos=[],
        note="매달 결제되지만 금액 변동이 커 구독이 아니다 (CV 조건에서 걸러져야 한다)",
    ))

    # 정상 연속결제: 10분 내 2건이지만 금액이 다르다 (중복결제가 아니다)
    conv = weighted_pick(rng, SPENDING_BRANDS, [b.frequency for b in SPENDING_BRANDS])
    when = rng.choice(days).replace(hour=rng.randrange(12, 21), minute=rng.randrange(0, 40))
    a1 = _amount(rng, conv)
    a2 = int(round(a1 * rng.uniform(1.5, 3.0) / 100) * 100)   # 100원 단위 유지
    acc.drafts.append(Draft(cid, conv, when, a1, tag="decoy:consec"))
    acc.drafts.append(Draft(cid, conv, when + timedelta(minutes=rng.randrange(3, 9)),
                            max(1000, a2), tag="decoy:consec"))
    acc.decoys.append(DecoyTruth(
        customer_id=cid, kind="normal_consecutive", canonical=conv.brand, approval_nos=[],
        note="짧은 간격 2건이지만 금액이 달라 중복결제가 아니다",
    ))


# ── 조립 ────────────────────────────────────────────────────────────────────
def build(
    seed: int = 42,
    customers: int = 30,
    months: int = 6,
    period_end: datetime | None = None,
    per_month: int = 42,
    providers: ProviderSet | None = None,
) -> tuple[list[RawTx], GroundTruth]:
    """합성 거래와 정답셋을 만든다.

    `period_end` 를 인자로 받는 이유: 실행 시각에 의존하면 같은 시드라도 데이터가 달라진다.
    """
    pset = providers or load_providers()
    # 기본 종료일 상수. tz 없는 KST 로 통일한다 (D-009) — 표준API 가 오프셋 없이 주고
    # 데이터가 한국 단일 시간대다. `now()` 를 쓰지 않는 이유는 §docstring 참고.
    default_end = datetime(2026, 8, 31)  # noqa: DTZ001
    end = (period_end or default_end).replace(hour=0, minute=0, second=0, microsecond=0)
    start = _add_months(end.replace(day=1), -months + 1)
    days = _month_days(start, months)

    acc = _Acc()
    # 고객별 보유 카드 세트. 한 사람이 모든 기관을 쓰지는 않는다 — 기관 배정을 이 안에서만 한다.
    cards: dict[str, list[str]] = {}
    #: (고객, 기관) → 신용/체크. **카드의 속성이므로 거래마다 달라지지 않는다** (D-023).
    #: 규격의 `pay_type` 이 이 값이다 — 할부 여부가 아니다.
    card_types: dict[tuple[str, str], CardType] = {}
    for i in range(customers):
        cid = f"SYNTH_{i:04d}"
        crng = child_rng(seed, "cards", cid)
        k = crng.choices((1, 2, 3), weights=(0.25, 0.5, 0.25), k=1)[0]
        picked: list[str] = []
        pool, pw = list(pset.ids()), list(pset.weights())
        for _ in range(min(k, len(pool))):
            chosen = weighted_pick(crng, pool, pw)
            j = pool.index(chosen)
            pool.pop(j); pw.pop(j)
            picked.append(chosen)
        cards[cid] = picked
        for pid in picked:
            trng = child_rng(seed, "cardtype", cid, pid)
            card_types[(cid, pid)] = (
                CardType.CHECK if trng.random() < 0.3 else CardType.CREDIT
            )

        _daily(child_rng(seed, "daily", cid), cid, days, acc, per_month)
        _subscriptions(child_rng(seed, "sub", cid), cid, start, months, acc)
        _spending_anomalies(child_rng(seed, "spend", cid), cid, days, acc)
        _tx_anomalies(child_rng(seed, "txanom", cid), cid, days, acc)
        _decoys(child_rng(seed, "decoy", cid), cid, start, months, days, acc)

    # 시간순으로 정렬해 승인번호를 안정적으로 부여한다
    acc.drafts.sort(key=lambda d: (d.when, d.customer_id, d.brand.brand, d.amount))

    txs: list[RawTx] = []
    gt = GroundTruth(
        seed=seed, customers=customers,
        period_start=start.strftime("%Y-%m-%d"), period_end=end.strftime("%Y-%m-%d"),
        tx_total=0, distinct_raw_names=0,
    )
    by_tag: dict[tuple[str, str], list[str]] = {}
    raw_names: set[str] = set()

    for idx, dr in enumerate(acc.drafts):
        prng = child_rng(seed, "provider", dr.customer_id, idx)
        owned = cards[dr.customer_id]
        if dr.tag.startswith("sub:"):
            # 구독은 **등록된 카드 한 장**으로 계속 결제된다. 매달 다른 기관으로 나가면
            # 비현실적일 뿐 아니라 구독 탐지가 카드 축으로 묶이지 않아 평가가 왜곡된다.
            pinned = child_rng(seed, "subcard", dr.customer_id, dr.tag).choice(owned)
            profile = pset.by_id(pinned)
        else:
            w = [pset.by_id(pid).weight for pid in owned]
            profile = pset.by_id(weighted_pick(prng, owned, w))
        cn = corrupt(child_rng(seed, "corrupt", dr.customer_id, idx), dr.brand, profile)

        regno: str | None = None
        if prng.random() < profile.regno_fill_rate:
            regno = make_regno(prng, dr.brand, cn.branch, pg=cn.pg_masked)

        approval_no = f"{seed:02d}{idx:08d}"
        raw_names.add(cn.raw)
        card_type = card_types[(dr.customer_id, profile.id)]
        # 체크카드는 할부가 되지 않는다. 카드 배정이 초안(draft)보다 뒤에 오므로 여기서 맞춘다 —
        # 규격상 `pay_type`(신용/체크)과 `total_install_cnt` 는 독립이지만 현실에서는 이 제약이 있다.
        install = dr.install_months if card_type is CardType.CREDIT else 0
        txs.append(RawTx(
            provider_id=profile.id,
            customer_id=dr.customer_id,
            approval_no=approval_no,
            tx_datetime=dr.when,
            merchant_name_raw=cn.raw,
            amount=dr.amount,
            tx_type=TxType.APPROVE,
            installment_months=install,
            card_type=card_type,
            merchant_regno=regno,
        ))

        # PG 마스킹이면 정답도 미분류다 — 억지 추측을 정답으로 만들지 않는다
        if cn.pg_masked:
            label = TxLabel(
                canonical="", category_l1=CategoryL1.UNCLASSIFIED,
                category_l2=CategoryL2.UNCLASSIFIED_PG_MASKED,
                provider_id=profile.id, pg_masked=True, style=cn.style,
            )
        else:
            # `gold_*` 를 쓴다 — 택소노미에 자리가 없는 업종은 **미분류가 정답**이다 (D-026).
            # `l1`/`l2` 는 행동 모델용 값이라 정답으로 쓰면 억지 라벨이 정답이 된다.
            label = TxLabel(
                canonical=dr.brand.brand, category_l1=dr.brand.gold_l1,
                category_l2=dr.brand.gold_l2, provider_id=profile.id, style=cn.style,
            )
        gt.tx_labels[approval_no] = label
        by_tag.setdefault((dr.customer_id, dr.tag), []).append(approval_no)

        # 취소 페어 — 동일 승인번호, 부호 반대.
        # ⚠️ 규격은 취소를 *부호 없는 금액 + status=02 + `trans_dtime`* 으로 표현한다. 여기서는
        # 음수 금액 별행으로 두고 `trans_dtime` 을 내지 않는다 — 의도된 단순화이며 D-023 에
        # 기록했다. `ingest` 는 실데이터의 `trans_dtime` 을 읽을 수 있다.
        if dr.cancelled:
            txs.append(RawTx(
                provider_id=profile.id, customer_id=dr.customer_id,
                approval_no=approval_no,
                tx_datetime=dr.when + timedelta(days=prng.randrange(1, 4)),
                merchant_name_raw=cn.raw, amount=-dr.amount,
                tx_type=TxType.CANCEL, card_type=card_type, merchant_regno=regno,
            ))

    # 심은 패턴에 승인번호를 연결한다
    for s in acc.subs:
        s.approval_nos = by_tag.get((s.customer_id, f"sub:{s.canonical}"), [])
    for sa in acc.spend:
        tag = "anom:surge" if sa.kind == "category_surge" else "anom:large"
        sa.approval_nos = by_tag.get((sa.customer_id, tag), [])
    for ta in acc.txanom:
        tag = "anom:dup" if ta.kind == "duplicate_charge" else "anom:burst"
        ta.approval_nos = by_tag.get((ta.customer_id, tag), [])
    for dc in acc.decoys:
        tag = "decoy:sublike" if dc.kind == "subscription_like" else "decoy:consec"
        dc.approval_nos = by_tag.get((dc.customer_id, tag), [])

    gt.subscriptions = acc.subs
    gt.spending_anomalies = acc.spend
    gt.tx_anomalies = acc.txanom
    gt.decoys = acc.decoys
    gt.tx_total = len(txs)
    gt.distinct_raw_names = len(raw_names)
    return txs, gt
