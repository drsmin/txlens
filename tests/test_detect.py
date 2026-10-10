"""2부 탐지 계약 테스트.

탐지기도 지표와 같은 성질을 갖는다 — **틀려도 에러를 내지 않고 그냥 다른 알림을 낸다.** 그래서
여기서 막는 것은 크래시가 아니라 다음이다.

1. **오탐 함정이 통과하는 것.** 금액이 들쭉날쭉한 단골 가맹점이 구독이 되고, 금액이 다른 연속
   결제가 중복결제가 되면 알림은 소음이 된다. 규칙이 느슨해지는 방향의 변경을 여기서 잡는다.
2. **취소된 거래가 다시 세어지는 것.** 전처리가 붙인 `excluded_reason` 을 탐지 입구에서 존중하지
   않으면 "쓰지 않은 돈"이 급증으로 뜬다.
3. **전이가 사라지는 것.** §5.2 의 주장은 "분류를 고치면 탐지가 는다"이므로, 같은 탐지기가 더 나쁜
   분류 위에서 **더 나쁜 결과**를 내야 한다. 그 부등호가 뒤집히면 논거가 무너진다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from txlens.detect import detect, detect_from_rows, spend_rows
from txlens.detect.common import SpendRow, robust_z
from txlens.detect.patterns import detect_tx_anomalies
from txlens.detect.spending import SURGE_MIN_EXCESS, detect_spending_anomalies
from txlens.detect.subscription import (
    MAX_AMOUNT_CV,
    MIN_OCCURRENCES,
    detect_subscriptions,
    monthly_fixed_cost,
)
from txlens.eval.harness import DEFAULT_BUNDLE_DIR, DEFAULT_TRUTH_PATH
from txlens.eval.transfer import run_transfer
from txlens.preprocess import preprocess
from txlens.schemas import (
    CategoryL1,
    CategoryL2,
    ClassifiedTx,
    MatchStage,
    RawTx,
    TxType,
)

START = datetime(2026, 3, 10, 9, 0)


def _row(
    no: str,
    *,
    when: datetime,
    amount: int,
    canonical: str = "넷플릭스",
    category: CategoryL1 = CategoryL1.CULTURE,
    customer: str = "C0001",
) -> SpendRow:
    return SpendRow(
        customer_id=customer,
        approval_no=no,
        at=when,
        amount=amount,
        merchant_canonical=canonical,
        category_l1=category,
    )


def _monthly(amounts: list[int], *, canonical: str = "넷플릭스", start: datetime = START):
    return [
        _row(f"S{i}", when=start + timedelta(days=30 * i), amount=a, canonical=canonical)
        for i, a in enumerate(amounts)
    ]


# ── ① 구독 ─────────────────────────────────────────────────────────────────
def test_monthly_subscription_is_detected_with_next_payment_date():
    subs = detect_subscriptions(_monthly([14900] * 6))
    assert len(subs) == 1
    sub = subs[0]
    assert sub.period_kind == "monthly"
    assert sub.occurrences == 6
    assert sub.median_amount == 14900
    assert sub.next_expected == sub.last_at + timedelta(days=30)
    assert "다음 결제 예상" in sub.evidence


def test_two_payments_are_not_a_subscription():
    """간격이 하나뿐이면 '주기'라고 부를 근거가 없다."""
    assert detect_subscriptions(_monthly([14900] * (MIN_OCCURRENCES - 1))) == []


def test_wobbling_amounts_are_not_a_subscription():
    """오탐 함정 — 매달 같은 가맹점이지만 금액 변동이 크면 구독이 아니다."""
    subs = detect_subscriptions(_monthly([9000, 21000, 12000, 30000, 15000, 24000]))
    assert subs == []


def test_price_increase_is_flagged_but_still_a_subscription():
    subs = detect_subscriptions(_monthly([13500, 13500, 13500, 15900, 15900, 15900]))
    assert len(subs) == 1
    assert subs[0].price_increased is True
    assert subs[0].amount_cv < MAX_AMOUNT_CV


def test_price_drop_is_not_an_increase():
    """한 번이라도 내렸으면 요동이지 인상이 아니다."""
    subs = detect_subscriptions(_monthly([15900, 15900, 13500, 13500, 15900, 15900]))
    assert subs and subs[0].price_increased is False


def test_unused_subscription_is_flagged_against_period_end_not_today():
    rows = _monthly([14900] * 4)
    subs = detect_subscriptions(rows, period_end=rows[-1].at + timedelta(days=95))
    assert subs and subs[0].unused is True
    assert "결제 없음" in subs[0].evidence
    # 기준 시각을 주지 않으면 데이터의 마지막 거래가 기준이라 미사용이 아니다.
    assert detect_subscriptions(rows)[0].unused is False


def test_yearly_subscription_needs_three_occurrences():
    """6개월 창의 연 구독은 1회뿐이라 원리상 찾을 수 없다 — 규칙을 느슨하게 하지 않는다."""
    rows = [_row("Y0", when=START, amount=3300)]
    assert detect_subscriptions(rows) == []


def test_weekly_and_yearly_periods_convert_to_monthly_cost():
    weekly = [
        _row(f"W{i}", when=START + timedelta(days=7 * i), amount=5000, canonical="주간구독")
        for i in range(6)
    ]
    subs = detect_subscriptions(weekly)
    assert subs and subs[0].period_kind == "weekly"
    # 주 5,000원 → 월 환산은 약 21,400원. 주기가 달라도 같은 자로 더할 수 있어야 한다.
    assert 21_000 <= monthly_fixed_cost(subs)["C0001"] <= 21_800


def test_irregular_interval_is_not_a_subscription():
    """간격 중앙값이 주/월/연 창 밖이면 구독이 아니다 (격주 청구 등)."""
    rows = [
        _row(f"I{i}", when=START + timedelta(days=15 * i), amount=14900) for i in range(6)
    ]
    assert detect_subscriptions(rows) == []


# ── ② 이상지출 ─────────────────────────────────────────────────────────────
def _baseline_months(total: int, *, months: int = 5, category=CategoryL1.FOOD) -> list[SpendRow]:
    rows = []
    for m in range(months):
        for k in range(4):
            rows.append(
                _row(
                    f"B{m}-{k}",
                    when=datetime(2026, 3 + m, 5 + k, 12, 0),
                    amount=total // 4,
                    canonical=f"식당{k}",
                    category=category,
                )
            )
    return rows


def test_category_surge_needs_both_ratio_and_absolute_excess():
    rows = _baseline_months(100_000)
    # 평소 10만원인 카테고리가 15만원이 된 달: 배수는 1.5 지만 증가액이 문턱 미만이라 알리지 않는다.
    small = [
        _row("X", when=datetime(2026, 8, 10, 12, 0), amount=50_000, canonical="식당0",
             category=CategoryL1.FOOD)
    ]
    assert [a for a in detect_spending_anomalies(rows + small) if a.kind == "category_surge"] == []

    big = [
        _row("X", when=datetime(2026, 8, 10, 12, 0), amount=SURGE_MIN_EXCESS * 3,
             canonical="식당0", category=CategoryL1.FOOD)
    ]
    surges = [a for a in detect_spending_anomalies(rows + big) if a.kind == "category_surge"]
    assert len(surges) == 1
    assert surges[0].month == "2026-08"
    assert "평소 월" in surges[0].evidence


def test_unclassified_category_never_produces_a_spending_alert():
    """미분류는 여러 업종이 섞인 자루라 그 안의 변동은 소비 패턴이 아니다."""
    rows = _baseline_months(100_000, category=CategoryL1.UNCLASSIFIED)
    rows.append(
        _row("X", when=datetime(2026, 8, 10, 12, 0), amount=5_000_000, canonical="식당0",
             category=CategoryL1.UNCLASSIFIED)
    )
    assert detect_spending_anomalies(rows) == []


def test_single_large_uses_merchant_baseline_and_needs_a_ratio_floor():
    """금액이 촘촘한 가맹점은 MAD 가 작아 z 가 폭발한다 — 배수 하한이 그걸 막는다."""
    tight = [
        _row(f"T{i}", when=START + timedelta(days=i), amount=2000, canonical="지하철",
             category=CategoryL1.TRANSPORT)
        for i in range(30)
    ]
    doubled = tight + [
        _row("D", when=START + timedelta(days=40), amount=4000, canonical="지하철",
             category=CategoryL1.TRANSPORT)
    ]
    assert [a for a in detect_spending_anomalies(doubled) if a.kind == "single_large"] == []

    huge = tight + [
        _row("H", when=START + timedelta(days=40), amount=60_000, canonical="지하철",
             category=CategoryL1.TRANSPORT)
    ]
    alerts = [a for a in detect_spending_anomalies(huge) if a.kind == "single_large"]
    assert len(alerts) == 1
    assert alerts[0].approval_nos == ["H"]
    assert "이 가맹점의 통상" in alerts[0].evidence


def test_single_large_stays_quiet_when_the_merchant_has_too_few_transactions():
    """기준선을 못 세우는 가맹점에서는 알리지 않는다 — 모르는 것을 이상이라 하지 않는다."""
    few = [
        _row(f"F{i}", when=START + timedelta(days=i), amount=2000, canonical="희귀가맹점",
             category=CategoryL1.SHOPPING)
        for i in range(5)
    ] + [
        _row("F9", when=START + timedelta(days=9), amount=500_000, canonical="희귀가맹점",
             category=CategoryL1.SHOPPING)
    ]
    assert detect_spending_anomalies(few) == []


# ── ③ 이상거래패턴 ─────────────────────────────────────────────────────────
def test_duplicate_charge_needs_same_amount_within_the_window():
    when = datetime(2026, 5, 2, 13, 0)
    same = [
        _row("D1", when=when, amount=16_400, canonical="김밥천국", category=CategoryL1.FOOD),
        _row("D2", when=when + timedelta(minutes=4), amount=16_400, canonical="김밥천국",
             category=CategoryL1.FOOD),
    ]
    alerts = detect_tx_anomalies(same)
    assert [a.kind for a in alerts] == ["duplicate_charge"]
    assert alerts[0].approval_nos == ["D1", "D2"]


def test_consecutive_payments_with_different_amounts_are_not_duplicates():
    """오탐 함정 — 짧은 간격 2건이지만 금액이 다르면 중복결제가 아니다."""
    when = datetime(2026, 5, 2, 13, 0)
    rows = [
        _row("N1", when=when, amount=8_000, canonical="GS25", category=CategoryL1.FOOD),
        _row("N2", when=when + timedelta(minutes=5), amount=19_000, canonical="GS25",
             category=CategoryL1.FOOD),
    ]
    assert detect_tx_anomalies(rows) == []


def test_same_amount_far_apart_is_not_a_duplicate():
    when = datetime(2026, 5, 2, 13, 0)
    rows = [
        _row("F1", when=when, amount=16_400, canonical="김밥천국", category=CategoryL1.FOOD),
        _row("F2", when=when + timedelta(hours=3), amount=16_400, canonical="김밥천국",
             category=CategoryL1.FOOD),
    ]
    assert detect_tx_anomalies(rows) == []


def test_micro_burst_is_reported_once_not_as_many_duplicates():
    """소액 다발은 같은 금액이 반복되므로 손대지 않으면 중복결제로도 세어진다."""
    base = datetime(2026, 8, 17, 3, 0)
    rows = [
        _row(f"M{k}", when=base + timedelta(minutes=2 * k), amount=200, canonical="서울교통공사",
             category=CategoryL1.TRANSPORT)
        for k in range(5)
    ]
    alerts = detect_tx_anomalies(rows)
    assert [a.kind for a in alerts] == ["micro_burst"]
    assert alerts[0].amount == 1000


# ── 입구 계약 ──────────────────────────────────────────────────────────────
def _classified(no: str, name: str, when: datetime, amount: int, tx_type=TxType.APPROVE):
    raw = RawTx(
        provider_id="CARD_A",
        customer_id="C0001",
        approval_no=no,
        tx_datetime=when,
        merchant_name_raw=name,
        amount=amount,
        tx_type=tx_type,
    )
    return raw


def test_cancelled_transactions_never_reach_the_detectors():
    """전액취소된 거래를 탐지가 다시 세면 '쓰지 않은 돈'이 급증으로 뜬다."""
    when = datetime(2026, 6, 1, 12, 0)
    raws = [
        _classified("A1", "스타벅스", when, 500_000),
        _classified("A1", "스타벅스", when + timedelta(days=1), -500_000, TxType.CANCEL),
        _classified("A2", "스타벅스", when, 5_000),
    ]
    classified = [
        ClassifiedTx(
            normalized=n,
            merchant_canonical="스타벅스",
            category_l1=CategoryL1.CAFE,
            category_l2=CategoryL2.CAFE_COFFEE,
            confidence=1.0,
            match_stage=MatchStage.EXACT,
        )
        for n in preprocess(raws)
    ]
    rows = spend_rows(classified)
    assert [r.approval_no for r in rows] == ["A2"]


def test_detect_runs_all_three_and_survives_empty_input():
    empty = detect([])
    assert empty.subscriptions == [] and empty.spending_alerts == [] and empty.tx_alerts == []
    assert empty.total_alerts == 0
    assert detect_from_rows([]).summary()


# ── 통계 도구 ──────────────────────────────────────────────────────────────
def test_robust_z_falls_back_to_a_ratio_when_mad_is_zero():
    """MAD 가 0 일 때 0 을 주면 '정상'이라 단정하고, 무한대를 주면 전부 알림이 된다."""
    assert robust_z(100.0, [100.0] * 6) == 0.0
    assert robust_z(400.0, [100.0] * 6) == pytest.approx(9.0)
    assert robust_z(50.0, [100.0]) == 0.0


def test_robust_z_is_not_dragged_by_the_outlier_it_is_looking_for():
    sample = [100.0, 105.0, 95.0, 100.0, 1000.0]
    assert robust_z(1000.0, sample) > 100


# ── 통합: 실제 합성 데이터 위의 §5.2 전이 ──────────────────────────────────
pytestmark_data = pytest.mark.skipif(
    not DEFAULT_BUNDLE_DIR.exists() or not DEFAULT_TRUTH_PATH.exists(),
    reason="합성 데이터가 없다 — `python -m txlens.generator --seed 42` (docs/recovery.md)",
)


@pytest.fixture(scope="module")
def transfer_rows():
    return run_transfer()


@pytestmark_data
def test_분류가_나빠지면_탐지도_나빠진다(transfer_rows):
    """§5.2 의 부등호. 이게 뒤집히면 1부와 2부를 묶는 논거가 사라진다 (D-010 · D-035).

    비교는 **재현율**로 한다. 정밀도는 정답셋이 우연히 생긴 진짜 급증에 라벨을 붙이지 않아
    하한으로만 읽히지만, 재현율의 분모는 심은 것의 수라 흔들리지 않는다.
    """
    by_label = {r.label: r.detection for r in transfer_rows}
    normalize_only = by_label["정규화만"]
    rules_only = by_label["A · 룰만"]
    oracle = by_label["정답 분류(상한)"]

    assert normalize_only.subscription.recall < rules_only.subscription.recall
    assert rules_only.subscription.recall < oracle.subscription.recall
    # 이상지출은 카테고리 자체가 없으면 성립하지 않는다 — 사전을 빼면 0 이 되어야 맞다.
    assert normalize_only.spending.detectable > 0
    assert normalize_only.spending.recall == 0.0
    assert rules_only.spending.recall < oracle.spending.recall


@pytestmark_data
def test_탐지기는_모든_줄에서_같다(transfer_rows):
    """분모(심은 정답의 수)가 줄마다 다르면 '분류만 바꿨다'는 전제가 이미 깨진 것이다."""
    denominators = {
        (r.detection.subscription.detectable, r.detection.tx_pattern.detectable)
        for r in transfer_rows
    }
    assert len(denominators) == 1


@pytestmark_data
def test_이상거래_탐지는_분류에_가장_덜_의존한다(transfer_rows):
    """금액·시각만으로 서는 규칙이라 사전을 빼도 대부분 남는다 — 세 탐지기의 낙폭이 다르다."""
    by_label = {r.label: r.detection for r in transfer_rows}
    drop_tx = 1.0 - by_label["정규화만"].tx_pattern.recall / by_label["정답 분류(상한)"].tx_pattern.recall
    drop_sub = 1.0 - by_label["정규화만"].subscription.recall / by_label["정답 분류(상한)"].subscription.recall
    assert drop_tx < drop_sub


@pytestmark_data
def test_고객당_월_알림이_상한을_넘지_않는다(transfer_rows):
    """알림은 많을수록 좋은 게 아니다 — 읽히지 않으면 없는 것과 같다 (blueprint §3.6)."""
    for row in transfer_rows:
        assert row.detection.alerts_per_customer_month <= 3.0, row.label
