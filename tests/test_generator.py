"""생성기 계약 테스트.

여기서 막고 싶은 것은 크래시가 아니라 **데이터가 조용히 비현실적으로 만들어지는 것**이다.
그런 데이터로 낸 지표는 통과처럼 보이면서 아무 의미가 없다.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime

import pytest

from txlens.generator.build import build
from txlens.generator.emit import to_approved_item, to_bundles
from txlens.generator.profiles import load_providers
from txlens.schemas import CardType, CategoryL1, CategoryL2, TxType

END = datetime(2026, 8, 31)


@pytest.fixture(scope="module")
def dataset():
    """작게 뽑아 재사용 — 전 테스트가 같은 데이터를 본다."""
    return build(seed=42, customers=8, months=4, period_end=END)


# ── 재현성 ──────────────────────────────────────────────────────────────────
def test_same_seed_is_identical(dataset):
    txs, gt = dataset
    txs2, gt2 = build(seed=42, customers=8, months=4, period_end=END)
    assert [t.model_dump() for t in txs] == [t.model_dump() for t in txs2]
    assert gt.model_dump() == gt2.model_dump()


def test_different_seed_differs(dataset):
    txs, _ = dataset
    txs2, _ = build(seed=43, customers=8, months=4, period_end=END)
    assert [t.merchant_name_raw for t in txs] != [t.merchant_name_raw for t in txs2]


def test_period_end_is_explicit_not_now(dataset):
    """실행 시각에 의존하면 같은 시드라도 데이터가 달라진다."""
    _, gt = dataset
    assert gt.period_end == "2026-08-31"
    assert gt.period_start == "2026-05-01"


# ── 롱테일 (이 프로젝트가 풀려는 문제가 실제로 존재하는가) ──────────────────
def test_distinct_names_far_exceed_brands(dataset):
    """오염이 실제로 일어났는가 — 표기 수가 브랜드 수보다 훨씬 많아야 한다."""
    from txlens.generator.brands import ALL_BRANDS

    _, gt = dataset
    assert gt.distinct_raw_names > len(ALL_BRANDS) * 2, (
        f"원본 표기 {gt.distinct_raw_names}개 / 브랜드 {len(ALL_BRANDS)}개 — 오염이 부족하다"
    )


def test_name_distribution_is_long_tailed(dataset):
    """상위 소수 표기가 큰 비중을 차지하고 꼬리가 길어야 한다."""
    txs, _ = dataset
    counts = Counter(t.merchant_name_raw for t in txs).most_common()
    top10_share = sum(c for _, c in counts[:10]) / len(txs)
    singletons = sum(1 for _, c in counts if c == 1) / len(counts)
    assert top10_share > 0.10, f"상위 10개 표기 비중 {top10_share:.2f} — head 가 없다"
    assert singletons > 0.30, f"1회성 표기 비율 {singletons:.2f} — tail 이 없다"


# ── 제공기관 프로파일 준수 ──────────────────────────────────────────────────
def test_regno_fill_rate_follows_profile(dataset):
    txs, _ = dataset
    pset = load_providers()
    by = defaultdict(lambda: [0, 0])
    for t in txs:
        by[t.provider_id][0] += 1
        if t.merchant_regno is not None:
            by[t.provider_id][1] += 1
    for pid, (n, filled) in by.items():
        want = pset.by_id(pid).regno_fill_rate
        got = filled / n
        assert abs(got - want) < 0.12, f"{pid} regno 채움 {got:.2f} vs 설정 {want:.2f} (n={n})"


def test_providers_differ_widely(dataset):
    """편차를 모델링하는 게 목적이므로, 기관별 결과가 비슷하면 의미가 없다."""
    txs, _ = dataset
    by = defaultdict(lambda: [0, 0])
    for t in txs:
        by[t.provider_id][0] += 1
        if t.merchant_regno is not None:
            by[t.provider_id][1] += 1
    rates = [f / n for n, f in by.values()]
    assert max(rates) - min(rates) > 0.5, f"기관 간 regno 채움 편차 {max(rates)-min(rates):.2f}"


def test_customer_holds_few_cards(dataset):
    """한 사람이 모든 기관을 쓰지는 않는다."""
    txs, _ = dataset
    per = defaultdict(set)
    for t in txs:
        per[t.customer_id].add(t.provider_id)
    assert max(len(v) for v in per.values()) <= 3


def test_subscription_billed_on_single_card(dataset):
    """구독은 등록된 카드 한 장으로 계속 결제된다."""
    txs, gt = dataset
    prov = {t.approval_no: t.provider_id for t in txs}
    for s in gt.subscriptions:
        provs = {prov[a] for a in s.approval_nos if a in prov}
        assert len(provs) <= 1, f"{s.canonical} 이 {provs} 여러 기관에 흩어졌다"


# ── PG 마스킹은 미분류가 정답 ───────────────────────────────────────────────
def test_pg_masked_label_is_unclassified(dataset):
    """표기에서 브랜드를 복원할 수 없으면 정답도 미분류다 — 억지 추측을 정답으로 만들지 않는다."""
    _, gt = dataset
    pg = [lb for lb in gt.tx_labels.values() if lb.pg_masked]
    assert pg, "PG 마스킹 케이스가 하나도 없다 — 오염기가 동작하지 않았다"
    for lb in pg:
        assert lb.category_l1 is CategoryL1.UNCLASSIFIED
        assert lb.category_l2 is CategoryL2.UNCLASSIFIED_PG_MASKED
        assert lb.canonical == "", "PG 건에 원 브랜드를 적으면 정답이 추측을 정답화한다"


# ── 정답셋 정합성 ───────────────────────────────────────────────────────────
def test_every_planted_pattern_links_to_real_transactions(dataset):
    txs, gt = dataset
    known = {t.approval_no for t in txs}
    groups = (
        [(s.canonical, s.approval_nos) for s in gt.subscriptions],
        [(a.kind, a.approval_nos) for a in gt.spending_anomalies],
        [(a.kind, a.approval_nos) for a in gt.tx_anomalies],
        [(d.kind, d.approval_nos) for d in gt.decoys],
    )
    for group in groups:
        for name, aps in group:
            assert aps, f"{name} 에 승인번호가 연결되지 않았다"
            assert set(aps) <= known, f"{name} 이 존재하지 않는 거래를 가리킨다"


def test_labels_cover_all_approvals(dataset):
    txs, gt = dataset
    approved = {t.approval_no for t in txs if t.tx_type is TxType.APPROVE}
    assert approved == set(gt.tx_labels), "승인 거래와 라벨 집합이 다르다"


# ── 심은 패턴이 탐지 규칙을 실제로 만족하는가 ───────────────────────────────
def test_detectable_subscriptions_meet_the_rule(dataset):
    """`detectable=True` 는 "최소 3회 + 금액 CV<0.15" 를 실제로 만족해야 한다."""
    txs, gt = dataset
    amt = {t.approval_no: abs(t.amount) for t in txs}
    for s in gt.subscriptions:
        if not s.detectable:
            continue
        amounts = [amt[a] for a in s.approval_nos if a in amt]
        assert len(amounts) >= 3, f"{s.canonical} {len(amounts)}회 — 3회 미만인데 detectable"
        if not s.price_increased and len(amounts) > 1:
            cv = statistics.pstdev(amounts) / statistics.mean(amounts)
            assert cv < 0.15, f"{s.canonical} CV={cv:.3f} — 구독인데 금액 변동이 크다"


def test_undetectable_are_reported_not_hidden(dataset):
    """연 구독은 6개월 창에서 원리상 못 찾는다. 숨기지 않고 detectable=False 로 남긴다."""
    _, gt = dataset
    und = [s for s in gt.subscriptions if not s.detectable]
    assert und, "탐지 불가 케이스가 없다 — 연 구독이 심기지 않았다"
    for s in und:
        assert s.occurrences < 3


# ── 오탐 함정이 진짜 함정인가 (precision 을 재려면 필수) ────────────────────
def test_subscription_like_decoy_has_high_cv(dataset):
    """구독처럼 보이지만 금액 변동이 커서 CV 조건에 걸러져야 한다."""
    txs, gt = dataset
    amt = {t.approval_no: abs(t.amount) for t in txs}
    checked = 0
    for d in gt.decoys:
        if d.kind != "subscription_like":
            continue
        amounts = [amt[a] for a in d.approval_nos if a in amt]
        if len(amounts) < 3:
            continue
        cv = statistics.pstdev(amounts) / statistics.mean(amounts)
        assert cv > 0.15, f"{d.canonical} CV={cv:.3f} — 함정이 아니라 진짜 구독처럼 보인다"
        checked += 1
    assert checked, "검사할 subscription_like 함정이 없다"


def test_normal_consecutive_decoy_has_differing_amounts(dataset):
    """짧은 간격 2건이지만 금액이 달라 중복결제가 아니어야 한다."""
    txs, gt = dataset
    amt = {t.approval_no: abs(t.amount) for t in txs}
    checked = 0
    for d in gt.decoys:
        if d.kind != "normal_consecutive":
            continue
        amounts = [amt[a] for a in d.approval_nos if a in amt]
        assert len(set(amounts)) == len(amounts), f"{d.canonical} 금액이 같아 진짜 중복결제가 됐다"
        checked += 1
    assert checked


def test_duplicate_charge_has_identical_amounts(dataset):
    """반대로 중복결제 정답은 금액이 같아야 한다."""
    txs, gt = dataset
    amt = {t.approval_no: abs(t.amount) for t in txs}
    for a in gt.tx_anomalies:
        if a.kind != "duplicate_charge":
            continue
        amounts = [amt[x] for x in a.approval_nos if x in amt]
        assert len(set(amounts)) == 1, f"{a.canonical} 중복결제인데 금액이 다르다"


# ── 취소 페어 ───────────────────────────────────────────────────────────────
def test_cancellations_pair_by_approval_no(dataset):
    txs, _ = dataset
    cancels = [t for t in txs if t.tx_type is TxType.CANCEL]
    assert cancels, "취소 거래가 없다"
    approvals = defaultdict(list)
    for t in txs:
        approvals[t.approval_no].append(t)
    for c in cancels:
        pair = approvals[c.approval_no]
        assert len(pair) == 2
        assert sum(t.amount for t in pair) == 0, "취소 페어의 합이 0이 아니다"
        assert c.amount < 0


# ── 금액 현실성 ─────────────────────────────────────────────────────────────
def test_amounts_are_plausible_per_category(dataset):
    """9.5억원 편의점 결제 같은 값이 나오면 구독·이상지출 탐지가 성립하지 않는다."""
    txs, gt = dataset
    amt = {t.approval_no: abs(t.amount) for t in txs}
    by = defaultdict(list)
    for ap, lb in gt.tx_labels.items():
        if not lb.pg_masked and ap in amt:
            by[lb.category_l2].append(amt[ap])

    bounds = {
        CategoryL2.FOOD_CONVENIENCE: (1_000, 60_000),
        CategoryL2.CAFE_COFFEE: (3_000, 40_000),
        CategoryL2.TRANSPORT_TRANSIT: (1_000, 10_000),
    }
    for cat, (lo, hi) in bounds.items():
        vals = by.get(cat)
        assert vals, f"{cat} 거래가 없다"
        med = statistics.median(vals)
        assert lo <= med <= hi, f"{cat} 중앙값 {med:,}원 이 현실 범위를 벗어난다"


def test_amounts_are_positive_and_rounded(dataset):
    """변동 금액은 100원 단위로 떨어진다.

    **정액 구독료는 예외다** — 쿠팡와우 7,890원처럼 100원 단위가 아닌 실제 가격이 있고,
    그걸 반올림하면 오히려 비현실적이 된다. 카탈로그에서 `lo == hi` 인 브랜드가 정액이다.
    """
    from txlens.generator.brands import ALL_BRANDS

    fixed_price = {b.brand for b in ALL_BRANDS if b.lo == b.hi}
    txs, gt = dataset
    for t in txs:
        assert abs(t.amount) >= 100
        label = gt.tx_labels.get(t.approval_no)
        if label and label.canonical in fixed_price:
            continue
        assert abs(t.amount) % 100 == 0, (
            f"{t.amount}원 ({t.merchant_name_raw}) 이 100원 단위가 아니다"
        )


def test_fixed_price_subscriptions_keep_exact_amounts(dataset):
    """정액 구독은 매회 정확히 같은 금액이어야 한다 — CV 조건의 전제다."""
    from txlens.generator.brands import ALL_BRANDS

    fixed = {b.brand: b.lo for b in ALL_BRANDS if b.lo == b.hi}
    txs, gt = dataset
    amt = {t.approval_no: abs(t.amount) for t in txs}
    for s in gt.subscriptions:
        if s.canonical not in fixed or s.price_increased:
            continue
        seen = {amt[a] for a in s.approval_nos if a in amt}
        assert seen == {fixed[s.canonical]}, f"{s.canonical} 정액인데 금액이 {seen}"


# ── Layer 1: 표준API 봉투 ───────────────────────────────────────────────────
def test_envelope_uses_spec_field_names_and_string_amounts(dataset):
    txs, _ = dataset
    item = to_approved_item(txs[0])
    assert set(item) >= {
        "approved_num", "approved_dtime", "approved_amt", "status",
        "pay_type", "merchant_name",
    }
    assert isinstance(item["approved_amt"], str), "규격은 금액을 문자열로 준다"
    assert len(item["approved_dtime"]) == 14, "YYYYMMDDHHMMSS 형식이어야 한다"


def test_envelope_has_no_fields_outside_the_spec(dataset):
    """규격에 없는 필드를 지어내면 "실제 규격대로 만들었다"는 주장이 무너진다.

    `merchant_category` 가 그런 필드였다 — 어느 업권 규격에도 없는데 봉투에 있었다.
    """
    txs, _ = dataset
    allowed = {
        "approved_num", "approved_dtime", "approved_amt", "status",
        "pay_type", "total_install_cnt", "merchant_name", "merchant_regno",
    }
    for tx in txs:
        extra = set(to_approved_item(tx)) - allowed
        assert not extra, f"규격에 없는 필드: {extra}"


def test_installment_count_is_omitted_for_lump_sum(dataset):
    """규격: `total_install_cnt` 는 **일시불일 경우 미회신**.

    `"0"` 을 채워 보내면 수신 측이 "할부 정보가 왔다"고 착각한다 — 조건부 미회신의 요점.
    """
    txs, _ = dataset
    lump = [t for t in txs if not t.installment_months]
    install = [t for t in txs if t.installment_months]
    assert lump and install, "일시불·할부 양쪽 표본이 없다"
    assert "total_install_cnt" not in to_approved_item(lump[0])
    assert to_approved_item(install[0])["total_install_cnt"] == str(install[0].installment_months)


def test_pay_type_is_credit_or_check_only(dataset):
    """`card_pay_type` = {01 신용, 02 체크}. 일시불/할부 구분이 아니다."""
    txs, _ = dataset
    seen = {to_approved_item(t)["pay_type"] for t in txs}
    assert seen == {"01", "02"}, f"신용·체크 양쪽이 나와야 한다: {seen}"

    # 체크카드는 할부가 없다 — 카드 단위 속성이므로 같은 카드 안에서 흔들려서도 안 된다.
    per_card: dict[tuple[str, str], set[str]] = {}
    for t in txs:
        key = (t.customer_id, t.provider_id)
        per_card.setdefault(key, set()).add(to_approved_item(t)["pay_type"])
        if t.card_type is CardType.CHECK:
            assert not t.installment_months, "체크카드에 할부가 붙었다"
    for key, types in per_card.items():
        assert len(types) == 1, f"{key} 의 pay_type 이 거래마다 다르다: {types}"


def test_missing_regno_omits_the_key(dataset):
    """None 을 채워 보내면 전처리가 '값이 있다'고 착각한다 — 조건부 미회신을 재현한다."""
    txs, _ = dataset
    without = [t for t in txs if t.merchant_regno is None]
    assert without, "regno 누락 케이스가 없다"
    assert "merchant_regno" not in to_approved_item(without[0])
    with_ = [t for t in txs if t.merchant_regno is not None]
    assert "merchant_regno" in to_approved_item(with_[0])


def test_bundles_are_grouped_and_serializable(dataset):
    txs, _ = dataset
    bundles = to_bundles(txs)
    assert bundles
    for (provider, customer), payload in bundles.items():
        assert payload["rsp_code"] == "00000"
        assert int(payload["approved_cnt"]) == len(payload["approved_list"])
        json.dumps(payload, ensure_ascii=False)  # 직렬화 가능해야 한다
        dtimes = [i["approved_dtime"] for i in payload["approved_list"]]
        assert dtimes == sorted(dtimes), f"{provider}/{customer} 이 시간순이 아니다"
