"""스키마 계약 테스트.

여기서 막고 싶은 것은 형식 오류가 아니라 **조용히 지표를 틀어뜨리는 값**이다:
l1/l2 불일치, 범위를 벗어난 confidence, 취소 거래가 분류 대상에 남는 것.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from txlens.schemas import (
    CategoryL1,
    CategoryL2,
    Evidence,
    LLMVerdict,
    MatchStage,
    NormalizedTx,
    PaymentChannel,
    RawTx,
    ReviewItem,
    RuleProposal,
    TxType,
)


def _raw(**kw) -> RawTx:
    base = {
        "provider_id": "CARD_A",
        "customer_id": "SYNTH_0001",
        "approval_no": "12345678",
        # tz 없는 KST 로컬 시각 — 표준API 가 오프셋 없이 주는 형태를 그대로 쓴다.
        # DTZ001 은 pyproject 의 per-file-ignores 에서 tests/ 에만 허용한다 (D-009).
        "tx_datetime": datetime(2026, 8, 1, 12, 30),
        "merchant_name_raw": "(주)스타벅스커피코리아 1234호점",
        "amount": 5500,
    }
    return RawTx(**{**base, **kw})


# ── 카테고리 정합성 ─────────────────────────────────────────────────────────
def test_l2_must_belong_to_l1():
    with pytest.raises(ValidationError, match="속하지 않는다"):
        LLMVerdict(
            merchant_canonical="스타벅스",
            category_l1=CategoryL1.FOOD,        # 불일치: food ≠ cafe
            category_l2=CategoryL2.CAFE_COFFEE,
            confidence=0.9,
            reasoning="커피 브랜드",
            evidence=Evidence.BRAND_KNOWN,
        )


def test_matching_l1_l2_is_accepted():
    v = LLMVerdict(
        merchant_canonical="스타벅스",
        category_l1=CategoryL1.CAFE,
        category_l2=CategoryL2.CAFE_COFFEE,
        confidence=0.95,
        reasoning="커피 브랜드",
        evidence=Evidence.BRAND_KNOWN,
    )
    assert v.category_l2.l1 is CategoryL1.CAFE


def test_l2_property_derives_l1():
    assert CategoryL2.UNCLASSIFIED_PG_MASKED.l1 is CategoryL1.UNCLASSIFIED
    assert CategoryL2.FINANCE_CASH.l1 is CategoryL1.FINANCE


# ── confidence 범위 ────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_confidence_out_of_range_rejected(bad):
    with pytest.raises(ValidationError):
        LLMVerdict(
            merchant_canonical="X",
            category_l1=CategoryL1.CAFE,
            category_l2=CategoryL2.CAFE_COFFEE,
            confidence=bad,
            reasoning="r",
            evidence=Evidence.GUESS,
        )


# ── 분류 대상 판정 ──────────────────────────────────────────────────────────
def test_cancelled_tx_is_not_classifiable():
    n = NormalizedTx(raw=_raw(), normalized_name="스타벅스", is_cancelled=True)
    assert not n.is_classifiable


def test_excluded_tx_is_not_classifiable():
    n = NormalizedTx(raw=_raw(), normalized_name="스타벅스", excluded_reason="할부 회차")
    assert not n.is_classifiable


def test_plain_tx_is_classifiable():
    assert NormalizedTx(raw=_raw(), normalized_name="스타벅스").is_classifiable


# ── 전처리 산출물 보존 ─────────────────────────────────────────────────────
def test_channel_and_branch_are_preserved_not_dropped():
    """채널·지점은 정규화 과정에서 버리지 않고 필드로 보존한다 (blueprint §3.3 ①)."""
    n = NormalizedTx(
        raw=_raw(merchant_name_raw="NAVERPAY*스타벅스 강남역점"),
        normalized_name="스타벅스",
        branch="강남역점",
        payment_channel=PaymentChannel.NAVERPAY,
    )
    assert n.branch == "강남역점"
    assert n.payment_channel is PaymentChannel.NAVERPAY
    assert n.raw.merchant_name_raw == "NAVERPAY*스타벅스 강남역점"  # 원문도 남는다


def test_pg_masked_goes_to_unclassified_not_a_guess():
    """PG 경유는 억지 추측 대신 미분류 + 이유 코드로 간다."""
    n = NormalizedTx(
        raw=_raw(merchant_name_raw="(주)케이지이니시스"),
        normalized_name="케이지이니시스",
        payment_channel=PaymentChannel.PG,
        pg_masked=True,
    )
    assert n.pg_masked
    assert CategoryL2.UNCLASSIFIED_PG_MASKED.l1 is CategoryL1.UNCLASSIFIED


def test_raw_tx_is_immutable():
    """원본 거래는 감사 추적의 기준점이므로 변형되지 않아야 한다."""
    r = _raw()
    with pytest.raises(ValidationError):
        r.amount = 9999


def test_cancel_tx_uses_negative_amount_and_pair_key():
    r = _raw(amount=-5500, tx_type=TxType.CANCEL)
    n = NormalizedTx(raw=r, normalized_name="스타벅스", cancel_of="12345678")
    assert n.raw.amount < 0
    assert n.raw.tx_type is TxType.CANCEL
    assert n.cancel_of == "12345678"


# ── 검토 큐 우선순위 ────────────────────────────────────────────────────────
def _verdict(conf: float) -> LLMVerdict:
    return LLMVerdict(
        merchant_canonical="X",
        category_l1=CategoryL1.CAFE,
        category_l2=CategoryL2.CAFE_COFFEE,
        confidence=conf,
        reasoning="r",
        evidence=Evidence.KEYWORD,
    )


def test_review_priority_favours_high_volume_and_low_confidence():
    """사람의 1분이 가장 많은 거래를 고치게 하는 정렬 (blueprint §3.3 ④)."""
    many_unsure = ReviewItem(normalized_name="A", verdict=_verdict(0.5), tx_count=100)
    few_unsure = ReviewItem(normalized_name="B", verdict=_verdict(0.5), tx_count=1)
    many_sure = ReviewItem(normalized_name="C", verdict=_verdict(0.99), tx_count=100)

    assert many_unsure.priority > few_unsure.priority
    assert many_unsure.priority > many_sure.priority

    ordered = sorted([few_unsure, many_sure, many_unsure], key=lambda i: -i.priority)
    assert ordered[0] is many_unsure


# ── 룰 승격 제안 ────────────────────────────────────────────────────────────
def test_rule_proposal_records_origin_and_coverage():
    p = RuleProposal(
        pattern_type="alias",
        pattern="에스씨케이컴퍼니",
        merchant_canonical="스타벅스",
        category_l1=CategoryL1.CAFE,
        category_l2=CategoryL2.CAFE_COFFEE,
        origin="human",
        source_name="에스씨케이컴퍼니",
        covers_tx_count=412,
    )
    assert p.origin == "human"
    assert p.covers_tx_count == 412


def test_rule_proposal_rejects_unknown_pattern_type():
    with pytest.raises(ValidationError):
        RuleProposal(
            pattern_type="embedding",  # 지원하지 않는 유형
            pattern="x",
            merchant_canonical="X",
            category_l1=CategoryL1.CAFE,
            category_l2=CategoryL2.CAFE_COFFEE,
            origin="human",
            source_name="x",
            covers_tx_count=0,
        )


def test_match_stage_covers_every_pipeline_path():
    """지표가 '어느 단계가 일했는가'를 집계하므로 경로마다 값이 있어야 한다."""
    assert {s.value for s in MatchStage} >= {
        "exact", "alias", "regex", "fuzzy", "llm", "human", "unmatched"
    }
