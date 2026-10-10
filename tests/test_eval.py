"""평가 지표 계약 테스트.

지표는 틀려도 **에러를 내지 않는다** — 그냥 더 좋은 숫자를 낸다. 그래서 여기서 막고 싶은 것은
크래시가 아니라 다음 세 가지다.

1. **모수가 조용히 줄어드는 것.** 분모에서 무엇이 빠졌는지 결과에 남지 않으면 정확도는 올라간다.
2. **미분류 두 종류가 합산되는 것.** `미매칭`(줄일 수 있다) + `PG마스킹`(줄지 않는다) 을 합치면
   개선 여지를 실제보다 크게 보고한다.
3. **롱테일이 건수 가중에 덮이는 것.** 정규화 키 기준이 거래 기준과 같은 계산이 되면 이 프로젝트의
   논거 자체가 사라진다.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from txlens.eval import evaluate
from txlens.eval.harness import DEFAULT_BUNDLE_DIR, DEFAULT_TRUTH_PATH, run_rules_baseline
from txlens.ground_truth import GroundTruth, TxLabel
from txlens.preprocess import preprocess_one
from txlens.rules import RuleMatcher
from txlens.schemas import (
    CategoryL1,
    CategoryL2,
    ClassifiedTx,
    MatchStage,
    NormalizedTx,
    RawTx,
)

WHEN = datetime(2026, 8, 1, 12, 0)


def _raw(approval_no: str, name: str, *, provider_id: str = "CARD_A", **kw) -> RawTx:
    return RawTx(
        provider_id=provider_id,
        customer_id="C0001",
        approval_no=approval_no,
        tx_datetime=WHEN,
        merchant_name_raw=name,
        amount=10_000,
        **kw,
    )


def _classified(
    approval_no: str,
    name: str,
    *,
    stage: MatchStage,
    l1: CategoryL1,
    l2: CategoryL2,
    provider_id: str = "CARD_A",
) -> ClassifiedTx:
    normalized = preprocess_one(_raw(approval_no, name, provider_id=provider_id))
    return ClassifiedTx(
        normalized=normalized,
        merchant_canonical=name,
        confidence=1.0,
        match_stage=stage,
        category_l1=l1,
        category_l2=l2,
    )


def _hit(approval_no: str, name: str, **kw) -> ClassifiedTx:
    """맞게 분류된 카페 거래."""
    return _classified(
        approval_no, name, l1=CategoryL1.CAFE, l2=CategoryL2.CAFE_COFFEE, **kw
    )


def _miss(approval_no: str, name: str, *, stage: MatchStage = MatchStage.UNMATCHED) -> ClassifiedTx:
    """미분류로 남은 거래."""
    l2 = (
        CategoryL2.UNCLASSIFIED_PG_MASKED
        if stage is MatchStage.PG_MASKED
        else CategoryL2.UNCLASSIFIED_INSUFFICIENT
    )
    return _classified(approval_no, name, stage=stage, l1=CategoryL1.UNCLASSIFIED, l2=l2)


def _truth(*labels: tuple[str, str, CategoryL1, CategoryL2]) -> GroundTruth:
    return GroundTruth(
        seed=42,
        customers=1,
        period_start="2026-08-01",
        period_end="2026-08-31",
        tx_total=len(labels),
        distinct_raw_names=len({name for _, name, _, _ in labels}),
        tx_labels={
            approval_no: TxLabel(
                canonical=canonical, category_l1=l1, category_l2=l2, provider_id="CARD_A"
            )
            for approval_no, canonical, l1, l2 in labels
        },
    )


CAFE = (CategoryL1.CAFE, CategoryL2.CAFE_COFFEE)


# ── ① 모수 ──────────────────────────────────────────────────────────────────
def test_제외된_거래는_정확도_분모에_들어가지_않는다():
    """취소로 상계된 거래를 분모에 넣으면 같은 지출이 두 번 세어진다."""
    cancelled = NormalizedTx(
        raw=_raw("A2", "스타벅스"),
        normalized_name="스타벅스",
        is_cancelled=True,
        excluded_reason="취소_상계",
    )
    result = evaluate(
        [_hit("A1", "스타벅스", stage=MatchStage.EXACT)],
        _truth(("A1", "스타벅스", *CAFE), ("A2", "스타벅스", *CAFE)),
        label="테스트",
        all_normalized=[preprocess_one(_raw("A1", "스타벅스")), cancelled],
    )
    assert result.tx_total == 2
    assert result.tx_classified == 1
    assert result.tx_excluded == 1
    assert result.overall.labeled == 1
    assert result.overall.accuracy_l2 == 1.0


def test_제외_사유가_결과에_남는다():
    """"제외했다"만 있고 이유가 없으면 분모가 줄어든 것을 검증할 수 없다."""
    excluded = NormalizedTx(
        raw=_raw("A2", "스타벅스"), normalized_name="스타벅스", excluded_reason="할부_회차"
    )
    result = evaluate(
        [_hit("A1", "스타벅스", stage=MatchStage.EXACT)],
        _truth(("A1", "스타벅스", *CAFE)),
        label="테스트",
        all_normalized=[preprocess_one(_raw("A1", "스타벅스")), excluded],
    )
    assert result.excluded_reasons["할부_회차"] == 1
    assert "할부_회차" in result.report()


def test_정답이_없는_거래는_분모에서_빠지고_따로_보고된다():
    """정답 없는 거래를 오답으로 세면 지표가 데이터 결함을 분류기 탓으로 돌린다."""
    result = evaluate(
        [_hit("A1", "스타벅스", stage=MatchStage.EXACT), _hit("A9", "빈스로드", stage=MatchStage.EXACT)],
        _truth(("A1", "스타벅스", *CAFE)),
        label="테스트",
    )
    assert (result.overall.labeled, result.overall.unlabeled) == (1, 1)
    assert result.overall.accuracy_l2 == 1.0
    assert "정답 라벨이 없는 거래 1건" in result.report()


def test_모수_설명이_리포트에_반드시_찍힌다():
    """발표 슬라이드의 숫자를 남이 검산할 수 있어야 한다 — 전체·대상·정답 세 수를 모두 낸다."""
    result = evaluate(
        [_hit("A1", "스타벅스", stage=MatchStage.EXACT)],
        _truth(("A1", "스타벅스", *CAFE)),
        label="테스트",
    )
    report = result.report()
    assert "전체 1건" in report and "분류 대상 1건" in report and "정답 있음 1건" in report


# ── ② 미분류 두 종류 ────────────────────────────────────────────────────────
def test_PG마스킹과_미매칭은_합산되지_않는다():
    """둘을 합치면 사전을 키워 줄일 수 있는 몫이 실제보다 커 보인다."""
    result = evaluate(
        [
            _hit("A1", "스타벅스", stage=MatchStage.EXACT),
            _miss("A2", "을지로호프"),
            _miss("A3", "네이버페이", stage=MatchStage.PG_MASKED),
            _miss("A4", "카카오페이", stage=MatchStage.PG_MASKED),
        ],
        _truth(
            ("A1", "스타벅스", *CAFE),
            ("A2", "을지로호프", CategoryL1.CAFE, CategoryL2.CAFE_ALCOHOL),
            ("A3", "미분류(PG경유)", CategoryL1.UNCLASSIFIED, CategoryL2.UNCLASSIFIED_PG_MASKED),
            ("A4", "미분류(PG경유)", CategoryL1.UNCLASSIFIED, CategoryL2.UNCLASSIFIED_PG_MASKED),
        ),
        label="테스트",
    )
    assert result.overall.unmatched_rate == 0.25
    assert result.overall.pg_masked_rate == 0.5
    assert result.overall.coverage == 0.25


def test_PG마스킹은_미분류가_정답이라_정확도를_깎지_않는다():
    """실가맹점명이 오지 않은 거래에서 미분류는 **맞는 답**이다."""
    result = evaluate(
        [_miss("A3", "네이버페이", stage=MatchStage.PG_MASKED)],
        _truth(("A3", "미분류(PG경유)", CategoryL1.UNCLASSIFIED, CategoryL2.UNCLASSIFIED_PG_MASKED)),
        label="테스트",
    )
    assert result.overall.accuracy_l2 == 1.0
    assert result.overall.coverage == 0.0  # 맞았지만 브랜드를 확정한 것은 아니다


def test_커버리지는_정확도와_다르다():
    """확정했지만 틀린 것도 커버리지에는 들어간다 — 둘을 같은 수로 보고하면 오분류가 숨는다."""
    wrong = _classified(
        "A1", "김밥천국", stage=MatchStage.FUZZY, l1=CategoryL1.CAFE, l2=CategoryL2.CAFE_COFFEE
    )
    result = evaluate(
        [wrong],
        _truth(("A1", "김밥천국", CategoryL1.FOOD, CategoryL2.FOOD_RESTAURANT)),
        label="테스트",
    )
    assert result.overall.coverage == 1.0
    assert result.overall.accuracy_l2 == 0.0
    assert result.confusions == [("food", "cafe", 1)]


def test_혼동표는_맞은_조합을_담지_않는다():
    result = evaluate(
        [_hit("A1", "스타벅스", stage=MatchStage.EXACT)],
        _truth(("A1", "스타벅스", *CAFE)),
        label="테스트",
    )
    assert result.confusions == []


# ── ③ 정규화 키 기준 (롱테일) ────────────────────────────────────────────────
def test_정규화_키_기준은_건수_가중이_아니다():
    """상위 브랜드 1건이 표기 100개를 덮지 못하게 — 키마다 가중치가 같다."""
    classified = [_hit(f"A{i}", "스타벅스", stage=MatchStage.EXACT) for i in range(50)]
    classified += [_miss("B1", "을지로호프"), _miss("B2", "성수동카페")]
    truth = _truth(
        *[(f"A{i}", "스타벅스", *CAFE) for i in range(50)],
        ("B1", "을지로호프", CategoryL1.CAFE, CategoryL2.CAFE_ALCOHOL),
        ("B2", "성수동카페", *CAFE),
    )
    result = evaluate(classified, truth, label="테스트")

    assert result.overall.accuracy_l2 > 0.96  # 거래 기준으로는 거의 만점
    assert result.distinct.keys == 3
    assert result.distinct.accuracy_l2 == pytest.approx(1 / 3)  # 표기 기준으로는 1/3
    assert result.distinct.unmatched == 2


def test_같은_표기가_여러_거래면_키_하나로_센다():
    result = evaluate(
        [_hit(f"A{i}", "스타벅스", stage=MatchStage.EXACT) for i in range(5)],
        _truth(*[(f"A{i}", "스타벅스", *CAFE) for i in range(5)]),
        label="테스트",
    )
    assert result.distinct.keys == 1


# ── 축 분해 · 승격 후보 ─────────────────────────────────────────────────────
def test_제공기관_축_분해의_합이_전체와_같다():
    """축을 쪼갠 표와 전체 표의 모수가 어긋나면 둘 중 하나는 거짓이다."""
    classified = [
        _hit("A1", "스타벅스", stage=MatchStage.EXACT, provider_id="CARD_A"),
        _hit("A2", "스타벅스", stage=MatchStage.EXACT, provider_id="CARD_B"),
        _classified(
            "A3", "을지로호프", stage=MatchStage.UNMATCHED,
            l1=CategoryL1.UNCLASSIFIED, l2=CategoryL2.UNCLASSIFIED_INSUFFICIENT,
            provider_id="CARD_B",
        ),
    ]
    result = evaluate(classified, _truth(("A1", "스타벅스", *CAFE)), label="테스트")

    assert set(result.by_provider) == {"CARD_A", "CARD_B"}
    assert sum(s.labeled + s.unlabeled for s in result.by_provider.values()) == 3
    assert result.by_provider["CARD_B"].unmatched_rate == 0.5


def test_미매칭_상위는_정답_대표명을_함께_준다():
    """사전에 무엇을 넣어야 하는지가 바로 보여야 승격 루프가 돈다 (blueprint §3.3⑤)."""
    result = evaluate(
        [_miss("A1", "제이디에프컴퍼니"), _miss("A2", "제이디에프컴퍼니"), _miss("A3", "낙원떡집")],
        _truth(
            ("A1", "빈스로드", *CAFE),
            ("A2", "빈스로드", *CAFE),
            ("A3", "낙원떡집", CategoryL1.CAFE, CategoryL2.CAFE_BAKERY),
        ),
        label="테스트",
    )
    assert result.unmatched_top[0] == ("제이디에프컴퍼니", 2, "빈스로드")
    assert ("낙원떡집", 1, "낙원떡집") in result.unmatched_top


def test_단계별_정확도는_단계마다_따로_낸다():
    """평균 하나로 뭉치면 fuzzy 가 조용히 틀리는 것이 exact 의 만점에 가려진다."""
    result = evaluate(
        [
            _hit("A1", "스타벅스", stage=MatchStage.EXACT),
            _classified(
                "A2", "김밥천국", stage=MatchStage.FUZZY,
                l1=CategoryL1.CAFE, l2=CategoryL2.CAFE_COFFEE,
            ),
        ],
        _truth(("A1", "스타벅스", *CAFE), ("A2", "김밥천국", CategoryL1.FOOD, CategoryL2.FOOD_RESTAURANT)),
        label="테스트",
    )
    assert result.overall.stage_accuracy("exact") == 1.0
    assert result.overall.stage_accuracy("fuzzy") == 0.0


# ── 통합: 실제 합성 데이터 위의 베이스라인 (A) ──────────────────────────────
pytestmark_data = pytest.mark.skipif(
    not DEFAULT_BUNDLE_DIR.exists() or not DEFAULT_TRUTH_PATH.exists(),
    reason="합성 데이터가 없다 — `python -m txlens.generator --seed 42` (docs/recovery.md)",
)


@pytest.fixture(scope="module")
def baseline():
    return run_rules_baseline()


@pytestmark_data
def test_룰이_확정한_것은_전부_정답이다(baseline):
    """룰의 실패 양식은 "틀림"이 아니라 "모름"이어야 한다.

    이 테스트가 깨지면 **어느 단계가 남의 브랜드를 삼켰다**는 뜻이다 (D-024 의 `token_set_ratio`
    사고가 정확히 이 형태였다). L2 평균 정확도로는 안 보인다 — 삼킨 쪽과 삼켜진 쪽이 같은 L2 를
    공유하면 지표는 그대로이고 2부(구독 탐지)만 조용히 오염된다 (D-010).
    """
    for stage in (MatchStage.EXACT, MatchStage.ALIAS, MatchStage.REGEX, MatchStage.FUZZY):
        if baseline.overall.by_stage.get(stage.value):
            assert baseline.overall.stage_accuracy(stage.value) == 1.0, (
                f"{stage.value} 단계가 틀린 브랜드로 확정했다 — 패턴이 남의 상호를 삼킨다"
            )


@pytestmark_data
def test_모든_단계가_적어도_한_건씩_돈다(baseline):
    """단계를 만들어 놓고 한 건도 안 도는 것을 "구현했다"고 세지 않는다."""
    for stage in (MatchStage.EXACT, MatchStage.ALIAS, MatchStage.REGEX, MatchStage.FUZZY):
        assert baseline.overall.by_stage.get(stage.value, 0) > 0, f"{stage.value} 단계가 0건이다"


@pytestmark_data
def test_정규화_키_기준이_거래_기준보다_한참_낮다(baseline):
    """이 격차가 곧 이 프로젝트의 존재 이유다. 격차가 사라졌다면 데이터가 쉬워진 것이다."""
    assert baseline.overall.accuracy_l2 > 0.85
    assert baseline.distinct.accuracy_l2 < baseline.overall.accuracy_l2 - 0.3


@pytestmark_data
def test_PG마스킹이_미매칭보다_작지_않게_뭉뚱그려지지_않는다(baseline):
    """둘이 모두 실재해야 "원리적 한계 / 개선 가능"의 구분이 데이터로 뒷받침된다."""
    assert baseline.overall.pg_masked_rate > 0.0
    assert baseline.overall.unmatched_rate > 0.0
    assert baseline.overall.coverage + baseline.overall.unmatched_rate + (
        baseline.overall.pg_masked_rate
    ) == pytest.approx(1.0)


@pytestmark_data
def test_tail_브랜드가_미매칭_상위를_채운다(baseline):
    """사전이 head 만 담았다는 D-022 의 결과가 지표에도 나타나야 한다."""
    names = {name for name, _, _ in baseline.unmatched_top}
    assert {"을지로호프", "낙원떡집"} <= names


@pytestmark_data
def test_기관별로_난이도가_다르다(baseline):
    """제공기관을 한 덩어리로 보면 오염이 심한 기관의 실패가 평균에 묻힌다 (D-012)."""
    coverages = {k: s.coverage for k, s in baseline.by_provider.items()}
    assert len(coverages) >= 3
    assert max(coverages.values()) - min(coverages.values()) > 0.1


@pytestmark_data
def test_사전을_바꾸면_지표가_따라_움직인다(tmp_path):
    """평가 하네스가 사전을 실제로 갈아끼우는지 — v2 비교 실험의 전제다."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("version: 99\nfuzzy_threshold: 90\nentries: []\n", encoding="utf-8")
    result = run_rules_baseline(dictionary_path=empty, label="사전 없음")
    assert result.dictionary_version == 99
    assert result.overall.coverage == 0.0


def test_같은_입력이면_같은_지표가_나온다():
    """지표에 난수·시각이 섞이면 베이스라인 비교가 의미를 잃는다."""
    matcher = RuleMatcher.from_path()
    txs = [preprocess_one(_raw("A1", "스타벅스 성수점")), preprocess_one(_raw("A2", "을지로호프"))]
    truth = _truth(("A1", "스타벅스", *CAFE))
    first = evaluate(matcher.match_all(txs), truth, label="테스트")
    second = evaluate(matcher.match_all(txs), truth, label="테스트")
    assert first == second
