"""LangGraph 배선 — 노드·분기·상태·실행 지표 (blueprint §3.1 · §3.2).

여기서 막고 싶은 것은 넷이다.

① **그림이 거짓말하는 것.** `classifier=None` 인데 LLM 노드가 그래프에 있으면
   `draw_ascii()` 가 실제로 도는 경로와 다른 것을 보여준다. 발표 자료가 그 그림을 쓴다.
② **미분류가 커버리지로 세어지는 것.** LLM 이 "모른다"고 답한 것에 `MatchStage.LLM` 을 붙이면
   `RESOLVED_STAGES` 에 들어가 커버리지가 부푼다. 지표 하나로는 안 보이는 종류의 거짓이다
   (D-024① 의 `token_set_ratio` 사고와 같은 형태).
③ **조용한 잘라내기.** 상한 때문에 안 보낸 표기가 "LLM 이 못 잡았다"로 섞이면 부분 측정이
   전수 측정으로 보고된다.
④ **A/B/C 가 서로 다른 파이프라인을 도는 것.** 셋이 같은 그래프를 돌지 않으면 비교가 조건
   차이가 아니라 코드 차이를 재게 된다.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from scripted_model import ScriptedChatModel, verdict_turn
from txlens.eval.harness import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_TRUTH_PATH,
    EMPTY_DICTIONARY,
    compare_table,
    run_hybrid_baseline,
    run_llm_baseline,
    run_rules_baseline,
)
from txlens.graph import (
    apply_verdict,
    build_graph,
    collect_targets,
    finalize_node,
    normalize_node,
    pair_cancellations_node,
    route_after_rules,
)
from txlens.preprocess import preprocess, preprocess_one
from txlens.schemas import (
    CategoryL1,
    CategoryL2,
    ClassifiedTx,
    Evidence,
    LLMVerdict,
    MatchStage,
    RawTx,
)

#: 분류기 유무와 무관하게 항상 도는 노드들. 이름이 바뀌면 발표 그림도 같이 바뀌어야 한다.
_PIPELINE = {
    "ingest",
    "pair_cancellations",
    "normalize",
    "rule_match",
    "collect_targets",
    "finalize",
}


def _raw(approval_no: str, name: str, *, provider: str = "CARD_A") -> RawTx:
    return RawTx(
        provider_id=provider,
        customer_id="SYNTH_0001",
        approval_no=approval_no,
        tx_datetime=datetime(2026, 3, 1, 12, 0),
        merchant_name_raw=name,
        amount=10000,
    )


def _unmatched(approval_no: str, name: str) -> ClassifiedTx:
    normalized = preprocess_one(_raw(approval_no, name))
    return ClassifiedTx(
        normalized=normalized,
        merchant_canonical=normalized.normalized_name,
        category_l1=CategoryL1.UNCLASSIFIED,
        category_l2=CategoryL2.UNCLASSIFIED_INSUFFICIENT,
        confidence=0.0,
        match_stage=MatchStage.UNMATCHED,
    )


def _verdict(**overrides) -> LLMVerdict:
    fields = {
        "merchant_canonical": "을지로호프",
        "brand": None,
        "category_l1": CategoryL1.CAFE,
        "category_l2": CategoryL2.CAFE_ALCOHOL,
        "confidence": 0.9,
        "reasoning": "호프집 표기",
        "evidence": Evidence.KEYWORD,
    }
    return LLMVerdict(**{**fields, **overrides})


def _scripted(**scripts_by_name) -> ScriptedChatModel:
    """이름별 대본. 대본이 없는 표기는 **모른다고 답한다** — 롱테일의 기본값이다."""
    return ScriptedChatModel(
        scripts={
            name: [verdict_turn(**fields)] for name, fields in scripts_by_name.items()
        },
        default_script=[
            verdict_turn(
                merchant_canonical="",
                brand=None,
                category_l1="unclassified",
                category_l2="unclassified.insufficient",
                confidence=0.1,
                reasoning="업종 단서가 없다",
                evidence="guess",
            )
        ],
    )


# ── ① 그래프 모양 ───────────────────────────────────────────────────────────
def test_분류기가_없으면_LLM_노드가_그래프에_없다():
    """베이스라인 (A) 의 그림에 LLM 노드가 그려지면 그 그림이 곧 잘못된 설명이 된다."""
    nodes = set(build_graph().get_graph().nodes)
    assert "llm_classify" not in nodes
    assert _PIPELINE <= nodes


def test_분류기가_있으면_조건부_분기가_생긴다():
    classifier = _classifier(_scripted())
    graph = build_graph(classifier=classifier).get_graph()
    assert "llm_classify" in set(graph.nodes)
    sources = {edge.source for edge in graph.edges if edge.target == "llm_classify"}
    assert sources == {"collect_targets"}, "분기는 대상을 다 모은 뒤에 갈린다"


def test_전처리와_대상수집이_각각_제_노드다():
    """노드 경계가 그림의 설명력이다 (D-031).

    `pair_cancellations`(집합 판단) 와 `normalize`(건 단위) 를 한 상자에 두면 "전처리"라는 한
    단어가 성질이 다른 둘을 덮는다. `collect_targets` 는 **거래 → 고유 표기**로 단위가 바뀌는
    전송 경계라 그림에서 짚을 수 있어야 한다.
    """
    graph = build_graph().get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("ingest", "pair_cancellations") in edges
    assert ("pair_cancellations", "normalize") in edges
    assert ("normalize", "rule_match") in edges
    assert ("rule_match", "collect_targets") in edges
    assert "preprocess" not in set(graph.nodes), "쪼갠 노드가 옛 이름으로 남아 있으면 안 된다"


def test_쪼갠_두_노드가_preprocess_와_같은_답을_낸다():
    """`preprocess()` 는 순수 함수로 남아 있다 — 두 경로가 갈라지면 그래프가 조용히 달라진다."""
    raws = [
        _raw("A1", "스타벅스 강남점"),
        _raw("A2", "(주)을지로호프"),
        _raw("A3", "성수동카페", provider="CARD_B"),
    ]
    state = pair_cancellations_node({"raw": raws})
    got = normalize_node({"raw": raws, **state})["normalized"]
    assert got == preprocess(raws)


def _classifier(model):
    from txlens.agents import build_classifier

    return build_classifier(model, dictionary=EMPTY_DICTIONARY)


def test_미매칭이_없으면_LLM_을_건너뛴다():
    assert route_after_rules({"llm_targets": []}) == "finalize"
    assert route_after_rules({}) == "finalize"
    assert route_after_rules({"llm_targets": [object()]}) == "llm_classify"


# ── 대상 수집 ───────────────────────────────────────────────────────────────
def test_대상은_건수_많은_것부터_이름순으로_고정된다():
    """순서가 흔들리면 상한을 걸었을 때 보내는 표기가 실행마다 달라진다."""
    matched = [
        *[_unmatched(f"A{i}", "을지로호프") for i in range(3)],
        *[_unmatched(f"B{i}", "낙원떡집") for i in range(3)],
        _unmatched("C1", "가나다세탁"),
    ]
    targets = collect_targets(matched)
    assert [t.normalized_name for t in targets] == ["낙원떡집", "을지로호프", "가나다세탁"]
    assert [t.tx_count for t in targets] == [3, 3, 1]


def test_대표_원본표기는_가장_흔한_것이다():
    """아무거나 고르면 실행마다 달라져 캐시가 빗나간다."""
    matched = [
        _unmatched("A1", "(주)을지로호프"),
        _unmatched("A2", "을지로호프 2호점"),
        _unmatched("A3", "을지로호프 3호점"),
    ]
    # 세 표기가 같은 키로 모이지만 원본은 다르다 — 최빈값이 없으면 첫 항목이 대표가 된다.
    (target,) = collect_targets(matched)
    assert target.normalized_name == "을지로호프"
    assert target.sample_raw_name in {"(주)을지로호프", "을지로호프 2호점", "을지로호프 3호점"}


def test_이미_매칭된_거래는_대상이_아니다():
    matched = [
        ClassifiedTx(
            normalized=preprocess_one(_raw("A1", "스타벅스")),
            merchant_canonical="스타벅스",
            category_l1=CategoryL1.CAFE,
            category_l2=CategoryL2.CAFE_COFFEE,
            confidence=1.0,
            match_stage=MatchStage.EXACT,
        ),
        ClassifiedTx(
            normalized=preprocess_one(_raw("A2", "(주)케이지이니시스")),
            merchant_canonical="미분류(PG경유)",
            category_l1=CategoryL1.UNCLASSIFIED,
            category_l2=CategoryL2.UNCLASSIFIED_PG_MASKED,
            confidence=1.0,
            match_stage=MatchStage.PG_MASKED,
        ),
    ]
    assert collect_targets(matched) == []


# ── ② 미분류를 커버리지로 세지 않는다 ───────────────────────────────────────
def test_LLM_이_모른다고_답하면_단계를_올리지_않는다():
    """`MatchStage.LLM` 은 `RESOLVED_STAGES` 다 — 미분류에 붙이면 커버리지가 부푼다.

    판정 자체는 `source_verdict` 로 남긴다. 사람이 검토 화면에서 "LLM 은 왜 모른다고 했나"를
    읽어야 하기 때문이다.
    """
    tx = _unmatched("A1", "가나다물산")
    unknown = _verdict(
        merchant_canonical="",
        category_l1=CategoryL1.UNCLASSIFIED,
        category_l2=CategoryL2.UNCLASSIFIED_INSUFFICIENT,
        confidence=0.1,
        evidence=Evidence.GUESS,
    )
    applied = apply_verdict(tx, unknown)

    assert applied.match_stage is MatchStage.UNMATCHED
    assert applied.source_verdict is unknown
    assert applied.merchant_canonical == "가나다물산", "못 잡은 문자열은 그대로 남아야 한다"


def test_LLM_이_확정하면_단계가_LLM_이_된다():
    applied = apply_verdict(_unmatched("A1", "을지로호프"), _verdict())
    assert applied.match_stage is MatchStage.LLM
    assert applied.category_l2 is CategoryL2.CAFE_ALCOHOL
    assert applied.confidence == 0.9


# ── ③ 상한 보고 ─────────────────────────────────────────────────────────────
def test_분류기가_없으면_미전송으로_세지_않는다():
    """(A) 는 "보내려다 못 보낸" 게 아니라 "보낼 계획이 없는" 것이다."""
    matched = [_unmatched("A1", "을지로호프")]
    state = {
        "matched": matched,
        "normalized": [tx.normalized for tx in matched],
        "llm_targets": collect_targets(matched),
    }
    metrics = finalize_node(state, classifier=None)["metrics"]
    assert metrics.llm_targets == 0
    assert metrics.llm_targets_skipped == 0


# ── 통합: 실제 합성 데이터 ──────────────────────────────────────────────────
pytestmark_data = pytest.mark.skipif(
    not DEFAULT_BUNDLE_DIR.exists() or not DEFAULT_TRUTH_PATH.exists(),
    reason="합성 데이터가 없다 — `python -m txlens.generator --seed 42` (docs/recovery.md)",
)

#: 미매칭 상위 표기와 그 정답 (`test_eval.py` 가 상위 목록을 따로 잠그고 있다).
#: `우리동네세탁` 은 쓰지 않는다 — 정답이 **미분류**라 (D-026) "LLM 이 확정했다"는 시나리오를
#: 만들 수 없다. 여기 넣으면 커버리지가 오르는 것을 확인하려는 테스트가 그 브랜드에서 실패한다.
_TOP_UNMATCHED = {
    "을지로호프": ("cafe", "cafe.alcohol"),
    "성수동카페": ("cafe", "cafe.coffee"),
    "연세이비인후과": ("health", "health.clinic"),
}


def _top3_model() -> ScriptedChatModel:
    return _scripted(**{
        name: {
            "merchant_canonical": name,
            "brand": name,
            "category_l1": l1,
            "category_l2": l2,
            "confidence": 0.92,
            "reasoning": "표기에 업종이 드러난다",
            "evidence": "keyword",
        }
        for name, (l1, l2) in _TOP_UNMATCHED.items()
    })


@pytestmark_data
def test_하이브리드는_룰이_못_잡은_것만_LLM_에_보낸다():
    """(C) 의 정의 그대로다 — 룰이 잡은 8,000여 건은 모델을 거치지 않는다."""
    result, metrics = run_hybrid_baseline(_top3_model(), llm_target_limit=3)

    assert metrics.llm_targets == 3
    assert metrics.llm_targets_skipped == 152, "상한으로 자른 몫이 보고되지 않았다"
    assert metrics.rule_resolved == 7262  # 베이스라인 (A) 와 같은 룰 결과
    assert result.overall.coverage > run_rules_baseline().overall.coverage


@pytestmark_data
def test_LLM_이_모른다고_답한_표기는_미매칭에_남는다():
    """모든 표기에 "모른다"고 답하는 모델을 붙여도 커버리지가 오르면 지표가 거짓말하는 것이다."""
    baseline = run_rules_baseline()
    result, metrics = run_hybrid_baseline(_scripted(), llm_target_limit=5)

    assert metrics.llm_unknown == 5
    assert result.overall.coverage == pytest.approx(baseline.overall.coverage)
    assert result.overall.unmatched_rate == pytest.approx(baseline.overall.unmatched_rate)


@pytestmark_data
def test_LLM만_베이스라인은_사전을_전혀_쓰지_않는다():
    """(B) — 룰 커버리지가 0 이어야 한다. 도구가 v1 사전을 보면 그건 이미 하이브리드다."""
    result, metrics = run_llm_baseline(_top3_model(), llm_target_limit=2)

    assert result.dictionary_version is None or result.dictionary_version == 0
    assert metrics.rule_resolved == 0
    assert metrics.llm_targets == 2
    # 사전이 비어 있으니 미매칭 표기가 (A) 보다 훨씬 많다 — 그게 사전이 하는 일의 크기다.
    assert metrics.llm_targets + metrics.llm_targets_skipped > 200


@pytestmark_data
def test_검토_큐는_신뢰도가_낮은_것과_모르는_것을_담는다():
    """τ 미만·미분류는 사람에게 간다. 큐에 있다고 결과에서 빼지는 않는다 (그게 (C) 다)."""
    result, metrics = run_hybrid_baseline(_scripted(), llm_target_limit=4)
    assert metrics.review_queued == 4
    # 큐에 들어간 판정도 결과에 반영된다 — 모수가 조용히 줄지 않는다.
    assert result.tx_classified == 8619


@pytestmark_data
def test_같은_입력이면_같은_지표가_나온다():
    """표기 순서·캐시 히트까지 결정론이어야 베이스라인 비교가 성립한다."""
    first, m1 = run_hybrid_baseline(_top3_model(), llm_target_limit=3)
    second, m2 = run_hybrid_baseline(_top3_model(), llm_target_limit=3)

    assert first.report() == second.report()
    assert m1.model_dump() == m2.model_dump()


@pytestmark_data
def test_지표를_JSON_으로_낼_수_있고_비율이_함께_담긴다():
    """`dataclasses.asdict()` 가 `Counter` 를 `{("exact", 5474): 1}` 로 망가뜨려 전량 실행이
    **끝에서** 죽었다 (30분을 돌린 뒤였다). 그리고 `accuracy_l2` 는 property 라 asdict 가
    빠뜨려, 지표 파일에 지표가 없었다. 둘 다 `to_dict()` 로 막는다 (D-026).
    """
    result, metrics = run_hybrid_baseline(_top3_model(), llm_target_limit=2)
    payload = json.dumps(
        {"result": result.to_dict(), "metrics": metrics.model_dump()}, ensure_ascii=False
    )
    back = json.loads(payload)

    assert back["result"]["overall"]["by_stage"]["exact"] > 0, "Counter 가 망가졌다"
    assert back["result"]["overall"]["accuracy_l2"] == pytest.approx(result.overall.accuracy_l2)
    assert back["result"]["overall"]["coverage"] == pytest.approx(result.overall.coverage)
    assert back["result"]["distinct"]["accuracy_l2"] == pytest.approx(
        result.distinct.accuracy_l2
    )


@pytestmark_data
def test_비교표는_상한으로_자른_사실을_숨기지_않는다():
    rows = [
        (run_rules_baseline(), None),
        run_hybrid_baseline(_top3_model(), llm_target_limit=3),
    ]
    table = compare_table(rows)
    assert "A · 룰만" in table
    assert "C · 하이브리드" in table
    assert "부분 측정" in table
