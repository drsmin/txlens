"""HITL 검토 루프 — `confidence_gate` → `human_review` → `rule_promote` (설계 §3.3 ④).

여기서 막고 싶은 것은 다섯이다.

① **사람 없는 곳에서 멈추는 것.** 평가 하네스는 이 그래프를 수십 번 돌린다. 그 실행이
   `interrupt()` 를 만나면 지표가 안 나오는데, 그것은 "정확도가 낮다"가 아니라 "측정이 없다"로
   나타난다. 그래서 `review` 는 기본이 꺼짐이고, 켜도 큐가 비면 사람을 부르지 않는다.
② **사람이 고친 것이 조용히 무시되는 것.** 검토 큐(τ 0.8 미만)와 자동 승격(0.7 이상)은 구간이
   겹친다. 그 겹친 표기에서 자동이 이기면 화면은 "반영됨"이라 적고 사전에는 모델의 답이 남는다.
③ **지표가 두 배로 보고되는 것.** 서브그래프가 받은 큐를 그대로 돌려주면 부모의 `operator.add`
   가 한 번 더 더한다. 실측에서 3개가 6개로 보고됐다 — 많이 검토한 것처럼 보이는 버그다.
④ **사람 교정이 사전을 깨뜨리는 것.** 사람만이 새 브랜드를 만들 수 있으므로(자동은 못 한다),
   사전 로더의 검증을 통과하지 못하는 항목이 여기서만 만들어질 수 있다.
⑤ **그림이 거짓말하는 것.** `review=False` 인데 HITL 상자가 그려지거나, 분류기가 없는데
   (베이스라인 A) HITL 을 붙여 영원히 빈 큐를 보는 상자를 그리는 것.
"""

from __future__ import annotations

import pytest

from txlens.graph import (
    build_checkpointer,
    build_graph,
    build_review_subgraph,
    checkpoint_types,
    confidence_gate_node,
    pending_review,
    resume_review,
    route_after_gate,
    rule_promote_node,
)
from txlens.graph.nodes import _coerce_corrections
from txlens.rules import load_dictionary
from txlens.rules.promote import apply_proposals, dump_dictionary, propose_from_corrections
from txlens.schemas import (
    CategoryL1,
    CategoryL2,
    Evidence,
    HumanCorrection,
    LLMTarget,
    LLMVerdict,
    ReviewItem,
)

#: v1 사전에 있는 브랜드 / 없는 브랜드. 사람 교정의 두 갈래가 여기서 갈린다.
KNOWN = "스타벅스"
NEW_BRAND = "을지로호프"


def _verdict(confidence: float = 0.4) -> LLMVerdict:
    return LLMVerdict(
        merchant_canonical="",
        brand=None,
        category_l1=CategoryL1.UNCLASSIFIED,
        category_l2=CategoryL2.UNCLASSIFIED_INSUFFICIENT,
        confidence=confidence,
        reasoning="단서가 부족하다",
        evidence=Evidence.GUESS,
    )


def _item(name: str, tx_count: int, confidence: float = 0.4) -> ReviewItem:
    return ReviewItem(
        normalized_name=name, verdict=_verdict(confidence), tx_count=tx_count
    )


def _correction(name: str, canonical: str, **overrides) -> HumanCorrection:
    fields = {
        "normalized_name": name,
        "merchant_canonical": canonical,
        "category_l1": CategoryL1.FOOD,
        "category_l2": CategoryL2.FOOD_RESTAURANT,
        "tx_count": 7,
    }
    return HumanCorrection(**{**fields, **overrides})


class _AnyClassifier:
    """`build_graph` 가 보는 것은 `classifier is None` 뿐이다 — 모양 검사에는 이걸로 충분하다."""


# ── ① 그래프 모양 ───────────────────────────────────────────────────────────
def test_기본값은_꺼짐이고_그래프가_예전_그대로다():
    """평가 하네스가 도는 그래프다. 여기에 HITL 이 끼면 실측이 통째로 멈춘다."""
    nodes = set(build_graph(classifier=_AnyClassifier()).get_graph().nodes)
    assert "rule_promote" in nodes
    assert "review" not in nodes


def test_켜면_승격_자리가_서브그래프로_바뀐다():
    """승격의 자리는 하나다 — 바깥에도 두면 같은 후보가 두 번 만들어진다."""
    graph = build_graph(classifier=_AnyClassifier(), review=True).get_graph()
    nodes = set(graph.nodes)
    edges = {(e.source, e.target) for e in graph.edges}

    assert "review" in nodes
    assert "rule_promote" not in nodes, "바깥에 남으면 사람 없이 만든 후보가 먼저 키를 차지한다"
    assert ("llm_classify", "review") in edges
    assert ("review", "finalize") in edges


def test_분류기가_없으면_켤_수_없다():
    """(A) 는 검토 큐를 채우는 노드가 없다 — 붙이면 그림에만 있는 HITL 이 된다."""
    with pytest.raises(ValueError, match="분류기"):
        build_graph(review=True)


def test_서브그래프는_세_노드와_두_갈래다():
    """경계는 **멈출 수 있는 부분**이다. 그 셋만 체크포인터가 필요하다."""
    graph = build_review_subgraph(dictionary=load_dictionary()).get_graph()
    nodes = set(graph.nodes)
    edges = {(e.source, e.target) for e in graph.edges}

    assert {"confidence_gate", "human_review", "rule_promote"} <= nodes
    assert ("confidence_gate", "human_review") in edges
    assert ("confidence_gate", "rule_promote") in edges, "큐가 비면 사람을 건너뛰는 갈래"
    assert ("human_review", "rule_promote") in edges


# ── ② 큐 정렬과 배치 ────────────────────────────────────────────────────────
def test_정렬은_건수와_불확실성의_곱이다():
    """건수만 보면 확신 있는 대형 브랜드가, 불확실성만 보면 한 건짜리 꼬리가 화면을 채운다."""
    queue = [
        _item("확신하는대형", tx_count=100, confidence=0.95),  # 우선순위 5.0
        _item("애매한중형", tx_count=40, confidence=0.5),  # 우선순위 20.0
        _item("모르는꼬리", tx_count=1, confidence=0.1),  # 우선순위 0.9
    ]
    batch = confidence_gate_node({"review_queue": queue})["review_batch"]
    assert [i.normalized_name for i in batch] == ["애매한중형", "확신하는대형", "모르는꼬리"]


def test_동순위는_이름순으로_고정된다():
    """흔들리면 데모에서 같은 장면을 두 번 만들 수 없다."""
    queue = [_item("나가맹점", 10), _item("가가맹점", 10), _item("다가맹점", 10)]
    batch = confidence_gate_node({"review_queue": queue})["review_batch"]
    assert [i.normalized_name for i in batch] == ["가가맹점", "나가맹점", "다가맹점"]


def test_배치_상한이_큐_길이와_다르다():
    """큐는 276개까지 간다. 한 화면에 쏟으면 사람은 고르는 대신 스크롤을 한다."""
    queue = [_item(f"가맹점{i:03d}", tx_count=i + 1) for i in range(50)]
    batch = confidence_gate_node({"review_queue": queue}, batch_size=5)["review_batch"]

    assert len(batch) == 5
    assert batch[0].normalized_name == "가맹점049", "잘라도 우선순위 위쪽이 남아야 한다"


def test_볼_것이_없으면_사람을_부르지_않는다():
    """이 한 줄이 배치 실행을 지킨다 — 답할 사람이 없는 곳에서 멈추면 지표가 없다."""
    assert route_after_gate({"review_batch": []}) == "rule_promote"
    assert route_after_gate({}) == "rule_promote"
    assert route_after_gate({"review_batch": [_item("을지로호프", 3)]}) == "human_review"


# ── ③ 재개 값 해석 ──────────────────────────────────────────────────────────
def test_재개_값은_세_형태를_다_받는다():
    """형태를 하나로 강제하면 화면 쪽에 포장 코드가 생기고, 그 코드에는 테스트가 없다."""
    raw = {
        "normalized_name": "을지로호프",
        "merchant_canonical": NEW_BRAND,
        "category_l1": "food",
        "category_l2": "food.restaurant",
    }
    assert _coerce_corrections(None) == []
    assert len(_coerce_corrections([raw])) == 1
    assert len(_coerce_corrections({"corrections": [raw]})) == 1
    assert len(_coerce_corrections([HumanCorrection(**raw)])) == 1


def test_재개_값이_교정_목록이_아니면_거절한다():
    """여기를 통과한 것은 곧 사전에 쓰인다 — 느슨하게 받으면 안 되는 자리다."""
    with pytest.raises(TypeError):
        _coerce_corrections("을지로호프")
    with pytest.raises(ValueError, match="category_l2"):
        _coerce_corrections(
            [
                {
                    "normalized_name": "을지로호프",
                    "merchant_canonical": NEW_BRAND,
                    "category_l1": "food",
                    "category_l2": "cafe.coffee",  # l1 에 속하지 않는다
                }
            ]
        )


# ── ④ 멈춤과 재개 ───────────────────────────────────────────────────────────
def _paused(queue: list[ReviewItem], thread: str):
    """서브그래프를 큐만 넣고 돌려 멈춘 상태를 만든다. **번들도 거래도 없다**."""
    graph = build_review_subgraph(dictionary=load_dictionary(), checkpointer=build_checkpointer())
    config = {"configurable": {"thread_id": thread}}
    graph.invoke({"review_queue": queue}, config)
    return graph, config


def test_서브그래프는_큐만으로_혼자_돈다():
    """Streamlit S1 이 파이프라인을 다시 돌리지 않고 이것만 부를 수 있어야 한다."""
    graph, config = _paused([_item("을지로호프", 71)], "solo")
    pending = pending_review(graph, config)

    assert pending is not None, "멈추지 않았다 — 사람에게 물을 것이 있는데 그냥 지나갔다"
    assert pending["kind"] == "review_batch"
    assert pending["items"][0]["normalized_name"] == "을지로호프"
    assert pending["items"][0]["tx_count"] == 71


def test_나가는_것은_표기와_판정뿐이다():
    """거래는 서브그래프 상태에 애초에 없다 — 전송 경계와 같은 선택이다."""
    graph, config = _paused([_item("을지로호프", 71)], "payload")
    item = pending_review(graph, config)["items"][0]

    assert set(item) == {"normalized_name", "tx_count", "priority", "verdict"}


def test_교정을_넣으면_이어_돌고_승격_후보가_나온다():
    graph, config = _paused([_item("을지로호프", 71)], "resume")
    final = resume_review(graph, config, [_correction("을지로호프", NEW_BRAND, tx_count=0)])

    assert pending_review(graph, config) is None, "재개했는데 여전히 멈춰 있다"
    proposal = final["rule_proposals"][0]
    assert proposal.origin == "human"
    assert proposal.pattern == "을지로호프"
    assert proposal.covers_tx_count == 71, "교정이 덮는 거래 건수는 배치가 안다"


def test_화면이_되돌려_보내지_않아도_건수와_원판정이_채워진다():
    """화면이 상태의 사본을 들면 그 사본은 어긋난다 — 배치가 아는 것은 배치가 채운다."""
    graph, config = _paused([_item("을지로호프", 71, confidence=0.2)], "fill")
    final = resume_review(graph, config, [_correction("을지로호프", NEW_BRAND, tx_count=0)])

    correction = final["human_corrections"][0]
    assert correction.tx_count == 71
    assert correction.reviewed_verdict is not None, "무엇을 보고 고쳤는지가 감사 추적이다"


def test_배치에_없는_표기를_고쳐도_버리지_않는다():
    """사람이 화면 밖에서 아는 것을 반영하는 정당한 경로다 — 다만 채워 줄 것이 없다."""
    graph, config = _paused([_item("을지로호프", 71)], "outside")
    final = resume_review(graph, config, [_correction("성수동카페", NEW_BRAND, tx_count=0)])

    correction = final["human_corrections"][0]
    assert correction.normalized_name == "성수동카페"
    assert correction.tx_count == 0 and correction.reviewed_verdict is None


def test_빈_교정도_정당한_답이다():
    """사람을 부른 것이 사람이 무언가를 고쳤다는 뜻은 아니다."""
    graph, config = _paused([_item("을지로호프", 71)], "empty")
    final = resume_review(graph, config, [])

    assert final["human_corrections"] == []
    assert pending_review(graph, config) is None


def test_큐가_비면_멈추지_않고_끝난다():
    """켜 두고 돌려도 볼 것이 없으면 배치처럼 끝나야 한다."""
    graph, config = _paused([], "quiet")
    assert pending_review(graph, config) is None


def test_검토_큐가_두_배로_보고되지_않는다():
    """서브그래프가 받은 큐를 되돌려주면 부모의 `operator.add` 가 한 번 더 더한다.

    실측에서 3개가 6개로 보고됐다. 두 배가 된 지표는 "많이 검토했다"처럼 보여 성공으로
    오해된다 — `ReviewOutput` 이 나가는 것을 새로 만든 둘로 제한해 막는다.
    """
    graph, config = _paused([_item("을지로호프", 71), _item("성수동카페", 65)], "double")
    final = resume_review(graph, config, [])

    assert "review_queue" not in final, "받은 큐를 되돌려주면 부모에서 두 배가 된다"


def test_체크포인터가_우리_타입을_모델로_되살린다():
    """허용 목록은 `(모듈, 이름)` **완전 일치**다 — 빠진 타입은 `dict` 로 되살아난다.

    예외가 아니라 조용한 형변환이라, 처음 겪었을 때 증상은 저 아래 노드의 `AttributeError`
    였다. 목록을 모듈에서 뽑는 이유가 이것이고, 이 테스트가 그 계약이다.
    """
    graph, config = _paused([_item("을지로호프", 71)], "types")
    restored = graph.get_state(config).values["review_queue"][0]

    assert isinstance(restored, ReviewItem), "체크포인트에서 dict 로 돌아왔다"
    assert isinstance(restored.verdict, LLMVerdict)


def test_허용_목록에_상태_스키마가_다_들어_있다():
    """스키마가 하나 늘 때 손으로 적은 목록은 조용히 빠진다 — 모듈에서 뽑는 것이 그 방지책이다."""
    names = {t.__name__ for t in checkpoint_types()}

    assert {"ReviewItem", "HumanCorrection", "RuleProposal", "LLMVerdict"} <= names
    assert "RunMetrics" in names, "finalize 가 낸 지표도 체크포인트에 실린다"


# ── ⑤ 사람만이 새 브랜드를 만든다 ───────────────────────────────────────────
def test_사전에_없는_브랜드는_새_항목이_된다():
    """자동 승격이 **원리적으로** 못 하는 일이고, 남은 276건을 여는 열쇠다."""
    base = load_dictionary()
    proposals = propose_from_corrections([_correction("을지로호프", NEW_BRAND)], dictionary=base)

    assert [(p.pattern_type, p.origin) for p in proposals] == [("exact", "human")]
    promoted, accepted = apply_proposals(base, proposals)
    assert len(accepted) == 1
    assert promoted.exact_index[NEW_BRAND].canonical == NEW_BRAND
    assert len(promoted.entries) == len(base.entries) + 1


def test_사전에_있는_브랜드로_고치면_별칭이다():
    """새 항목을 만들면 같은 브랜드가 두 이름으로 쪼개져 2부의 집계가 그 위에서 무너진다."""
    base = load_dictionary()
    proposals = propose_from_corrections([_correction("타벅스", KNOWN)], dictionary=base)

    assert [(p.pattern_type, p.merchant_canonical) for p in proposals] == [("alias", KNOWN)]
    promoted, _ = apply_proposals(base, proposals)
    assert promoted.alias_index["타벅스"].canonical == KNOWN
    assert len(promoted.entries) == len(base.entries), "아는 브랜드에 새 항목이 생겼다"


def test_라벨은_사전이_이긴다():
    """사람이 스타벅스의 라벨에 이견이 있다면 사전 수정 사안이지 이 표기의 교정이 아니다."""
    base = load_dictionary()
    entry = base.exact_index[KNOWN]
    proposals = propose_from_corrections(
        [
            _correction(
                "타벅스",
                KNOWN,
                category_l1=CategoryL1.FOOD,
                category_l2=CategoryL2.FOOD_RESTAURANT,
            )
        ],
        dictionary=base,
    )
    assert proposals[0].category_l2 is entry.category_l2


def test_대표명은_정규화해서_담는다():
    """`canonical` 은 그 자신이 exact 키가 된다 — 정규화 형태가 아니면 로더가 사전을 거부한다."""
    base = load_dictionary()
    proposals = propose_from_corrections(
        [_correction("을지로호프", f"(주){NEW_BRAND}")], dictionary=base
    )
    assert proposals[0].merchant_canonical == NEW_BRAND


def test_자동은_여전히_새_브랜드를_못_만든다():
    """정책을 `propose_rules` 한 곳에만 두면 다른 생산자가 붙을 때 조용히 새 브랜드가 생긴다."""
    base = load_dictionary()
    forged = [
        p.model_copy(update={"origin": "llm_high_confidence"})
        for p in propose_from_corrections([_correction("을지로호프", NEW_BRAND)], dictionary=base)
    ]

    _, accepted = apply_proposals(base, forged)
    assert accepted == [], "출처가 모델인데 새 브랜드가 만들어졌다"


def test_사람이_만든_사전도_로더_검증을_통과한다(tmp_path):
    """round-trip 이 계약이다 — 파일로 나갔다가 다시 들어와야 다음 라운드가 쓴다."""
    base = load_dictionary()
    proposals = propose_from_corrections(
        [_correction("을지로호프", NEW_BRAND), _correction("타벅스", KNOWN)], dictionary=base
    )
    promoted, accepted = apply_proposals(base, proposals)

    path = tmp_path / f"v{promoted.version}.yaml"
    dump_dictionary(
        promoted, path, parent_version=base.version, accepted=accepted, model="human-review"
    )
    reloaded = load_dictionary(path)

    assert reloaded.exact_index[NEW_BRAND].canonical == NEW_BRAND
    assert reloaded.alias_index["타벅스"].canonical == KNOWN
    # 출처가 머리말에 남아야 한다: 자동은 모델을 바꿔 다시 만들 수 있고 사람 판단은 그럴 수 없다.
    assert "사람 2건" in path.read_text(encoding="utf-8")


def test_대표명_자신이_표기면_별칭을_또_만들지_않는다():
    """`exact_keys` 가 이미 대표명을 색인한다 — 또 넣으면 로더가 키 중복으로 죽는다."""
    base = load_dictionary()
    proposals = propose_from_corrections([_correction(NEW_BRAND, NEW_BRAND)], dictionary=base)
    promoted, accepted = apply_proposals(base, proposals)

    assert len(accepted) == 1
    entry = promoted.exact_index[NEW_BRAND]
    assert entry.alias == [] and entry.exact == []


# ── ⑥ 사람이 자동을 이긴다 ──────────────────────────────────────────────────
def test_겹치는_구간에서는_사람이_이긴다():
    """τ(0.8)과 승격 문턱(0.7)이 겹친다 — 그 표기는 큐에도 있고 자동 후보에도 있다.

    순서가 뒤집히면 사람이 화면에서 고친 답이 조용히 버려지고, 화면은 "반영됨"이라 적는다.
    """
    dictionary = load_dictionary()
    verdict = LLMVerdict(
        merchant_canonical=KNOWN,
        brand=KNOWN,
        category_l1=CategoryL1.CAFE,
        category_l2=CategoryL2.CAFE_COFFEE,
        confidence=0.75,  # 큐에도 들어가고(τ 0.8 미만) 자동 승격도 된다(0.7 이상)
        reasoning="브랜드를 안다",
        evidence=Evidence.BRAND_KNOWN,
    )
    state = {
        "llm_targets": [
            LLMTarget(normalized_name="애매한표기", sample_raw_name="애매한표기", tx_count=9)
        ],
        "llm_results": {"애매한표기": verdict},
        "human_corrections": [_correction("애매한표기", NEW_BRAND)],
    }
    proposals = rule_promote_node(state, dictionary=dictionary)["rule_proposals"]
    _, accepted = apply_proposals(dictionary, proposals)

    assert [p.origin for p in accepted] == ["human"], "모델의 답이 사람의 답을 덮었다"
    assert accepted[0].merchant_canonical == NEW_BRAND
