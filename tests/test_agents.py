"""분류 Agent — 도구·프롬프트·가드레일·캐시 (blueprint §3.0 · D-017).

API 키 없이 돈다. 스크립트된 모델(`tests/scripted_model.py`)이 도구 호출과 구조화 출력을 내므로
**검증하려는 경로 자체**는 진짜로 돈다 — 모델의 판단력만 대역이다.

여기서 막고 싶은 것은 넷이다.

① **few-shot 이 정답 카탈로그로 자라는 것** (D-022 의 연장). 예시에 데이터셋 브랜드를 넣으면
   베이스라인 (B) 가 그 브랜드를 공짜로 맞혀 A/B/C 비교가 무의미해진다. 사람의 기억이 아니라
   테스트가 기계적으로 막는다.
② **전송 경계가 넓어지는 것.** Agent 에 가는 메시지에 금액·일시·고객 식별자가 섞이면
   "표기 단위 전송"이라는 설계가 무너진다. 가드레일이 실제로 던지는지 확인한다.
③ **가드레일 발동이 조용히 성공으로 세어지는 것.** 상한에 닿으면 판정이 없어야 하고, 없으면
   실패로 세어야 한다 — 추측으로 채우면 안 된다.
④ **택소노미가 프롬프트에서 굳는 것.** YAML 을 고쳤는데 프롬프트가 옛 목록을 보면 원인을 찾기
   어려운 오류가 난다.
"""

from __future__ import annotations

import pytest
from langchain.agents.middleware import PIIDetectionError

from scripted_model import STRUCTURED_TOOL_NAME, ScriptedChatModel, Turn, verdict_turn
from txlens.agents import (
    FEW_SHOTS,
    build_classifier,
    build_system_prompt,
    make_tools,
    target_message,
)
from txlens.generator.brands import ALL_BRANDS
from txlens.middleware import ModelCallCache
from txlens.rules import RuleDictionary, load_dictionary
from txlens.schemas import CategoryL1, CategoryL2, LLMTarget
from txlens.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def dictionary() -> RuleDictionary:
    return load_dictionary()


@pytest.fixture(scope="module")
def tools(dictionary):
    return {t.name: t for t in make_tools(dictionary=dictionary, taxonomy=load_taxonomy())}


def _script(**verdict_fields) -> list[Turn]:
    """도구 한 번 → 구조화 출력 한 번. 가장 흔한 경로를 대본으로 굳혀 둔다."""
    return [
        Turn(tool_name="get_brand_hint", args={"name": "무엇이든"}),
        verdict_turn(**verdict_fields),
    ]


_DEFAULT_VERDICT = {
    "merchant_canonical": "테스트가맹점",
    "brand": None,
    "category_l1": "cafe",
    "category_l2": "cafe.alcohol",
    "confidence": 0.72,
    "reasoning": "테스트용 판정",
    "evidence": "keyword",
}


# ── ① few-shot 누출 가드 ────────────────────────────────────────────────────
def test_few_shot_에_데이터셋_브랜드가_없다():
    """예시가 사전 노릇을 하면 베이스라인 (B) 가 부풀어 A/B/C 비교가 죽는다 (D-022 의 연장).

    브랜드명·법인명·로마자 표기를 모두 본다 — 하나만 검사하면 나머지 경로로 새어 들어온다.
    """
    forbidden: set[str] = set()
    for brand in ALL_BRANDS:
        forbidden.update(
            value.upper() for value in (brand.brand, brand.corporate, brand.romanized) if value
        )

    for name, verdict in FEW_SHOTS:
        text = f"{name} {verdict.merchant_canonical} {verdict.brand or ''}".upper()
        hits = sorted(token for token in forbidden if token in text)
        assert not hits, f"few-shot '{name}' 이 카탈로그 표기를 담고 있다: {hits}"


def test_few_shot_은_모른다고_답하는_예시를_포함한다():
    """"모르면 미분류" 를 예시로 보여주지 않으면 모델은 늘 무언가에 붙인다 (D-010 의 사슬)."""
    unknown = [
        name for name, v in FEW_SHOTS if v.category_l1 is CategoryL1.UNCLASSIFIED
    ]
    assert len(unknown) >= 2, "미분류 예시가 없으면 억지 추측을 가르치는 프롬프트가 된다"


# ── ④ 택소노미 주입 ─────────────────────────────────────────────────────────
def test_프롬프트가_택소노미를_YAML_에서_받아온다():
    """카테고리를 프롬프트에 적어 두면 YAML 을 고쳐도 모델은 옛 목록을 본다."""
    taxonomy = load_taxonomy()
    prompt = build_system_prompt(taxonomy)
    missing = [cid for cid in taxonomy.l2_ids() if cid not in prompt]
    assert not missing, f"프롬프트에 없는 L2 카테고리: {missing}"


# ── 도구 ────────────────────────────────────────────────────────────────────
def test_도구는_후보를_확정으로_말하지_않는다(tools):
    """`lookup_alias` 는 이미 매칭에 실패한 이름에만 불린다 — 후보는 전부 임계값 미달이다."""
    out = tools["lookup_alias"].invoke({"name": "스타박스"})
    assert "스타벅스" in out
    assert "확정 아님" in out


def test_빈_사전이면_사전으로는_판단할_수_없다고_답한다():
    """베이스라인 (B) 에서 도구가 조용히 아무것도 안 하면, 왜 못 맞혔는지 설명할 수 없다."""
    tools = {
        t.name: t for t in make_tools(dictionary=RuleDictionary(version=0, entries=[]),
                                      taxonomy=load_taxonomy())
    }
    assert "사전이 비어 있다" in tools["lookup_alias"].invoke({"name": "스타벅스"})


def test_택소노미_검색은_없는_낱말에_없다고_답한다(tools):
    out = tools["search_taxonomy"].invoke({"keyword": "우주선"})
    assert "unclassified.insufficient" in out

    hit = tools["search_taxonomy"].invoke({"keyword": "약국"})
    assert "health.pharmacy" in hit


def test_힌트_도구는_전처리가_분리한_사실을_읽는다(tools):
    """이미 결정론으로 계산한 것을 LLM 에게 다시 추론시키지 않는다."""
    branch = tools["get_brand_hint"].invoke({"name": "(주)미래봄커피 강남점"})
    assert "강남점" in branch
    assert "MIRAE" not in branch  # 법인표기만 떼고 이름을 번역하지 않는다

    pg = tools["get_brand_hint"].invoke({"name": "(주)케이지이니시스"})
    assert "unclassified.pg_masked" in pg
    assert "추측하지 마라" in pg

    channel = tools["get_brand_hint"].invoke({"name": "카카오페이*무명식당"})
    assert "kakaopay" in channel


# ── ② 전송 경계 ─────────────────────────────────────────────────────────────
def test_Agent_에_보내는_것은_표기뿐이다():
    """금액·일시·고객 식별자·건수는 프롬프트에 넣지 않는다 (blueprint §1.1)."""
    message = target_message(
        LLMTarget(normalized_name="을지로호프", sample_raw_name="을지로호프 2호점", tx_count=71)
    )
    assert "을지로호프" in message.text
    assert "71" not in message.text


def test_고객_식별자가_섞이면_예외로_멈춘다():
    """`strategy="block"` 을 쓰는 이유 — 조용히 가리면 사고가 있었다는 사실이 안 남는다.

    ⚠️ 이 예외를 잡아 미분류로 처리하면 안 된다. 표기 하나의 문제가 아니라 개인정보가 새는
    경로가 생겼다는 신호다.
    """
    model = ScriptedChatModel(default_script=_script(**_DEFAULT_VERDICT))
    classifier = build_classifier(model)
    with pytest.raises(PIIDetectionError):
        classifier.classify(LLMTarget(normalized_name="SYNTH_0001 을지로호프", tx_count=1))
    assert model.calls == 0, "차단은 모델을 부르기 **전에** 일어나야 한다"


# ── 구조화 출력 · 도구 바인딩 ───────────────────────────────────────────────
def test_구조화_출력_도구가_실제로_바인딩된다():
    """스키마를 프롬프트로 부탁하지 않고 `ToolStrategy` 로 강제한다."""
    model = ScriptedChatModel(default_script=_script(**_DEFAULT_VERDICT))
    classifier = build_classifier(model)
    classifier.classify(LLMTarget(normalized_name="을지로호프", tx_count=1))

    assert model.bound_tools == [
        "lookup_alias",
        "search_taxonomy",
        "get_brand_hint",
        STRUCTURED_TOOL_NAME,
    ]


def test_판정은_스키마를_통과한_객체로_온다():
    model = ScriptedChatModel(default_script=_script(**_DEFAULT_VERDICT))
    verdict = build_classifier(model).classify(LLMTarget(normalized_name="을지로호프", tx_count=1))
    assert verdict is not None
    assert verdict.category_l2 is CategoryL2.CAFE_ALCOHOL
    assert verdict.category_l1 is CategoryL1.CAFE  # l1/l2 정합성은 스키마가 보장한다


# ── ③ 가드레일이 발동했을 때 ────────────────────────────────────────────────
def test_모델_호출_상한에_닿으면_판정이_없다():
    """상한에 닿았는데 그럴듯한 값이 채워지면, 비용 통제가 곧 오분류가 된다."""
    model = ScriptedChatModel(
        default_script=[Turn(tool_name="get_brand_hint", args={"name": "x"})] * 6
    )
    classifier = build_classifier(model, model_call_limit=2)
    assert classifier.classify(LLMTarget(normalized_name="무명가맹점", tx_count=1)) is None


def test_실패한_표기는_이름으로_보고된다():
    """개수만 세면 어느 표기가 왜 실패했는지 물어볼 수 없다."""
    model = ScriptedChatModel(
        scripts={"실패표기": [Turn(tool_name="get_brand_hint", args={"name": "x"})] * 6},
        default_script=_script(**_DEFAULT_VERDICT),
    )
    classifier = build_classifier(model, model_call_limit=2)
    outcome = classifier.classify_all([
        LLMTarget(normalized_name="실패표기", tx_count=3),
        LLMTarget(normalized_name="정상표기", tx_count=1),
    ])
    assert outcome.failed == ["실패표기"]
    assert list(outcome.verdicts) == ["정상표기"]


# ── 캐시 ────────────────────────────────────────────────────────────────────
def test_같은_표기를_두_번_물으면_모델을_다시_부르지_않는다():
    """같은 실행 안의 중복을 없앤다. 데모를 몇 번 돌려도 같은 화면이 나오는 근거다."""
    cache = ModelCallCache()
    model = ScriptedChatModel(default_script=_script(**_DEFAULT_VERDICT))
    classifier = build_classifier(model, cache=cache)
    target = LLMTarget(normalized_name="을지로호프", tx_count=1)

    first = classifier.classify(target)
    calls_after_first = model.calls
    second = classifier.classify(target)

    assert model.calls == calls_after_first, "캐시 히트가 모델 호출을 건너뛰지 않았다"
    assert cache.hits == calls_after_first
    assert first == second


def test_캐시_히트가_대화를_짧게_만들지_않는다():
    """같은 id 의 메시지를 두 번 넣으면 `add_messages` 가 덧붙이지 않고 덮어쓴다."""
    cache = ModelCallCache()
    model = ScriptedChatModel(default_script=_script(**_DEFAULT_VERDICT))
    classifier = build_classifier(model, cache=cache)
    target = LLMTarget(normalized_name="을지로호프", tx_count=1)
    classifier.classify(target)

    ids = [m.id for m in cache.entries[next(iter(cache.entries))].result]
    reissued = classifier.classify(target)
    assert reissued is not None
    assert all(i is not None for i in ids)  # 원본은 id 를 갖고 있고


# ── LLMToolEmulator (D-017) ─────────────────────────────────────────────────
def test_도구_실행을_에뮬레이터로_대신할_수_있다():
    """실연동에서 사전 조회가 외부 서비스가 되는 순간, 그래프 경로만 검증하는 자리다.

    우리 도구 3종은 순수 로컬 함수라 평소에는 필요하지 않다. 배선이 실제로 동작하는지만 잠근다.
    에뮬레이터 모델은 분류 모델과 **따로** 준다 — 질문이 다르면 대본도 달라야 한다.
    """
    model = ScriptedChatModel(
        default_script=[
            Turn(tool_name="lookup_alias", args={"name": "무명가맹점"}),
            verdict_turn(**_DEFAULT_VERDICT),
        ]
    )
    emulator = ScriptedChatModel(
        default_script=[Turn(text="사전 조회 결과: 비슷한 항목 없음 (에뮬레이터가 만든 응답)")]
    )
    classifier = build_classifier(
        model, emulate_tools=["lookup_alias"], emulator_model=emulator
    )
    verdict = classifier.classify(LLMTarget(normalized_name="무명가맹점", tx_count=1))

    assert verdict is not None
    assert emulator.calls == 1, "도구가 실제로 에뮬레이터로 대체되지 않았다"
