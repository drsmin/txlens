"""룰 사전 + 5단계 매칭 테스트 (설계 §3.3 ②).

여기서 막고 싶은 것은 셋이다.

① **사전이 정답 카탈로그로 자라는 것**. 사전을 손으로 고치는 일이 앞으로 계속 생기고,
   그때마다 "이것도 넣으면 점수가 오르는데" 라는 유혹이 있다. tail 브랜드나 가상 브랜드의
   법인명이 사전에 들어오면 베이스라인 (A) 가 부풀고 §5.2 전이 실험의 개선폭이 무너지므로,
   사람의 기억이 아니라 **테스트가 기계적으로 막는다.**
② **죽은 키** — 정규화 전 형태로 적힌 키는 예외 없이 그냥 안 맞는다. 로더가 실패시키는지 확인한다.
③ **단계가 조용히 사라지는 것** — 5단계 각각이 실제로 도는지, 순서가 지켜지는지 잠근다.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from txlens.generator.brands import ALL_BRANDS, PG_NAMES
from txlens.preprocess import normalize_name, preprocess_one
from txlens.rules import (
    FUZZY_CONFIDENCE_CAP,
    PG_MASKED_CANONICAL,
    STAGE_CONFIDENCE,
    RuleDictionary,
    RuleDictionaryError,
    RuleMatcher,
    load_dictionary,
)
from txlens.schemas import CategoryL1, CategoryL2, MatchStage, RawTx

#: head/tail 경계 — 사전에 들어갈 수 있는 브랜드의 유일한 기준이다.
HEAD_FREQUENCY = 1.0


@pytest.fixture(scope="module")
def dictionary() -> RuleDictionary:
    return load_dictionary()


@pytest.fixture(scope="module")
def matcher(dictionary: RuleDictionary) -> RuleMatcher:
    return RuleMatcher(dictionary)


def _tx(name: str, **kw):
    base = {
        "provider_id": "CARD_A",
        "customer_id": "SYNTH_0000",
        "approval_no": "A0001",
        "tx_datetime": datetime(2026, 3, 15, 14, 30),
        "merchant_name_raw": name,
        "amount": 12000,
    }
    base.update(kw)
    return preprocess_one(RawTx(**base))


def _all_keys(dictionary: RuleDictionary) -> list[str]:
    """사전에 적힌 모든 문자열 (regex 패턴까지). 누출 검사는 빠짐없이 봐야 한다."""
    out: list[str] = []
    for entry in dictionary.entries:
        out.extend(entry.exact_keys)
        out.extend(entry.alias)
        out.extend(entry.regex)
        out.append(entry.canonical)
    return out


# ── ① 사전의 범위를 기계적으로 강제한다 ─────────────────────────────────────
def test_사전은_head_브랜드만_담는다(dictionary: RuleDictionary):
    """tail 이 들어오면 LLM·HITL 이 일할 자리가 사라져 §5.2 전이 실험이 무의미해진다."""
    head = {b.brand for b in ALL_BRANDS if b.frequency >= HEAD_FREQUENCY}
    listed = {e.canonical for e in dictionary.entries}
    assert listed <= head, f"head 가 아닌 브랜드가 사전에 있다: {sorted(listed - head)}"
    # 반대 방향도 잠근다 — head 를 빼먹으면 베이스라인이 실제보다 나쁘게 나온다.
    assert listed == head, f"사전에서 빠진 head 브랜드: {sorted(head - listed)}"


def test_tail_브랜드의_어떤_표기도_룰에_걸리지_않는다(matcher: RuleMatcher):
    """키 목록을 문자열로 훑지 않고 **매칭을 실제로 돌려** 확인한다.

    문자열 검사(`tail 이름이 키 안에 들어 있나`)로는 두 방향의 오류가 난다. ① `스텝짐`(tail)은
    `스텝짐멤버십`(head)의 부분문자열이라 정상인 사전이 실패한다. ② 반대로 `^…` regex 는 키에
    tail 이름이 없어도 tail 표기를 삼킬 수 있어 검사를 통과해버린다. 막고 싶은 것은 "tail 이
    룰에 걸리는가"이므로 그것을 직접 묻는다.
    """
    for brand in ALL_BRANDS:
        if brand.frequency >= HEAD_FREQUENCY:
            continue
        for token in filter(None, (brand.brand, brand.corporate, brand.romanized)):
            hit = matcher.match(_tx(token))
            assert hit.match_stage is MatchStage.UNMATCHED, (
                f"tail 브랜드 {brand.brand} 의 표기 {token!r} 가 "
                f"{hit.match_stage.value} 로 잡혔다 → {hit.merchant_canonical}"
            )


def test_가상_브랜드의_법인명은_사전에_없다(dictionary: RuleDictionary):
    """가상 브랜드는 "표기만 보고는 사람도 모르는" 최고난도로 만든 함정이다.

    법인명 alias 를 넣으면 스스로 함정을 해제한다. 로마자(`BEANS ROAD`)는 사람이 읽어 옮길 수
    있으므로 허용 — 즉 여기서 막는 것은 **법인명뿐**이다.
    """
    keys = _all_keys(dictionary)
    for brand in ALL_BRANDS:
        if not brand.is_fictional or not brand.corporate:
            continue
        corp = normalize_name(brand.corporate).normalized_name
        assert corp not in keys, f"가상 브랜드 {brand.brand} 의 법인명 {corp!r} 가 사전에 있다"


def test_PG사명은_사전에_없다(dictionary: RuleDictionary):
    """PG 경유는 실가맹점이 소실된 표기다. 사전에 넣으면 브랜드를 추측하게 된다."""
    keys = _all_keys(dictionary)
    for pg in PG_NAMES:
        normalized = normalize_name(pg).normalized_name
        assert normalized not in keys, f"PG사명 {normalized!r} 가 사전에 있다"


def test_사전_라벨은_카탈로그_정답과_일치한다(dictionary: RuleDictionary):
    """사전에 라벨을 손으로 적으므로 오타가 난다. 오타 하나가 그 브랜드 전 거래를 틀리게 한다.

    카탈로그를 **비교 대상으로만** 쓴다 — 사전을 카탈로그에서 생성하지는 않는다.
    """
    truth = {b.brand: (b.l1, b.l2) for b in ALL_BRANDS}
    for entry in dictionary.entries:
        expected = truth[entry.canonical]
        assert (entry.category_l1, entry.category_l2) == expected, (
            f"{entry.canonical} 라벨이 정답과 다르다: "
            f"{entry.category_l1.value}/{entry.category_l2.value} != "
            f"{expected[0].value}/{expected[1].value}"
        )


# ── ② 로더 — 죽은 키·중복을 로딩 시점에 막는다 ──────────────────────────────
def test_사전의_모든_키는_정규화_형태다(dictionary: RuleDictionary):
    """매칭 대상은 `normalized_name` 이다. 정규화 전 형태의 키는 영원히 안 맞는다."""
    for entry in dictionary.entries:
        for key in (*entry.exact_keys, *entry.alias):
            assert normalize_name(key).normalized_name == key, f"정규화되지 않은 키: {key!r}"


def test_정규화_전_형태의_키는_로딩에서_막는다():
    with pytest.raises(RuleDictionaryError, match="정규화 형태가 아니다"):
        RuleDictionary.model_validate(
            {
                "version": 1,
                "entries": [
                    {"canonical": "스타벅스", "l1": "cafe", "l2": "cafe.coffee",
                     "alias": ["(주)에스씨케이컴퍼니"]},
                ],
            }
        )


def test_중복_키는_로딩에서_막는다():
    """나중 항목이 앞 항목을 덮어쓰면 라벨이 조용히 바뀐다 — 예외로 드러낸다."""
    with pytest.raises(RuleDictionaryError, match="중복"):
        RuleDictionary.model_validate(
            {
                "version": 1,
                "entries": [
                    {"canonical": "쿠팡", "l1": "shopping", "l2": "shopping.online",
                     "alias": ["쿠팡"]},
                    {"canonical": "쿠팡와우", "l1": "shopping", "l2": "shopping.online",
                     "alias": ["쿠팡"]},
                ],
            }
        )


def test_l2가_l1에_속하지_않으면_막는다():
    """LLM 뿐 아니라 사람도 이 실수를 한다. 스키마에서 막지 않으면 지표가 조용히 틀어진다."""
    with pytest.raises(ValueError, match="속하지 않는다"):
        RuleDictionary.model_validate(
            {
                "version": 1,
                "entries": [
                    {"canonical": "스타벅스", "l1": "food", "l2": "cafe.coffee"},
                ],
            }
        )


def test_canonical은_적지_않아도_exact_키가_된다():
    """`exact: [자기이름]` 을 매 항목에 반복시키면 반드시 빠뜨리는 항목이 생긴다."""
    d = RuleDictionary.model_validate(
        {
            "version": 1,
            "entries": [{"canonical": "스타벅스", "l1": "cafe", "l2": "cafe.coffee"}],
        }
    )
    assert d.exact_index["스타벅스"].canonical == "스타벅스"


# ── ③ 5단계 — 각 단계가 실제로 도는지 ───────────────────────────────────────
def test_exact_완전일치(matcher: RuleMatcher):
    hit = matcher.match(_tx("스타벅스"))
    assert hit.match_stage is MatchStage.EXACT
    assert hit.merchant_canonical == "스타벅스"
    assert hit.category_l2 is CategoryL2.CAFE_COFFEE
    assert hit.confidence == STAGE_CONFIDENCE[MatchStage.EXACT] == 1.0


def test_지점과_채널이_붙어도_exact로_돌아온다(matcher: RuleMatcher):
    """전처리가 벗겨낸 뒤 매칭한다 — 두 층이 맞물리는 지점이라 여기서 함께 잠근다."""
    hit = matcher.match(_tx("NAVERPAY*(주)이마트 성수점"))
    assert (hit.match_stage, hit.merchant_canonical) == (MatchStage.EXACT, "이마트")


def test_alias는_법인명과_로마자를_브랜드로_잇는다(matcher: RuleMatcher):
    """전처리는 표기만 벗기고 의미 변환은 하지 않는다. 그 몫이 사전에 있다."""
    for raw in ("(주)에스씨케이컴퍼니", "STARBUCKS", "㈜에스씨케이컴퍼니 강남역점"):
        hit = matcher.match(_tx(raw))
        assert (hit.match_stage, hit.merchant_canonical) == (MatchStage.ALIAS, "스타벅스"), raw
        assert hit.confidence == STAGE_CONFIDENCE[MatchStage.ALIAS]


def test_regno는_사전에_등록된_경우에만_돈다(matcher: RuleMatcher, dictionary: RuleDictionary):
    """v1 사전의 regno 는 비어 있다 — 그래서 이 단계는 0건을 잡는다.

    번호가 있어도 매칭이 일어나지 않는 것이 **의도된 상태**다. 사전에 등록되면 하이픈 표기가
    달라도 잡혀야 하므로, 등록된 사전을 만들어 그 경로를 확인한다.
    """
    assert dictionary.regno_index == {}, "v1 사전에 regno 를 넣었다면 설계 근거를 다시 읽어야 한다"
    assert matcher.match(_tx("알수없는가게", merchant_regno="123-45-67890")).match_stage is (
        MatchStage.UNMATCHED
    )

    registered = RuleMatcher(
        RuleDictionary.model_validate(
            {
                "version": 99,
                "entries": [
                    {"canonical": "스타벅스", "l1": "cafe", "l2": "cafe.coffee",
                     "regno": ["1234567890"]},
                ],
            }
        )
    )
    hit = registered.match(_tx("정체불명커피", merchant_regno="123-45-67890"))
    assert (hit.match_stage, hit.merchant_canonical) == (MatchStage.REGNO, "스타벅스")
    assert hit.confidence == STAGE_CONFIDENCE[MatchStage.REGNO]


def test_regex는_공백없이_붙은_지점명을_흡수한다(matcher: RuleMatcher):
    """오타로 공백이 사라지면 전처리의 지점 규칙(`\\s+…점$`)이 잡지 못한다. 그 잔여를 받는다."""
    hit = matcher.match(_tx("김밥천국종로3가점"))
    assert (hit.match_stage, hit.merchant_canonical) == (MatchStage.REGEX, "김밥천국")
    assert hit.confidence == STAGE_CONFIDENCE[MatchStage.REGEX]


def test_regex는_다른_상호를_삼키지_않는다(matcher: RuleMatcher):
    """`^CU` 를 그대로 두면 `CULTURELAND` 같은 무관한 상호를 편의점으로 만든다."""
    assert matcher.match(_tx("CULTURELAND")).match_stage is MatchStage.UNMATCHED
    assert matcher.match(_tx("CU판교점")).merchant_canonical == "CU"


def test_fuzzy는_오타를_흡수하고_신뢰도를_낮춘다(matcher: RuleMatcher):
    """접두 regex 가 없는 키(법인명 alias)의 글자 중복 오타 — 유사도만 잡을 수 있는 자리다.

    `스타벅스스` 처럼 **꼬리에 붙은** 오타는 `^스타벅스` regex 가 먼저 잡는다(더 확실한 신호다).
    유사도의 실제 몫은 이렇게 이름 중간이 망가진 경우다.
    """
    hit = matcher.match(_tx("우아한한형제들"))
    assert (hit.match_stage, hit.merchant_canonical) == (MatchStage.FUZZY, "배달의민족")
    assert 0 < hit.confidence <= FUZZY_CONFIDENCE_CAP < 1.0


def test_fuzzy_임계값_미달은_미매칭으로_남긴다(matcher: RuleMatcher):
    """가장 비슷한 것에 억지로 붙이면 정확도는 오르지만 **틀린 답이 조용히 섞인다**."""
    hit = matcher.match(_tx("처음보는가게이름"))
    assert hit.match_stage is MatchStage.UNMATCHED
    assert hit.category_l1 is CategoryL1.UNCLASSIFIED
    assert hit.category_l2 is CategoryL2.UNCLASSIFIED_INSUFFICIENT
    assert hit.confidence == 0.0
    # 못 잡은 문자열을 남겨야 HITL·승격 후보가 그것을 본다.
    assert hit.merchant_canonical == "처음보는가게이름"


def test_유사도가_모르는_가맹점을_사전의_긴_이름에_흡수하지_않는다(matcher: RuleMatcher):
    """`token_set_ratio` 를 되돌리면 여기서 깨진다.

    부분집합에 100 점을 주는 스코어러는 `STEP GYM`(사전에 없는 tail 브랜드)을
    `STEP GYM MEMBERSHIP`(사전의 구독 브랜드)으로 만든다. L2 가 같아서 **분류 지표로는 보이지도
    않지만**, 구독 탐지가 일회성 결제를 구독 결제로 세게 되어 2부가 조용히 거짓말을 한다.
    """
    for token, swallowed in (("STEP GYM", "스텝짐멤버십"), ("YOUTUBE", "유튜브프리미엄")):
        hit = matcher.match(_tx(token))
        assert hit.match_stage is MatchStage.UNMATCHED, (
            f"{token!r} 가 {hit.merchant_canonical!r} 에 흡수됐다 (기대: 미매칭, "
            f"흡수 위험 대상 {swallowed})"
        )


def test_짧은_브랜드의_오타는_유사도로_잡히지_않는다(matcher: RuleMatcher):
    """`GS225` vs `GS25` 는 token_set_ratio 88.9 로 임계값 미달이다.

    유사도의 원리적 한계이므로 **임계값을 낮춰 덮지 않는다** — 90 아래로 내리면 짧은 문자열이
    서로 붙기 시작한다. 이 케이스는 LLM·사람의 몫으로 남긴다.
    """
    assert matcher.match(_tx("GS225")).match_stage is MatchStage.UNMATCHED


def test_PG경유는_매칭하지_않고_미분류가_정답이다(matcher: RuleMatcher):
    """`UNMATCHED` 와 반드시 구분한다 — 사전을 키워도 줄지 않는 항목이기 때문이다."""
    hit = matcher.match(_tx("(주)케이지이니시스"))
    assert hit.match_stage is MatchStage.PG_MASKED
    assert hit.category_l2 is CategoryL2.UNCLASSIFIED_PG_MASKED
    assert hit.merchant_canonical == PG_MASKED_CANONICAL
    # 추측이 아니라 확정된 답이므로 신뢰도가 높다.
    assert hit.confidence == 1.0


def test_tail_브랜드는_룰이_잡지_못한다(matcher: RuleMatcher):
    """이것이 **의도된 실패**다. 여기가 LLM·HITL 이 증명해야 하는 자리다."""
    for name in ("을지로호프", "성수동카페", "영어의달인학원"):
        assert matcher.match(_tx(name)).match_stage is MatchStage.UNMATCHED, name


def test_단계_순서가_뒤집히지_않는다(matcher: RuleMatcher):
    """같은 문자열이 여러 단계에 걸릴 때 **앞 단계가 이겨야** 한다.

    `이마트` 는 exact 키이면서 `^이마트` regex 에도 걸린다. regex 가 먼저 이기면 신뢰도가
    0.85 로 떨어지고, HITL 검토 큐가 확실한 거래를 위로 올린다.
    """
    assert matcher.match(_tx("이마트")).match_stage is MatchStage.EXACT
    # alias 도 regex 보다 앞이다 — `EMART` 는 alias 키다.
    assert matcher.match(_tx("EMART")).match_stage is MatchStage.ALIAS


def test_입력_순서를_유지한다(matcher: RuleMatcher):
    names = ["스타벅스", "처음보는가게이름", "(주)이마트"]
    got = matcher.match_all([_tx(n) for n in names])
    assert [c.normalized.raw.merchant_name_raw for c in got] == names
