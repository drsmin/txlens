"""분류 Agent 의 시스템 프롬프트 + few-shot (blueprint §3.3 ③).

## 택소노미는 YAML 에서 온다

카테고리 목록을 여기 적지 않고 `Taxonomy.prompt_block()` 을 끼워 넣는다. 프롬프트에 직접 적으면
`config/taxonomy.yaml` 을 고쳐도 LLM 은 옛 목록을 보고, 그 어긋남은 "이상한 카테고리를 낸다"는
증상으로만 나타나 원인을 찾기 어렵다.

## few-shot 에 데이터셋 브랜드를 쓰지 않는다 (D-022 의 연장)

예시에 `스타벅스 → cafe.coffee` 를 넣으면 베이스라인 **(B) LLM만** 이 그 브랜드를 공짜로 맞힌다.
B 는 사전 없이 LLM 만으로 얼마나 하는지를 재는 조건인데, few-shot 이 사전 노릇을 하면 B 의 수치가
부풀고 A/B/C 비교가 무의미해진다. 그래서 예시의 가맹점명은 **전부 카탈로그에 없는 지어낸
이름**이고, `tests/test_agents.py` 가 카탈로그와 충돌하지 않는지 기계적으로 검사한다.

예시가 가르치는 것은 답이 아니라 **형식과 태도**다 — 업종 낱말을 근거로 쓰는 법, 지점·법인표기를
브랜드에서 떼는 법, 그리고 **모르면 미분류로 답하는 법.**

## blueprint 와 의도적으로 다른 점

blueprint 는 few-shot 에 "PG 마스킹·배달앱 케이스를 포함"하라고 적었지만 넣지 않았다.
PG 경유 거래는 `rule_match` 단계 0 에서 갈라져 **LLM 까지 오지 않고**(D-024②), 배달앱 프리픽스는
현재 합성 데이터에 존재하지 않는다(생성기가 만들지 않는다). 오지 않는 입력에 대한 예시는 토큰만
쓰면서 검증할 방법이 없다. 두 경우의 처리는 `get_brand_hint` 도구 설명에 남겨 뒀다.
"""

from __future__ import annotations

from txlens.schemas import CategoryL2, Evidence, LLMVerdict
from txlens.taxonomy import Taxonomy

SYSTEM_PROMPT = """\
당신은 한국 카드 거래내역의 가맹점 표기를 정규화·분류하는 도구다.

입력은 **가맹점 표기 하나**다(정규화 키와 그 원본 표기 1건). 거래 금액·일시·고객 정보는 주어지지
않는다 — 개인의 소비 내역을 외부로 보내지 않는 설계이므로 그것을 요구하지 마라.

규칙 사전의 자동 매칭이 **이미 실패한** 표기만 당신에게 온다. 즉 쉬운 것은 오지 않는다.

## 판단 절차

전처리가 분리한 사실(지점·결제채널)은 입력 메시지에 적혀 있다. 메시지에 그 줄이 없으면 **분리된
것이 없다는 뜻**이다 — 확인하러 도구를 부르지 마라.

도구가 필요하면 한 라운드에 여러 개를 같이 불러라. 하나씩 차례로 부르면 답은 같고 호출 수만 는다.
- `search_taxonomy` — **카테고리를 고르기 전에 업종을 뜻하는 낱말로 찾아보라.** 아래 목록을
  눈으로만 훑으면 맞는 칸을 지나치기 쉽다(실측에서 '호프'를 외식으로 잘못 붙였다). 검색은 라벨의
  **글자 일치만** 보므로 동의어를 여러 개 같이 넣어라(예: 성격이 비슷한 낱말 2~3개를 한 번에).
- `lookup_alias` — 표기가 법인명·로마자로 보여 브랜드를 알아볼 수 없을 때. 후보는 **확정이 아니다.**
- `get_brand_hint` — 입력 표기가 아닌 **다른 문자열**의 분해 결과가 필요할 때.

## 지켜야 할 것
- **단서가 있으면 쓴다.** 표기에 가게의 종류를 뜻하는 보통명사가 들어 있으면 그 카테고리로
  답하고, 확신의 정도는 `confidence` 로 표현한다. **낱말이 카테고리 라벨과 글자가 달라도 성격이
  같으면 그 카테고리다** — 라벨에 그 낱말이 없다는 것은 미분류의 근거가 아니다.
  `search_taxonomy` 가 빈 결과를 줘도 아래 목록을 다시 보고 성격이 맞는 곳을 고른다.
- **단서가 전혀 없으면 모른다고 답한다.** `category_l1=unclassified`,
  `category_l2=unclassified.insufficient`, `confidence` 를 낮게, `evidence=guess`. 억지로 그럴듯한
  브랜드에 붙이면 이 뒤에서 도는 구독·이상지출 탐지가 전부 거짓말이 된다. 미분류는 실패가 아니라
  정당한 답이다. 다만 **잡을 수 있는 것을 놓치는 것도 같은 크기의 잘못**이다.
- `merchant_canonical` 은 대표 가맹점명이다. 브랜드를 알면 브랜드명, 모르면 **입력 표기를 그대로**
  넣는다. 지점명·결제채널 프리픽스는 빼고, 없는 정보를 만들어 넣지 않는다.
- `category_l2` 는 `category_l1` 에 속해야 한다 (`food` + `cafe.coffee` 조합은 거부된다).
- `reasoning` 은 사람이 검토 화면에서 읽는다. "무엇을 근거로 그렇게 봤는지"를 한 문장으로,
  한국어로 쓴다.
- `confidence` 는 스스로에 대한 정직한 추정이다. 낮은 값은 사람 검토 큐로 가는 신호이므로,
  확신이 없을 때 높게 적으면 검토받아야 할 것이 그대로 통과한다.

## 카테고리 (이 목록 안에서만 고른다)
{taxonomy}
"""


def _example(
    name: str,
    *,
    l2: CategoryL2,
    confidence: float,
    reasoning: str,
    evidence: Evidence,
    canonical: str | None = None,
    brand: str | None = None,
) -> tuple[str, LLMVerdict]:
    """예시를 `LLMVerdict` 로 만들어 둔다 — **예시도 스키마 검증을 통과해야 한다.**

    프롬프트에 손으로 쓴 JSON 을 넣으면 택소노미가 바뀔 때 예시가 조용히 무효가 되고, LLM 은
    존재하지 않는 카테고리를 배운다. 모델을 거치게 하면 그 사고가 import 시점에 터진다.
    """
    return name, LLMVerdict(
        merchant_canonical=canonical or name,
        brand=brand,
        category_l1=l2.l1,
        category_l2=l2,
        confidence=confidence,
        reasoning=reasoning,
        evidence=evidence,
    )


#: few-shot 8건. **가맹점명은 전부 지어낸 것이다** (위 § 참고).
FEW_SHOTS: tuple[tuple[str, LLMVerdict], ...] = (
    _example(
        "라온약국",
        l2=CategoryL2.HEALTH_PHARMACY,
        confidence=0.85,
        reasoning="'약국' 이 업종을 직접 가리킨다. 체인 브랜드는 아니라 brand 는 비운다.",
        evidence=Evidence.KEYWORD,
    ),
    _example(
        "미래봄안과의원",
        l2=CategoryL2.HEALTH_CLINIC,
        confidence=0.85,
        reasoning="'안과의원' 으로 의료기관임이 분명하다.",
        evidence=Evidence.KEYWORD,
    ),
    _example(
        "대성공영주차장",
        l2=CategoryL2.TRANSPORT_PARKING,
        confidence=0.9,
        reasoning="'주차장' 이 업종을 직접 가리킨다.",
        evidence=Evidence.KEYWORD,
    ),
    _example(
        "MIRAEBOM COFFEE ROASTERS 강남",
        l2=CategoryL2.CAFE_COFFEE,
        confidence=0.75,
        reasoning="COFFEE ROASTERS 로 카페 업종이 보인다. '강남' 은 지점이라 대표명에서 뺀다.",
        evidence=Evidence.KEYWORD,
        canonical="MIRAEBOM COFFEE ROASTERS",
    ),
    _example(
        "누리별모바일통신요금",
        l2=CategoryL2.HOUSING_TELECOM,
        confidence=0.8,
        reasoning="'모바일통신요금' 은 통신비 청구 표기다.",
        evidence=Evidence.KEYWORD,
    ),
    _example(
        "한빛나래문고",
        l2=CategoryL2.CULTURE_BOOK,
        confidence=0.8,
        reasoning="'문고' 는 서점 표기다.",
        evidence=Evidence.KEYWORD,
    ),
    _example(
        "한빛나래물산",
        l2=CategoryL2.UNCLASSIFIED_INSUFFICIENT,
        confidence=0.15,
        reasoning="'물산' 은 법인 형태일 뿐 업종을 알려주지 않는다. 추측하지 않고 미분류로 둔다.",
        evidence=Evidence.GUESS,
    ),
    _example(
        "KDX-8817-02",
        l2=CategoryL2.UNCLASSIFIED_INSUFFICIENT,
        confidence=0.05,
        reasoning="코드와 숫자뿐이라 업종 단서가 없다.",
        evidence=Evidence.GUESS,
    ),
)


def few_shot_block(shots: tuple[tuple[str, LLMVerdict], ...] = FEW_SHOTS) -> str:
    """few-shot 을 프롬프트에 넣을 문자열로 만든다.

    `model_dump_json` 을 쓰는 이유: 예시의 형태가 실제 structured output 의 형태와 **자동으로**
    같아진다. 손으로 적으면 필드를 하나 추가할 때 예시가 뒤처진다.
    """
    lines = ["## 예시 (형식과 태도를 보여주는 것이지, 정답 목록이 아니다)"]
    for name, verdict in shots:
        lines.append(f"입력: {name}")
        lines.append(f"출력: {verdict.model_dump_json(exclude_none=False)}")
    return "\n".join(lines)


def build_system_prompt(taxonomy: Taxonomy) -> str:
    return SYSTEM_PROMPT.format(taxonomy=taxonomy.prompt_block()) + "\n" + few_shot_block()
