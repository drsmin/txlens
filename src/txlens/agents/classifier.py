"""`llm_classify` 의 속 — `create_agent` + `ToolStrategy(LLMVerdict)` (blueprint §3.0 · D-017).

## 여기가 이 프로젝트에서 유일하게 LLM 을 쓰는 자리다

문자열 정규화·룰 매칭·지표 계산·표 생성은 전부 결정론 코드다. LLM 에 묻는 질문은 하나뿐이다 —
**"이 표기는 어느 브랜드·어느 카테고리인가."** 그 하나를 위해 Agent 를 쓰는 이유는 사전과
택소노미를 프롬프트에 붓지 않고 **조회하게** 하기 위해서다 (`agents/tools.py`).

## 표기 단위로 한 번씩 invoke 한다

거래 8,969건이 아니라 미매칭 고유 표기(현재 155개)가 단위다. 배치로 묶어 한 번에 보내지 않는
이유는 셋이다. ① 표기 하나가 스키마 검증에 실패해도 나머지가 살아 있다. ② 캐시 키가 표기 하나라
승격 후 재실행에서 그대로 히트한다. ③ 실패한 표기를 이름으로 지목할 수 있다 — 배치는 "어딘가
틀렸다"만 알려준다.

`Send` 로 병렬 fan-out 하는 것은 지금 하지 않는다 (blueprint §3.3 ③ 은 그렇게 적혀 있다).
표기 순서를 고정해야 캐시 히트 수와 호출 수가 결정론이 되고, 그 수치가 곧 발표 지표다. 병렬은
지연을 줄이지만 지금 재려는 것은 지연이 아니다 — 실제 키가 붙고 155개의 지연이 문제가 되면
그때 바꾼다.

## 판정을 못 얻는 경우를 정상 경로로 다룬다

가드레일 상한(`exit_behavior="end"`)에 닿거나 모델이 구조화 출력을 내지 않으면
`structured_response` 가 없다. 그때 **추측으로 채우지 않고** 실패로 세어 미분류로 착지시킨다
(`RunMetrics.llm_failed`). 반대로 `PIIDetectionError` 는 잡지 않는다 — 그건 표기 하나의 문제가
아니라 개인정보가 새는 경로가 생겼다는 신호다 (`middleware/guardrails.py`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from txlens.agents.prompt import build_system_prompt
from txlens.agents.tools import make_tools
from txlens.middleware import ModelCallCache, build_guardrails
from txlens.rules import RuleDictionary, load_dictionary
from txlens.schemas import LLMTarget, LLMVerdict
from txlens.taxonomy import Taxonomy, load_taxonomy


def target_message(target: LLMTarget) -> HumanMessage:
    """Agent 에 보내는 유일한 입력. **표기 말고는 아무것도 넣지 않는다.**

    금액·일시·고객 식별자를 넣지 않는 것이 이 프로젝트의 전송 경계다 (blueprint §1.1).
    `tx_count` 도 넣지 않는다 — 검토 우선순위용 값이고, 모델에게는 "이게 많이 쓰인 가맹점"이라는
    편향만 준다.

    한때 이 함수는 키만 보내고 전처리 결과는 `get_brand_hint` 로 **조회하게** 했다. 실측해 보니
    그 한 번의 왕복이 표기당 모델 호출을 4회로 만들었고(시스템 프롬프트 2,600토큰이 매 호출마다
    재전송된다), 돌려받는 것은 우리가 이미 계산해 둔 값이었다. 이미 아는 사실은 **묻지 말고 적어
    준다** — 도구는 모르는 것을 찾을 때만 쓴다 (D-026).
    """
    lines = [f"정규화 키: {target.normalized_name}"]
    if target.sample_raw_name and target.sample_raw_name != target.normalized_name:
        lines.append(f"원본 표기(대표 1건): {target.sample_raw_name}")
    if target.branch:
        lines.append(f"전처리가 분리한 지점: {target.branch} (브랜드 판단에서 제외하라)")
    if target.payment_channel is not None:
        lines.append(f"전처리가 분리한 결제채널: {target.payment_channel.value}")
    return HumanMessage("\n".join(lines))


@dataclass
class ClassifyOutcome:
    """표기 묶음을 돌린 결과. **실패를 결과와 함께 돌려준다.**

    실패를 예외로 던지면 배치가 죽고, 조용히 버리면 모수가 줄어 정확도가 저절로 오른다.
    """

    verdicts: dict[str, LLMVerdict] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)


@dataclass
class MerchantClassifier:
    """분류 Agent 한 벌 + 그 실행에 딸린 캐시."""

    agent: object  # CompiledStateGraph — 제네릭 인자가 길어 여기서는 좁히지 않는다
    cache: ModelCallCache

    def classify(self, target: LLMTarget) -> LLMVerdict | None:
        """표기 하나를 분류한다. 판정을 못 얻으면 `None` — 억지로 채우지 않는다."""
        result = self.agent.invoke({"messages": [target_message(target)]})  # type: ignore[attr-defined]
        verdict = result.get("structured_response")
        return verdict if isinstance(verdict, LLMVerdict) else None

    def classify_all(self, targets: Sequence[LLMTarget]) -> ClassifyOutcome:
        """입력 순서대로 돌린다 (재현성 — `ingest`·`preprocess`·`rule_match` 와 같은 이유)."""
        outcome = ClassifyOutcome()
        for target in targets:
            verdict = self.classify(target)
            if verdict is None:
                outcome.failed.append(target.normalized_name)
            else:
                outcome.verdicts[target.normalized_name] = verdict
        return outcome


def build_classifier(
    model: str | BaseChatModel,
    *,
    dictionary: RuleDictionary | None = None,
    taxonomy: Taxonomy | None = None,
    tool_call_limit: int | None = None,
    model_call_limit: int | None = None,
    emulate_tools: list[str] | None = None,
    emulator_model: str | BaseChatModel | None = None,
    cache: ModelCallCache | None = None,
) -> MerchantClassifier:
    """분류 Agent 를 조립한다.

    `dictionary` 를 인자로 받는 것이 베이스라인 비교의 전제다 — **(B) LLM만** 은 빈 사전을 주고,
    **(C) 하이브리드** 는 v1 을 준다. 도구가 모듈 전역의 사전을 읽으면 B 를 만들 수 없다.

    `emulate_tools`: 도구 실행을 LLM 으로 대신하게 한다(`LLMToolEmulator` — D-017). 우리 도구
    3종은 순수 로컬 함수라 테스트에 에뮬레이터가 필요하지 않다. 실연동에서 사전 조회가 외부
    서비스로 바뀌는 순간 그 호출을 끊고 그래프 경로만 검증하기 위한 자리로 남겨 둔다.
    `emulator_model` 을 따로 받는 이유: 에뮬레이터는 **분류와 다른 질문**("이 도구라면 무엇을
    돌려줄까")을 하므로, 같은 모델 객체를 공유하면 분류 호출과 에뮬레이션 호출이 한 카운터·한
    대본에 섞인다.
    """
    dictionary = dictionary if dictionary is not None else load_dictionary()
    taxonomy = taxonomy if taxonomy is not None else load_taxonomy()
    cache = cache if cache is not None else ModelCallCache()

    limits: dict[str, int] = {}
    if tool_call_limit is not None:
        limits["tool_call_limit"] = tool_call_limit
    if model_call_limit is not None:
        limits["model_call_limit"] = model_call_limit

    middleware = [*build_guardrails(**limits), cache.middleware()]
    if emulate_tools:
        middleware.append(
            LLMToolEmulator(tools=emulate_tools, model=emulator_model or model)
        )

    agent = create_agent(
        model=model,
        tools=make_tools(dictionary=dictionary, taxonomy=taxonomy),
        system_prompt=build_system_prompt(taxonomy),
        # 스키마를 강제한다 — 파싱을 LLM 에게 맡기지 않는다. 택소노미 밖의 값과
        # l1/l2 불일치는 여기서 막히므로 억지 통과가 없다 (schemas._CategoryConsistency).
        response_format=ToolStrategy(LLMVerdict),
        middleware=middleware,
        name="merchant_classifier",
    )
    return MerchantClassifier(agent=agent, cache=cache)
