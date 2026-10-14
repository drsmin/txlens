"""LangGraph state — 노드가 서로에게 넘기는 것 (설계 §3.2).

## reducer 를 습관적으로 붙이지 않는다

`Annotated[..., operator.add]` 는 편하지만 **누적**이다. 거래 리스트는 8,969건이라 단계마다
누적하면 같은 거래가 상태 안에 여러 벌 쌓이고, 체크포인터를 붙이면 그 복사본이 그대로 저장된다.
그래서 셋으로 나눈다.

| 성질 | 필드 | reducer |
| --- | --- | --- |
| 단계별 산출물 — 다음 단계가 **대체**한다 | `raw`·`pairings`·`normalized`·`matched`·`confirmed` | 없음(교체) |
| 여러 노드·여러 라운드가 조금씩 **보탠다** | `review_queue`·`human_corrections`·`rule_proposals` | `operator.add` |
| 표기 단위로 쪼개 돌린 결과를 **합친다** | `llm_results` | `merge_verdicts` |

## 아직 없는 필드는 넣지 않는다

설계 §3.2 는 `insights` 도 적어 두었지만, 그것을 쓰는 노드가 생기면 그때 넣는다.
**생산자도 소비자도 없는 필드는 파이프라인이 하는 일을 과장한다** — 상태 정의만 보고
"이미 붙어 있다"고 읽히면 그게 곧 거짓 보고다. `human_corrections` 는 5주차에 생산자
(`human_review`)와 소비자(`rule_promote`)가 동시에 생겨서 들어왔다.

## 상태를 둘로 가른다 — `TxState` 와 `ReviewState`

검토 서브그래프는 거래를 보지 않는다. 보는 것은 `review_queue`(표기 수십 개)와 그것을 승격에
잇는 데 필요한 `llm_targets`/`llm_results` 뿐이다. 그래서 상태 스키마를 따로 둔다.

이유는 두 가지고, 둘째가 더 크다.
① `interrupt()` 로 멈춘 지점은 체크포인터에 저장되는데, 서브그래프까지 거래 8,969건을 들고
   들어가면 사람이 한 줄 고칠 때마다 그 전량이 직렬화된다.
② **화면과 배치가 같은 것을 부를 수 있다.** Streamlit S1 은 번들이 없어도 검토 큐만 있으면
   서브그래프를 그대로 돌릴 수 있다 — 화면용 코드를 따로 쓰면 데모와 배치가 갈라진다.

겹치는 키는 LangGraph 가 부모↔서브 사이에서 이름으로 이어 준다. 그래서 서브그래프가 돌려준
`rule_proposals`·`human_corrections` 는 부모 상태에도 그대로 얹힌다.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from txlens.schemas import (
    CancelPairing,
    ClassifiedTx,
    HumanCorrection,
    LLMTarget,
    LLMVerdict,
    NormalizedTx,
    RawTx,
    ReviewItem,
    RuleProposal,
)


def merge_verdicts(
    left: dict[str, LLMVerdict], right: dict[str, LLMVerdict]
) -> dict[str, LLMVerdict]:
    """표기 → 판정 사전을 병합한다. 나중 값이 이긴다 (재분류·사람 교정이 덮어쓴다)."""
    return {**left, **right}


class RunMetrics(BaseModel):
    """한 번의 실행이 **무엇을 얼마나 했는지.** 정확도 지표(`EvalResult`)와 다른 물건이다.

    정확도는 "얼마나 맞혔나"고 이쪽은 "얼마나 들었나"다. 둘을 한 객체에 섞으면 비용을 줄인
    변경이 정확도 표에 섞여 들어와 무엇이 개선인지 못 읽는다. 승격 루프의 핵심 주장인
    **"라운드가 돌면 LLM 호출이 줄어든다"** 는 여기 `llm_targets` 로 증명한다 — 사전이 커지면
    미매칭이 줄고, 미매칭이 줄면 보낼 표기가 줄어든다.
    """

    tx_total: int = 0
    tx_classifiable: int = 0
    #: 룰이 브랜드를 확정한 거래 수 (PG 마스킹·미매칭 제외).
    rule_resolved: int = 0
    #: LLM 에 보낸 고유 표기 수. 승격 효과가 나타나는 자리다.
    llm_targets: int = 0
    #: 상한(`llm_target_limit`) 때문에 **보내지 않은** 표기 수. 조용히 잘라내면 "LLM 이 못 잡았다"로
    #: 오해되므로 반드시 함께 보고한다.
    llm_targets_skipped: int = 0
    #: 실제 모델 호출 수. 표기 수와 다르다 — 도구를 쓰면 표기 하나에 두 번 이상 호출된다.
    llm_model_calls: int = 0
    llm_cache_hits: int = 0
    #: LLM 이 **모른다고 답한** 표기 수 (`category_l1=unclassified`). 실패가 아니라 정직한 답이다.
    llm_unknown: int = 0
    #: 가드레일 발동·스키마 검증 실패로 판정을 얻지 못한 표기 수. **0 이라고 가정하지 않는다.**
    llm_failed: int = 0
    review_queued: int = 0
    #: 사람이 확정한 교정 수. `review_queued` 와 나란히 둬야 **큐에 쌓인 것과 실제로 처리한 것**이
    #: 한 줄에서 비교된다 — 큐 길이만 보고하면 HITL 이 도는 것처럼 보인다.
    human_corrected: int = 0
    stage_counts: dict[str, int] = Field(default_factory=dict)

    def report(self) -> str:
        skipped = f" · 상한으로 미전송 {self.llm_targets_skipped}개" if self.llm_targets_skipped else ""
        return (
            f"  실행  전체 {self.tx_total}건 → 분류 대상 {self.tx_classifiable}건 "
            f"(룰 확정 {self.rule_resolved}건)\n"
            f"        LLM 표기 {self.llm_targets}개{skipped} · 모델 호출 "
            f"{self.llm_model_calls}회 (캐시 히트 {self.llm_cache_hits})\n"
            f"        LLM 이 모른다고 답함 {self.llm_unknown}개 · 판정 실패 {self.llm_failed}개 · "
            f"검토 큐 {self.review_queued}개 (사람 교정 {self.human_corrected}개)"
        )


class TxState(TypedDict, total=False):
    """그래프 전역 상태. `total=False` 로 둬 노드가 **바꾼 것만** 돌려주게 한다."""

    #: 입력 — 번들 디렉터리. `ingest` 노드가 여기서 읽는다.
    bundle_dir: str
    raw: list[RawTx]
    #: `pair_cancellations` → `normalize` 사이의 표식. `raw` 인덱스로 행을 지목한다
    #: (`CancelPairing` — id() 는 직렬화를 못 넘는다).
    pairings: list[CancelPairing]
    normalized: list[NormalizedTx]
    #: 룰 단계까지의 결과. 미매칭·PG 마스킹도 들어 있다 (모수를 잃지 않는다).
    matched: list[ClassifiedTx]
    llm_targets: list[LLMTarget]
    llm_results: Annotated[dict[str, LLMVerdict], merge_verdicts]
    #: 판정을 얻지 못한 표기 이름. 개수만 세지 않고 **이름을 남긴다** — 어느 표기가 왜 실패했는지
    #: 물어볼 수 없으면 고칠 수도 없다.
    llm_failed: Annotated[list[str], operator.add]
    review_queue: Annotated[list[ReviewItem], operator.add]
    #: 사람이 확정한 교정. 검토 서브그래프가 낳고 `rule_promote` 가 먹는다.
    human_corrections: Annotated[list[HumanCorrection], operator.add]
    rule_proposals: Annotated[list[RuleProposal], operator.add]
    #: 출력 — 룰 결과에 LLM 판정을 얹은 최종 분류. 평가 하네스가 이것을 본다.
    confirmed: list[ClassifiedTx]
    metrics: RunMetrics


class ReviewState(TypedDict, total=False):
    """검토 서브그래프의 상태 — **거래가 없다**.

    필드 이름은 `TxState` 와 일부러 같게 둔다. LangGraph 는 부모 그래프와 서브그래프를 **키
    이름으로** 잇기 때문에, 이름이 같으면 배선 코드가 한 줄도 필요 없고 이름이 다르면 변환
    함수를 둬야 한다 — 그 변환 함수가 곧 데모와 배치가 갈라지는 자리다.

    `llm_targets`·`llm_results` 를 들고 있는 것은 `rule_promote` 가 자동 승격(고신뢰 판정)과
    사람 승격(교정)을 **한 자리에서** 합치기 때문이다. 둘을 다른 노드로 쪼개면 같은 사전을
    두 번 읽고, 두 결과가 어긋났을 때 어느 쪽이 이겼는지 알 수 없다.
    """

    llm_targets: list[LLMTarget]
    llm_results: Annotated[dict[str, LLMVerdict], merge_verdicts]
    review_queue: Annotated[list[ReviewItem], operator.add]
    #: `confidence_gate` 가 잘라낸 **이번에 사람에게 보일 몫.** 큐 전체가 아니다 — 사람이 한
    #: 번에 볼 수 있는 양을 넘기면 화면이 답을 요구하지 않는 목록으로 바뀐다.
    review_batch: list[ReviewItem]
    human_corrections: Annotated[list[HumanCorrection], operator.add]
    rule_proposals: Annotated[list[RuleProposal], operator.add]


class ReviewOutput(TypedDict, total=False):
    """서브그래프가 부모에게 **돌려주는 것.** 받은 것을 되돌려주지 않는다.

    출력 스키마를 따로 적지 않으면 서브그래프는 최종 상태 전체를 돌려주고, 부모는 그것을
    자기 reducer 로 받는다. `review_queue` 는 `operator.add` 라서 **받은 큐가 그대로 한 번 더
    더해진다** — 실측에서 검토 큐 3개가 6개로 보고됐다. 지표가 두 배가 되는 종류의 버그는
    "많이 검토했다"처럼 보이기 때문에 실행이 성공한 것으로 오해된다.

    그래서 나가는 것은 이 서브그래프가 **새로 만든 둘**뿐이다.
    """

    human_corrections: Annotated[list[HumanCorrection], operator.add]
    rule_proposals: Annotated[list[RuleProposal], operator.add]
