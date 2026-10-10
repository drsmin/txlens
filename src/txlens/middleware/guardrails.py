"""가드레일 미들웨어 3종 (blueprint §3.0 · D-017).

## 무엇을 막는가

| 미들웨어 | 막는 사고 |
| --- | --- |
| `PIIMiddleware("credit_card", block)` | 카드번호가 프롬프트에 섞여 외부로 나간다 |
| `PIIMiddleware("customer_id", block)` | 고객 식별자가 섞여 나간다 — 정규화 키 단위 전송의 전제가 깨진다 |
| `ToolCallLimitMiddleware` | 도구 호출 루프가 돌며 비용이 무한히 늘어난다 |
| `ModelCallLimitMiddleware` | 모델 호출 루프가 같은 방식으로 늘어난다 |

## `strategy="block"` 을 쓰는 이유 — 조용히 지우면 사고를 못 본다

`redact`/`mask` 는 값을 가리고 **호출을 계속한다.** 그러면 "가맹점 표기만 보낸다"는 설계가 깨진
사실이 로그에도 지표에도 남지 않는다. `block` 은 `PIIDetectionError` 를 던져 그 자리에서 멈춘다.
신용정보법 아래에서 다루는 데이터라 "일단 보내고 나중에 가리기"는 선택지가 아니다.

⚠️ 그래서 이 예외는 **잡아서 미분류로 처리하지 않는다.** 발동했다면 우리 파이프라인에 개인정보가
새는 경로가 생겼다는 뜻이고, 그건 미분류 한 건이 아니라 고쳐야 할 버그다.

## 호출 상한은 `thread_limit` 이 아니라 `run_limit` 으로 준다

blueprint 초안은 `thread_limit=20` 이었다. 하지만 분류 Agent 는 **표기 하나당 한 번 invoke** 하고
체크포인터를 붙이지 않으므로 스레드가 곧 런이다 — `thread_limit` 을 적으면 "20" 이 무엇에 대한
상한인지가 코드에서 안 보인다. 표기 하나에 도구를 몇 번까지 쓰게 할지가 실제로 관리하려는 값이라
`run_limit` 으로 적는다. HITL 체크포인터는 **그래프 층**에 붙고 Agent 안에는 없다 (blueprint §3.0).
"""

from __future__ import annotations

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
)

#: 합성 고객 식별자 형태 (`SYNTH_0000`). 생성기가 이 형태로만 만든다.
#: 실연동 시에는 기관이 주는 식별자 형태로 **바꾸는 것이 아니라 추가한다** — 형태가 늘 뿐
#: 막아야 하는 대상은 줄지 않는다.
CUSTOMER_ID_PATTERN = r"\bSYNTH_\d{4}\b"

#: 표기 하나를 판단하는 데 허용하는 도구 호출 수.
DEFAULT_TOOL_CALL_LIMIT = 8
#: 표기 하나에 허용하는 모델 호출 수. **정상 경로는 1~2회**다(프롬프트에 카테고리 목록이 이미
#: 있으므로 도구 없이 답할 수 있다). 상한을 그 값에 딱 맞추면 안 된다 — 실측에서 4로 두었더니
#: 동의어를 찾아 두 라운드 탐색한 표기가 **정답을 손에 쥐고도** 상한에서 잘렸다(D-026).
#: 상한은 폭주를 막는 장치이고, 정상 경로를 자르는 장치가 아니다.
DEFAULT_MODEL_CALL_LIMIT = 6


def build_guardrails(
    *,
    tool_call_limit: int = DEFAULT_TOOL_CALL_LIMIT,
    model_call_limit: int = DEFAULT_MODEL_CALL_LIMIT,
) -> list[AgentMiddleware]:
    """가드레일 목록. 순서는 유지한다 — PII 검사가 상한 검사보다 먼저 돌아야 한다.

    `exit_behavior="end"` 는 상한에 닿으면 **에러 대신 종료**한다. 그러면
    `structured_response` 가 없는 결과가 돌아오고, 호출자(`MerchantClassifier`)가 그것을
    판정 실패로 세어 미분류로 착지시킨다. 8,969건을 도는 배치가 표기 하나 때문에 죽으면
    안 되지만, 그 표기가 조용히 "분류됨"으로 세어지는 것도 안 된다.
    """
    return [
        PIIMiddleware("credit_card", strategy="block"),
        PIIMiddleware("customer_id", detector=CUSTOMER_ID_PATTERN, strategy="block"),
        ToolCallLimitMiddleware(run_limit=tool_call_limit, exit_behavior="end"),
        ModelCallLimitMiddleware(run_limit=model_call_limit, exit_behavior="end"),
    ]
