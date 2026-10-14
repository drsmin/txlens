"""모델 호출 캐시 — `wrap_model_call` 미들웨어 (설계 §3.0).

## 왜 캐시가 미들웨어인가

같은 표기를 두 번 물으면 같은 답이 와야 한다. `temperature=0` 으로도 완전한 재현은 보장되지
않으므로(공급자 쪽 변동이 있다), **요청이 같으면 모델을 부르지 않는 층**을 하나 둔다. 이 층이
있으면 데모를 몇 번 돌려도 같은 화면이 나오고, 평가를 반복해도 지표가 흔들리지 않는다.

`wrap_model_call` 로 만드는 이유는 캐시 히트가 **모델 호출 자체를 건너뛴다**는 걸 코드 모양으로
보이기 위해서다. Agent 안에서 호출 직전에 가로채므로 도구 호출 루프 중간의 반복도 잡힌다.

## 이 캐시가 증명하지 않는 것

"라운드가 돌면 LLM 호출이 줄어든다"는 프로젝트의 주장은 **이 캐시가 아니라 룰 승격**이 만든다
(설계 §3.3 ⑤). 사전이 커지면 미매칭이 줄고, 미매칭이 줄면 애초에 보낼 표기가 줄어든다.
캐시는 같은 실행 안의 중복만 없앤다 — 둘을 섞어 보고하면 승격 효과를 부풀리게 된다.
그래서 `RunMetrics` 는 `llm_targets`(승격이 줄이는 값)와 `llm_cache_hits` 를 따로 센다.

## 캐시 키

`(모델 이름, 시스템 메시지, 대화 메시지의 종류+텍스트, 도구 이름 목록)`. 메시지 id 나 호출 id 는
매번 달라지므로 키에 넣지 않는다 — 넣으면 캐시가 영원히 비어 있고, 그 사실이 눈에 안 띈다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain_core.messages import BaseMessage


def _model_id(request: ModelRequest) -> str:
    """모델을 **안정된 문자열**로 식별한다.

    `str(model)` 을 쓰면 안 된다 — pydantic 모델의 repr 은 필드를 전부 펼치므로, 카운터 하나만
    들고 있는 모델도 호출마다 다른 문자열이 된다. 그러면 캐시 키가 매번 달라져 **캐시가 영원히
    비어 있고**, 히트가 0 이라는 사실은 지표를 들여다보지 않으면 눈에 띄지 않는다.
    (테스트를 쓰면서 실제로 이렇게 났다.)
    """
    for attr in ("model_name", "model", "name"):
        value = getattr(request.model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(request.model).__name__


def _message_key(message: BaseMessage) -> list[object]:
    """메시지 하나를 캐시 키로 만든다. **id 는 제외한다** (매번 새로 발급된다)."""
    tool_calls = [
        [call.get("name"), json.dumps(call.get("args"), ensure_ascii=False, sort_keys=True)]
        for call in getattr(message, "tool_calls", []) or []
    ]
    return [message.type, message.text, tool_calls]


@dataclass
class ModelCallCache:
    """모델 호출 결과를 요청 단위로 기억한다. 히트·호출 수를 센다.

    미들웨어 객체가 카운터를 들고 있는 이유: 그래프 상태에 넣으면 노드가 아닌 곳(Agent 내부)에서
    증가하는 값이 상태에 섞여 reducer 를 복잡하게 만든다. 실행 하나에 캐시 하나를 만들고
    `RunMetrics` 로 옮겨 담는 편이 단순하다.
    """

    #: 요청 해시 → 응답. 프로세스 안에서만 산다 (`InMemory*` 와 같은 수명).
    entries: dict[str, ModelResponse] = field(default_factory=dict)
    calls: int = 0
    hits: int = 0

    def key(self, request: ModelRequest) -> str:
        payload = json.dumps(
            {
                "model": _model_id(request),
                "system": request.system_message.text if request.system_message else None,
                "messages": [_message_key(m) for m in request.messages],
                "tools": sorted(t.name for t in request.tools),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def middleware(self) -> AgentMiddleware:
        @wrap_model_call(name="ModelCallCache")
        def cached(request: ModelRequest, handler):  # type: ignore[no-untyped-def]
            key = self.key(request)
            cached_response = self.entries.get(key)
            if cached_response is not None:
                self.hits += 1
                return _reissue(cached_response)
            self.calls += 1
            response = handler(request)
            self.entries[key] = response
            return response

        return cached


def _reissue(response: ModelResponse) -> ModelResponse:
    """캐시된 응답의 메시지 id 를 새로 발급해 돌려준다.

    같은 id 의 메시지를 상태에 두 번 넣으면 `add_messages` reducer 가 **덧붙이지 않고 덮어쓴다.**
    한 실행 안에서 같은 요청이 두 번 나는 경우(도구 루프가 제자리를 도는 경우)에 대화가 조용히
    짧아지는 것을 막는다.
    """
    return ModelResponse(
        result=[m.model_copy(update={"id": None}) for m in response.result],
        structured_response=response.structured_response,
    )
