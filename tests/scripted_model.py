"""스크립트된 채팅 모델 — API 키 없이 Agent 를 돌리는 테스트 대역.

## 왜 픅스처가 아니라 모델을 만드는가

`langchain_core` 의 `FakeListChatModel` 류는 `bind_tools` 를 지원하지 않아 `create_agent` 에
넣으면 바로 죽는다. 도구 호출과 구조화 출력이 **이 프로젝트에서 검증하려는 바로 그 경로**이므로
그 둘을 낼 수 있는 대역이 필요하다.

## 어떻게 결정론이 되는가

입력 메시지에 이미 들어 있는 AI 메시지 수를 세어 대본의 몇 번째 차례인지 정한다. 그래서 같은
표기를 다시 물으면 같은 순서로 같은 답이 나오고, 도구 호출 → 구조화 출력의 **순서**를 잠글 수
있다. 난수도 시각도 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

#: `ToolStrategy(LLMVerdict)` 가 만드는 인공 도구의 이름 = 스키마 클래스 이름.
STRUCTURED_TOOL_NAME = "LLMVerdict"


@dataclass
class Turn:
    """모델이 한 번 호출됐을 때 낼 것. 도구 호출 또는 최종 판정 중 하나."""

    tool_name: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def to_message(self) -> AIMessage:
        if self.tool_name is None:
            return AIMessage(content=self.text)
        return AIMessage(
            content=self.text,
            tool_calls=[{"name": self.tool_name, "args": self.args, "id": f"call_{self.tool_name}"}],
        )


def verdict_turn(**fields: Any) -> Turn:
    """구조화 출력(= `LLMVerdict` 도구 호출) 차례를 만든다."""
    return Turn(tool_name=STRUCTURED_TOOL_NAME, args=fields)


class ScriptedChatModel(BaseChatModel):
    """표기별 대본대로 답한다. 대본이 없는 표기는 `default_script` 를 쓴다.

    `bound_tools` 에 바인딩된 도구 이름을 남긴다 — "구조화 출력 도구가 실제로 바인딩됐는가"를
    테스트가 확인할 수 있어야 한다. 프롬프트에 스키마를 적어 두고 되기를 바라는 것과 다르다.

    (`BaseChatModel` 이 pydantic 모델이라 `@dataclass` 를 겹치면 생성자가 깨진다 — 필드는
    pydantic 방식으로 선언한다.)
    """

    scripts: dict[str, list[Turn]] = Field(default_factory=dict)
    default_script: list[Turn] = Field(default_factory=list)
    calls: int = 0
    prompts: list[str] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    # ── BaseChatModel 규약 ──────────────────────────────────────────────────
    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any:
        self.bound_tools = [
            getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools
        ]
        # 자기 자신을 돌려준다 — 카운터·대본을 한 객체에 모아 두려면 새 인스턴스를 만들면 안 된다.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        prompt = next(
            (m.text for m in messages if isinstance(m, HumanMessage)),
            "",
        )
        self.prompts.append(prompt)
        script = self._script_for(prompt)
        turn_index = sum(1 for m in messages if isinstance(m, AIMessage))
        if turn_index >= len(script):
            raise AssertionError(
                f"대본이 {len(script)}차례뿐인데 {turn_index + 1}번째 호출이 왔다 — "
                f"Agent 가 예상보다 오래 돌고 있다 (프롬프트: {prompt!r})"
            )
        return ChatResult(generations=[ChatGeneration(message=script[turn_index].to_message())])

    # ── 내부 ────────────────────────────────────────────────────────────────
    def _script_for(self, prompt: str) -> list[Turn]:
        for name, script in self.scripts.items():
            if name in prompt:
                return script
        return self.default_script
