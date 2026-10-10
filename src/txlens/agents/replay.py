"""판정 저장·재생 — 같은 질문에 두 번 결제하지 않기 위한 층.

표기 하나를 분류하는 데 모델 호출이 3~4회 든다. 미매칭 155개면 한 번 돌 때마다 500회다.
그런데 **문턱을 정하는 일**(승격 하한을 얼마로 둘 것인가)은 판정을 새로 얻는 일이 아니라 이미
받은 판정을 정답과 대조하는 일이다. 문턱 후보마다 모델을 다시 부르면 같은 답에 같은 돈을
반복해서 낸다.

그래서 판정을 파일로 남기고, 그 파일로 파이프라인을 다시 돌린다.

## 재생은 실행 경로를 바꾸지 않는다

`ReplayClassifier` 는 `MerchantClassifier` 와 같은 자리에 꽂힌다 — 그래프도, 노드도, 지표
계산도 그대로다. 바뀌는 것은 "판정을 어디서 얻는가" 하나뿐이라, 재생으로 낸 지표와 실호출로
낸 지표를 나란히 놓을 수 있다.

**대본에 없는 표기는 실패로 센다.** 없는 표기를 조용히 건너뛰면 그 표기가 분모에서 사라져
정확도가 저절로 오르고, 재생 결과가 실호출 결과보다 좋아 보인다.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from txlens.agents.classifier import ClassifyOutcome
from txlens.schemas import LLMTarget, LLMVerdict


@dataclass
class ReplayCounters:
    """`ModelCallCache` 자리에 꽂히는 계수기. 재생이므로 **모델 호출은 0 이다.**

    0 을 그대로 보고한다 — 재생 실행의 지표에 실호출 수를 베껴 넣으면 "이번 실행이 얼마나
    들었나"라는 질문에 거짓으로 답하게 된다.
    """

    calls: int = 0
    hits: int = 0


@dataclass
class ReplayClassifier:
    """저장된 판정으로 답하는 분류기. API 를 부르지 않는다."""

    verdicts: Mapping[str, LLMVerdict]
    cache: ReplayCounters = field(default_factory=ReplayCounters)

    def classify(self, target: LLMTarget) -> LLMVerdict | None:
        return self.verdicts.get(target.normalized_name)

    def classify_all(self, targets: Sequence[LLMTarget]) -> ClassifyOutcome:
        outcome = ClassifyOutcome()
        for target in targets:
            verdict = self.verdicts.get(target.normalized_name)
            if verdict is None:
                outcome.failed.append(target.normalized_name)
            else:
                outcome.verdicts[target.normalized_name] = verdict
                self.cache.hits += 1
        return outcome


def save_verdicts(verdicts: Mapping[str, LLMVerdict], path: Path, *, model: str) -> None:
    """판정을 JSON 으로 남긴다. **어느 모델의 답인지 함께 적는다.**

    모델이 바뀌면 판정도 바뀐다. 모델 이름이 없는 판정 파일은 6개월 뒤에 무엇으로 잰 수치인지
    말할 수 없고, 그러면 그 위에서 정한 문턱도 근거를 잃는다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "verdicts": {name: v.model_dump(mode="json") for name, v in sorted(verdicts.items())},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_verdicts(path: Path) -> tuple[dict[str, LLMVerdict], str]:
    """판정 파일을 읽어 `(판정, 모델명)` 을 돌려준다."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdicts = {
        name: LLMVerdict.model_validate(raw) for name, raw in payload["verdicts"].items()
    }
    return verdicts, payload.get("model", "unknown")
