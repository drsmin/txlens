"""화면이 배치와 **같은 코드**를 부르게 하는 얇은 층 (설계 §3.5).

여기 있는 것은 네 가지뿐이다 — 무엇을 불러올지(재생 대본·사전·실측 스냅샷), 표와 객체
사이의 변환, 한 라운드의 결과를 요약하는 것, 그리고 축이 틀리지 않은 차트를 만드는 것.
**분류도 승격도 지표 계산도 여기서 하지 않는다.** 그 셋을 화면 쪽에 한 줄이라도 베껴 두면
데모에서 본 숫자와 `python -m txlens.eval` 이 낸 숫자가 갈라지고, 그때 어느 쪽이 맞는지
말할 방법이 없다.

## 화면은 API 를 부르지 않는다

기본 분류기는 `ReplayClassifier` 다 (`data/gold/verdicts-*.json`). 발표장에서 네트워크·API
오류가 데모를 죽이는 것을 막는 장치이자(설계 §3.5 데모 안정성), 같은 장면을 몇 번이고
똑같이 재현하기 위한 장치다. 재생과 실호출의 차이는 **라운드 2 에서만** 나타난다 — 재생
대본은 v1 사전 위에서 받은 답이라 커진 사전을 도구로 조회하지 못한다(`agents/replay.py`).

## 순수 함수로 갈라 둔 이유

`rows_to_corrections`·`summarize_promotion` 은 Streamlit 을 import 하지 않는다. 화면 코드
안에 두면 테스트할 수 있는 자리가 없어지는데, 이 둘은 **사람이 고친 것이 사전에 실제로
반영되는가**를 결정하는 지점이라 조용히 틀리면 화면만 "반영됨"이라고 적는다.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from txlens.agents.replay import ReplayClassifier, load_verdicts
from txlens.detect import spend_rows
from txlens.detect.subscription import monthly_fixed_cost
from txlens.graph import build_checkpointer, build_graph
from txlens.rules import RuleDictionary, RuleMatcher
from txlens.schemas import CategoryL2, HumanCorrection, RuleProposal
from txlens.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "data" / "synthetic" / "bundle"
#: 기본 재생 대본. 파일 이름에 모델이 들어 있고, `load_verdicts` 가 그 이름을 함께 돌려준다.
VERDICTS_PATH = ROOT / "data" / "gold" / "verdicts-v1-gpt-4o-mini.json"

#: 표에 보일 열 이름. 화면과 테스트가 같은 문자열을 쓰게 상수로 둔다.
COL_APPLY = "반영"
COL_NAME = "표기"
COL_CANONICAL = "대표명(교정)"
COL_CATEGORY = "분류(교정)"


# ── 택소노미 선택지 ─────────────────────────────────────────────────────────
def category_choices() -> dict[str, str]:
    """`"외식 · food.restaurant"` → `"food.restaurant"`.

    한글 라벨만 보이면 발표 청중에게 읽히고, id 를 함께 붙여 두면 고른 값이 그대로 골드셋·
    프롬프트와 같은 키라는 것이 화면에서 보인다. 라벨은 `config/taxonomy.yaml` 이 원본이다 —
    화면에 한글을 다시 적으면 택소노미를 고칠 때 화면만 옛 이름을 말한다.
    """
    taxonomy = load_taxonomy()
    return {f"{taxonomy.label(c.value)} · {c.value}": c.value for c in CategoryL2}


def category_choice_of(category_id: str) -> str:
    """id → 화면 표시 문자열. `category_choices()` 의 역방향."""
    return f"{load_taxonomy().label(category_id)} · {category_id}"


# ── 실행 로그 ───────────────────────────────────────────────────────────────
#: 로그 줄에서 노드 이름이 차지하는 폭. 왼쪽이 맞아야 단계가 눈으로 읽힌다.
NODE_WIDTH = 18

#: 룰이 "해결했다"고 셀 수 있는 매칭 단계. `pg_masked`·`unmatched` 는 룰이 못 푼 것이다.
_RESOLVED_STAGES = ("exact", "alias", "fuzzy", "regex")


def _stage_value(stage) -> str:
    """`MatchStage.EXACT` 든 `"exact"` 든 같은 문자열로 센다."""
    return getattr(stage, "value", stage)


def log_line(node: str, update: object) -> str | None:
    """스트리밍 청크 하나 → 실행 로그 한 줄. 찍을 것이 없으면 `None`.

    화면이 결과만 보여주면 8노드가 돈다는 것이 말로만 남는다 — 배점이 Agent 설계 구체성
    35점인데 화면에 그 증거가 없다. `graph.stream(stream_mode="updates")` 가 주는 상태
    조각을 사람이 읽는 문장으로 바꾼다.

    **노드가 실제로 낸 것에서 센다.** `metrics` 는 `finalize` 에서야 채워지므로 그 전
    단계는 상태에 들어온 목록을 직접 센다 — 화면이 아직 계산되지 않은 값을 미리 말하지
    않게. 모르는 노드는 `None` 이고, 그러면 로그에 줄이 생기지 않는다.
    """
    if node == "__interrupt__":
        return "사람을 기다린다 — 그래프가 interrupt() 로 멈춘 자리다"
    if not isinstance(update, dict):
        return None

    if node == "ingest":
        return f"원본 거래 {len(update.get('raw') or []):,}건을 읽었다"

    if node == "pair_cancellations":
        return f"취소·환불 {len(update.get('pairings') or []):,}쌍을 원거래에 묶었다"

    if node == "normalize":
        rows = update.get("normalized") or []
        keys = {row.normalized_name for row in rows}
        return f"표기를 정규화했다 — 거래 {len(rows):,}건이 정규화 키 {len(keys):,}개로"

    if node == "rule_match":
        matched = update.get("matched") or []
        counts = Counter(_stage_value(row.match_stage) for row in matched)
        resolved = sum(counts[stage] for stage in _RESOLVED_STAGES)
        detail = " · ".join(f"{s} {counts[s]:,}" for s in _RESOLVED_STAGES if counts[s])
        return f"룰 사전이 {resolved:,}건을 해결했다 — {detail}"

    if node == "collect_targets":
        return f"룰이 못 푼 표기 {len(update.get('llm_targets') or []):,}개를 LLM 몫으로 모았다"

    if node == "llm_classify":
        verdicts = update.get("llm_results") or {}
        failed = update.get("llm_failed") or []
        queued = update.get("review_queue") or []
        parts = [f"판정 {len(verdicts):,}개"]
        if failed:
            parts.append(f"실패 {len(failed)}건")
        if queued:
            parts.append(f"신뢰도가 낮은 {len(queued)}개는 사람 몫")
        return " · ".join(parts)

    if node == "human_review":
        corrections = update.get("human_corrections") or []
        if not corrections:
            return "사람이 고친 줄이 없다 — 그것도 정당한 답이다"
        return f"사람 교정 {len(corrections)}건을 받았다"

    if node == "rule_promote":
        return f"승격 후보 {len(update.get('rule_proposals') or []):,}개를 만들었다"

    if node == "finalize":
        metrics = update.get("metrics")
        if metrics is None:
            return f"분류를 확정했다 — {len(update.get('confirmed') or []):,}건"
        stages = metrics.stage_counts or {}
        return (
            f"분류 확정 {metrics.tx_classifiable:,}건 — 룰 {metrics.rule_resolved:,} · "
            f"LLM {stages.get('llm', 0):,} · PG마스킹 {stages.get('pg_masked', 0):,} · "
            f"미분류 {stages.get('unmatched', 0):,}"
        )

    # `review` 는 서브그래프를 감싼 부모 상자다 — 안쪽 노드가 이미 찍었으므로 또 찍지 않는다.
    return None


#: 로그 상세도. 화면이 고르게 하고, 값은 **테스트도 같은 상수를 쓴다**.
#: 발표에서는 `요약` 으로 읽고, 질문이 오면 그 자리에서 내린다 — 상세도를 바꿔도 그래프는
#: 다시 돌지 않는다(다음 라운드부터 적용). 지금 화면에 남은 로그를 소급해 늘리지 않는 이유는
#: 로그가 **그때 실제로 흐른 것**이어야 하기 때문이다 (같은 이유).
DETAIL_SUMMARY = "요약 — 노드마다 한 줄"
DETAIL_RAW = "요약 + 상태 원문"
DETAIL_DEBUG = "debug — task · checkpoint 까지"
DETAIL_LEVELS = (DETAIL_SUMMARY, DETAIL_RAW, DETAIL_DEBUG)

#: 원문 한 줄에 보일 값의 최대 길이. 넘으면 자른다 — `normalized` 8,969건을 그대로 찍으면
#: 로그가 아니라 덤프가 되고, 발표 화면에서 스크롤만 남는다.
RAW_CLIP = 88


def _raw_value(value: object) -> str:
    """상태 채널 하나의 값을 한 줄로 줄인다. **길이가 있는 것은 길이로 말한다.**"""
    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__} {len(value):,}개"
    if isinstance(value, dict):
        return f"dict {len(value):,}키"
    text = str(value).replace("\n", " ")
    return text if len(text) <= RAW_CLIP else text[: RAW_CLIP - 1] + "…"


def raw_lines(update: object) -> list[str]:
    """노드가 쓴 상태 채널을 채널마다 한 줄로. 요약이 감춘 것을 그대로 본다.

    `log_line` 은 사람이 읽는 **한 문장**이라 노드가 실제로 어느 채널에 썼는지는 보이지
    않는다. 설계를 말하는 화면이 "무엇이 상태에 들어갔나"를 못 보여주면 State 설계(S06)는
    슬라이드에만 남는다. 그래서 요약 아래에 채널 이름과 크기를 붙인다.
    """
    if not isinstance(update, dict):
        return []
    return [f"      · {key} = {_raw_value(value)}" for key, value in update.items()]


def _result_channels(result: object) -> list[str]:
    """`task_result` 가 쓴 채널 이름들. **형태가 버전마다 다르다.**

    실측(langgraph 1.x)에서는 채널→값 `dict` 가 온다. 문서·구버전에서는 `[(채널, 값), …]`
    쌍 목록이라, 한 쪽만 가정하면 화면이 `ValueError` 로 죽는다 — 로그를 보여주려다 화면을
    떨어뜨리는 것은 최악의 교환이다. 그래서 둘 다 받고, 모르는 형태는 조용히 비운다.
    """
    if isinstance(result, dict):
        return [str(k) for k in result]
    if isinstance(result, (list, tuple)):
        names = []
        for item in result:
            if isinstance(item, (list, tuple)) and item:
                names.append(str(item[0]))
        return names
    return []


def debug_line(chunk: object) -> str | None:
    """`stream_mode="debug"` 청크 하나 → 로그 한 줄. 찍을 것이 없으면 `None`.

    `updates` 는 노드가 **끝난 뒤** 무엇을 썼는지만 준다. 그래서 요약만 보면 노드가 언제
    시작했고 무엇 때문에 깨어났는지(`triggers`)가 안 보이고, `interrupt` 가 걸린 태스크가
    어느 것인지도 안 보인다. debug 는 그 둘을 준다 — 대신 줄이 두 배로 늘어난다.
    """
    if not isinstance(chunk, dict):
        return None
    kind, payload = chunk.get("type"), chunk.get("payload") or {}
    step = chunk.get("step")

    if kind == "task":
        triggers = ", ".join(payload.get("triggers") or []) or "시작"
        return f"step {step} · 태스크 시작 ← {triggers}"
    if kind == "task_result":
        if payload.get("error"):
            return f"step {step} · 태스크 실패 — {payload['error']}"
        channels = ", ".join(_result_channels(payload.get("result"))) or "쓴 것 없음"
        stopped = " · interrupt" if payload.get("interrupts") else ""
        return f"step {step} · 태스크 끝 → {channels}{stopped}"
    if kind == "checkpoint":
        nxt = ", ".join(payload.get("next") or []) or "끝"
        return f"step {step} · 체크포인트 저장 · 다음: {nxt}"
    return None


# ── 검토 큐 ↔ 표 ────────────────────────────────────────────────────────────
def queue_rows(payload: dict) -> list[dict]:
    """`interrupt()` 페이로드 → `st.data_editor` 가 받는 행 목록.

    페이로드의 모양을 아는 곳을 여기 하나로 묶는다. 화면이 `item["verdict"]["confidence"]`
    를 직접 파고들면 노드가 페이로드를 바꾸는 날 화면이 조용히 빈 칸을 그린다.

    교정 칸(`대표명`·`분류`)은 **LLM 이 낸 답으로 미리 채운다.** 빈 칸으로 두면 사람이 고칠
    것이 없는 줄에서도 전부 입력해야 하고, 그러면 배치 20개가 화면에서 끝나지 않는다.

    **열 순서가 곧 손이 가는 순서다.** 교정 칸을 오른쪽 끝에 두면 폭 넓은 `근거` 열이 그것을
    화면 밖으로 밀어내고, 발표자가 가로로 스크롤한 뒤에야 고칠 수 있다 — 1920px 에서도 잘렸다.
    그래서 `반영·표기·건수` 다음이 교정 칸이고, 판단 재료(근거·LLM 대표명)는 그 뒤에 온다.
    """
    rows = []
    for item in payload.get("items", []):
        verdict = item["verdict"]
        rows.append(
            {
                COL_APPLY: False,
                COL_NAME: item["normalized_name"],
                "건수": item["tx_count"],
                COL_CANONICAL: verdict["merchant_canonical"],
                COL_CATEGORY: category_choice_of(verdict["category_l2"]),
                "신뢰도": verdict["confidence"],
                "LLM 대표명": verdict["merchant_canonical"],
                "근거": verdict["reasoning"],
                "판단재료": verdict["evidence"],
                "우선순위": round(item["priority"], 1),
            }
        )
    return rows


def rows_to_corrections(
    rows, *, reviewer: str = "human"
) -> tuple[list[HumanCorrection], list[str]]:
    """표에서 **반영에 체크된 줄만** 교정 객체로 바꾼다. 두 번째 값은 못 만든 줄의 사유다.

    `category_l1` 을 화면이 고르게 하지 않는다 — L2 의 소속에서 끌어온다. 사람이 둘을 따로
    고르면 `food` + `cafe.coffee` 같은 조합이 만들어지고, 그것은 스키마 검증에서 예외가 되어
    **제출 버튼이 통째로 실패한다.** 고를 수 없는 것은 틀릴 수도 없다.

    못 만든 줄을 조용히 버리지 않고 사유로 돌려주는 이유: 체크는 했는데 대표명이 빈 줄을
    말없이 건너뛰면 화면은 "N건 반영"이라 적고 사전에는 그 항목이 없다. 이 프로젝트가 가장
    경계하는 형태의 거짓 보고다.
    """
    choices = category_choices()
    corrections: list[HumanCorrection] = []
    problems: list[str] = []
    for row in rows:
        if not row.get(COL_APPLY):
            continue
        name = str(row.get(COL_NAME, "")).strip()
        canonical = str(row.get(COL_CANONICAL) or "").strip()
        if not canonical:
            problems.append(f"{name}: 대표명이 비어 있다")
            continue
        category_id = choices.get(str(row.get(COL_CATEGORY, "")))
        if category_id is None:
            problems.append(f"{name}: 분류를 고르지 않았다")
            continue
        category = CategoryL2(category_id)
        corrections.append(
            HumanCorrection(
                normalized_name=name,
                merchant_canonical=canonical,
                category_l1=category.l1,
                category_l2=category,
                reviewer=reviewer,
                note=str(row.get("메모") or ""),
            )
        )
    return corrections, problems


# ── 라운드 요약 ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PromotionSummary:
    """한 번의 승격이 무엇을 바꿨는가. **사람 몫과 자동 몫을 끝까지 갈라서 센다.**

    합계만 보고하면 사람이 한 일이 자동 승격 34건에 섞여 보이지 않는다. 이 화면이 증명하려는
    것은 "사전이 자란다"가 아니라 **"사람이 사전을 자라게 한다"** 이므로(자동은 새 브랜드를
    못 만든다), 두 수는 끝까지 다른 칸에 있어야 한다.
    """

    human: list[RuleProposal] = field(default_factory=list)
    auto: list[RuleProposal] = field(default_factory=list)
    #: 제안됐지만 사전이 받지 않은 것. 0 이 아닐 수 있으므로 화면에도 자리를 준다.
    rejected: list[RuleProposal] = field(default_factory=list)

    @property
    def new_brands(self) -> list[RuleProposal]:
        """새 항목을 만든 승격 — `exact` 는 사전에 없던 브랜드라는 뜻이다."""
        return [p for p in self.human if p.pattern_type == "exact"]

    @property
    def human_tx(self) -> int:
        return sum(p.covers_tx_count for p in self.human)

    @property
    def auto_tx(self) -> int:
        return sum(p.covers_tx_count for p in self.auto)


def summarize_promotion(
    proposals, accepted
) -> PromotionSummary:
    """제안 목록과 사전이 받아들인 목록을 사람/자동/거절로 가른다.

    `accepted` 를 따로 받는 이유는 **제안 ≠ 승격**이기 때문이다. `apply_proposals` 는 이미
    쓰인 키를 건너뛰고, 대표명이 사전에 없는 모델 출처 후보를 버린다. 제안 수를 승격 수로
    보고하면 사전 항목 증가분과 표의 숫자가 어긋난다.
    """
    accepted_keys = {(p.pattern_type, p.pattern) for p in accepted}
    summary_human, summary_auto, rejected = [], [], []
    for proposal in proposals:
        if (proposal.pattern_type, proposal.pattern) not in accepted_keys:
            rejected.append(proposal)
        elif proposal.origin == "human":
            summary_human.append(proposal)
        else:
            summary_auto.append(proposal)
    return PromotionSummary(human=summary_human, auto=summary_auto, rejected=rejected)


# ── 실행 ────────────────────────────────────────────────────────────────────
def load_replay_classifier(path: Path | None = None) -> tuple[ReplayClassifier, str, int]:
    """재생 대본을 읽어 분류기를 만든다. `(분류기, 모델명, 판정 수)`."""
    verdicts, model = load_verdicts(path or VERDICTS_PATH)
    return ReplayClassifier(verdicts), model, len(verdicts)


def build_review_graph(dictionary: RuleDictionary, classifier, *, batch: int, tau: float):
    """검토가 켜진 그래프를 만든다 — 화면이 직접 `build_graph` 를 부르는 자리는 여기뿐이다.

    라운드마다 **사전이 다르므로** 그래프도 다시 만든다. 캐시로 묶어 두면 승격된 사전이
    반영되지 않은 채 라운드 2 가 돌고, 그러면 "호출이 줄지 않았다"는 잘못된 결론이 나온다.
    컴파일 자체는 밀리초 단위라 아낄 것이 없다.
    """
    return build_graph(
        matcher=RuleMatcher(dictionary),
        classifier=classifier,
        review=True,
        review_batch=batch,
        tau=tau,
        checkpointer=build_checkpointer(),
    )


# ── 지표 대시보드 (S2) ──────────────────────────────────────────────────────
#: 실호출 실측 스냅샷. **(B) LLM만은 여기서만 온다.** 재생 대본은 (C) 가 LLM 에 보낸 표기
#: 149개뿐이라, 사전 없이 표기 전체를 보내는 (B) 를 되살릴 방법이 없다 — 없는 수를 화면이
#: 지어내지 않게 파일이 없으면 그 칸을 비우고 무엇을 돌려야 하는지 적는다.
SNAPSHOT_PATH = ROOT / "data" / "gold" / "baselines-v1-gpt-4o-mini.json"

#: 스냅샷을 만드는 명령. 화면이 이 문자열을 그대로 보여준다 — 발표 전날 "그 명령이 뭐였지"
#: 를 세션 로그에서 찾지 않기 위한 자리다. `--baseline b` 다: (A)·(C) 는 이 화면이 그 자리에서
#: 다시 재므로 `all` 로 돌리면 이미 있는 수를 587회 더 사 오는 것이다.
SNAPSHOT_CMD = (
    "python -m txlens.eval --baseline b --model openai:gpt-4o-mini "
    f"--json {SNAPSHOT_PATH.relative_to(ROOT).as_posix()}"
)

#: 출처 딱지. 표에 열로 들어간다.
SRC_LIVE = "이 자리에서 계산"
SRC_REPLAY = "이 자리에서 계산 (재생)"
SRC_SNAPSHOT = "실호출 스냅샷"


@dataclass(frozen=True)
class BaselineRow:
    """§5.1 비교표 한 줄. **어디서 온 수인지를 값과 같은 자리에 들고 다닌다.**

    이 표는 세 종류의 수가 섞인다 — 지금 계산한 것(A), 대본으로 되살린 것(C), 그리고 API 키가
    있어야만 나오는 것(B). 출처를 표 아래 각주로만 적으면 "이 숫자 어디서 나온 거냐"는 질문에
    표의 어느 칸도 가리킬 수 없다. 그래서 `source` 가 열이다.
    """

    label: str
    source: str
    accuracy_l2: float
    distinct_l2: float
    coverage: float
    unmatched_rate: float
    pg_masked_rate: float
    llm_targets: int
    llm_model_calls: int

    def to_row(self) -> dict:
        """표 한 줄. 비율은 여기서 문자열로 굳힌다 — 화면마다 자릿수가 달라지지 않게."""
        return {
            "구성": self.label,
            "거래 L2": f"{self.accuracy_l2:.1%}",
            "키 L2": f"{self.distinct_l2:.1%}",
            "확정 커버리지": f"{self.coverage:.1%}",
            "미매칭": f"{self.unmatched_rate:.1%}",
            "PG마스킹": f"{self.pg_masked_rate:.1%}",
            "LLM 표기": self.llm_targets,
            "모델 호출": self.llm_model_calls,
            "출처": self.source,
        }


def row_from_result(result, metrics, *, source: str) -> BaselineRow:
    """하네스가 낸 `EvalResult` → 표 한 줄. **비율을 여기서 다시 계산하지 않는다.**

    `Slice` 의 property 를 그대로 읽는다. 화면이 `correct_l2 / labeled` 를 스스로 나누면
    분모를 하나 잘못 고른 날 슬라이드와 화면이 다른 수를 말하고, 둘 다 그럴듯해서 어느 쪽이
    틀렸는지 발표 중에는 알 수 없다.
    """
    o = result.overall
    return BaselineRow(
        label=result.label,
        source=source,
        accuracy_l2=o.accuracy_l2,
        distinct_l2=result.distinct.accuracy_l2,
        coverage=o.coverage,
        unmatched_rate=o.unmatched_rate,
        pg_masked_rate=o.pg_masked_rate,
        llm_targets=metrics.llm_targets if metrics else 0,
        llm_model_calls=metrics.llm_model_calls if metrics else 0,
    )


def load_snapshot_rows(path: Path | None = None) -> list[BaselineRow]:
    """`python -m txlens.eval --json` 산출물을 읽는다. 없으면 **빈 목록** (예외 아님).

    파일이 없는 것은 고장이 아니라 "아직 실호출을 안 돌렸다"는 정상 상태다. 화면은 그 상태를
    그리는 책임이 있으므로 여기서 죽으면 안 된다. 대신 모양이 틀린 파일은 조용히 넘기지
    않는다 — 스냅샷이 반쯤 읽히면 표에 빈 줄이 생기고 그게 측정값처럼 보인다.
    """
    path = path or SNAPSHOT_PATH
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for entry in payload:
        result, metrics = entry["result"], entry.get("metrics")
        o, d = result["overall"], result["distinct"]
        rows.append(
            BaselineRow(
                label=result["label"],
                source=SRC_SNAPSHOT,
                accuracy_l2=o["accuracy_l2"],
                distinct_l2=d["accuracy_l2"],
                coverage=o["coverage"],
                unmatched_rate=o["unmatched_rate"],
                pg_masked_rate=o["pg_masked_rate"],
                llm_targets=(metrics or {}).get("llm_targets", 0),
                llm_model_calls=(metrics or {}).get("llm_model_calls", 0),
            )
        )
    return rows


#: 같은 구성인데 값이 이만큼 넘게 벌어지면 화면이 말한다. 재생과 실호출은 원리상 완전히
#: 같을 수 없어(대본에 없는 표기는 실패로 남는다) 0 으로 두면 매번 경고가 뜬다.
DRIFT_TOLERANCE = 0.001


def merge_baseline_rows(
    live: list[BaselineRow], snapshot: list[BaselineRow]
) -> tuple[list[BaselineRow], list[str]]:
    """지금 계산한 줄과 스냅샷을 **A/B/C 순서로** 합치고, 어긋난 곳을 함께 돌려준다.

    합치는 규칙은 하나다 — **지금 계산할 수 있는 것은 지금 계산한 값을 쓴다.** 스냅샷은
    되살릴 수 없는 줄(B)만 채운다. 반대로 하면 화면이 코드와 무관한 옛 수를 말하게 되고,
    사전이나 매칭을 고친 날 그 사실이 화면 어디에도 나타나지 않는다.

    같은 구성이 양쪽에 다 있으면 값을 대조한다. 어긋남을 감추지 않는 이유는 그 어긋남이
    두 가지 중 하나이기 때문이다 — 재생과 실호출의 원리적 차이(정상)이거나, 스냅샷을 뜬
    뒤에 코드·사전이 바뀐 것(슬라이드가 낡았다는 신호)이다. 둘 다 발표 전에 알아야 한다.
    """
    merged = {row.label: row for row in snapshot}
    notes: list[str] = []
    for row in live:
        old = merged.get(row.label)
        if old is not None:
            for field_name, title in (("accuracy_l2", "거래 L2"), ("distinct_l2", "키 L2")):
                new_v, old_v = getattr(row, field_name), getattr(old, field_name)
                if abs(new_v - old_v) > DRIFT_TOLERANCE:
                    notes.append(
                        f"**{row.label}** {title} — 지금 계산 {new_v:.1%} · "
                        f"실호출 스냅샷 {old_v:.1%}"
                    )
        merged[row.label] = row
    return [merged[k] for k in sorted(merged)], notes


def delta_pp(new: float, old: float) -> float:
    """두 비율의 차이를 **화면에 찍히는 자릿수로 먼저 반올림한 뒤** 뺀다.

    그냥 빼면 타일이 `98.9%` 와 `+5.8%p` 를 나란히 적고 표에는 `93.0%` 가 있다 — 셋이 서로
    안 맞는다(실제 값은 98.85 와 92.99 라 차이가 5.86 이다). 청중은 화면에 보이는 수로
    검산하므로, 보이지 않는 소수점 때문에 산수가 틀린 것처럼 보이면 그 표 전체를 의심한다.
    그래서 **보이는 수끼리 빼서 보이는 수와 맞춘다.**
    """
    return round(new * 100, 1) - round(old * 100, 1)


def round_rows(reports) -> list[dict]:
    """승격 라운드 보고 → 표·꺾은선이 함께 쓰는 행 목록.

    **비용 축과 정확도 축을 한 행에 둔다** (`promote.RoundReport` 와 같은 이유). 호출이 줄어든
    표만 따로 그리면 자동 승격이 오답을 굳혀 정확도를 깎은 라운드도 성공처럼 보인다.
    """
    return [
        {
            "라운드": r.round,
            "사전": f"v{r.dictionary_version}",
            "사전 항목": r.dictionary_keys,
            "LLM 표기": r.metrics.llm_targets,
            "미매칭 표기": r.pending_targets,
            "승격": r.promoted,
            "덮는 거래": r.promoted_tx,
            "거래 L2": r.result.overall.accuracy_l2,
        }
        for r in reports
    ]


def round_chart(rows: list[dict], *, value: str, title: str):
    """라운드 꺾은선. **x 축은 순서형(`:O`)이다** — 라운드는 정수인데 1.000000 으로 찍혔다.

    Streamlit 의 `st.line_chart` 는 축 종류를 값에서 추정한다. 라운드 번호는 정수라 연속형
    으로 잡히고, 그러면 눈금이 `1.000000 · 1.500000` 처럼 **있지도 않은 라운드 1.5** 를 그린다.
    `:O` 로 못 박으면 눈금이 실제로 존재하는 라운드에만 선다. 정렬도 문자열이 아니라 숫자로
    돈다 — 라운드가 10을 넘으면 `"10" < "2"` 가 되는 자리를 미리 막는다.

    altair 를 직접 쓰는 이유는 이 축 지정을 `st.line_chart` 로는 할 수 없어서다. 차트를
    함수로 갈라 둔 덕에 축 종류를 테스트가 확인할 수 있다 — 화면을 눈으로 봐야만 알 수 있는
    회귀였다.
    """
    import altair as alt

    frame = pd.DataFrame(rows)
    base = alt.Chart(frame, title=title).encode(
        x=alt.X("라운드:O", axis=alt.Axis(labelAngle=0, title="승격 라운드")),
        y=alt.Y(f"{value}:Q", title=value, scale=alt.Scale(zero=True)),
        tooltip=list(frame.columns),
    )
    return base.mark_line(point=True, strokeWidth=2)


# ── S4 인사이트 ─────────────────────────────────────────────────────────────
# 탐지 결과를 표로 바꾸는 층. 계산은 `txlens.detect` 에만 있고 여기서는 **보여줄 모양**만
# 만든다 — 화면에서 만들면 A 와 C 두 벌을 그릴 때 두 코드가 갈라진다 (같은 이유).
#
# 근거 문장(`evidence`)은 손대지 않고 그대로 싣는다. 탐지기가 규칙을 옮겨 적은 문장이고,
# 고객에게 "왜 이 알림이 왔나"를 설명하는 것이 2부의 요점이다. 화면에서 요약하면
# 그 문장이 검증된 적 없는 다른 문장이 된다.

PERIOD_LABEL = {"monthly": "월간", "weekly": "주간", "yearly": "연간"}
SPEND_KIND_LABEL = {"category_surge": "카테고리 급증", "single_large": "단발 고액"}
TX_KIND_LABEL = {"duplicate_charge": "중복 결제", "micro_burst": "소액 다발"}


def subscription_rows(subs) -> list[dict]:
    """구독 표. 정렬은 **월 고정비 내림차순** — 고객이 먼저 볼 것은 큰 돈이다."""
    rows = []
    for s in sorted(subs, key=lambda x: (-x.monthly_cost, x.merchant_canonical)):
        flags = " · ".join(
            f for f in ("미사용 의심" if s.unused else "", "가격 인상" if s.price_increased else "") if f
        )
        rows.append(
            {
                "고객": s.customer_id,
                "가맹점": s.merchant_canonical,
                "주기": PERIOD_LABEL.get(s.period_kind, s.period_kind),
                "결제": s.occurrences,
                "중앙값": s.median_amount,
                "월 환산": s.monthly_cost,
                "다음 결제": s.next_expected.strftime("%Y-%m-%d"),
                "표시": flags,
                "근거": s.evidence,
            }
        )
    return rows


def spending_rows(alerts) -> list[dict]:
    """이상지출 표. robust z 가 큰 순 — 얼마나 튀었나가 정렬 기준이다."""
    return [
        {
            "고객": a.customer_id,
            "종류": SPEND_KIND_LABEL.get(a.kind, a.kind),
            "카테고리": a.category_l1,
            "월": a.month,
            "금액": a.amount,
            "robust z": round(a.robust_z, 1),
            "근거": a.evidence,
        }
        for a in sorted(alerts, key=lambda x: (-x.robust_z, x.customer_id))
    ]


def tx_alert_rows(alerts) -> list[dict]:
    return [
        {
            "고객": a.customer_id,
            "종류": TX_KIND_LABEL.get(a.kind, a.kind),
            "가맹점": a.merchant_canonical,
            "금액": a.amount,
            "시각": a.at.strftime("%Y-%m-%d %H:%M"),
            "근거": a.evidence,
        }
        for a in sorted(alerts, key=lambda x: (x.at, x.customer_id))
    ]


def customer_months(classified) -> tuple[int, int]:
    """고객 수와 관측 개월 수. **탐지기가 실제로 본 행에서** 센다.

    알림 수만 찍으면 많은지 적은지 알 수 없다. **고객당 월 알림**이 설계 §3.6 의 기준선
    (0.5건)이고, 그 분모를 정답셋이 아니라 화면이 들고 있는 거래에서 구한다 — 정답이 없는
    실제 운영에서도 같은 식으로 재야 하기 때문이다.

    `spend_rows` 를 다시 부르는 이유는 분자와 분모를 **같은 모집단**에서 세기 위해서다.
    취소 상계·0원 행은 탐지 입력에서 빠지는데, 분모만 전체 거래로 세면 알림률이 조용히
    낮아진다. 필드 경로(`tx.normalized.raw.…`)를 여기 베끼지 않는 효과도 있다.
    """
    rows = spend_rows(classified)
    customers = {r.customer_id for r in rows}
    months = {(r.at.year, r.at.month) for r in rows}
    return len(customers), len(months)


def insight_tiles(det, classified) -> dict:
    """S4 타일. 값과 **표시 문자열을 함께** 돌려준다 — 증감은 보이는 수끼리 빼야 한다."""
    n_customers, n_months = customer_months(classified)
    denom = max(1, n_customers * n_months)
    alerts = len(det.spending_alerts) + len(det.tx_alerts)
    fixed = sum(monthly_fixed_cost(det.subscriptions).values())
    return {
        "구독": len(det.subscriptions),
        "월 고정비": fixed,
        "알림": alerts,
        "고객당 월 알림": round(alerts / denom, 1),
        "고객": n_customers,
        "개월": n_months,
    }
