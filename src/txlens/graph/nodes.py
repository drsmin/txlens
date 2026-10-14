"""그래프 노드 — 하나만 빼고 전부 결정론 (설계 §3.1 · §3.3).

노드는 **순수 함수에 가깝게** 둔다. 사전·분류기 같은 의존물은 인자로 받고(`build.py` 가 묶는다),
상태에서 읽은 것만 보고 상태 조각만 돌려준다. 그래야 노드 하나를 그래프 없이 테스트할 수 있다.

## LLM 이 "모른다"고 답하면 `UNMATCHED` 로 남긴다

`MatchStage.LLM` 을 붙이면 `metrics.RESOLVED_STAGES` 에 들어가 **커버리지로 세어진다.** 그러면
"LLM 이 미분류라고 답한 것"이 "브랜드를 확정한 것"과 같은 칸에 들어가 커버리지가 부풀고,
그 위에서 도는 2부는 여전히 아무것도 못 한다. 판정 자체는 `source_verdict` 로 보존해 감사
추적을 남기고, 몇 개가 그랬는지는 `RunMetrics.llm_unknown` 으로 따로 센다.

## 확정한 판정은 신뢰도가 낮아도 답으로 쓴다

τ 미만은 검토 큐에 넣지만 결과에서 빼지는 않는다. 사람이 고치기 전까지는 그게 최선의 답이고,
**그 상태의 지표가 베이스라인 (C)** 다. 검토 전/후의 차이는 5주차에 (C) ↔ (C+HITL) 로 보인다.
큐에 있다고 결과에서 빼면 분모가 조용히 줄어 정확도가 저절로 오른다.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from txlens.agents.classifier import MerchantClassifier
from txlens.graph.state import ReviewState, RunMetrics, TxState
from txlens.ingest import load_dataset
from txlens.preprocess import pair_cancellations, preprocess_one
from txlens.rules import RuleDictionary, RuleMatcher
from txlens.rules.promote import (
    MIN_PROMOTE_CONFIDENCE,
    propose_from_corrections,
    propose_rules,
)
from txlens.schemas import (
    CancelPairing,
    CategoryL1,
    ClassifiedTx,
    HumanCorrection,
    LLMTarget,
    LLMVerdict,
    MatchStage,
    NormalizedTx,
    ReviewItem,
)

#: 검토 큐로 보내는 신뢰도 경계. 잠정값이다 — 실제 판정 분포를 보고 조정한다(4주차 τ 튜닝).
#: 높이면 큐가 길어져 사람이 지치고, 낮추면 틀린 판정이 검토 없이 통과한다.
DEFAULT_CONFIDENCE_TAU = 0.8

#: 한 번 멈출 때 사람에게 보일 검토 항목 수. **큐 길이와 다른 물건이다** — 큐는 276개까지 가고
#: 그것을 한 화면에 쏟으면 사람은 답을 고르는 대신 스크롤을 한다. 우선순위로 자른 상위 몫만
#: 보이고 나머지는 다음 라운드에 다시 줄을 선다.
DEFAULT_REVIEW_BATCH = 20


# ── ① ingest ────────────────────────────────────────────────────────────────
def ingest_node(state: TxState) -> TxState:
    """표준API 응답 번들 → `RawTx`. 실연동 시 **교체되는 지점은 여기 하나다.**"""
    bundle_dir = state.get("bundle_dir")
    if not bundle_dir:
        raise ValueError("bundle_dir 가 없다 — 그래프 입력에 번들 디렉터리를 넣어야 한다")
    txs, _report = load_dataset(Path(bundle_dir))
    return {"raw": txs}


# ── ② pair_cancellations ────────────────────────────────────────────────────
def pair_cancellations_node(state: TxState) -> TxState:
    """취소·정정을 상계한다. **거래 집합 전체를 봐야 답이 나오는 유일한 전처리다.**

    `normalize` 와 한 노드였다가 쪼갰다. 둘은 성질이 다르다 — 이쪽은 (기관, 고객,
    승인번호)로 묶어 합계를 보는 **fan-in** 이고, 저쪽은 행 하나만 보는 **map** 이다. 한 상자에
    있으면 "전처리"라는 한 단어에 가려 그 차이가 안 보이고, 실제로 이 프로젝트에서 가장 자주
    틀리는 곳(모수가 조용히 부푸는 곳)이 이쪽이다.

    순수 함수는 `id(tx)` 를 키로 돌려주지만 상태에는 인덱스로 바꿔 싣는다 (`CancelPairing`).
    """
    raw = state["raw"]
    marks = pair_cancellations(raw)
    pairings = []
    for index, tx in enumerate(raw):
        mark = marks.get(id(tx))
        if mark is None:
            continue
        cancel_of, reason = mark
        pairings.append(CancelPairing(index=index, cancel_of=cancel_of, excluded_reason=reason))
    return {"pairings": pairings}


# ── ③ normalize ─────────────────────────────────────────────────────────────
def normalize_node(state: TxState) -> TxState:
    """행 단위 결정론 정규화. 여기서 정확도의 절반이 결정된다.

    거래 하나만 보고 답이 나오므로 원리적으로 병렬·스트리밍이 가능하다 — 앞 노드와 달리
    **다른 거래를 볼 필요가 없다.** 그 사실을 그래프가 말하게 하려고 노드를 나눴다.
    """
    raw = state["raw"]
    marks = {p.index: p for p in state.get("pairings", [])}
    return {
        "normalized": [
            preprocess_one(
                tx,
                cancel_of=marks[i].cancel_of if i in marks else None,
                excluded_reason=marks[i].excluded_reason if i in marks else None,
            )
            for i, tx in enumerate(raw)
        ]
    }


# ── ④ rule_match ────────────────────────────────────────────────────────────
def rule_match_node(state: TxState, *, matcher: RuleMatcher) -> TxState:
    """5단계 룰 매칭. **분류 대상만** 넣는다 — 취소 상계된 거래를 분류하면 모수가 부푼다."""
    target = [n for n in state["normalized"] if n.is_classifiable]
    return {"matched": matcher.match_all(target)}


# ── ⑤ collect_targets ───────────────────────────────────────────────────────
def collect_targets(matched: list[ClassifiedTx]) -> list[LLMTarget]:
    """미매칭을 고유 표기로 묶어 LLM 대상 목록을 만든다.

    **건수 많은 것부터** 정렬한다. 상한을 걸어 자르든(데모), 사람이 위에서부터 검토하든,
    같은 노력으로 가장 많은 거래가 고쳐지는 순서가 된다. 동순위는 이름순으로 고정해 실행마다
    순서가 흔들리지 않게 한다 — 순서가 흔들리면 캐시 히트 수도 흔들린다.
    """
    counts: collections.Counter[str] = collections.Counter()
    raw_forms: dict[str, collections.Counter[str]] = {}
    first: dict[str, NormalizedTx] = {}

    for tx in matched:
        if tx.match_stage is not MatchStage.UNMATCHED:
            continue
        key = tx.normalized.normalized_name
        counts[key] += 1
        raw_forms.setdefault(key, collections.Counter())[tx.normalized.raw.merchant_name_raw] += 1
        first.setdefault(key, tx.normalized)

    targets = [
        LLMTarget(
            normalized_name=key,
            # 대표 원본 표기 = 그 키로 모인 표기 중 **가장 흔한 것.** 아무거나 고르면
            # 실행마다 달라져 캐시가 빗나간다.
            sample_raw_name=raw_forms[key].most_common(1)[0][0],
            tx_count=count,
            payment_channel=first[key].payment_channel,
            branch=first[key].branch,
        )
        for key, count in counts.items()
    ]
    targets.sort(key=lambda t: (-t.tx_count, t.normalized_name))
    return targets


def collect_targets_node(state: TxState) -> TxState:
    """미매칭을 고유 표기로 묶는다. **단위가 거래에서 문자열로 바뀌는 유일한 지점이다.**

    한때 `rule_match` 안에 있었다. 밖으로 꺼낸 이유는 이 한 줄이 이 프로젝트의 **전송 경계**이기
    때문이다 (설계 §1.1) — 여기 들어가는 것은 거래 8,969건이고 나오는 것은 문자열 155개다.
    LLM 이 보는 것은 나온 쪽뿐이라 "누가 언제 얼마를 썼는가"는 경계를 넘지 않는다. 경계를 노드로
    세우면 그림에서 손가락으로 짚을 수 있고, 규제 질문에 그림으로 답할 수 있다.

    비용 축으로도 같은 자리다 — 승격 루프가 줄이는 것은 거래 수가 아니라 **여기서 나오는 개수**다.
    """
    return {"llm_targets": collect_targets(state["matched"])}


def route_after_rules(state: TxState) -> str:
    """조건부 분기 ① — 미매칭이 없으면 LLM 을 건너뛴다.

    베이스라인 (A) 는 분류기를 붙이지 않으므로 `build.py` 가 이 분기를 아예 만들지 않는다.
    여기서 `classifier is None` 을 보지 않는 이유: 그래프 모양이 조건에 따라 달라지는 것을
    **그림으로** 확인할 수 있어야 한다 (`get_graph().draw_ascii()`).
    """
    return "llm_classify" if state.get("llm_targets") else "finalize"


# ── ⑥ llm_classify ──────────────────────────────────────────────────────────
def llm_classify_node(
    state: TxState,
    *,
    classifier: MerchantClassifier,
    tau: float = DEFAULT_CONFIDENCE_TAU,
    limit: int | None = None,
) -> TxState:
    """미매칭 고유 표기를 Agent 에게 하나씩 묻는다 (`agents/classifier.py`).

    `limit` 은 데모·비용 통제용 상한이다. 자른 만큼을 `llm_targets_skipped` 로 보고하므로
    "LLM 이 못 잡았다"와 "안 보냈다"가 섞이지 않는다.
    """
    targets = state.get("llm_targets", [])
    sent = targets if limit is None else targets[:limit]

    outcome = classifier.classify_all(sent)
    queue = [
        ReviewItem(
            normalized_name=name,
            verdict=verdict,
            tx_count=next(t.tx_count for t in sent if t.normalized_name == name),
        )
        for name, verdict in outcome.verdicts.items()
        if verdict.confidence < tau or verdict.category_l1 is CategoryL1.UNCLASSIFIED
    ]
    return {
        "llm_results": outcome.verdicts,
        "llm_failed": outcome.failed,
        "review_queue": queue,
    }


# ── ⑦ finalize ──────────────────────────────────────────────────────────────
def apply_verdict(tx: ClassifiedTx, verdict: LLMVerdict) -> ClassifiedTx:
    """미매칭 거래에 LLM 판정을 얹는다.

    판정이 미분류면 **단계를 올리지 않는다** (모듈 docstring 참고). 그래도 `source_verdict` 는
    남긴다 — 사람이 검토 화면에서 "LLM 은 왜 모른다고 했나"를 읽어야 한다.
    """
    unknown = verdict.category_l1 is CategoryL1.UNCLASSIFIED
    return ClassifiedTx(
        normalized=tx.normalized,
        merchant_canonical=tx.merchant_canonical if unknown else verdict.merchant_canonical,
        category_l1=verdict.category_l1,
        category_l2=verdict.category_l2,
        confidence=verdict.confidence,
        match_stage=MatchStage.UNMATCHED if unknown else MatchStage.LLM,
        source_verdict=verdict,
    )


def finalize_node(state: TxState, *, classifier: MerchantClassifier | None = None) -> TxState:
    """룰 결과에 LLM 판정을 합치고 실행 지표를 낸다. 그래프의 유일한 출구다."""
    matched = state["matched"]
    verdicts = state.get("llm_results", {})

    confirmed: list[ClassifiedTx] = []
    for tx in matched:
        verdict = (
            verdicts.get(tx.normalized.normalized_name)
            if tx.match_stage is MatchStage.UNMATCHED
            else None
        )
        confirmed.append(tx if verdict is None else apply_verdict(tx, verdict))

    stage_counts = collections.Counter(tx.match_stage.value for tx in confirmed)
    normalized = state.get("normalized", [])
    failed = state.get("llm_failed", [])
    cache = classifier.cache if classifier is not None else None
    # 상한 때문에 자른 몫. **분류기가 없으면 0 이다** — 베이스라인 (A) 는 LLM 노드가 그래프에
    # 아예 없으니 "보내려다 못 보낸" 표기가 아니라 "보낼 계획이 없는" 표기다. 이걸 미전송으로
    # 세면 A 가 무언가를 누락한 것처럼 보고된다.
    sent = len(verdicts) + len(failed)
    skipped = max(0, len(state.get("llm_targets", [])) - sent) if classifier is not None else 0

    metrics = RunMetrics(
        tx_total=len(normalized),
        tx_classifiable=len(matched),
        rule_resolved=sum(
            1
            for tx in matched
            if tx.match_stage not in {MatchStage.UNMATCHED, MatchStage.PG_MASKED}
        ),
        llm_targets=sent,
        llm_targets_skipped=skipped,
        llm_model_calls=cache.calls if cache else 0,
        llm_cache_hits=cache.hits if cache else 0,
        llm_unknown=sum(1 for v in verdicts.values() if v.category_l1 is CategoryL1.UNCLASSIFIED),
        llm_failed=len(failed),
        review_queued=len(state.get("review_queue", [])),
        human_corrected=len(state.get("human_corrections", [])),
        stage_counts=dict(stage_counts),
    )
    return {"confirmed": confirmed, "metrics": metrics}


# ── ⑧ confidence_gate ───────────────────────────────────────────────────────
def confidence_gate_node(
    state: ReviewState, *, batch_size: int = DEFAULT_REVIEW_BATCH
) -> ReviewState:
    """검토 큐를 우선순위로 세우고 **이번에 사람이 볼 몫**을 자른다.

    큐를 채우는 것은 `llm_classify` 지만 순서를 정하는 것은 여기다. 채우는 쪽은 판정이 나오는
    순서대로 넣을 뿐이고, 무엇을 먼저 볼지는 사람의 시간에 대한 판단이라 다른 질문이다.

    정렬 키는 `ReviewItem.priority` = **거래 건수 × 불확실성**이다. 건수만 보면 확신 있는 대형
    브랜드가 위로 올라와 사람이 고칠 것이 없고, 불확실성만 보면 한 건짜리 꼬리가 화면을 채운다.
    곱을 쓰면 "고치면 가장 많은 거래가 움직이는 것"이 위로 온다. 동순위는 이름순으로 고정해
    실행마다 화면이 흔들리지 않게 한다 — 흔들리면 데모에서 같은 장면을 두 번 못 만든다.
    """
    queue = sorted(
        state.get("review_queue", []),
        key=lambda item: (-item.priority, item.normalized_name),
    )
    return {"review_batch": queue[:batch_size]}


def route_after_gate(state: ReviewState) -> str:
    """조건부 분기 ② — 볼 것이 없으면 **사람을 부르지 않는다.**

    이 한 줄이 배치 실행을 지킨다. 큐가 비었는데도 `human_review` 에 들르면 `interrupt()` 가
    걸려 평가 하네스가 통째로 멈춘다 — 아무도 답을 줄 사람이 없는 곳에서.
    """
    return "human_review" if state.get("review_batch") else "rule_promote"


# ── ⑨ human_review ──────────────────────────────────────────────────────────
def _coerce_corrections(payload: Any) -> list[HumanCorrection]:
    """재개 값 → `HumanCorrection` 목록. 화면이 주는 형태 셋을 다 받는다.

    `None`(아무것도 안 고치고 넘김) · 목록 · `{"corrections": [...]}`. 형태를 하나로 강제하면
    Streamlit 쪽에 포장 코드가 생기고, 그 포장 코드는 테스트가 없는 자리다.

    **검증은 느슨하게 하지 않는다.** 사람 교정만이 새 브랜드를 만들 수 있으므로 여기서
    통과한 것은 곧 사전에 쓰인다. `model_validate` 가 택소노미 정합성까지 본다.
    """
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        payload = payload.get("corrections", [])
    if isinstance(payload, HumanCorrection):
        return [payload]
    if not isinstance(payload, Sequence) or isinstance(payload, str | bytes):
        raise TypeError(f"검토 재개 값이 교정 목록이 아니다: {type(payload).__name__}")
    return [
        item if isinstance(item, HumanCorrection) else HumanCorrection.model_validate(item)
        for item in payload
    ]


def human_review_node(state: ReviewState) -> ReviewState:
    """**그래프가 멈추는 유일한 지점.** `interrupt()` 로 사람의 답을 기다린다.

    멈춤이 예외가 아니라 노드인 것이 요점이다. 파이프라인을 반쪽 돌려 놓고 밖에서 사람을 부른
    뒤 다시 이어 붙이면, 이어 붙이는 코드가 배치와 화면에서 각각 생기고 둘이 갈라진다. 여기서
    멈추면 재개는 `Command(resume=...)` 한 줄이고 상태는 체크포인터가 들고 있다.

    나가는 것은 **표기와 판정뿐**이다 — 거래는 서브그래프 상태에 애초에 없다(`ReviewState`).
    화면이 "누가 언제 얼마를 썼는가"를 못 보는 것은 제약이 아니라 전송 경계와 같은 선택이다.

    돌아온 교정은 배치에서 `tx_count`·원판정을 채워 넣는다. 화면이 그 둘을 되돌려 보내게 하면
    화면이 상태의 사본을 들게 되고, 사본은 어긋난다. 배치에 없는 표기를 고쳐 보내는 것은 막지
    않는다 — 사람이 화면 밖에서 아는 것을 반영하는 정당한 경로이고, 다만 채워 줄 것이 없다.
    """
    batch = state.get("review_batch", [])
    answer = interrupt(
        {
            "kind": "review_batch",
            "queued": len(state.get("review_queue", [])),
            "items": [
                {
                    "normalized_name": item.normalized_name,
                    "tx_count": item.tx_count,
                    "priority": item.priority,
                    "verdict": item.verdict.model_dump(mode="json"),
                }
                for item in batch
            ],
        }
    )

    known = {item.normalized_name: item for item in batch}
    corrections = []
    for correction in _coerce_corrections(answer):
        item = known.get(correction.normalized_name)
        if item is None:
            corrections.append(correction)
            continue
        corrections.append(
            correction.model_copy(
                update={
                    "tx_count": correction.tx_count or item.tx_count,
                    "reviewed_verdict": correction.reviewed_verdict or item.verdict,
                }
            )
        )
    return {"human_corrections": corrections}


# ── ⑩ rule_promote ──────────────────────────────────────────────────────────
def rule_promote_node(
    state: TxState | ReviewState,
    *,
    dictionary: RuleDictionary,
    min_confidence: float = MIN_PROMOTE_CONFIDENCE,
) -> TxState:
    """고신뢰 판정을 사전 승격 후보로 바꾼다 — **이 실행이 남기는 유일한 지식이다.**

    다른 노드의 산출물은 이 실행의 답(`confirmed`)으로 끝나지만, 여기서 나온 `rule_proposals`
    는 파일이 되어 **다음 라운드의 룰 단계**로 간다. 승격 조건과 그 이유는
    `txlens.rules.promote` 의 모듈 docstring에 있다.

    노드로 세운 이유는 이 그래프에서 사전을 **읽지 않고 쓰는** 자리가 여기 하나뿐이기 때문이다.
    사람 교정도 같은 자리로 들어온다(`origin="human"`) — 두 출처를 다른 노드로 쪼개면 같은
    사전을 두 번 읽고, 두 결과가 같은 표기를 두고 어긋났을 때 어느 쪽이 이겼는지 알 수 없다.

    **사람 교정을 목록 앞에 둔다.** 두 문턱이 겹치기 때문이다 — 검토 큐는 τ(0.8) 미만을 담고
    자동 승격은 0.7 이상을 올리므로, 신뢰도 0.7~0.8 인 표기는 **큐에도 있고 자동 후보에도 있다.**
    `apply_proposals` 가 이미 쓰인 키를 건너뛰므로 목록 순서가 곧 우선순위이고, 그 표기에 대해
    사람이 실제로 화면에서 보고 고친 답이 모델의 답을 이긴다. 순서를 뒤집으면 사람이 고친 것이
    조용히 무시되고, 화면은 "반영됨"이라고 표시한다.

    사전을 인자로 받는 것이 승격 정책의 핵심이다 — 모델이 댄 대표명을 **사전에서 찾을 수
    있을 때만** 승격한다. 사전이 판정의 검증자가 되고, 자동 승격은 새 브랜드를 만들지
    않는다. 룰 매칭과 같은 사전을 봐야 하므로 `build.py` 가 `matcher.dictionary` 를 넘긴다.

    후보를 만들 뿐 파일을 쓰지 않는다. 파일을 쓰는 것은 되돌리기 어렵고, 그래프는 평가 하네스가
    수십 번 돌리는 물건이다 — 한 번 돌 때마다 사전이 바뀌면 무엇으로 잰 지표인지 알 수 없다.
    쓰는 것은 라운드 드라이버의 몫이다 (`python -m txlens.promote`).
    """
    proposals = propose_from_corrections(
        state.get("human_corrections", []), dictionary=dictionary
    )
    proposals += propose_rules(
        state.get("llm_targets", []),
        state.get("llm_results", {}),
        dictionary=dictionary,
        min_confidence=min_confidence,
    )
    return {"rule_proposals": proposals}
