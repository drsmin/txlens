"""파이프라인 골격 — LangGraph `StateGraph` (blueprint §3.1 · D-017).

**골격은 그래프, Agent 는 노드 안.** 설계 13개 노드 중 12개가 결정론이므로 전부를 ReAct 루프에
넣으면 테스트 가능성만 잃는다. 분기·HITL·루프는 그래프가 맡고, LLM 은 `llm_classify` 안에만 있다.

지금 컴파일되는 것은 8개다 (C — A 는 `llm_classify`·`rule_promote` 가 빠져 6개).
`review=True` 면 승격 자리가 검토 서브그래프로 바뀌어 바깥은 7상자, 그 안이 3노드다 (D-036②).
노드 경계는 **한 상자가 하는 일이 한 문장으로 끝나는가**로 정한다 (D-031).
"""

from __future__ import annotations

from txlens.graph.build import (
    build_checkpointer,
    build_graph,
    build_review_subgraph,
    checkpoint_types,
    pending_review,
    resume_command,
    resume_review,
    run_pipeline,
)
from txlens.graph.nodes import (
    DEFAULT_CONFIDENCE_TAU,
    DEFAULT_REVIEW_BATCH,
    apply_verdict,
    collect_targets,
    collect_targets_node,
    confidence_gate_node,
    finalize_node,
    human_review_node,
    ingest_node,
    llm_classify_node,
    normalize_node,
    pair_cancellations_node,
    route_after_gate,
    route_after_rules,
    rule_match_node,
    rule_promote_node,
)
from txlens.graph.state import (
    ReviewOutput,
    ReviewState,
    RunMetrics,
    TxState,
    merge_verdicts,
)

__all__ = [
    "DEFAULT_CONFIDENCE_TAU",
    "DEFAULT_REVIEW_BATCH",
    "ReviewOutput",
    "ReviewState",
    "RunMetrics",
    "TxState",
    "apply_verdict",
    "build_checkpointer",
    "build_graph",
    "build_review_subgraph",
    "checkpoint_types",
    "collect_targets",
    "collect_targets_node",
    "confidence_gate_node",
    "finalize_node",
    "human_review_node",
    "ingest_node",
    "llm_classify_node",
    "merge_verdicts",
    "normalize_node",
    "pair_cancellations_node",
    "pending_review",
    "resume_command",
    "resume_review",
    "route_after_gate",
    "route_after_rules",
    "rule_match_node",
    "rule_promote_node",
    "run_pipeline",
]
