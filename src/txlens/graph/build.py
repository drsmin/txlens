"""그래프 조립 (설계 §3.1).

## 같은 그래프로 A/B/C 를 돌린다

베이스라인이 셋인데 파이프라인이 셋이면 비교가 성립하지 않는다 — 어디까지가 조건 차이고 어디부터
코드 차이인지 알 수 없다. 그래서 바뀌는 것은 **주입물 둘**뿐이다.

| 베이스라인 | 사전 | 분류기 |
| --- | --- | --- |
| A · 룰만 | `rules/v1.yaml` | 없음 (`classifier=None` → LLM 노드가 그래프에 아예 없다) |
| B · LLM만 | 빈 사전 | 있음 (도구도 빈 사전을 본다 — 아니면 B 가 아니다) |
| C · 하이브리드 | `rules/v1.yaml` | 있음 |

`classifier=None` 일 때 노드를 만들지 않고 분기도 걸지 않는 이유: `get_graph().draw_ascii()` 가
**실제로 도는 경로**를 그린다. 노드를 만들어 두고 안 들르게 하면 그림이 거짓말을 한다.

## 승격은 노드로 있고, 루프는 그래프 밖에 있다

`rule_promote` 는 `llm_classify` 뒤에 있다 — 고신뢰 판정을 사전 후보로 바꾸는 자리다. 하지만
`rule_match` 로 되돌아가는 **사이클은 만들지 않는다.** 같은 실행에서 되돌면 방금 LLM 이 잡은
거래가 다음 바퀴에서 EXACT 로 잡혀 단계 분포가 흔들리고, 그 분포가 곧 베이스라인 비교표다.
라운드 사이에 두면(v1 → 승격 → v2 → 재측정) 라운드 1 은 그대로 두고 라운드 2 가 더해진다.
루프와 그 종료 조건은 `python -m txlens.promote` 가 갖는다.

## HITL 은 **서브그래프**이고, 기본값은 꺼져 있다

`confidence_gate` → `human_review` → `rule_promote` 셋은 `review` 서브그래프로 묶인다.
경계는 취향이 아니라 **"멈출 수 있는 부분"** 이다 — `interrupt()` 와 체크포인터가
필요한 노드가 정확히 이 셋이고, 나머지 경로는 배치로 끝까지 돈다.

`review=False` 가 기본값인 것은 이 그래프를 가장 많이 돌리는 것이 **평가 하네스**이기
때문이다. 사람이 없는 곳에서 `interrupt()` 를 만나면 실행이 멈춘 채로 끝나고, 그 멈춤은
"정확도가 낮다"가 아니라 "지표가 없다"로 나타난다. 켜고 끄는 것이 A/B/C 비교표의 전제를
지키는 장치이기도 하다 — 켠 그래프와 끈 그래프는 모양이 다르고, 발표에 쓰는 그림은 실제로
도는 쪽이어야 한다(`scripts/render-graph.py`).

`review=True` 면 `rule_promote` 는 서브그래프 안으로 들어간다. 바깥에도 두면 같은 후보가 두 번
만들어지고, 사람 교정 없이 만들어진 쪽이 먼저 사전 키를 차지한다.
"""

from __future__ import annotations

import functools
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from txlens.agents.classifier import MerchantClassifier
from txlens.graph import nodes
from txlens.graph.state import ReviewOutput, ReviewState, TxState
from txlens.rules import RuleDictionary, RuleMatcher
from txlens.rules.promote import MIN_PROMOTE_CONFIDENCE


def build_review_subgraph(
    *,
    dictionary: RuleDictionary,
    min_promote_confidence: float = MIN_PROMOTE_CONFIDENCE,
    review_batch: int = nodes.DEFAULT_REVIEW_BATCH,
    checkpointer: object | None = None,
):
    """검토 서브그래프 — `confidence_gate` → `human_review` → `rule_promote`.

    **혼자서도 돈다.** 번들도 거래도 필요 없고 `review_queue` 하나면 된다(`ReviewState`).
    그래서 Streamlit S1 은 파이프라인을 다시 돌리지 않고 이것만 부를 수 있고, 배치는 같은 것을
    노드로 품는다 — 화면과 배치가 같은 코드를 부르는 것이 이 분리의 목적이다.

    `checkpointer` 는 **단독으로 쓸 때만** 넘긴다. 부모 그래프의 노드로 들어갈 때는 부모의
    체크포인터가 서브그래프까지 덮으므로(LangGraph 가 네임스페이스를 나눠 준다) 여기서 또
    주면 같은 실행이 두 곳에 저장된다.
    """
    graph = StateGraph(ReviewState, output_schema=ReviewOutput)
    graph.add_node(
        "confidence_gate",
        functools.partial(nodes.confidence_gate_node, batch_size=review_batch),
    )
    graph.add_node("human_review", nodes.human_review_node)
    graph.add_node(
        "rule_promote",
        functools.partial(
            nodes.rule_promote_node,
            dictionary=dictionary,
            min_confidence=min_promote_confidence,
        ),
    )

    graph.add_edge(START, "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate",
        nodes.route_after_gate,
        {"human_review": "human_review", "rule_promote": "rule_promote"},
    )
    graph.add_edge("human_review", "rule_promote")
    graph.add_edge("rule_promote", END)
    return graph.compile(checkpointer=checkpointer)  # type: ignore[arg-type]


def build_graph(
    *,
    matcher: RuleMatcher | None = None,
    classifier: MerchantClassifier | None = None,
    tau: float = nodes.DEFAULT_CONFIDENCE_TAU,
    llm_target_limit: int | None = None,
    min_promote_confidence: float = MIN_PROMOTE_CONFIDENCE,
    review: bool = False,
    review_batch: int = nodes.DEFAULT_REVIEW_BATCH,
    checkpointer: object | None = None,
    store: object | None = None,
):
    """파이프라인 그래프를 만들어 컴파일한다.

    `review=True` 면 승격 자리에 검토 서브그래프가 들어간다 — 그때부터 이 그래프는 **멈출 수
    있는 물건**이 되고, 재개하려면 `checkpointer` 가 있어야 한다 (`InMemorySaver`).
    `store` 는 사전 namespace 용으로 그대로 넘긴다.

    `review=True` 인데 분류기가 없으면 **에러다.** 분류기가 없으면 검토 큐를 채우는 노드가
    없어 서브그래프가 언제나 빈 큐를 보고 지나간다 — 그림에는 HITL 이 붙어 있고 실제로는 아무
    일도 하지 않는, 이 프로젝트가 가장 경계하는 형태의 거짓 보고가 된다.
    """
    if review and classifier is None:
        raise ValueError(
            "review=True 는 분류기가 있어야 한다 — 검토 큐를 채우는 것은 `llm_classify` 이고, "
            "분류기가 없으면(베이스라인 A) 큐가 영원히 비어 HITL 이 그림에만 남는다"
        )
    matcher = matcher if matcher is not None else RuleMatcher.from_path()

    graph = StateGraph(TxState)
    graph.add_node("ingest", nodes.ingest_node)
    graph.add_node("pair_cancellations", nodes.pair_cancellations_node)
    graph.add_node("normalize", nodes.normalize_node)
    graph.add_node("rule_match", functools.partial(nodes.rule_match_node, matcher=matcher))
    graph.add_node("collect_targets", nodes.collect_targets_node)
    graph.add_node("finalize", functools.partial(nodes.finalize_node, classifier=classifier))

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "pair_cancellations")
    graph.add_edge("pair_cancellations", "normalize")
    graph.add_edge("normalize", "rule_match")
    graph.add_edge("rule_match", "collect_targets")

    if classifier is None:
        graph.add_edge("collect_targets", "finalize")
    else:
        graph.add_node(
            "llm_classify",
            functools.partial(
                nodes.llm_classify_node,
                classifier=classifier,
                tau=tau,
                limit=llm_target_limit,
            ),
        )
        # 승격의 자리는 하나다 — `review` 를 켜면 서브그래프가, 끄면 노드가 그 자리에 온다.
        if review:
            graph.add_node(
                "review",
                build_review_subgraph(
                    dictionary=matcher.dictionary,
                    min_promote_confidence=min_promote_confidence,
                    review_batch=review_batch,
                ),
            )
            promote_entry = "review"
        else:
            graph.add_node(
                "rule_promote",
                functools.partial(
                    nodes.rule_promote_node,
                    dictionary=matcher.dictionary,
                    min_confidence=min_promote_confidence,
                ),
            )
            promote_entry = "rule_promote"

        graph.add_conditional_edges(
            "collect_targets",
            nodes.route_after_rules,
            {"llm_classify": "llm_classify", "finalize": "finalize"},
        )
        graph.add_edge("llm_classify", promote_entry)
        graph.add_edge(promote_entry, "finalize")

    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer, store=store)  # type: ignore[arg-type]


def run_pipeline(
    bundle_dir: Path | str,
    *,
    matcher: RuleMatcher | None = None,
    classifier: MerchantClassifier | None = None,
    tau: float = nodes.DEFAULT_CONFIDENCE_TAU,
    llm_target_limit: int | None = None,
    min_promote_confidence: float = MIN_PROMOTE_CONFIDENCE,
) -> TxState:
    """그래프를 한 번 돌려 최종 상태를 돌려준다.

    평가 하네스·노트북·Streamlit 이 모두 이 함수를 부른다 — 실행 경로를 세 벌 만들지 않는다.
    그래프를 매번 새로 컴파일하는 비용은 무시할 수준이고, 대신 실행마다 캐시·카운터가
    깨끗한 상태에서 시작해 지표가 이전 실행에 오염되지 않는다.
    """
    graph = build_graph(
        matcher=matcher,
        classifier=classifier,
        tau=tau,
        llm_target_limit=llm_target_limit,
        min_promote_confidence=min_promote_confidence,
    )
    return graph.invoke({"bundle_dir": str(bundle_dir)})


# ── 체크포인터 ──────────────────────────────────────────────────────────────
def checkpoint_types() -> list[type]:
    """체크포인트에 실리는 우리 타입 전부.

    손으로 열거하지 않고 모듈에서 뽑는다. 허용 목록은 `(모듈, 이름)` **완전 일치**라 접두어로
    묶을 수 없고, 손으로 적으면 스키마가 하나 늘 때마다 조용히 빠진다 — 빠진 타입은 예외가
    아니라 **`dict` 로 되살아나서** 노드가 `AttributeError` 로 죽는다(실제로 겪었다).
    """
    from txlens import schemas
    from txlens.graph import state

    return [
        value
        for module in (schemas, state)
        for value in vars(module).values()
        if isinstance(value, type) and value.__module__ == module.__name__
    ]


def build_checkpointer():
    """검토 재개용 `InMemorySaver`.

    기본 직렬화기는 **모르는 타입도 일단 되살리고 경고를 남긴다.** 우리 상태는 거의 전부
    pydantic 모델이라 그 경고가 재개 한 번에 십수 줄 쏟아지고(실측 14줄), 데모 콘솔에서 그것은
    곧 "무언가 잘못됐다"로 읽힌다. 게다가 LangGraph 는 "다음 버전에서 막는다"고 예고한다 —
    막히는 날 재개가 통째로 깨지므로 경고인 동안 등록해 둔다. 명시하면 조용해지는 동시에 **더
    좁은** 설정이 된다: 목록 밖의 타입은 되살리지 않는다.

    `InMemorySaver` 인 것은 이 프로젝트가 한 프로세스 안에서 데모를 돌기 때문이다. 재시작을
    넘겨 살아남아야 하면 `langgraph-checkpoint-sqlite` 로 바꾸고 같은 `serde` 를 넘기면 된다.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=checkpoint_types()))


# ── 재개 ────────────────────────────────────────────────────────────────────
def pending_review(graph, config) -> dict | None:
    """멈춰 있는 검토 요청을 꺼낸다. 멈춰 있지 않으면 `None`.

    화면이 `interrupt` 예외를 직접 다루지 않게 하려고 둔다. Streamlit 은 새로고침마다 스크립트를
    처음부터 다시 도는데, 그때마다 "이 thread 가 지금 멈춰 있나"를 물어볼 곳이 필요하다 —
    그것을 `graph.get_state(...).tasks[...].interrupts[0].value` 로 매번 쓰게 하면 그 경로가
    화면 코드 안에 박혀 그래프 구조를 바꿀 때 화면이 조용히 깨진다.
    """
    for task in graph.get_state(config).tasks:
        for pending in task.interrupts:
            return pending.value
    return None


def resume_review(graph, config, corrections) -> TxState:
    """사람 교정을 넣어 멈춘 실행을 이어 돌린다 (`Command(resume=...)`).

    `{"corrections": [...]}` 로 감싸서 보낸다. 목록을 그대로 넘기면 LangGraph 쪽에서 "여러
    interrupt 에 하나씩 나눠 주는 값"과 모양이 겹친다 — 지금은 멈추는 노드가 하나뿐이라 구분이
    되지만, 그 사실에 기대는 코드는 노드가 늘어나는 날 조용히 어긋난다.

    빈 목록도 정당한 답이다. "이번 배치에서 고칠 것이 없다" 는 판단이고, 그때 승격되는 것은
    자동 후보뿐이다 — 사람을 부른 것이 곧 사람이 무언가를 고쳤다는 뜻은 아니다.
    """
    return graph.invoke(resume_command(corrections), config)


def resume_command(corrections) -> Command:
    """사람 교정 → `Command(resume=...)`. `resume_review` 와 화면이 같은 것을 보낸다.

    화면은 재개를 `stream` 으로 돌린다 (노드마다 로그를 찍기 위해서다). 그때 이 감싸는
    규칙을 화면이 다시 적으면, 규칙이 바뀌는 날 화면만 옛 모양을 보낸다 — 그래서 감싸는
    자리는 여기 하나다.
    """
    payload = [
        item if isinstance(item, dict) else item.model_dump(mode="json") for item in corrections
    ]
    return Command(resume={"corrections": payload})
