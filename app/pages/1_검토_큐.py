"""S1 · 검토 큐 (HITL) — 이 프로젝트만 보여줄 수 있는 화면 (설계 §3.5).

여기서 일어나는 일은 한 문장이다. **그래프가 `interrupt()` 로 멈춰 있고, 사람이 표에서 몇 줄을
고치면, 그 교정이 룰 사전 v{n+1} 이 되어 거래 N건을 즉시 덮는다.**

## 화면이 상태를 들고 있지 않다

Streamlit 은 위젯을 건드릴 때마다 스크립트를 처음부터 다시 돈다. 보통은 그것이 상태 관리
문제가 되는데, 여기서는 그래프 상태가 체크포인터에 있어서 화면이 들고 있을 것이 `thread_id`
와 이번 라운드의 그래프뿐이다 — HITL 설계의 부수 효과다 (설계 §3.5).

그래프를 `@st.cache_resource` 가 아니라 세션에 두는 이유는 **라운드마다 사전이 다르기**
때문이다. 캐시로 묶으면 승격된 사전이 반영되지 않은 채 라운드 2 가 돌고, 그러면 이 화면이
증명하려는 "호출이 줄어든다"가 거짓이 된다.

## 재생으로 돈다 — 이 화면은 API 를 부르지 않는다

분류기는 저장된 판정(`data/gold/verdicts-*.json`)이다. 발표장에서 네트워크가 죽어도 같은
장면이 나오고, 라운드를 몇 번 돌려도 비용이 0 이다. 대신 라운드 2 의 재생은 실호출과 다르다 —
대본은 v1 사전 위에서 받은 답이라 커진 사전을 도구로 조회하지 못한다. 화면에도 그렇게 적는다.
"""

from __future__ import annotations

import time
import uuid

import pandas as pd
import streamlit as st

import txlens_ui as ui
from txlens.graph import pending_review, resume_command
from txlens.graph.nodes import DEFAULT_CONFIDENCE_TAU, DEFAULT_REVIEW_BATCH
from txlens.rules import load_dictionary
from txlens.rules.promote import apply_proposals

st.set_page_config(page_title="TxLens · 검토 큐", page_icon="🧾", layout="wide")
st.title("S1 · 검토 큐 (HITL)")


#: `data_editor` 한 행의 높이(px)와 머리글. 배치가 20이면 표도 20줄을 보여야 한다 —
#: 화면이 "이번 배치 20개"라고 적어 놓고 10줄만 보이면 발표자가 스크롤로 나머지를 찾는다.
ROW_PX, HEADER_PX, MAX_TABLE_PX = 35, 45, 780


def _table_height(rows: int) -> int:
    return min(rows * ROW_PX + HEADER_PX, MAX_TABLE_PX)


@st.cache_resource(show_spinner=False)
def _classifier():
    """재생 대본은 라운드가 바뀌어도 같으므로 프로세스에 한 번만 읽는다."""
    return ui.load_replay_classifier()


def _stream(graph, payload, config, *, title: str, pane, detail: str) -> list[str]:
    """그래프를 돌리며 노드마다 로그 한 줄을 `pane` 에 흘린다. 돌려주는 것은 찍은 줄 전부.

    `invoke` 로 돌리면 8노드가 스피너 하나 뒤에 통째로 숨는다 — 화면이 결과만 말하고 과정을
    안 보여주면, 이 프로젝트가 설계한 그래프는 발표에서 말로만 남는다. `subgraphs=True` 를
    주는 이유는 HITL 이 **서브그래프 안에서** 멈추기 때문이다 — 없으면 `interrupt`
    직전의 `confidence_gate` 가 안 보인다.

    **로그는 본문 옆(`pane`)에 그린다.** 아래에 두면 사람이 표를 고치는 동안 그래프가 무엇을
    했는지가 화면 밖으로 밀려난다 — 이 화면이 증명하려는 것이 그 둘의 인과라서, 같은 화면에
    동시에 보여야 한다.

    `stream_mode` 는 **항상 리스트로** 준다. 문자열로 주면 `subgraphs=True` 와 묶여 2-튜플이
    오고, `debug` 를 켤 때만 3-튜플이 되어 코드가 두 갈래가 된다.
    """
    modes = ["updates", "debug"] if detail == ui.DETAIL_DEBUG else ["updates"]
    lines: list[str] = []
    stopped = False  # interrupt 는 서브그래프와 부모에서 두 번 온다 — 한 번만 찍는다
    with pane, st.status(title, expanded=True) as status:
        holder = st.empty()
        t0 = time.perf_counter()

        def emit(label: str, text: str) -> None:
            # 라벨은 ASCII 로 둔다 — 한글은 글자 폭이 2 라 `:<{NODE_WIDTH}` 정렬이 어긋난다
            lines.append(f"[{time.perf_counter() - t0:5.2f}s] {label:<{ui.NODE_WIDTH}} {text}")
            holder.code("\n".join(lines), language="text")

        for _ns, mode, chunk in graph.stream(payload, config, stream_mode=modes, subgraphs=True):
            if mode == "debug":
                text = ui.debug_line(chunk)
                if text is not None:
                    emit(str((chunk.get("payload") or {}).get("name") or "graph"), f"· {text}")
                continue
            for node, update in chunk.items():
                if node == "__interrupt__":
                    if stopped:
                        continue
                    stopped = True
                text = ui.log_line(node, update)
                if text is None:
                    continue
                label = "interrupt" if node == "__interrupt__" else node
                emit(label, text)
                if detail != ui.DETAIL_SUMMARY:
                    lines.extend(ui.raw_lines(update))
                    holder.code("\n".join(lines), language="text")
        status.update(label=f"{title} — {time.perf_counter() - t0:.1f}초", state="complete")
    return lines


def _start_round(dictionary, *, batch: int, tau: float, pane, detail: str) -> None:
    """라운드 하나를 시작한다 — 그래프를 만들고 `interrupt()` 에 걸릴 때까지 돌린다."""
    classifier, _model, _n = _classifier()
    graph = ui.build_review_graph(dictionary, classifier, batch=batch, tau=tau)
    config = {"configurable": {"thread_id": f"s1-{uuid.uuid4().hex[:8]}"}}
    round_no = len(st.session_state.get("rounds", [])) + 1
    lines = _stream(
        graph,
        {"bundle_dir": str(ui.BUNDLE_DIR)},
        config,
        title=f"라운드 {round_no} · 사전 v{dictionary.version} — 파이프라인 실행",
        pane=pane,
        detail=detail,
    )
    st.session_state.update(
        graph=graph, config=config, dictionary=dictionary, summary=None, final=None
    )
    st.session_state.setdefault("logs", []).append(
        {"title": f"라운드 {round_no} · 사전 v{dictionary.version} — 파이프라인 실행", "lines": lines}
    )


# ── 레이아웃 ────────────────────────────────────────────────────────────────
# **본문 | 로그**. 6:4 로 나눈 이유는 로그 한 줄이 70자 남짓이라서다 — 3할로 줄이면 줄마다
# 접혀 노드 이름 정렬(`NODE_WIDTH`)이 무너진다. 사이드바보다 먼저 만드는 것은 사이드바의
# "처음부터 다시" 버튼이 **이 칸에** 로그를 흘려야 하기 때문이다.
body, logpane = st.columns([6, 4], gap="medium")

# ── 사이드바 ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("실행 설정")
    batch = st.slider("배치 상한 (한 번에 볼 줄 수)", 5, 50, DEFAULT_REVIEW_BATCH, step=5)
    tau = st.slider("검토 문턱 τ", 0.5, 0.95, DEFAULT_CONFIDENCE_TAU, step=0.05)
    reviewer = st.text_input("검토자", value="demo")
    detail = st.selectbox(
        "로그 상세도", ui.DETAIL_LEVELS, index=ui.DETAIL_LEVELS.index(ui.DETAIL_SUMMARY)
    )
    st.caption(
        "**기본이 `요약` 인 것은 이 화면이 녹화돼 발표에 들어가기 때문이다** — 영상이 "
        "슬라이드 크기로 줄면 45줄은 안 읽힌다. `debug` 는 LangGraph 의 태스크 시작·종료와 "
        "체크포인트까지 찍는다: 질문을 받았을 때, 또는 보조 영상을 따로 뜰 때 올린다."
    )
    _, model, n_verdicts = _classifier()
    st.caption(f"재생 대본 · {model} · 판정 {n_verdicts}개 — **API 를 부르지 않는다**")
    if st.button("처음부터 다시 (사전 v1)", width="stretch"):
        st.session_state.rounds = []
        st.session_state.logs = []
        _start_round(load_dictionary(), batch=batch, tau=tau, pane=logpane, detail=detail)
        st.rerun()

if "graph" not in st.session_state:
    st.session_state.rounds = []
    st.session_state.logs = []
    _start_round(load_dictionary(), batch=batch, tau=tau, pane=logpane, detail=detail)
    # 실행 중에는 `st.status` 가 로그를 실시간으로 흘리고, 끝나면 아래 로그 패널이 같은 것을
    # 들고 있다 — 다시 그리지 않으면 화면에 같은 로그가 두 번 남는다.
    st.rerun()

graph = st.session_state.graph
config = st.session_state.config
dictionary = st.session_state.dictionary
payload = pending_review(graph, config)

# ── 본문 ────────────────────────────────────────────────────────────────────
with body:
    # ── 상단 지표 ──────────────────────────────────────────────────────────────────────────────
    round_no = len(st.session_state.rounds) + 1
    cols = st.columns(4)
    cols[0].metric("라운드", round_no)
    cols[1].metric("룰 사전", f"v{dictionary.version}", f"{len(dictionary.entries)}개 항목")
    if payload:
        cols[2].metric("검토 큐", f"{payload['queued']}개 표기")
        cols[3].metric("이번 배치", f"{len(payload['items'])}개", "우선순위 = 건수 × 불확실성")

    # ── 표 ──────────────────────────────────────────────────────────────────────────────────
    if payload:
        st.markdown(
            f"신뢰도 **τ={tau:.2f} 미만**인 표기가 큐에 있고, 그중 우선순위 상위 "
            f"**{len(payload['items'])}개**가 아래다. **반영**에 체크한 줄만 사전으로 올라간다."
        )
        edited = st.data_editor(
            pd.DataFrame(ui.queue_rows(payload)),
            key=f"review-{round_no}",
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            height=_table_height(len(payload["items"])),
            column_config={
                ui.COL_APPLY: st.column_config.CheckboxColumn("반영", width="small"),
                ui.COL_NAME: st.column_config.TextColumn("표기", disabled=True, width="medium"),
                "건수": st.column_config.NumberColumn("덮는 거래", disabled=True, width="small"),
                ui.COL_CANONICAL: st.column_config.TextColumn("대표명(교정)", width="medium"),
                ui.COL_CATEGORY: st.column_config.SelectboxColumn(
                    "분류(교정)", options=list(ui.category_choices()), width="large"
                ),
                "신뢰도": st.column_config.ProgressColumn(
                    "신뢰도", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "LLM 대표명": st.column_config.TextColumn("LLM 대표명", disabled=True),
                "근거": st.column_config.TextColumn("LLM 추론 근거", disabled=True, width="medium"),
                "판단재료": st.column_config.TextColumn("판단 재료", disabled=True, width="small"),
                "우선순위": st.column_config.NumberColumn("우선순위", disabled=True, width="small"),
            },
        )

        if st.button("교정 제출 → 사전 승격", type="primary"):
            corrections, problems = ui.rows_to_corrections(
                edited.to_dict("records"), reviewer=reviewer
            )
            for problem in problems:
                st.error(f"올리지 못했다 — {problem}")
            lines = _stream(
                graph,
                resume_command(corrections),
                config,
                title=f"라운드 {round_no} · 사람 교정 {len(corrections)}건을 넣고 재개",
                pane=logpane,
                detail=detail,
            )
            st.session_state.setdefault("logs", []).append(
                {"title": f"라운드 {round_no} — 재개 (사람 교정 {len(corrections)}건)", "lines": lines}
            )
            final = graph.get_state(config).values
            proposals = final.get("rule_proposals", [])
            promoted, accepted = apply_proposals(dictionary, proposals)
            st.session_state.update(
                final=final,
                summary=ui.summarize_promotion(proposals, accepted),
                promoted=promoted,
            )
            st.session_state.rounds.append(
                {
                    "라운드": round_no,
                    "사전": f"v{dictionary.version}",
                    "LLM 표기": final["metrics"].llm_targets,
                    "검토 큐": final["metrics"].review_queued,
                    "사람 교정": final["metrics"].human_corrected,
                }
            )
            st.rerun()

    # ── 승격 결과 ──────────────────────────────────────────────────────────────────────────────
    summary = st.session_state.get("summary")
    if summary is not None:
        promoted = st.session_state.promoted
        st.subheader("승격 결과 — 이 교정이 사전을 어떻게 바꿨나")
        cols = st.columns(4)
        cols[0].metric(
            "룰 사전",
            f"v{dictionary.version} → v{promoted.version}",
            f"항목 {len(promoted.entries) - len(dictionary.entries):+d}개",
        )
        cols[1].metric("사람이 만든 새 브랜드", f"{len(summary.new_brands)}개")
        cols[2].metric("사람 교정이 덮는 거래", f"{summary.human_tx:,}건")
        cols[3].metric("자동 승격이 덮는 거래", f"{summary.auto_tx:,}건", f"별칭 {len(summary.auto)}개")

        if summary.human:
            st.markdown(
                "**사람 교정이 만든 승격.** `exact` 는 사전에 없던 브랜드다 — "
                "**자동 승격은 이것을 만들지 못한다** (사전이 판정의 검증자이므로)."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "패턴": p.pattern_type,
                            "표기": p.pattern,
                            "대표명": p.merchant_canonical,
                            "분류": p.category_l2.value,
                            "즉시 반영 거래": p.covers_tx_count,
                        }
                        for p in summary.human
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("이번 배치에서 고친 것이 없다 — 그것도 정당한 답이고, 자동 승격만 올라갔다.")

        if summary.rejected:
            st.caption(
                f"제안 {len(summary.rejected)}개는 사전이 받지 않았다 — 이미 쓰인 키이거나, "
                "대표명이 사전에 없는 모델 출처 후보다. 제안 수를 승격 수로 보고하지 않는다."
            )

        if st.button("승격된 사전으로 다시 돌린다 (라운드 +1)"):
            _start_round(promoted, batch=batch, tau=tau, pane=logpane, detail=detail)
            st.rerun()

    # ── 라운드 이력 ─────────────────────────────────────────────────────────────────────────────
    if st.session_state.get("rounds"):
        st.subheader("라운드별 — LLM 에 보낸 표기 수")
        history = pd.DataFrame(st.session_state.rounds).set_index("라운드")
        st.dataframe(history, width="stretch")
        # `st.line_chart` 는 라운드를 연속형 축으로 잡아 **라운드 1.5** 에 눈금을 세운다.
        # 순서형으로 못 박은 차트를 S2 와 같은 함수에서 만든다 (`ui.round_chart`).
        st.altair_chart(
            ui.round_chart(
                st.session_state.rounds, value="LLM 표기", title="라운드별 LLM 에 보낸 표기 수"
            ),
            width="stretch",
        )
        st.caption(
            "재생 대본은 v1 사전 위에서 받은 답이라 라운드 2 의 Agent 는 커진 사전을 도구로 "
            "조회하지 못한다 — 실호출로 한 번 확인하는 것이 남아 있다 (`agents/replay.py`)."
        )

    if payload is None and summary is None:
        st.success("검토할 것이 없다 — 큐가 비면 그래프는 사람을 부르지 않고 끝까지 돈다.")


# ── 실행 로그 (오른쪽 칸) ───────────────────────────────────────────────────
# 화면은 결과만이 아니라 **과정**을 보여야 한다. 라운드 2 의 `collect_targets` 가 라운드 1 보다
# 작은 수를 찍는 것 — 사람이 고친 세 줄이 다음 라운드의 LLM 호출을 줄인다는 이 프로젝트의
# 주장이 그 두 줄 사이에서 눈으로 확인된다. 그래서 지난 라운드 로그를 지우지 않고 접어 둔다.
#
# 본문보다 **뒤에** 그린다 — 앞에 두면 제출 중 흐르는 live 로그가 지난 라운드 아래로 밀린다.
logs = st.session_state.get("logs", [])
with logpane:
    st.subheader("실행 로그 — 그래프가 무엇을 했나")
    if not logs:
        st.caption("아직 돌린 것이 없다.")
    for i, entry in enumerate(reversed(logs)):
        with st.expander(entry["title"], expanded=(i == 0)):
            st.code("\n".join(entry["lines"]), language="text")
    st.caption(f"상세도: {detail}")
