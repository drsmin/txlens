#!/usr/bin/env python3
"""컴파일된 LangGraph 를 그림으로 뽑는다 (발표 자료용).

왜 손으로 안 그리는가: 리포트의 S05 는 **설계도**다(9노드 — 5주차에 붙을 것 포함). 청중이
"그래서 지금 진짜 도는 건 뭐냐"고 물으면 답이 있어야 하는데, 그 답을 또 손으로 그리면 코드가
바뀔 때 조용히 어긋난다. 그래서 이 그림은 `build_graph()` 가 실제로 컴파일한 것에서만 나온다.

그림은 **세 층**이다. 파이프라인 그래프(`langgraph-a`·`langgraph-c`·`langgraph-c-hitl`)는
`build_graph()` 가 만든 바깥 골격이고, 나머지 둘은 그 안의 상자를 연 것이다.

| 그림 | 무엇 | 요점 |
| --- | --- | --- |
| `langgraph-agent` | `llm_classify` 노드 **안** (`create_agent`) | 가드레일은 설정이 아니라 **노드**다 (D-031) |
| `langgraph-review` | `review` 상자 **안** (`build_review_subgraph()`) | 멈출 수 있는 부분이 곧 서브그래프 경계다 (D-036②) |

`langgraph-c` 와 `langgraph-c-hitl` 을 **둘 다** 뽑는 이유: 평가 하네스가 도는 것은 앞의 것이고
(HITL 은 꺼져 있다) 데모가 도는 것은 뒤의 것이다. 하나만 그리면 그 그림이 둘 중 하나에 대해
거짓말한다.

산출물 (`assets/diagrams/`):
    langgraph-<v>.mmd   mermaid 원문 — **드리프트 판정 기준**. 오프라인이고 커밋한다
    langgraph-<v>.png   그림 — pptx/docx 가 넣는다. 커밋한다 (발표장에 네트워크가 없을 수 있다)
    langgraph-<v>.txt   draw_ascii() — MD 코드펜스에 붙이는 판 (GitHub·한글 폰트 없는 환경용)

사용:
    python scripts/render-graph.py            # 다시 그린다 (PNG 는 네트워크 필요)
    python scripts/render-graph.py --check    # 커밋된 .mmd 와 지금 그래프가 같은지만 본다 (오프라인)

⚠ PNG 렌더는 LangGraph 기본 경로라 **mermaid.ink 로 그래프 원문이 나간다**(노드 이름뿐 —
거래 데이터·가맹점명은 없다). 그 판단은 D-029. 그래서 `--check` 는 네트워크를 쓰지 않는다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# 출력 인코딩을 콘솔 코드페이지에서 뗀다. Windows 한국어 콘솔(cp949)에서는 "✓" 하나 때문에
# UnicodeEncodeError 로 죽고, 이 스크립트는 pre-commit 훅이 부르므로 **이유 없이 커밋이 막힌다**
# (실제로 훅을 고쳐 검사가 처음 돌자마자 이걸로 터졌다). 깨져 보이는 건 죽는 것보다 낫다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # 파이프로 감싸인 경우 등
        pass

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "diagrams"
sys.path.insert(0, str(ROOT / "src"))


class _AnyClassifier:
    """분류기 자리에 놓는 표식.

    `build_graph()` 가 보는 것은 `classifier is None` 하나뿐이다 (`llm_classify` 노드와 분기를
    만들지 말지). 그림을 뽑자고 API 키를 요구할 이유가 없어서 실물 대신 이것을 넘긴다.
    """


# (파일 이름, 라벨, classifier, review) — A 와 C 의 그래프가 실제로 다르다는 것 자체가 근거다.
VARIANTS = [
    ("langgraph-a", "A · 룰만 (classifier=None)", None, False),
    ("langgraph-c", "C · 하이브리드 (classifier 주입)", _AnyClassifier(), False),
    ("langgraph-c-hitl", "C + HITL (review=True — 데모가 도는 그래프)", _AnyClassifier(), True),
]

#: 그림을 뽑을 때만 쓰는 자리표시 키. `create_agent` 는 조립 시점에 모델 클라이언트를 만들지만
#: 우리는 `get_graph()` 만 부르고 **invoke 하지 않는다** — 네트워크도 과금도 없다. 진짜 키가
#: 환경에 있으면 그것을 그대로 쓴다(그래프 모양은 어차피 같다).
_RENDER_ONLY_KEY = "sk-render-only-not-used"


def _mermaid(graph) -> str:
    """그래프 객체에서 mermaid 원문을 만든다. **Agent 그래프 전용이다.**

    파이프라인 그래프는 `draw_mermaid()` 를 그대로 쓴다. Agent 그래프만 직접 만드는 이유는 하나 —
    미들웨어 노드 이름에 대괄호가 들어간다(`PIIMiddleware[credit_card].before_model`). LangChain 의
    `draw_mermaid()` 는 그 이름을 따옴표 없이 `id(label)` 에 넣어서 mermaid 파서가 대괄호를
    노드 문법으로 읽고, mermaid.ink 가 400 으로 거절한다. 라벨을 따옴표로 감싸면 통과한다.

    노드 이름과 간선은 **컴파일된 그래프에서만** 온다 — 손으로 적는 것은 따옴표뿐이다.
    """
    ids: dict[str, str] = {}
    for key in graph.nodes:
        safe = re.sub(r"[^0-9A-Za-z_]", "_", key)
        if safe in ids.values():
            raise ValueError(f"노드 id 충돌: {key} → {safe}")
        ids[key] = safe

    lines = ["---", "config:", "  flowchart:", "    curve: linear", "---", "graph TD;"]
    for key, node in graph.nodes.items():
        label = str(getattr(node, "name", key))
        if key in ("__start__", "__end__"):
            lines.append(f'\t{ids[key]}(["{label}"])')
        else:
            lines.append(f'\t{ids[key]}("{label}")')
    for edge in sorted(graph.edges, key=lambda e: (e.source, e.target)):
        arrow = "-.->" if edge.conditional else "-->"
        lines.append(f"\t{ids[edge.source]} {arrow} {ids[edge.target]};")
    return "\n".join(lines) + "\n"


def agent_graph():
    """`llm_classify` 안의 Agent 그래프. 바깥 그래프와 **다른 층**이라 따로 뽑는다."""
    import os

    os.environ.setdefault("OPENAI_API_KEY", _RENDER_ONLY_KEY)
    from txlens.agents.classifier import build_classifier

    return build_classifier("openai:gpt-4o-mini").agent.get_graph()


def review_graph():
    """`review` 상자 안. 바깥에서는 한 상자지만 안에는 분기 하나와 멈추는 노드가 있다."""
    from txlens.graph.build import build_review_subgraph
    from txlens.rules import load_dictionary

    return build_review_subgraph(dictionary=load_dictionary()).get_graph()


def graphs():
    from txlens.graph.build import build_graph  # 런타임이 없으면 ImportError — check() 가 받는다

    for stem, label, clf, review in VARIANTS:
        yield stem, label, build_graph(classifier=clf, review=review).get_graph()
    yield "langgraph-review", "review 서브그래프 (confidence_gate → human_review)", review_graph()
    yield "langgraph-agent", "llm_classify 안의 Agent (create_agent)", agent_graph()


def check() -> int:
    """커밋된 .mmd 와 지금 컴파일되는 그래프를 비교한다. 네트워크를 쓰지 않는다.

    런타임(langgraph)이 없으면 **건너뛴다** — pre-commit 훅이 이걸 부르는데, 가상환경 밖에서
    커밋한다고 커밋이 막히면 안 된다. 검사는 환경이 있는 곳에서 돌면 된다.
    """
    bad = []
    try:
        pairs = list(graphs())
    except ImportError as e:
        print(f"  · 그래프 그림 검사 건너뜀 (런타임 없음: {e.name})")
        return 0
    for stem, label, g in pairs:
        f = OUT / f"{stem}.mmd"
        now = _mermaid(g) if stem == "langgraph-agent" else g.draw_mermaid()
        if not f.exists():
            bad.append(f"{f.name} 없음 — python scripts/render-graph.py")
        elif f.read_text(encoding="utf-8") != now:
            bad.append(f"{f.name} 가 지금 그래프와 다르다 ({label}) — 다시 렌더해야 한다")
    if bad:
        print("✗ 발표 자료의 그래프 그림이 코드와 어긋났다:")
        for b in bad:
            print(f"    {b}")
        return 1
    print(f"✓ 그래프 그림이 코드와 일치 ({len(pairs)}종)")
    return 0


def _write(path: Path, text: str) -> None:
    r"""줄바꿈을 `\n` 으로 못박아 쓴다. **드리프트 판정 기준이 되는 파일이라 그렇다.**

    `write_text` 는 플랫폼 줄바꿈으로 번역한다. `draw_ascii()` 가 이미 `\r\n` 을 내는 판이면
    Windows 에서 `\r\r\n` 이 되고, `--check` 는 읽을 때 다시 번역돼 통과하지만 git 은 파일이
    통째로 바뀐 것으로 본다 — macOS 와 Windows 가 커밋할 때마다 이 파일을 서로 되돌리게 된다.
    """
    path.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


def _png(mermaid: str) -> bytes:
    """mermaid 원문 → PNG. `draw_mermaid_png()` 와 같은 경로(mermaid.ink)를 쓴다.

    연속 호출이 429/400 으로 튕기는 일이 있어 재시도를 넉넉히 준다 — 실패해도 기존 PNG 를
    그대로 두므로 발표 자료가 빈 자리로 나가지는 않는다.
    """
    from langchain_core.runnables.graph_mermaid import draw_mermaid_png

    return draw_mermaid_png(mermaid, max_retries=5, retry_delay=2.0)


def render() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, label, g in graphs():
        mermaid = _mermaid(g) if stem == "langgraph-agent" else g.draw_mermaid()
        _write(OUT / f"{stem}.mmd", mermaid)
        # ASCII 는 **파이프라인 그래프만** 낸다. Agent 그래프는 미들웨어가 앞뒤로 갈라져
        # `draw_ascii()` 폭이 수백 칸이 되고, 그건 코드펜스에 붙일 수 있는 물건이 아니다.
        if stem != "langgraph-agent":
            _write(OUT / f"{stem}.txt", g.draw_ascii() + "\n")
        try:
            png = _png(mermaid) if stem == "langgraph-agent" else g.draw_mermaid_png()
        except Exception as e:  # noqa: BLE001 — 실패 종류가 많다(네트워크·HTTP·타임아웃)
            print(f"  ! {stem}.png 실패 ({type(e).__name__}) — 기존 PNG 를 그대로 둔다")
            continue
        (OUT / f"{stem}.png").write_bytes(png)
        print(f"  · {stem}.png  {len(png) // 1024} KB   {label}")
    print(f"→ {OUT.relative_to(ROOT)}/  (.mmd 는 드리프트 판정 기준 — 셋 다 커밋한다)")
    return 0


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv[1:] else render())
