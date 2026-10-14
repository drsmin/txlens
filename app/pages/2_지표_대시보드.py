"""S2 · 지표 대시보드 — 주장 하나하나에 측정을 붙이는 화면 (설계 §5).

S1 이 "루프가 돈다"를 보여준다면 여기는 **"그래서 숫자가 어떻게 달라졌나"** 를 보여준다.
화면에 네 덩이가 있고, 각각이 발표의 한 주장에 대응한다.

① 타일 — 룰만(A) 에서 하이브리드(C) 로 가며 무엇이 좋아졌나
② §5.1 베이스라인 3종 — **B(LLM만) < A(룰만)**. 이 프로젝트의 가설이 틀렸던 자리다
③ §5.2 전이 실험 — 같은 탐지기를 다른 분류 위에 올린다. 1부와 2부를 잇는 논거
④ 승격 라운드 — 사전이 자라며 LLM 표기가 155 → 121 로 준다

## 숫자는 지금 계산한다 — 다만 (B) 만은 그럴 수 없다

①③④ 와 (A)·(C) 는 이 자리에서 하네스를 돌려 만든다. 전부 합쳐 몇 초고 API 를 부르지
않는다(C 는 저장된 판정을 재생한다). 화면이 파일에 적힌 수를 읽어 오면 사전이나 매칭을
고친 날 화면만 옛 수를 말하는데, 그 어긋남은 발표 중에는 보이지 않는다.

(B) 는 사전 없이 표기 **전체**를 LLM 에 보내는 구성이라 재생할 대본이 없다(대본에는 C 가
보낸 149개뿐이다). 그래서 (B) 한 줄만 실호출 스냅샷에서 읽고, **스냅샷에 함께 들어 있는
(A)·(C) 는 지금 계산한 값과 대조해 어긋나면 화면이 말한다** — 스냅샷이 낡았다는 것은 곧
슬라이드가 낡았다는 뜻이므로 발표 전에 알아야 한다.

## 여기서는 캐시가 옳다

S1 은 그래프를 캐시에 두면 안 됐다 — 라운드마다 사전이 다르기 때문이다. 이 화면은
반대다. 세 계산 모두 `rules/v1.yaml` 과 번들에 고정돼 있고 세션 중에 사전을 바꾸지 않는다.
그래서 `@st.cache_data` 가 맞고, 없으면 사이드바를 건드릴 때마다 7초를 다시 기다린다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import txlens_ui as ui
from txlens.eval import run_rules_baseline
from txlens.eval.harness import run_baseline_state
from txlens.eval.transfer import run_transfer
from txlens.promote import run_rounds
from txlens.rules import load_dictionary

st.set_page_config(page_title="TxLens · 지표", page_icon="📊", layout="wide")
st.title("S2 · 지표 대시보드")


# ── 계산 ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="베이스라인 (A)·(C) 를 재는 중 — API 는 부르지 않는다")
def _baselines() -> tuple[list[ui.BaselineRow], dict]:
    """(A) 와 (C) 를 지금 돌린다. 두 번째 값은 타일이 쓰는 원시 수치."""
    classifier, _model, _n = ui.load_replay_classifier()
    a = run_rules_baseline(bundle_dir=ui.BUNDLE_DIR)
    c, metrics_c, _state = run_baseline_state(
        label="C · 하이브리드",
        dictionary=load_dictionary(),
        classifier=classifier,
        bundle_dir=ui.BUNDLE_DIR,
    )
    rows = [
        ui.row_from_result(a, None, source=ui.SRC_LIVE),
        ui.row_from_result(c, metrics_c, source=ui.SRC_REPLAY),
    ]
    tiles = {
        "keys": a.distinct.keys,
        "tx": a.overall.labeled + a.overall.unlabeled,
        "a": rows[0],
        "c": rows[1],
    }
    # `BaselineRow` 는 dataclass 라 캐시가 그대로 저장한다 — 표는 화면에서 만든다.
    return rows, tiles


@st.cache_data(show_spinner="§5.2 전이 실험 — 같은 탐지기를 네 분류 위에서 돌린다")
def _transfer() -> list[dict]:
    rows = run_transfer(bundle_dir=ui.BUNDLE_DIR, verdicts_path=ui.VERDICTS_PATH)
    out = []
    for row in rows:
        d = row.detection
        out.append(
            {
                "분류 단계": row.label,
                "미묶음": f"{row.unmatched_rate:.1%}",
                "구독 정밀도": f"{d.subscription.precision:.1%}",
                "구독 재현율": f"{d.subscription.recall:.1%}",
                "이상지출 재현율": f"{d.spending.recall:.1%}",
                "이상거래 재현율": f"{d.tx_pattern.recall:.1%}",
                "알림/고객·월": round(d.alerts_per_customer_month, 1),
            }
        )
    return out


@st.cache_data(show_spinner="승격 루프를 고정점까지 — 저장된 판정으로 돈다")
def _rounds() -> list[dict]:
    classifier, model, _n = ui.load_replay_classifier()
    reports = run_rounds(model, replay=classifier.verdicts, bundle_dir=ui.BUNDLE_DIR)
    return ui.round_rows(reports)


live_rows, tiles = _baselines()
snapshot_rows = ui.load_snapshot_rows()
rows, drift = ui.merge_baseline_rows(live_rows, snapshot_rows)

st.caption(
    f"분류 대상 {tiles['tx']:,}건 · 정규화 키 {tiles['keys']}개 · 룰 사전 "
    f"v{load_dictionary().version} · 재생 대본 `{ui.VERDICTS_PATH.name}` — "
    "**이 화면은 API 를 부르지 않는다**"
)

# ── ① 타일 ──────────────────────────────────────────────────────────────────
st.subheader("룰만(A) → 하이브리드(C) — 무엇이 좋아졌나")
a, c = tiles["a"], tiles["c"]
cols = st.columns(4)
cols[0].metric(
    "거래 L2 정확도 (C)", f"{c.accuracy_l2:.1%}",
    f"{ui.delta_pp(c.accuracy_l2, a.accuracy_l2):+.1f}%p vs A",
)
cols[1].metric(
    "정규화 키 L2 정확도 (C)", f"{c.distinct_l2:.1%}",
    f"{ui.delta_pp(c.distinct_l2, a.distinct_l2):+.1f}%p vs A",
)
cols[2].metric(
    "미분류율 (C)", f"{c.unmatched_rate:.1%}",
    f"{ui.delta_pp(c.unmatched_rate, a.unmatched_rate):+.1f}%p vs A", delta_color="inverse",
)
cols[3].metric(
    "룰만의 확정 커버리지 (A)", f"{a.coverage:.1%}",
    f"{ui.delta_pp(c.coverage, a.coverage):+.1f}%p → C {c.coverage:.1%} (표기 {c.llm_targets}개)",
)
st.caption(
    "**키 기준 정확도가 거래 기준보다 훨씬 낮다** — 상위 브랜드가 거래 수를 덮기 때문이다. "
    f"그 격차({a.accuracy_l2:.0%} 대 {a.distinct_l2:.0%})가 롱테일이고, 이 프로젝트가 "
    f"LLM 과 사람을 붙이는 이유다. PG마스킹 {a.pg_masked_rate:.1%} 는 실가맹점명이 오지 "
    "않은 몫이라 어떤 구성으로도 줄지 않는다."
)

# ── ② §5.1 비교표 ───────────────────────────────────────────────────────────
st.subheader("§5.1 베이스라인 3종 — 내 가설이 틀렸던 자리")
st.dataframe(
    pd.DataFrame([r.to_row() for r in rows]), hide_index=True, width="stretch",
)

if not any(r.label.startswith("B") for r in rows):
    st.warning(
        "**(B) LLM만 줄이 비어 있다.** 사전 없이 표기 전체를 보내는 구성이라 재생 대본으로는 "
        "되살릴 수 없다 — 실호출 스냅샷이 있어야 한다. `.env` 에 `OPENAI_API_KEY` 를 넣고 "
        f"한 번 돌리면 이 줄이 채워진다:\n\n```\n{ui.SNAPSHOT_CMD}\n```"
    )
else:
    st.markdown(
        "- **가설:** \"(B) 가 정확도 상한이고 (C) 는 그 근처에서 비용을 아낀다.\" "
        "→ **(B) 가 (A) 보다 낮았다.** 사전 없이 LLM 만 쓰면 커버리지는 올라가는데 정확도가 "
        "떨어진다 — 답을 더 많이 내지만 그만큼 **자신 있게 틀린다**\n"
        "- 정확도와 커버리지를 한 수로 합쳐 보고하지 않는다. 하나만 내면 어느 쪽이든 왜곡된다"
    )

if drift:
    st.info(
        "**지금 계산한 값과 실호출 스냅샷이 다르다.**\n\n"
        + "\n".join(f"- {note}" for note in drift)
        + "\n\n(C) 의 미세한 차이는 재생의 원리적 한계다 — 대본에 없는 표기는 실패로 남는다. "
        "(A) 가 어긋났다면 스냅샷을 뜬 뒤 사전이나 매칭이 바뀐 것이고, **슬라이드의 수가 "
        "낡았다는 뜻**이다."
    )

# ── ③ §5.2 전이 ─────────────────────────────────────────────────────────────
st.subheader("§5.2 분류 → 탐지 전이 — \"가맹점명을 못 묶으면 구독도 못 찾는다\"")
st.dataframe(pd.DataFrame(_transfer()), hide_index=True, width="stretch")
st.caption(
    "**탐지기는 네 줄에서 완전히 같다.** 바뀌는 것은 그 위에 놓인 분류뿐이다 — 그래서 이 표의 "
    "차이는 전부 1부가 만든 것이다. '정답 분류' 는 1부가 완벽할 때의 상한이라, 그 아래 "
    "칸은 분류를 고쳐도 줄지 않는 탐지기 자체의 한계다."
)

# ── ④ 승격 라운드 ───────────────────────────────────────────────────────────
st.subheader("승격 라운드 — 사전이 자라며 LLM 몫이 준다")
rounds = _rounds()
left, right = st.columns([1, 1])
left.altair_chart(
    ui.round_chart(rounds, value="LLM 표기", title="라운드별 LLM 에 보낸 표기 수"),
    width="stretch",
)
# `미매칭 표기` 는 상한(`--llm-limit`) 을 걸지 않는 이 화면에서 `LLM 표기` 와 항상 같다.
# 두 열을 다 두면 정확도 열이 표 밖으로 밀려 나가는데, **정확도 열이 이 표의 요점이다.**
right.dataframe(
    pd.DataFrame(
        [
            {k: v for k, v in {**r, "거래 L2": f"{r['거래 L2']:.1%}"}.items() if k != "미매칭 표기"}
            for r in rounds
        ]
    ).set_index("라운드"),
    width="stretch",
)
first, last = rounds[0], rounds[-1]
st.caption(
    f"표기 {first['LLM 표기']} → {last['LLM 표기']} · 사전 {first['사전']}({first['사전 항목']}개) "
    f"→ {last['사전']}({last['사전 항목']}개) · 거래 L2 {first['거래 L2']:.1%} → "
    f"{last['거래 L2']:.1%}. **비용 축과 정확도 축을 같은 표에 둔다** — 호출만 따로 그리면 "
    "자동 승격이 오답을 굳혀 정확도를 깎은 라운드도 성공처럼 보인다. "
    f"마지막 라운드의 자동 승격이 {last['승격']}건인 것이 설계가 말한 고정점이다 — "
    "**자동은 새 브랜드를 못 만들고, 사전을 계속 키우는 것은 사람뿐이다** (S1 로 간다)."
)
