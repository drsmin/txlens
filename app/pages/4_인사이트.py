"""S4 · 인사이트 — 2부가 고객에게 보이는 자리 (blueprint §3.6 · D-010 · D-035).

S2 가 "탐지가 분류에 얼마나 기대는가"를 **표로** 말한다면, 여기는 같은 것을 **물건으로**
보여준다. 구독 목록과 알림이 나오고, 위의 토글로 분류 단계를 옮기면 그 목록이 눈앞에서 바뀐다.

## 이 화면의 요점은 토글이다

표지의 주장은 "가맹점명을 못 묶으면 구독도 못 찾는다" 이고, §5.2 는 그것을 재현율 숫자로
증명했다. 숫자표는 그 말을 **믿게** 하지 못한다 — 같은 고객의 구독 목록이 분류 단계에 따라
늘고 주는 것을 보면 인과가 눈에 남는다.

**A ↔ C 가 아니라 §5.2 의 네 단계를 그대로 건다.** 처음엔 A·C 둘만 뒀는데 구독 수가 한 건도
움직이지 않았다 — 구독 브랜드 7개가 이미 룰 사전 v1 에 있어서다(D-035 · S13 이 밝힌 그 사실).
움직임은 **정규화만 ↔ 정답 분류** 축에 있다. 그래서 §5.2 와 같은 네 줄을 쓰고, 어느 단계를
고르든 **바닥(정규화만)과의 차이**를 타일에 함께 적는다. 화면이 표와 같은 코드를 돈다.

## 근거 문장은 손대지 않는다

탐지기는 판정마다 규칙을 그대로 옮긴 문장을 만든다. 화면은 그것을 요약하지 않고 그대로
싣는다 — 요약하면 검증된 적 없는 다른 문장이 된다. 금융 앱에서 "왜 이 알림이 왔나"에
답하지 못하는 알림은 없는 것과 같다.

## 숫자는 그 자리에서 다시 잰다 (D-041①)

네 파이프라인을 실제로 돌려 그 결과 위에서 탐지한다. API 는 부르지 않는다 — (C) 는 저장된
판정을 재생한다. 입력이 `rules/v1.yaml` 과 번들에 고정돼 세션 중에 바뀌지 않으므로
`@st.cache_data` 가 맞다 (S1 과 반대인 이유는 D-041⑥).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import txlens_ui as ui
from txlens.eval.transfer import run_transfer
from txlens.rules import load_dictionary

st.set_page_config(page_title="TxLens · 인사이트", page_icon="🔎", layout="wide")
st.title("S4 · 인사이트 — 구독과 알림")

FLOOR = "정규화만"


@st.cache_data(show_spinner="네 분류 단계를 돌리고 그 위에서 탐지하는 중 — API 는 부르지 않는다")
def _stages() -> dict[str, dict]:
    """§5.2 와 **같은 함수**를 돌린다 — 표와 목록이 한 코드에서 나온다 (D-035).

    캐시에는 표로 바꾼 dict 만 담는다. 거래 8,619건짜리 분류 결과를 네 벌 얹으면 캐시가
    무거워지고, 화면이 필요한 것은 이미 줄로 바뀐 값뿐이다.
    """
    out: dict[str, dict] = {}
    for row in run_transfer(bundle_dir=ui.BUNDLE_DIR, verdicts_path=ui.VERDICTS_PATH):
        det = row.detections
        out[row.label] = {
            "tiles": ui.insight_tiles(det, row.classified),
            "subs": ui.subscription_rows(det.subscriptions),
            "spend": ui.spending_rows(det.spending_alerts),
            "tx": ui.tx_alert_rows(det.tx_alerts),
            "unmatched": row.unmatched_rate,
        }
    return out


# ── 분류 단계 토글 ───────────────────────────────────────────────────────────
stages = _stages()
labels = list(stages)
label = st.radio(
    "분류 단계 — **탐지기는 네 줄에서 완전히 같다.** 바뀌는 것은 그 위에 놓인 분류뿐이다 (§5.2)",
    labels,
    index=labels.index("C · 하이브리드") if "C · 하이브리드" in labels else len(labels) - 1,
    horizontal=True,
)
cur, floor = stages[label], stages[FLOOR]
tiles, subs, spend, txa = cur["tiles"], cur["subs"], cur["spend"], cur["tx"]

st.caption(
    f"고객 {tiles['고객']}명 · {tiles['개월']}개월 · 룰 사전 v{load_dictionary().version} · "
    f"이 단계의 미묶음 {cur['unmatched']:.1%} — 분류를 이 자리에서 돌리고 그 결과 위에서 "
    "탐지한다. **API 는 부르지 않는다**"
)

# ── 타일 ────────────────────────────────────────────────────────────────────
cols = st.columns(4)
base = floor["tiles"]
is_floor = label == FLOOR
# 증감은 **보이는 수끼리** 뺀다 (D-041⑤). 바닥 자신을 고르면 "+0" 대신 바닥이라고 말한다.
cols[0].metric(
    "찾은 구독", f"{tiles['구독']}건",
    "바닥" if is_floor else f"{tiles['구독'] - base['구독']:+d}건 vs {FLOOR}",
    delta_color="off" if is_floor else "normal",
)
cols[1].metric(
    "월 고정비 합계", f"{tiles['월 고정비']:,}원",
    "바닥" if is_floor else f"{tiles['월 고정비'] - base['월 고정비']:+,}원 vs {FLOOR}",
    delta_color="off" if is_floor else "normal",
)
cols[2].metric("이상지출 · 이상거래 알림", f"{tiles['알림']}건",
               "바닥" if is_floor else f"{tiles['알림'] - base['알림']:+d}건 vs {FLOOR}",
               delta_color="off" if is_floor else "normal")
cols[3].metric(
    "고객당 월 알림", f"{tiles['고객당 월 알림']}건",
    "기준선 0.5건 (blueprint §3.6)", delta_color="off",
)

ladder = " → ".join(f"{k.split(' ·')[0]} {v['tiles']['구독']}" for k, v in stages.items())
top = max(stages.values(), key=lambda v: v["tiles"]["구독"])["tiles"]["구독"]
st.success(
    f"**토글이 이 프로젝트의 주장이다** — 찾은 구독: {ladder}건. 탐지기는 네 줄에서 "
    f"완전히 같은데 **{base['구독']} → {top}건**이다. 가맹점명을 못 묶으면 결제가 여러 표기로 "
    "쪼개져 \"최소 3회\" 조건을 못 채운다 (S13 · D-035).\n\n"
    "**A → C 에서 거의 움직이지 않는 것도 결과다** — 구독 브랜드 7개가 이미 룰 사전 v1 에 "
    "있어서다. 남은 격차는 커버리지가 아니라 **오귀속**(`쿠팡와우`→`쿠팡`)이고 그건 사람이 "
    "고친다 (S1 · D-033). `정답 분류` 는 1부가 완벽할 때의 **상한**이지 달성치가 아니다."
)

# ── 구독 ────────────────────────────────────────────────────────────────────
st.subheader("구독 — 매달 나가는 돈")
customers = ["전체"] + sorted({r["고객"] for r in subs})
pick = st.selectbox("고객", customers, help="한 명을 고르면 실제 앱 화면과 같은 모양이 된다")
shown = subs if pick == "전체" else [r for r in subs if r["고객"] == pick]

if shown:
    st.dataframe(
        pd.DataFrame(shown),
        hide_index=True,
        width="stretch",
        column_config={
            "중앙값": st.column_config.NumberColumn(format="%,d원"),
            "월 환산": st.column_config.NumberColumn(format="%,d원"),
            "근거": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        f"{len(shown)}건 · 월 고정비 {sum(r['월 환산'] for r in shown):,}원. "
        "**`미사용 의심`** 은 마지막 결제 후 주기가 여러 번 지난 것, **`가격 인상`** 은 금액이 "
        "계단식으로 오른 것이다. 근거 문장은 탐지 규칙을 그대로 옮긴 것이라 고객에게 그대로 보인다."
    )
else:
    st.warning("이 고객에게는 찾은 구독이 없다.")

# ── 알림 ────────────────────────────────────────────────────────────────────
st.subheader("알림 — 확인해 보라고 말하는 것들")
st.caption(
    "**차단이 아니라 확인 요청이다.** 판단은 고객이 한다 — FDS 가 아니다 (부록 J). "
    "그래서 정밀도보다 **읽히는 양**이 중요하고, 고객당 월 0.5건을 기준선으로 잡았다."
)
left, right = st.columns(2)
with left:
    st.markdown(f"**이상지출** — {len(spend)}건")
    if spend:
        st.dataframe(
            pd.DataFrame(spend), hide_index=True, width="stretch",
            column_config={
                "금액": st.column_config.NumberColumn(format="%,d원"),
                "근거": st.column_config.TextColumn(width="medium"),
            },
        )
    else:
        st.info("이상지출 알림이 없다.")
with right:
    st.markdown(f"**이상거래 패턴** — {len(txa)}건")
    if txa:
        st.dataframe(
            pd.DataFrame(txa), hide_index=True, width="stretch",
            column_config={
                "금액": st.column_config.NumberColumn(format="%,d원"),
                "근거": st.column_config.TextColumn(width="medium"),
            },
        )
    else:
        st.info("이상거래 알림이 없다.")

st.caption(
    "**세 탐지기는 분류에 기대는 정도가 다르다.** 이상거래는 금액·시각만 보므로 분류가 없어도 "
    "대부분 남고, 이상지출은 기준선이 카테고리라 분류가 없으면 아예 서지 않는다. 그 낙폭 차이가 "
    "S2 의 §5.2 표이고, 이 화면은 같은 것을 목록으로 보여준다 (D-035)."
)
