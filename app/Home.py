"""TxLens 앱 진입점 — `streamlit run app/Home.py` (blueprint §3.5 · D-006).

**화면 미학에 예산을 쓰지 않는다.** 교육 과정에서 Streamlit 은 거의 모두가 쓰므로 화면 자체는
변별력이 없다. 그래서 이 앱이 보여주는 것은 하나로 좁힌다 — **사전이 사람 손을 거쳐 자라고,
자란 사전이 다음 라운드의 LLM 호출을 줄인다**는 루프다 (S1).

여기(Home)는 그 루프를 돌리기 전에 **무엇이 준비돼 있는지**만 확인한다. 데이터·사전·재생
대본 중 하나라도 없으면 화면은 빈 표를 그리는 대신 무엇이 없는지 말해야 한다 — 발표 5분 전에
"왜 아무것도 안 나오지"를 디버깅하지 않기 위한 자리다.
"""

from __future__ import annotations

import streamlit as st

import txlens_ui as ui
from txlens.rules import load_dictionary

st.set_page_config(page_title="TxLens", page_icon="🧾", layout="wide")

st.title("TxLens — 마이데이터 거래내역 인텔리전스")
st.markdown(
    "**가맹점명을 못 묶으면 구독도 못 찾는다.** 정규화·분류(1부)와 패턴 탐지(2부)를 "
    "하나의 인과 사슬로 잇는 에이전트다 (D-010)."
)

# ── 준비 상태 ───────────────────────────────────────────────────────────────
st.subheader("실행 준비 상태")
rows = []

bundle_files = sorted(ui.BUNDLE_DIR.glob("*.json")) if ui.BUNDLE_DIR.exists() else []
rows.append(
    (
        "합성 번들",
        f"{len(bundle_files)}개 파일",
        ui.BUNDLE_DIR.relative_to(ui.ROOT).as_posix(),
        bool(bundle_files),
    )
)

try:
    dictionary = load_dictionary()
    rows.append(
        ("룰 사전", f"v{dictionary.version} · 항목 {len(dictionary.entries)}개", "rules/v1.yaml", True)
    )
except Exception as exc:  # noqa: BLE001 — 화면은 죽지 않고 무엇이 잘못됐는지 말해야 한다
    rows.append(("룰 사전", f"읽지 못했다 — {exc}", "rules/v1.yaml", False))

try:
    _classifier, model, n_verdicts = ui.load_replay_classifier()
    rows.append(("재생 대본", f"{model} · 판정 {n_verdicts}개", ui.VERDICTS_PATH.name, True))
except Exception as exc:  # noqa: BLE001
    rows.append(("재생 대본", f"읽지 못했다 — {exc}", ui.VERDICTS_PATH.name, False))

for label, value, where, ok in rows:
    icon = "✅" if ok else "❌"
    st.write(f"{icon} **{label}** — {value}  ·  `{where}`")

if not bundle_files:
    st.error(
        "번들이 없다. `python -m txlens.generator --seed 42` 로 만든다 — 시드가 같으면 "
        "거래 8,969건·표기 1,218개가 그대로 재현된다."
    )

# ── 화면 안내 ───────────────────────────────────────────────────────────────
st.subheader("화면")
st.markdown(
    "- **S1 · 검토 큐 (HITL)** — 그래프가 `interrupt()` 로 멈춘 자리다. 표에서 몇 줄을 고치면 "
    "룰 사전 v{n+1} 로 승격되고 덮는 거래 건수가 그 자리에서 나온다. ⭐ 이 프로젝트의 차별점\n"
    "- **S2 · 지표 대시보드** — 베이스라인 3종 비교 · §5.2 전이 실험 · 라운드별 호출 감소. "
    "숫자는 그 자리에서 다시 잰다 — 실호출이 필요한 (B) 한 줄만 스냅샷에서 읽는다\n"
    "- **S4 · 인사이트** — 구독 목록과 알림을 근거 문장까지 보여준다. 위의 **분류 단계 토글**을 "
    "옮기면 찾는 구독이 달라진다 — 표지의 주장이 화면에서 움직이는 자리다 (§5.2 · D-035)"
)

st.info(
    "이 앱은 저장된 판정으로 돈다 (`data/gold/verdicts-*.json`) — **API 를 부르지 않는다.** "
    "발표장에서 네트워크가 죽어도 같은 장면이 나오고, 라운드를 몇 번 돌려도 비용이 0 이다."
)
