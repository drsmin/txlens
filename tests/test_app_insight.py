"""S4 인사이트 화면 — 화면이 **탐지기와 다른 말을 하지 않는가** (blueprint §3.6 · D-035).

S2 의 실패 모양이 "화면이 그럴듯한 숫자를 보여주는데 그게 측정값이 아니다" 였다면, 이 화면의
실패 모양은 **"화면이 탐지 결과를 다시 해석해 버린다"** 이고 자리가 넷 있다.

① 알림률의 **분모**를 탐지기가 본 적 없는 모집단에서 센다 — 취소 상계·0원 행까지 세면
   분자는 그대로인데 분모만 커져 알림률이 조용히 낮아진다.
② 근거 문장을 화면이 다듬는다 — 검증된 적 없는 다른 문장이 고객에게 나간다.
③ 목록 정렬이 아무 순서다 — 고객이 먼저 봐야 하는 건 큰 돈인데 그게 아래에 있다.
④ 토글이 §5.2 표와 **다른 코드**를 돈다 — 같은 화면 안에서 표와 목록이 어긋나도 아무도 모른다.

넷 다 화면이 멀쩡해 보이므로 눈으로는 잡히지 않는다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import txlens_ui as ui
from txlens.detect import detect, spend_rows
from txlens.preprocess import preprocess
from txlens.schemas import CategoryL1, CategoryL2, ClassifiedTx, MatchStage, RawTx, TxType

pytest.importorskip("streamlit")


def _classified(rows: list[tuple[str, datetime, int, TxType]], customer="C0001"):
    raws = [
        RawTx(
            provider_id="CARD_A",
            customer_id=customer,
            approval_no=no,
            tx_datetime=when,
            merchant_name_raw="스타벅스",
            amount=amount,
            tx_type=kind,
        )
        for no, when, amount, kind in rows
    ]
    return [
        ClassifiedTx(
            normalized=n,
            merchant_canonical="스타벅스",
            category_l1=CategoryL1.CAFE,
            category_l2=CategoryL2.CAFE_COFFEE,
            confidence=1.0,
            match_stage=MatchStage.EXACT,
        )
        for n in preprocess(raws)
    ]


# ── ① 알림률의 분모 ─────────────────────────────────────────────────────────
def test_알림률_분모는_탐지기가_본_행에서만_센다():
    """취소 상계된 달을 분모에 넣으면 알림률이 실제보다 낮게 보인다."""
    base = datetime(2026, 6, 1, 12, 0)
    classified = _classified(
        [
            ("A1", base, 5_000, TxType.APPROVE),
            # 아래 둘은 상계되어 탐지 입력에서 빠진다 — **다른 달**이라 분모를 늘릴 참이었다.
            ("A2", base + timedelta(days=40), 500_000, TxType.APPROVE),
            ("A2", base + timedelta(days=41), -500_000, TxType.CANCEL),
        ]
    )
    seen_months = {(r.at.year, r.at.month) for r in spend_rows(classified)}
    assert ui.customer_months(classified) == (1, len(seen_months) or 1)
    assert seen_months == {(2026, 6)}, "취소 상계된 7월이 분모에 들어갔다"


def test_빈_입력에서도_분모가_0으로_나눠지지_않는다():
    tiles = ui.insight_tiles(detect([]), [])
    assert tiles["고객당 월 알림"] == 0.0
    assert tiles["구독"] == 0 and tiles["알림"] == 0


# ── ② 근거 문장 ─────────────────────────────────────────────────────────────
def test_근거_문장은_한_글자도_바뀌지_않는다():
    """탐지기가 만든 문장이 곧 고객에게 나가는 설명이다 — 화면이 다듬으면 다른 문장이 된다."""
    det = _real_detections()
    assert det.subscriptions, "표본이 비어 검사가 무의미하다"
    for src, row in zip(
        sorted(det.subscriptions, key=lambda x: (-x.monthly_cost, x.merchant_canonical)),
        ui.subscription_rows(det.subscriptions),
        strict=True,
    ):
        assert row["근거"] == src.evidence
    for src, row in zip(
        sorted(det.spending_alerts, key=lambda x: (-x.robust_z, x.customer_id)),
        ui.spending_rows(det.spending_alerts),
        strict=True,
    ):
        assert row["근거"] == src.evidence
    for src, row in zip(
        sorted(det.tx_alerts, key=lambda x: (x.at, x.customer_id)),
        ui.tx_alert_rows(det.tx_alerts),
        strict=True,
    ):
        assert row["근거"] == src.evidence


# ── ③ 정렬 ──────────────────────────────────────────────────────────────────
def test_구독은_월_고정비가_큰_것부터_나온다():
    det = _real_detections()
    costs = [r["월 환산"] for r in ui.subscription_rows(det.subscriptions)]
    assert costs == sorted(costs, reverse=True)


# ── ④ 토글이 §5.2 와 같은 코드를 도는가 ──────────────────────────────────────
def test_전이_실험이_원본_탐지_결과를_들고_다닌다():
    """화면이 목록을 그리려고 네 단계를 **다시** 계산하면 표와 목록이 갈라진다 (D-018)."""
    from txlens.eval.transfer import run_transfer

    rows = run_transfer(bundle_dir=ui.BUNDLE_DIR, verdicts_path=ui.VERDICTS_PATH)
    assert rows, "번들이 없다 — `python -m txlens.generator --seed 42`"
    for row in rows:
        assert row.detections is not None and row.classified is not None
        # 지표의 분모(`detected`)가 곧 원본 목록의 길이여야 한다 — 다르면 표와 목록이
        # 서로 다른 계산에서 나온 것이다.
        assert row.detection.subscription.detected == len(row.detections.subscriptions)
        # 같은 분류를 다시 넣으면 같은 탐지가 나온다 (결정론 — D-035).
        assert len(detect(row.classified).subscriptions) == len(row.detections.subscriptions)


def test_분류_단계를_올리면_찾는_구독이_는다():
    """이 화면의 주장 자체 — 이름을 못 묶으면 구독을 못 찾는다 (S13 · D-035)."""
    from txlens.eval.transfer import run_transfer

    found = {
        r.label: len(r.detections.subscriptions)
        for r in run_transfer(bundle_dir=ui.BUNDLE_DIR, verdicts_path=ui.VERDICTS_PATH)
    }
    assert found["정규화만"] < found["A · 룰만"] <= found["C · 하이브리드"]
    assert found["C · 하이브리드"] < found["정답 분류(상한)"], "상한이 달성치보다 낮을 수 없다"


# ── 화면이 실제로 도는가 ─────────────────────────────────────────────────────
def test_화면이_네_단계를_모두_예외_없이_그린다():
    from streamlit.testing.v1 import AppTest

    path = ui.ROOT / "app" / "pages" / "4_인사이트.py"
    at = AppTest.from_file(str(path), default_timeout=300).run()
    assert not at.exception, at.exception[0].value if at.exception else ""
    assert at.radio[0].options == ["정규화만", "A · 룰만", "C · 하이브리드", "정답 분류(상한)"]

    seen = {}
    for stage in at.radio[0].options:
        at.radio[0].set_value(stage).run()
        assert not at.exception, at.exception[0].value if at.exception else ""
        tile = next(m for m in at.metric if m.label == "찾은 구독")
        seen[stage] = (tile.value, len(at.dataframe[0].value))
    # 타일에 적힌 수와 표의 줄 수가 같아야 한다 — 화면 안에서 산수가 맞아야 한다 (D-041⑤).
    for stage, (value, n_rows) in seen.items():
        assert value == f"{n_rows}건", f"{stage}: 타일 {value} ≠ 표 {n_rows}줄"


def _real_detections():
    """실제 파이프라인 출력 위의 탐지 — 화면이 받는 것과 같은 것."""
    from txlens.eval.harness import run_baseline_state
    from txlens.rules import load_dictionary

    _r, _m, state = run_baseline_state(
        label="A · 룰만", dictionary=load_dictionary(), bundle_dir=ui.BUNDLE_DIR
    )
    return detect(state["confirmed"])
