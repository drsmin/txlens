"""S2 지표 화면 — 화면이 **없는 수를 지어내지 않는가** (blueprint §5 · D-025 · D-033).

S1 의 실패 모양이 "화면은 반영됐다고 적고 사전에는 없다" 였다면, 이 화면의 실패 모양은
**"화면이 그럴듯한 숫자를 보여주는데 그게 측정값이 아니다"** 이고, 자리가 넷 있다.

① (B) 를 재생으로 되살릴 수 없는데 빈 줄이나 0 을 측정값처럼 그린다.
② 스냅샷이 낡아 화면과 슬라이드가 다른 수를 말하는데 아무도 모른다.
③ 타일이 `98.9%` 와 `+5.8%p` 를 나란히 적고 표에는 `93.0%` 가 있다 — 화면 안에서 산수가 안 맞는다.
④ 라운드 축이 연속형이라 있지도 않은 **라운드 1.5** 에 눈금이 선다.

넷 다 화면이 멀쩡해 보이므로, 눈으로 보는 것으로는 잡히지 않는다.
"""

from __future__ import annotations

import json

import pytest

import txlens_ui as ui

pytest.importorskip("streamlit")


def _row(label: str, *, accuracy_l2: float = 0.9, distinct_l2: float = 0.5, source: str = ui.SRC_LIVE):
    return ui.BaselineRow(
        label=label,
        source=source,
        accuracy_l2=accuracy_l2,
        distinct_l2=distinct_l2,
        coverage=0.84,
        unmatched_rate=0.077,
        pg_masked_rate=0.08,
        llm_targets=0,
        llm_model_calls=0,
    )


# ── 스냅샷 ──────────────────────────────────────────────────────────────────
def test_스냅샷이_없으면_빈_목록이지_예외가_아니다(tmp_path):
    """실호출을 아직 안 돌린 것은 고장이 아니다 — 화면은 그 상태도 그려야 한다."""
    assert ui.load_snapshot_rows(tmp_path / "없다.json") == []


def test_스냅샷의_비율을_다시_계산하지_않고_그대로_읽는다(tmp_path):
    path = tmp_path / "baselines.json"
    path.write_text(
        json.dumps(
            [
                {
                    "result": {
                        "label": "B · LLM만",
                        "overall": {
                            "accuracy_l2": 0.82,
                            "coverage": 0.866,
                            "unmatched_rate": 0.054,
                            "pg_masked_rate": 0.08,
                        },
                        "distinct": {"accuracy_l2": 0.648},
                    },
                    "metrics": {"llm_targets": 257, "llm_model_calls": 1059},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (row,) = ui.load_snapshot_rows(path)
    assert row.label == "B · LLM만"
    assert row.accuracy_l2 == 0.82
    assert row.llm_model_calls == 1059
    assert row.source == ui.SRC_SNAPSHOT


def test_스냅샷_모양이_틀리면_조용히_넘기지_않는다(tmp_path):
    """반쯤 읽힌 스냅샷은 표에 빈 줄을 만들고, 빈 줄은 측정값처럼 보인다."""
    path = tmp_path / "baselines.json"
    path.write_text(json.dumps([{"metrics": None}]), encoding="utf-8")
    with pytest.raises(KeyError):
        ui.load_snapshot_rows(path)


# ── 합치기 ──────────────────────────────────────────────────────────────────
def test_되살릴_수_있는_줄은_지금_계산한_값이_이긴다():
    """스냅샷이 이기면 사전을 고친 날 화면이 옛 수를 말하고 그 사실이 어디에도 안 나온다."""
    live = [_row("A · 룰만", accuracy_l2=0.930)]
    snapshot = [_row("A · 룰만", accuracy_l2=0.777, source=ui.SRC_SNAPSHOT)]
    rows, _notes = ui.merge_baseline_rows(live, snapshot)
    assert [r.accuracy_l2 for r in rows] == [0.930]
    assert rows[0].source == ui.SRC_LIVE


def test_되살릴_수_없는_줄은_스냅샷이_채운다():
    live = [_row("A · 룰만"), _row("C · 하이브리드")]
    snapshot = [_row("B · LLM만", source=ui.SRC_SNAPSHOT)]
    rows, _notes = ui.merge_baseline_rows(live, snapshot)
    assert [r.label for r in rows] == ["A · 룰만", "B · LLM만", "C · 하이브리드"]


def test_스냅샷이_없으면_B_줄을_지어내지_않는다():
    rows, notes = ui.merge_baseline_rows([_row("A · 룰만")], [])
    assert [r.label for r in rows] == ["A · 룰만"]
    assert notes == []


def test_같은_구성이_어긋나면_감추지_않고_말한다():
    live = [_row("A · 룰만", accuracy_l2=0.930)]
    snapshot = [_row("A · 룰만", accuracy_l2=0.900, source=ui.SRC_SNAPSHOT)]
    _rows, notes = ui.merge_baseline_rows(live, snapshot)
    assert len(notes) == 1
    assert "93.0%" in notes[0] and "90.0%" in notes[0]


def test_반올림에_묻히는_차이로는_경고하지_않는다():
    """재생과 실호출은 원리상 완전히 같을 수 없다 — 0 으로 두면 매번 경고가 뜬다."""
    live = [_row("C · 하이브리드", accuracy_l2=0.98851)]
    snapshot = [_row("C · 하이브리드", accuracy_l2=0.98860, source=ui.SRC_SNAPSHOT)]
    _rows, notes = ui.merge_baseline_rows(live, snapshot)
    assert notes == []


# ── 타일 산수 ───────────────────────────────────────────────────────────────
def test_타일의_증감은_화면에_보이는_수끼리_뺀_값이다():
    """98.9% 와 93.0% 가 화면에 있는데 증감이 +5.8%p 이면 청중이 표 전체를 의심한다."""
    a, c = 0.929976, 0.988513
    assert f"{c:.1%}" == "98.9%" and f"{a:.1%}" == "93.0%"
    assert ui.delta_pp(c, a) == pytest.approx(5.9)


# ── 라운드 ──────────────────────────────────────────────────────────────────
def test_라운드_축은_순서형이다():
    """연속형이면 있지도 않은 **라운드 1.5** 에 눈금이 선다 — 눈으로만 잡히던 회귀다."""
    rows = ui.round_rows([])
    assert rows == []
    chart = ui.round_chart(
        [{"라운드": 1, "LLM 표기": 155}, {"라운드": 2, "LLM 표기": 121}],
        value="LLM 표기",
        title="t",
    )
    assert chart.to_dict()["encoding"]["x"]["type"] == "ordinal"


def test_라운드_행은_비용과_정확도를_한_줄에_둔다():
    """호출만 따로 그리면 자동 승격이 오답을 굳혀 정확도를 깎은 라운드도 성공처럼 보인다."""
    from txlens.agents.replay import load_verdicts
    from txlens.promote import run_rounds

    verdicts, model = load_verdicts(ui.VERDICTS_PATH)
    rows = ui.round_rows(run_rounds(model, replay=verdicts, bundle_dir=ui.BUNDLE_DIR))
    assert len(rows) >= 2
    assert rows[0]["LLM 표기"] > rows[-1]["LLM 표기"]  # 사전이 자라면 LLM 몫이 준다
    for row in rows:
        assert {"라운드", "사전", "LLM 표기", "거래 L2"} <= set(row)


# ── 화면 ────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def s2():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(ui.ROOT / "app" / "pages" / "2_지표_대시보드.py"), default_timeout=300
    )
    return app.run()


def test_화면이_예외_없이_뜬다(s2):
    assert not s2.exception


def test_네_덩이가_다_그려진다(s2):
    """타일 · §5.1 · §5.2 · 승격 라운드 — 하나라도 비면 발표에서 말할 것이 없다."""
    assert len(s2.subheader) == 4  # 네 덩이가 각각 제목을 하나씩 쓴다
    assert len(s2.dataframe) == 3
    assert len(s2.metric) == 4


def test_B_가_없으면_비어_있다고_말하고_명령을_준다(s2):
    """스냅샷이 없을 때 0 이나 빈 칸을 측정값처럼 그리면 안 된다."""
    labels = list(s2.dataframe[0].value["구성"])
    if any(label.startswith("B") for label in labels):
        pytest.skip("실호출 스냅샷이 있는 환경 — 빈 줄 경고는 뜨지 않는다")
    assert any("(B)" in w.value and "--baseline b" in w.value for w in s2.warning)


def test_표는_출처를_값과_같은_줄에_들고_있다(s2):
    """세 종류의 수가 섞이는 표다 — 각주로만 적으면 어느 칸도 가리킬 수 없다."""
    frame = s2.dataframe[0].value
    assert "출처" in frame.columns
    assert set(frame["출처"]) <= {ui.SRC_LIVE, ui.SRC_REPLAY, ui.SRC_SNAPSHOT}


def test_전이표의_탐지기는_네_분류_위에서_같다(s2):
    """§5.2 의 전제 — 바뀌는 것은 분류뿐이다 (D-035)."""
    frame = s2.dataframe[1].value
    assert len(frame) == 4
    assert list(frame["분류 단계"])[-1].startswith("정답 분류")
