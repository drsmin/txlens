"""S1 검토 화면 — 화면이 **사람이 고친 것을 사전까지 옮기는가** (blueprint §3.5 · D-037).

화면 테스트가 필요한 이유는 미학이 아니다. 이 화면에는 실패해도 화면이 멀쩡해 보이는 자리가
넷 있고, 넷 다 같은 방식으로 나타난다 — **화면은 "반영됨"이라 적고 사전에는 없다.**

① 체크한 줄이 조용히 버려진다 (대표명이 비었거나 분류를 못 고른 줄).
② 사람이 L1·L2 를 따로 골라 `food` + `cafe.coffee` 같은 조합이 만들어져 제출이 통째로 죽는다.
③ 제안 수를 승격 수로 보고한다 — `apply_proposals` 는 후보를 거른다.
④ 라운드 2 가 옛 사전으로 돈다 (그래프를 캐시로 묶으면 그렇게 된다).

여기서는 순수 함수 셋과 화면 실행 자체를 함께 본다. 화면은 `AppTest` 로 실제로 돌린다 —
import 만 검사하면 위젯 설정 오류(있지도 않은 열 이름·잘못된 selectbox 옵션)를 못 잡는다.
"""

from __future__ import annotations

import pytest

import txlens_ui as ui
from txlens.rules import load_dictionary
from txlens.rules.promote import apply_proposals, propose_from_corrections
from txlens.schemas import CategoryL1, CategoryL2, Evidence, LLMVerdict, RuleProposal

pytest.importorskip("streamlit")

NEW_BRAND = "을지로호프"


def _payload_item(name: str = NEW_BRAND, *, confidence: float = 0.75) -> dict:
    verdict = LLMVerdict(
        merchant_canonical=name,
        brand=None,
        category_l1=CategoryL1.CAFE,
        category_l2=CategoryL2.CAFE_ALCOHOL,
        confidence=confidence,
        reasoning="상호에 호프가 있다",
        evidence=Evidence.KEYWORD,
    )
    return {
        "normalized_name": name,
        "tx_count": 71,
        "priority": 71 * (1 - confidence),
        "verdict": verdict.model_dump(mode="json"),
    }


# ── 택소노미 선택지 ─────────────────────────────────────────────────────────
def test_선택지는_한글_라벨과_id_를_함께_보여준다():
    choices = ui.category_choices()
    assert len(choices) == len(list(CategoryL2))
    assert choices["외식 · food.restaurant"] == "food.restaurant"


def test_표시_문자열과_id_가_왕복한다():
    choices = ui.category_choices()
    for category in CategoryL2:
        assert choices[ui.category_choice_of(category.value)] == category.value


# ── 큐 → 표 ─────────────────────────────────────────────────────────────────
def test_교정_칸은_LLM_답으로_미리_채운다():
    (row,) = ui.queue_rows({"queued": 1, "items": [_payload_item()]})
    assert row[ui.COL_APPLY] is False
    assert row[ui.COL_CANONICAL] == NEW_BRAND
    assert row[ui.COL_CATEGORY] == ui.category_choice_of("cafe.alcohol")
    assert row["건수"] == 71


def test_표에_거래는_없다():
    """화면이 보는 것은 표기와 판정뿐이다 — 전송 경계와 같은 선택이다 (D-036②)."""
    (row,) = ui.queue_rows({"queued": 1, "items": [_payload_item()]})
    assert not {k for k in row if "거래" in k and k != "건수"}


# ── 표 → 교정 ───────────────────────────────────────────────────────────────
def test_체크한_줄만_올라간다():
    rows = ui.queue_rows({"queued": 2, "items": [_payload_item(), _payload_item("낙원떡집")]})
    rows[0][ui.COL_APPLY] = True
    corrections, problems = ui.rows_to_corrections(rows)
    assert problems == []
    assert [c.normalized_name for c in corrections] == [NEW_BRAND]


def test_L1_은_고르는_것이_아니라_L2_에서_끌어온다():
    """고를 수 없는 것은 틀릴 수도 없다 — `_CategoryConsistency` 가 여기서 절대 실패하지 않는다."""
    (row,) = ui.queue_rows({"queued": 1, "items": [_payload_item()]})
    row[ui.COL_APPLY] = True
    row[ui.COL_CATEGORY] = ui.category_choice_of("food.restaurant")
    (correction,), problems = ui.rows_to_corrections(row and [row])
    assert problems == []
    assert correction.category_l1 is CategoryL1.FOOD
    assert correction.category_l2 is CategoryL2.FOOD_RESTAURANT


def test_대표명이_비면_버리지_않고_사유를_돌려준다():
    (row,) = ui.queue_rows({"queued": 1, "items": [_payload_item()]})
    row[ui.COL_APPLY] = True
    row[ui.COL_CANONICAL] = "   "
    corrections, problems = ui.rows_to_corrections([row])
    assert corrections == []
    assert problems and NEW_BRAND in problems[0]


def test_분류_칸이_망가지면_사유를_돌려준다():
    (row,) = ui.queue_rows({"queued": 1, "items": [_payload_item()]})
    row[ui.COL_APPLY] = True
    row[ui.COL_CATEGORY] = "없는 카테고리"
    corrections, problems = ui.rows_to_corrections([row])
    assert corrections == []
    assert problems and "분류" in problems[0]


def test_검토자_이름이_교정에_남는다():
    (row,) = ui.queue_rows({"queued": 1, "items": [_payload_item()]})
    row[ui.COL_APPLY] = True
    (correction,), _ = ui.rows_to_corrections([row], reviewer="kangmin")
    assert correction.reviewer == "kangmin"


# ── 승격 요약 ───────────────────────────────────────────────────────────────
def _proposal(origin: str, pattern: str, *, pattern_type: str = "exact") -> RuleProposal:
    return RuleProposal(
        pattern_type=pattern_type,
        pattern=pattern,
        merchant_canonical=pattern,
        category_l1=CategoryL1.FOOD,
        category_l2=CategoryL2.FOOD_RESTAURANT,
        origin=origin,
        source_name=pattern,
        covers_tx_count=10,
    )


def test_사람_몫과_자동_몫을_갈라_센다():
    human = _proposal("human", NEW_BRAND)
    auto = _proposal("llm_high_confidence", "지에스25", pattern_type="alias")
    summary = ui.summarize_promotion([human, auto], [human, auto])
    assert summary.human == [human] and summary.auto == [auto]
    assert summary.human_tx == 10 and summary.auto_tx == 10
    assert summary.new_brands == [human]


def test_사전이_받지_않은_제안은_승격으로_세지_않는다():
    accepted = _proposal("human", NEW_BRAND)
    rejected = _proposal("llm_high_confidence", "없는브랜드")
    summary = ui.summarize_promotion([accepted, rejected], [accepted])
    assert summary.rejected == [rejected]
    assert summary.auto == [] and summary.human_tx == 10


def test_화면_요약과_사전_증가분이_맞는다():
    """요약의 "새 브랜드" 수는 사전 항목이 실제로 늘어난 수와 같아야 한다."""
    dictionary = load_dictionary()
    (row,) = ui.queue_rows({"queued": 1, "items": [_payload_item()]})
    row[ui.COL_APPLY] = True
    corrections, _ = ui.rows_to_corrections([row])
    proposals = propose_from_corrections(corrections, dictionary=dictionary)
    promoted, accepted = apply_proposals(dictionary, proposals)
    summary = ui.summarize_promotion(proposals, accepted)
    assert len(summary.new_brands) == len(promoted.entries) - len(dictionary.entries)


# ── 실행 로그 ───────────────────────────────────────────────────────────────
def test_상태_원문은_길이가_있는_것을_길이로_말한다():
    """8,969건을 그대로 찍으면 로그가 아니라 덤프다 — 목록은 길이로만 말한다."""
    lines = ui.raw_lines({"raw": list(range(8969)), "llm_results": {"a": 1}, "note": "짧다"})
    assert lines == [
        "      · raw = list 8,969개",
        "      · llm_results = dict 1키",
        "      · note = 짧다",
    ]


def test_원문_한_줄은_잘린다():
    assert ui.raw_lines({"k": "가" * 200})[0].endswith("…")
    assert len(ui.raw_lines({"k": "가" * 200})[0]) <= len("      · k = ") + ui.RAW_CLIP


def test_노드가_안_쓴_것은_원문에도_없다():
    """`log_line` 이 `None` 인 노드는 원문도 비어야 한다 — 없는 채널을 지어내지 않는다."""
    assert ui.raw_lines(None) == []
    assert ui.raw_lines("문자열") == []


def test_debug_청크를_사람_말로_바꾼다():
    """`updates` 는 노드가 **끝난 뒤**만 준다. 시작·깨운 이유·체크포인트는 debug 에서만 온다."""
    task = ui.debug_line({"type": "task", "step": 1, "payload": {"name": "ingest", "triggers": ("branch:to:ingest",)}})
    assert task == "step 1 · 태스크 시작 ← branch:to:ingest"
    ckpt = ui.debug_line({"type": "checkpoint", "step": 3, "payload": {"next": ["rule_match"]}})
    assert ckpt == "step 3 · 체크포인트 저장 · 다음: rule_match"
    done = ui.debug_line({"type": "task_result", "step": 6, "payload": {"name": "llm_classify", "result": {"llm_results": {}, "review_queue": []}}})
    assert done == "step 6 · 태스크 끝 → llm_results, review_queue"


def test_interrupt_로_멈춘_태스크는_그렇게_찍는다():
    """HITL 이 어느 태스크에서 멈췄는지가 debug 로만 보인다 (D-037)."""
    line = ui.debug_line(
        {"type": "task_result", "step": 2, "payload": {"name": "human_review", "result": {}, "interrupts": [{"value": {}}]}}
    )
    assert line == "step 2 · 태스크 끝 → 쓴 것 없음 · interrupt"


def test_task_result_는_dict_든_쌍_목록이든_채널을_읽는다():
    """**버전 계약이다.** 실측(langgraph 1.x)은 dict 를 주고 문서는 쌍 목록을 말한다 —
    한 쪽만 가정했다가 화면이 `ValueError` 로 떨어졌다. 로그를 보여주려다 화면을 잃는 것은
    최악의 교환이라 둘 다 받는다."""
    assert ui._result_channels({"raw": [1]}) == ["raw"]
    assert ui._result_channels([("raw", [1]), ("matched", [])]) == ["raw", "matched"]
    assert ui._result_channels(None) == []


def test_모르는_debug_타입은_줄을_만들지_않는다():
    assert ui.debug_line({"type": "처음_보는_것", "payload": {}}) is None
    assert ui.debug_line("dict 도 아니다") is None


def test_실패한_태스크는_오류를_감추지_않는다():
    line = ui.debug_line({"type": "task_result", "step": 4, "payload": {"error": "boom"}})
    assert line is not None and "boom" in line


# ── 화면 실행 ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def s1():
    """S1 을 실제로 돌린다. 재생 대본으로 도므로 API 를 부르지 않는다."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ui.ROOT / "app" / "pages" / "1_검토_큐.py"), default_timeout=120)
    app.run()
    return app


def test_화면이_예외_없이_뜬다(s1):
    assert not s1.exception


def test_큐가_표로_그려진다(s1):
    """`AppTest` 는 `data_editor` 를 표로만 본다 — 체크는 못 하지만 **무엇이 그려졌는지**는 본다.

    편집 자체는 순수 함수 쪽(`rows_to_corrections`)이 이미 덮는다. 여기서 잡고 싶은 것은
    열 이름이 어긋나 편집 칸이 통째로 안 그려지는 경우다.
    """
    table = s1.dataframe[0].value
    assert list(table.columns)[:2] == [ui.COL_APPLY, ui.COL_NAME]
    assert {ui.COL_CANONICAL, ui.COL_CATEGORY} <= set(table.columns)
    assert len(table) == 20  # DEFAULT_REVIEW_BATCH


def test_고친_것이_없어도_제출이_돌고_자동_승격만_올라간다():
    """빈 교정도 정당한 답이다 — 사람을 부른 것이 곧 고쳤다는 뜻은 아니다 (D-037)."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ui.ROOT / "app" / "pages" / "1_검토_큐.py"), default_timeout=120)
    app.run()
    submit = next(b for b in app.button if "제출" in b.label)
    submit.click().run()

    assert not app.exception
    labels = {m.label: m.value for m in app.metric}
    assert labels["사람이 만든 새 브랜드"] == "0개"
    assert labels["자동 승격이 덮는 거래"] != "0건"
    assert any("고친 것이 없다" in info.value for info in app.info)


def test_로그는_본문_옆_칸에_있다(s1):
    """로그가 본문 아래면 사람이 표를 고치는 동안 화면 밖으로 밀린다 — 인과를 같이 봐야 한다.

    폭도 같이 못 박는다. 로그 한 줄이 70자 남짓이라 3할로 줄이면 줄마다 접혀 노드 이름
    정렬(`NODE_WIDTH`)이 무너진다.
    """
    outer = [c.weight for c in s1.columns if c.weight in (0.6, 0.4)]
    assert outer == [0.6, 0.4]


def test_기본_상세도는_요약이다(s1):
    """**화면이 녹화돼 발표에 들어가기 때문이다** (D-053). 영상이 슬라이드 크기로 줄면 debug
    45줄은 안 읽힌다. 기본값이 조용히 debug 로 돌아가면 녹화본이 그대로 못 쓰게 된다.
    """
    assert [s.value for s in s1.selectbox if s.label == "로그 상세도"] == [ui.DETAIL_SUMMARY]
    log = s1.code[0].value
    assert "원본 거래 8,969건을 읽었다" in log  # 노드마다 한 줄은 있다
    assert "체크포인트 저장" not in log  # debug 는 꺼져 있다
    assert "· raw = list" not in log  # 상태 원문도 꺼져 있다


def test_상세도를_올리면_debug_가_스트림까지_닿는다():
    """선택이 화면 문구만 바꾸고 `stream_mode` 에 안 닿으면 아무도 모른다 — 라운드를 다시
    돌려 체크포인트와 서브그래프 노드가 실제로 찍히는지 본다.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ui.ROOT / "app" / "pages" / "1_검토_큐.py"), default_timeout=180)
    app.run()
    next(s for s in app.selectbox if s.label == "로그 상세도").set_value(ui.DETAIL_DEBUG).run()
    next(b for b in app.button if "처음부터 다시" in b.label).click().run()

    assert not app.exception
    log = app.code[0].value
    assert "체크포인트 저장" in log and "태스크 시작" in log
    assert "confidence_gate" in log  # 서브그래프 안 — subgraphs=True 없으면 안 보인다
    assert "· raw = list 8,969개" in log  # 상태 원문


def test_상단이_큐_길이와_배치를_함께_보고한다(s1):
    """큐 길이만 보고하면 20개를 본 것이 64개를 본 것처럼 읽힌다."""
    labels = {m.label: m.value for m in s1.metric}
    assert "검토 큐" in labels and "이번 배치" in labels
    assert int(labels["검토 큐"].split("개")[0]) > int(labels["이번 배치"].split("개")[0])
