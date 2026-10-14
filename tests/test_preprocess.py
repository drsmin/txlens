"""preprocess 규칙별 회귀 테스트 (설계 §3.3 ①).

전처리는 정확도의 절반을 결정하므로 **규칙 하나하나를 따로 잠근다.** 여기서 막고 싶은 것:

- 정규화가 약해서 같은 가맹점이 갈라지는 것 (구독 탐지가 주기 3회를 못 채운다 — §5.2)
- 정규화가 **과해서** 정보를 잘라먹는 것 (`배달의민족`·`…점` 으로 끝나는 브랜드)
- 취소 거래가 분류 대상에 남아 이중 계산되는 것

`test_생성기_오염을_전처리가_되돌린다` 는 드리프트 가드다 — 생성기가 새 프리픽스·PG명을
쓰기 시작하면 전처리가 모르는 채로 넘어가지 않고 여기서 깨진다.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from txlens.preprocess import (
    REASON_CANCEL_ROW,
    REASON_CANCELLED,
    REASON_CORRECTED,
    REASON_PARTIAL_CANCEL_ROW,
    canonical_form,
    has_installment_rows,
    normalize_name,
    preprocess,
    regno_key,
    split_branch,
    strip_channel,
    strip_corporate,
)
from txlens.schemas import PaymentChannel, RawTx, TxType


def _raw(name: str, **kw) -> RawTx:
    base = {
        "provider_id": "CARD_A",
        "customer_id": "SYNTH_0000",
        "approval_no": "A0001",
        "tx_datetime": datetime(2026, 3, 15, 14, 30),
        "merchant_name_raw": name,
        "amount": 12000,
    }
    base.update(kw)
    return RawTx(**base)


# ── ① 유니코드·공백·대문자 ──────────────────────────────────────────────────
def test_전각을_반각으로_바꾼다():
    """NFKC 를 먼저 하지 않으면 아래 정규식이 전각 표기를 전부 놓친다."""
    assert canonical_form("ＧＳ２５") == "GS25"
    assert canonical_form("２호점") == "2호점"


def test_괄호주_기호를_펼친다():
    """`㈜` 는 한 글자다. NFKC 가 `(주)` 로 바꿔주므로 법인표기 규칙이 하나면 된다."""
    assert canonical_form("㈜에스씨케이컴퍼니") == "(주)에스씨케이컴퍼니"
    assert normalize_name("㈜에스씨케이컴퍼니").normalized_name == "에스씨케이컴퍼니"


def test_공백을_모으고_대문자로_통일한다():
    assert canonical_form("  starbucks   coffee  ") == "STARBUCKS COFFEE"


# ── ② 결제채널 프리픽스 ─────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("raw", "channel"),
    [
        ("NAVERPAY*스타벅스", PaymentChannel.NAVERPAY),
        ("네이버페이*스타벅스", PaymentChannel.NAVERPAY),
        ("카카오페이_GS25", PaymentChannel.KAKAOPAY),
        ("KAKAOPAY*GS25", PaymentChannel.KAKAOPAY),
        ("TOSSPAY*이마트", PaymentChannel.TOSSPAY),
        ("PAYCO*교촌치킨", PaymentChannel.PAYCO),
        ("SSGPAY*이마트", PaymentChannel.SSGPAY),
    ],
)
def test_간편결제_프리픽스를_분리해_보존한다(raw, channel):
    parts = normalize_name(raw)
    assert parts.payment_channel is channel
    assert "PAY" not in parts.normalized_name
    assert "페이" not in parts.normalized_name


def test_프리픽스_뒤에_법인표기가_와도_처리한다():
    """실제로 이렇게 겹쳐 온다. 그래서 채널 분리가 법인표기 제거보다 먼저다."""
    parts = normalize_name("NAVERPAY*(주)스타벅스커피코리아")
    assert parts.payment_channel is PaymentChannel.NAVERPAY
    assert parts.normalized_name == "스타벅스커피코리아"


def test_배달앱_자체가_가맹점이면_이름을_지킨다():
    """⚠️ 함정: `배달의민족` 은 브랜드 카탈로그의 항목이다. 프리픽스로 잘라내면
    정상 브랜드가 빈 문자열이 되어 미분류로 떨어진다."""
    parts = normalize_name("배달의민족")
    assert parts.normalized_name == "배달의민족"
    assert parts.payment_channel is PaymentChannel.BAEMIN


def test_배달앱_프리픽스_뒤에_가맹점이_있으면_분리한다():
    parts = normalize_name("배달의민족김밥천국")
    assert parts.normalized_name == "김밥천국"
    assert parts.payment_channel is PaymentChannel.BAEMIN


def test_구분자만_남은_프리픽스는_이름을_비우지_않는다():
    """원장이 깨져 프리픽스만 온 경우 — 채널은 살리고 이름은 원문을 둔다."""
    name, channel = strip_channel("NAVERPAY*")
    assert channel is PaymentChannel.NAVERPAY
    assert name == "NAVERPAY*"


# ── ③ 법인 표기 ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("(주)스타벅스커피코리아", "스타벅스커피코리아"),
        ("주식회사 카카오", "카카오"),
        ("(유)동네약국", "동네약국"),
        ("GS25 CO.,LTD", "GS25"),
        ("COUPANG INC.", "COUPANG"),
        ("MUSINSA LLC", "MUSINSA"),
        ("(주)이마트(주)", "이마트"),
    ],
)
def test_법인표기를_제거한다(raw, expected):
    assert strip_corporate(canonical_form(raw)) == expected


def test_법인명을_브랜드로_바꾸지는_않는다():
    """`에스씨케이컴퍼니 → 스타벅스` 는 사전(alias)의 일이다. 전처리는 표기만 벗긴다."""
    assert normalize_name("(주)에스씨케이컴퍼니").normalized_name == "에스씨케이컴퍼니"


# ── ④ PG 마스킹 ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "raw",
    ["(주)케이지이니시스", "케이지모빌리언스", "NHN한국사이버결제", "토스페이먼츠", "(주)다날", "KCP"],
)
def test_PG_경유는_추측하지_않고_표시만_한다(raw):
    parts = normalize_name(raw)
    assert parts.pg_masked is True
    assert parts.payment_channel is PaymentChannel.PG


def test_TOSSPAY_프리픽스와_토스페이먼츠_PG_를_구별한다():
    """둘은 다르다 — 하나는 간편결제 채널, 하나는 실가맹점이 소실된 PG 경유다."""
    channel = normalize_name("TOSSPAY*이마트")
    assert channel.payment_channel is PaymentChannel.TOSSPAY
    assert channel.pg_masked is False

    pg = normalize_name("토스페이먼츠")
    assert pg.pg_masked is True


def test_일반_가맹점은_PG_로_표시되지_않는다():
    assert normalize_name("스타벅스 강남역점").pg_masked is False


# ── ⑤ 지점 접미사 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("raw", "name", "branch"),
    [
        ("스타벅스 강남역점", "스타벅스", "강남역점"),
        ("이마트 성수점", "이마트", "성수점"),
        ("GS25 종로3가점", "GS25", "종로3가점"),
        ("스타벅스 1234호점", "스타벅스", "1234호점"),
        ("스타벅스1234호점", "스타벅스", "1234호점"),
        ("교촌치킨 판교지점", "교촌치킨", "판교지점"),
        ("SK에너지 여의도대리점", "SK에너지", "여의도대리점"),
    ],
)
def test_지점을_분리해_보존한다(raw, name, branch):
    parts = normalize_name(raw)
    assert parts.normalized_name == name
    assert parts.branch == branch


def test_전각_숫자_호점도_잡는다():
    parts = normalize_name("스타벅스 ２호점")
    assert parts.normalized_name == "스타벅스"
    assert parts.branch == "2호점"


def test_이름_전체가_점으로_끝나면_지점이_아니다():
    """⚠️ 과잉 정규화 방지. `\\S+점$` 로 넓히면 브랜드를 잘라먹는다 — 되돌릴 수 없다."""
    assert split_branch("낙원떡집") == ("낙원떡집", None)
    assert split_branch("동네약국") == ("동네약국", None)
    # 공백 없는 단일 토큰은 지점으로 보지 않는다
    assert split_branch("면세점") == ("면세점", None)


def test_지점만_있으면_자르지_않는다():
    """`강남역점` 만 온 경우 잘라내면 이름이 사라진다."""
    head, branch = split_branch("강남역점")
    assert head == "강남역점"
    assert branch is None


# ── ⑥ 취소 페어링 ───────────────────────────────────────────────────────────
def test_전액취소는_원거래와_취소행을_모두_뺀다():
    """남겨두면 두 번 세어진다 — 미분류율 분모가 부풀고 이상지출이 오탐된다."""
    approve = _raw("스타벅스", amount=12000)
    cancel = _raw("스타벅스", amount=-12000, tx_type=TxType.CANCEL)
    original, cancelled = preprocess([approve, cancel])

    assert original.is_cancelled is True
    assert original.excluded_reason == REASON_CANCELLED
    assert original.is_classifiable is False

    assert cancelled.excluded_reason == REASON_CANCEL_ROW
    assert cancelled.cancel_of == "A0001"
    assert cancelled.is_classifiable is False


def test_부분취소는_원거래를_남긴다():
    """순액이 실제 지출이므로 원거래는 분류 대상이다. 취소행만 뺀다."""
    approve = _raw("이마트", amount=50000)
    cancel = _raw("이마트", amount=-20000, tx_type=TxType.CANCEL)
    original, cancelled = preprocess([approve, cancel])

    assert original.is_classifiable is True
    assert cancelled.excluded_reason == REASON_PARTIAL_CANCEL_ROW
    assert cancelled.is_classifiable is False


def test_승인번호가_같아도_기관이_다르면_상계하지_않는다():
    """승인번호는 기관 내에서만 유일하다. 기관을 무시하면 엉뚱한 거래가 상계된다."""
    a = _raw("스타벅스", amount=12000, provider_id="CARD_A")
    b = _raw("이마트", amount=-12000, provider_id="CARD_B", tx_type=TxType.CANCEL)
    first, second = preprocess([a, b])

    assert first.is_classifiable is True          # 상계되지 않았다
    assert second.excluded_reason == REASON_CANCEL_ROW  # 짝 없는 취소행


def test_짝_없는_취소행은_상계할_원거래가_없다():
    """원거래가 조회 기간 밖인 경우. 분류 대상은 아니지만 상계도 하지 않는다."""
    (only_cancel,) = preprocess([_raw("스타벅스", amount=-12000, tx_type=TxType.CANCEL)])
    assert only_cancel.excluded_reason == REASON_CANCEL_ROW
    assert only_cancel.is_cancelled is False


def test_정정은_원거래를_빼고_정정행을_남긴다():
    """규격은 원 승인 행과 정정 행을 **둘 다** 준다. 실지출은 정정 행이다 — 취소의 반대다.

    원거래를 남기면 정정 전 금액이 지출로 잡혀 이상탐지가 "쓰지 않은 돈"을 본다.
    """
    approve = _raw("이마트", amount=50000)
    corrected = _raw("이마트", amount=30000, tx_type=TxType.CORRECTION)
    original, fixed = preprocess([approve, corrected])

    assert original.excluded_reason == REASON_CORRECTED
    assert original.is_classifiable is False
    assert original.is_cancelled is False       # 취소가 아니다 — 이유를 구분해 남긴다
    assert fixed.is_classifiable is True        # 정정 행이 실지출이다


def test_무승인매입도_분류_대상이다():
    """`status=04` 는 승인 절차가 없었을 뿐 돈은 나갔다."""
    (tx,) = preprocess([_raw("주차장", amount=3000, tx_type=TxType.UNAPPROVED)])
    assert tx.is_classifiable is True


def test_무승인매입도_취소와_짝을_이룬다():
    """승인 없이 매입된 건도 취소될 수 있다. 원거래로 다루지 않으면 상계가 새어나간다."""
    unapproved = _raw("주차장", amount=3000, tx_type=TxType.UNAPPROVED)
    cancel = _raw("주차장", amount=-3000, tx_type=TxType.CANCEL)
    original, cancelled = preprocess([unapproved, cancel])

    assert original.excluded_reason == REASON_CANCELLED
    assert cancelled.excluded_reason == REASON_CANCEL_ROW


# ── 사업자등록번호 키 ───────────────────────────────────────────────────────
def test_사업자등록번호는_표기가_달라도_같은_키가_된다():
    """규격은 하이픈 포함이지만 빼고 주는 기관이 있다. 표기로 매칭하면 사업자가 갈라진다."""
    assert regno_key("123-45-67890") == "1234567890"
    assert regno_key("1234567890") == "1234567890"
    assert regno_key(" 123 - 45 - 67890 ") == "1234567890"


@pytest.mark.parametrize("bad", [None, "", "123-45", "12345678901", "사업자없음"])
def test_사업자등록번호가_10자리가_아니면_키를_만들지_않는다(bad):
    """틀린 키로 매칭하면 엉뚱한 브랜드에 붙는다 — 조용히 틀린 분류가 된다.

    regno 는 보조 신호이므로 버려도 이름 매칭이 남는다.
    """
    assert regno_key(bad) is None


def test_입력_순서를_유지한다():
    names = ["스타벅스", "이마트", "GS25"]
    got = preprocess([_raw(n, approval_no=f"A{i}") for i, n in enumerate(names)])
    assert [t.raw.merchant_name_raw for t in got] == names


# ── ⑦ 할부 ──────────────────────────────────────────────────────────────────
def test_할부_회차_행은_아직_데이터에_없다():
    """생성기는 할부를 승인 1건으로 낸다. 회차 행이 생기는 날 집계를 구현한다."""
    txs = [_raw("이마트", amount=300000, installment_months=3)]
    assert has_installment_rows(txs) is False
    (normalized,) = preprocess(txs)
    assert normalized.is_classifiable is True   # 실제 지출이므로 분류 대상이다


def test_같은_승인번호로_회차가_여러_행_오면_감지한다():
    txs = [_raw("이마트", amount=100000, installment_months=3) for _ in range(3)]
    assert has_installment_rows(txs) is True


# ── 실데이터·드리프트 가드 ──────────────────────────────────────────────────
def test_생성기_오염을_전처리가_되돌린다():
    """드리프트 가드 — 생성기가 새 프리픽스·PG명을 쓰면 여기서 깨진다.

    두 모듈은 서로 import 하지 않는다(전처리가 정답 생성기에 의존하면 안 된다). 대신 이
    테스트가 두 목록이 어긋나지 않게 잡는다 — `test_taxonomy_sync.py` 와 같은 방식이다.
    """
    from txlens.generator.brands import CHANNEL_PREFIXES as GEN_PREFIXES
    from txlens.generator.brands import PG_NAMES as GEN_PG

    for prefix, channel in GEN_PREFIXES:
        parts = normalize_name(f"{prefix}스타벅스")
        assert parts.payment_channel is not None, f"생성기 프리픽스를 전처리가 모른다: {prefix}"
        assert parts.payment_channel.value == channel
        assert parts.normalized_name == "스타벅스"

    for pg in GEN_PG:
        parts = normalize_name(pg)
        assert parts.pg_masked is True, f"생성기 PG명을 전처리가 모른다: {pg}"


def test_실데이터_전량_전처리():
    """생성된 8,969건을 실제로 통과시킨다. 예외 없이 끝나고 이름이 남아야 한다."""
    from pathlib import Path

    from txlens.ingest import load_dataset

    bundle_dir = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "bundle"
    if not bundle_dir.exists():
        pytest.skip("합성 데이터가 없다 — `python -m txlens.generator --seed 42`")

    txs, _ = load_dataset(bundle_dir)
    normalized = preprocess(txs)

    assert len(normalized) == len(txs)
    assert all(n.normalized_name for n in normalized), "빈 매칭 키가 생겼다"

    # 정규화는 표기 수를 **줄여야** 한다. 늘어나면 규칙이 이름을 쪼개고 있다는 뜻이다.
    raw_distinct = len({t.merchant_name_raw for t in txs})
    norm_distinct = len({n.normalized_name for n in normalized})
    assert norm_distinct < raw_distinct

    # 취소 페어가 상계되어 분류 대상에서 빠졌는가
    cancels = [n for n in normalized if n.raw.tx_type is TxType.CANCEL]
    assert cancels, "취소 거래가 데이터에 있어야 한다"
    assert all(not n.is_classifiable for n in cancels)
