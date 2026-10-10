"""ingest 계약 테스트.

여기서 막고 싶은 것은 파싱 실패가 아니라 **조용히 모수를 틀어뜨리는 경로**다:
건수 불일치를 넘기는 것, 빈 문자열을 값으로 받아들이는 것, 깨진 레코드를 말없이 버리는 것.

생성기 ↔ ingest 왕복(round-trip)도 여기서 잠근다 — emit 이 규격 관행을 바꾸면 이 테스트가
먼저 깨져야 한다. 둘은 서로의 역함수라는 계약을 공유한다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from txlens.generator.emit import to_bundles
from txlens.ingest import (
    IngestError,
    IngestReport,
    load_bundle,
    load_dataset,
    parse_amount,
    parse_dtime,
    split_bundle_name,
    to_raw_tx,
)
from txlens.schemas import CardType, RawTx, TxType


def _item(**kw) -> dict:
    """CD03 `approved_list` 항목 하나. **규격의 기본형**을 쓴다 (D-023).

    `total_install_cnt` 를 넣지 않는 것이 의도다 — 규격이 "일시불일 경우 미회신"이라고
    못박았으므로 일시불 항목에는 그 키가 아예 없다. `"0"` 을 기본값으로 두면 규격에 없는
    형태를 테스트가 정상으로 굳혀버린다.
    """
    base = {
        "approved_num": "A0001",
        "approved_dtime": "20260315143000",
        "approved_amt": "12000",
        "status": "01",
        "pay_type": "01",
        "merchant_name": "스타벅스 강남역점",
    }
    base.update(kw)
    return base


def _payload(items: list[dict], **kw) -> dict:
    base = {
        "rsp_code": "00000",
        "rsp_msg": "정상처리되었습니다",
        "approved_cnt": str(len(items)),
        "approved_list": items,
    }
    base.update(kw)
    return base


def _load(items: list[dict], **kw) -> list[RawTx]:
    return load_bundle(_payload(items), provider_id="CARD_A", customer_id="SYNTH_0000", **kw)


# ── 규격 관행 되돌리기 ──────────────────────────────────────────────────────
def test_금액은_문자열로_온다():
    assert parse_amount("12000") == 12000
    assert parse_amount(" 12,000 ") == 12000


@pytest.mark.parametrize("bad", ["", "12000.5", "만이천원", "1_2"])
def test_해석할_수_없는_금액은_실패한다(bad):
    with pytest.raises(IngestError):
        parse_amount(bad)


def test_거래일시는_tz_없는_KST_로_읽는다():
    dt = parse_dtime("20260315143000")
    assert dt == datetime(2026, 3, 15, 14, 30, 0)
    assert dt.tzinfo is None


def test_취소는_금액을_음수로_뒤집는다():
    """규격은 부호 없는 금액 + status 로 표현한다. 파이프라인 안에서는 부호로 다룬다."""
    tx = to_raw_tx(
        _item(status="02", approved_amt="12000"),
        provider_id="CARD_A",
        customer_id="SYNTH_0000",
    )
    assert tx.tx_type is TxType.CANCEL
    assert tx.amount == -12000


def test_할부는_회차_회신_여부로만_판단한다():
    """규격이 "일시불일 경우 미회신"이므로 **키의 유무**가 신호다 (D-023).

    이전 판본은 `pay_type` 을 일시불/할부로 오해해 "pay_type 이 02 가 아니면 회차를 무시"
    했다. 규격상 02 는 **체크카드**라, 그 규칙은 "체크카드가 아니면 할부를 무시한다"가 된다.
    """
    plain = to_raw_tx(_item(), provider_id="CARD_A", customer_id="SYNTH_0000")
    assert plain.installment_months == 0

    for pay_type in ("01", "02"):
        tx = to_raw_tx(
            _item(pay_type=pay_type, total_install_cnt="3"),
            provider_id="CARD_A",
            customer_id="SYNTH_0000",
        )
        assert tx.installment_months == 3, f"pay_type={pay_type} 이 할부 판단에 끼어들었다"


def test_pay_type_은_신용_체크_구분이다():
    """규격 코드셋 `card_pay_type` = {01 신용, 02 체크} (D-023)."""
    credit = to_raw_tx(_item(pay_type="01"), provider_id="CARD_A", customer_id="SYNTH_0000")
    check = to_raw_tx(_item(pay_type="02"), provider_id="CARD_A", customer_id="SYNTH_0000")
    assert credit.card_type is CardType.CREDIT
    assert check.card_type is CardType.CHECK

    absent = to_raw_tx(_item(pay_type=""), provider_id="CARD_A", customer_id="SYNTH_0000")
    assert absent.card_type is None

    with pytest.raises(IngestError, match="pay_type"):
        to_raw_tx(_item(pay_type="09"), provider_id="CARD_A", customer_id="SYNTH_0000")


# ── status 4종 (규격 코드셋 card_approval_status) ────────────────────────────
def test_정정은_modified_amt_가_실지출이다():
    """`status=03` 의 `approved_amt` 는 정정 **전** 금액이다. 믿으면 지출이 틀어진다 (D-023)."""
    tx = to_raw_tx(
        _item(status="03", approved_amt="12000", modified_amt="8000"),
        provider_id="CARD_A",
        customer_id="SYNTH_0000",
    )
    assert tx.tx_type is TxType.CORRECTION
    assert tx.amount == 8000


def test_무승인매입도_실지출이라_받는다():
    """`status=04` 는 승인 절차가 없었을 뿐 돈은 나갔다. 거부하면 실데이터에서 죽는다."""
    tx = to_raw_tx(_item(status="04"), provider_id="CARD_A", customer_id="SYNTH_0000")
    assert tx.tx_type is TxType.UNAPPROVED
    assert tx.amount == 12000


def test_취소_정정은_trans_dtime_을_사건_시각으로_쓴다():
    """`approved_dtime` 은 원 승인일시다. 그걸 거래일시로 쓰면 취소가 원거래 달에 잡힌다."""
    tx = to_raw_tx(
        _item(status="02", approved_dtime="20260315143000", trans_dtime="20260402101500"),
        provider_id="CARD_A",
        customer_id="SYNTH_0000",
    )
    assert tx.tx_datetime == datetime(2026, 4, 2, 10, 15)
    assert tx.original_datetime == datetime(2026, 3, 15, 14, 30)

    # 승인 행에 trans_dtime 이 와도 거래일시는 승인일시다 (같은 사건이므로 바꿀 이유가 없다).
    approved = to_raw_tx(
        _item(trans_dtime="20260402101500"),
        provider_id="CARD_A",
        customer_id="SYNTH_0000",
    )
    assert approved.tx_datetime == datetime(2026, 3, 15, 14, 30)
    assert approved.original_datetime is None


# ── 조건부 미회신 ───────────────────────────────────────────────────────────
def test_없는_필드와_빈_문자열을_같게_다룬다():
    """빈 문자열을 값으로 받으면 "regno 가 있다"고 착각해 REGNO 단계가 헛돈다 (D-012)."""
    absent = to_raw_tx(_item(), provider_id="CARD_A", customer_id="SYNTH_0000")
    blank = to_raw_tx(
        _item(merchant_regno="  "),
        provider_id="CARD_A",
        customer_id="SYNTH_0000",
    )
    assert absent.merchant_regno is None
    assert blank.merchant_regno is None


def test_사업자등록번호는_규격_형태로_보존한다():
    """규격은 `aN(12)` **하이픈 포함**이다. 표기를 고치지 않고 보존하고 매칭 때만 키로 만든다."""
    from txlens.preprocess import regno_key

    tx = to_raw_tx(
        _item(merchant_regno="123-45-67890"),
        provider_id="CARD_A",
        customer_id="SYNTH_0000",
    )
    assert tx.merchant_regno == "123-45-67890"
    assert regno_key(tx.merchant_regno) == "1234567890"


def test_필수_필드가_없으면_실패한다():
    for key in ("approved_num", "approved_dtime", "approved_amt", "merchant_name"):
        item = _item()
        del item[key]
        with pytest.raises(IngestError):
            to_raw_tx(item, provider_id="CARD_A", customer_id="SYNTH_0000")


def test_알_수_없는_status_는_실패한다():
    with pytest.raises(IngestError):
        to_raw_tx(_item(status="99"), provider_id="CARD_A", customer_id="SYNTH_0000")


# ── 봉투 검증 ───────────────────────────────────────────────────────────────
def test_건수_불일치는_전송_누락으로_막는다():
    """여기서 놓치면 "원래 그만큼이었다"고 착각한 채 지표를 낸다."""
    payload = _payload([_item()], approved_cnt="2")
    with pytest.raises(IngestError, match="전송 누락"):
        load_bundle(payload, provider_id="CARD_A", customer_id="SYNTH_0000")


def test_오류_응답은_거부한다():
    payload = _payload([], rsp_code="40001", rsp_msg="권한 없음")
    with pytest.raises(IngestError, match="정상 응답이 아니다"):
        load_bundle(payload, provider_id="CARD_A", customer_id="SYNTH_0000")


def test_approved_list_가_없으면_실패한다():
    with pytest.raises(IngestError, match="approved_list"):
        load_bundle(
            {"rsp_code": "00000", "approved_cnt": "0"},
            provider_id="CARD_A",
            customer_id="SYNTH_0000",
        )


def test_next_page_가_있으면_조용히_잘라내지_않는다():
    """규격은 마지막 페이지면 `next_page` 를 미회신한다 (D-023).

    있는데 무시하면 거래의 **일부만** 보고 미분류율·이상탐지를 계산한다 — 조용히 틀리는 경로라
    막는다. 페이지네이션 자체는 실연동 때 호출 측이 구현할 몫이다.
    """
    payload = _payload([_item()], next_page="MjAyNjAzMTU=")
    with pytest.raises(IngestError, match="next_page"):
        load_bundle(payload, provider_id="CARD_A", customer_id="SYNTH_0000")

    # 마지막 페이지 표현(키 없음 · 빈 문자열)은 통과해야 한다.
    assert len(_load([_item()])) == 1
    last = load_bundle(
        _payload([_item()], next_page=""),
        provider_id="CARD_A",
        customer_id="SYNTH_0000",
    )
    assert len(last) == 1


# ── 엄격/관용 모드 ──────────────────────────────────────────────────────────
def test_기본은_엄격_실패다():
    with pytest.raises(IngestError):
        _load([_item(), _item(approved_amt="???")])


def test_관용_모드는_버린_건수를_보고한다():
    """건너뛰기를 허용하되 조용히 사라지는 경로는 만들지 않는다."""
    report = IngestReport()
    txs = _load(
        [_item(), _item(approved_num="A0002", approved_amt="???")],
        strict=False,
        report=report,
    )
    assert len(txs) == 1
    assert report.tx_loaded == 1
    assert report.tx_skipped == 1
    assert not report.is_clean
    assert report.skipped[0][1] == "A0002"
    assert "버림 1건" in report.summary()


# ── 파일명 규약 ─────────────────────────────────────────────────────────────
def test_파일명에서_제공기관과_고객을_분리한다():
    """제공기관·고객 id 양쪽에 `_` 가 있으므로 단순 split 으로는 안 된다."""
    assert split_bundle_name(Path("CARD_A_SYNTH_0000_CD03.json")) == ("CARD_A", "SYNTH_0000")
    assert split_bundle_name(Path("/x/y/EFIN_D_SYNTH_0029_CD03.json")) == ("EFIN_D", "SYNTH_0029")


@pytest.mark.parametrize("bad", ["CARD_A.json", "CARD_A_CD03.json"])
def test_규약에_맞지_않는_파일명은_실패한다(bad):
    with pytest.raises(IngestError):
        split_bundle_name(Path(bad))


# ── 생성기 왕복 ─────────────────────────────────────────────────────────────
def test_생성기_왕복은_원본을_복원한다():
    """emit → ingest 가 역함수임을 잠근다. 규격 관행이 바뀌면 여기가 먼저 깨진다."""
    from txlens.generator.build import build

    txs, _ = build(seed=7, customers=2, months=2, per_month=8, period_end=datetime(2026, 8, 31))
    bundles = to_bundles(txs)

    restored: list[RawTx] = []
    for (provider, customer), payload in bundles.items():
        restored.extend(load_bundle(payload, provider_id=provider, customer_id=customer))

    assert len(restored) == len(txs)
    by_key = {(t.approval_no, t.tx_type): t for t in restored}
    for original in txs:
        got = by_key[(original.approval_no, original.tx_type)]
        assert got.merchant_name_raw == original.merchant_name_raw
        assert got.amount == original.amount
        assert got.tx_datetime == original.tx_datetime
        assert got.merchant_regno == original.merchant_regno
        assert got.installment_months == original.installment_months
        assert got.provider_id == original.provider_id
        assert got.customer_id == original.customer_id


def test_빈_디렉터리는_생성_명령을_안내한다(tmp_path):
    with pytest.raises(IngestError, match="txlens.generator"):
        load_dataset(tmp_path)
