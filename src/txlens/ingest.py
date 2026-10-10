"""파이프라인 1단계 — 표준API 응답을 읽어 `RawTx` 로 올린다.

`generator/emit.py` 의 역함수다. 생성기가 규격 관행(문자열 금액·`YYYYMMDDHHMMSS`·조건부
미회신)을 지켜 썼으므로 여기서 그걸 되돌린다. **실제 연동으로 교체할 지점이 이 모듈 하나다** —
`load_bundle()` 이 dict 를 받도록 되어 있어, 파일 대신 HTTP 응답을 넣으면 그대로 동작한다.

## 왜 관용적으로 파싱하지 않는가

깨진 항목을 조용히 건너뛰면 미분류율·정확도가 **모수부터** 틀어진다 (분모가 줄어드는데
아무도 모른다). 그래서 기본은 엄격 실패다. 다만 실무의 원장에는 실제로 깨진 레코드가 섞여
들어오므로, 건너뛰기를 원하면 `strict=False` 로 **명시**하고 `IngestReport` 에서 몇 건이
버려졌는지 확인하게 했다 — 조용히 사라지는 경로를 만들지 않는 것이 요점이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from txlens.schemas import CardType, RawTx, TxType

#: 표준API 카드 승인내역(CD03) 정상 응답 코드.
RSP_CODE_OK = "00000"

#: `status` = 결제상태. **네 값을 모두 받는다** — 규격 코드셋 `card_approval_status` (D-023).
#: 01/02 만 알던 판본은 실데이터의 정정·무승인매입을 만나면 죽었다.
STATUS_APPROVED = "01"
STATUS_CANCELLED = "02"
STATUS_CORRECTED = "03"
STATUS_UNAPPROVED = "04"

_TX_TYPE_BY_STATUS = {
    STATUS_APPROVED: TxType.APPROVE,
    STATUS_CANCELLED: TxType.CANCEL,
    STATUS_CORRECTED: TxType.CORRECTION,
    STATUS_UNAPPROVED: TxType.UNAPPROVED,
}

#: `pay_type` = **신용/체크** 구분이다 (`card_pay_type`). 할부 여부가 아니다 — D-023.
_CARD_TYPE_BY_PAY_TYPE = {"01": CardType.CREDIT, "02": CardType.CHECK}

#: 번들 파일명 규약 — `{provider_id}_{customer_id}_CD03.json` (emit.py).
#: 봉투 안에 제공기관·고객 식별자가 없다. 실제 표준API 도 마찬가지로 **요청 문맥**에 있고
#: 응답 본문에는 없기 때문에, 파일명을 그 문맥의 대역으로 쓴다.
BUNDLE_SUFFIX = "_CD03.json"


class IngestError(ValueError):
    """번들을 신뢰할 수 없다. 조용히 넘기지 않고 멈춘다."""


@dataclass
class IngestReport:
    """무엇을 읽고 무엇을 버렸는가. 지표의 모수를 설명하기 위해 남긴다."""

    bundles: int = 0
    tx_loaded: int = 0
    tx_skipped: int = 0
    #: (파일명, 승인번호 또는 인덱스, 이유)
    skipped: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.tx_skipped == 0

    def summary(self) -> str:
        s = f"번들 {self.bundles}개 · 거래 {self.tx_loaded}건"
        if self.tx_skipped:
            s += f" · 버림 {self.tx_skipped}건"
        return s


def parse_amount(text: str) -> int:
    """`"12000"` → `12000`. 규격이 금액을 **문자열**로 주기 때문에 필요하다.

    쉼표는 관용한다(제공기관에 따라 `"12,000"` 이 오는 일이 있다). 소수점은 원화에 무의미하므로
    허용하지 않는다 — 있으면 형식을 잘못 이해한 것이니 드러나야 한다.
    """
    cleaned = text.strip().replace(",", "")
    if not cleaned.lstrip("-").isdigit():
        raise IngestError(f"금액을 해석할 수 없다: {text!r}")
    return int(cleaned)


def parse_dtime(text: str) -> datetime:
    """`"20260315143000"` → `datetime`. tz 없는 KST 로 둔다 (D-009)."""
    try:
        return datetime.strptime(text.strip(), "%Y%m%d%H%M%S")  # noqa: DTZ007
    except ValueError as exc:
        raise IngestError(f"거래일시를 해석할 수 없다: {text!r}") from exc


def to_raw_tx(item: dict, *, provider_id: str, customer_id: str) -> RawTx:
    """CD03 `approved_list` 항목 하나 → `RawTx`.

    **취소는 금액을 음수로 뒤집는다.** 규격은 부호 없는 금액 + `status` 코드로 표현하지만,
    파이프라인 안에서는 부호로 다루는 편이 상계·집계가 단순하다. 이 변환을 한곳에 모아둔다.

    `status` 4종을 모두 받는다 (D-023) — 정정(03)은 `modified_amt` 가 실지출이고,
    무승인매입(04)은 승인이 없었을 뿐 실지출이라 분류·탐지 대상이다.
    """
    for key in ("approved_num", "approved_dtime", "approved_amt", "merchant_name"):
        if not item.get(key):
            raise IngestError(f"필수 필드 누락: {key}")

    status = str(item.get("status", STATUS_APPROVED))
    tx_type = _TX_TYPE_BY_STATUS.get(status)
    if tx_type is None:
        raise IngestError(f"알 수 없는 status: {status!r}")

    amount = abs(parse_amount(str(item["approved_amt"])))
    if tx_type is TxType.CORRECTION and _optional(item.get("modified_amt")) is not None:
        # 정정 행의 실지출은 `modified_amt` 다. `approved_amt` 는 정정 전 금액이므로
        # 그대로 쓰면 지출이 부풀거나 줄어든다.
        amount = abs(parse_amount(str(item["modified_amt"])))
    if tx_type is TxType.CANCEL:
        amount = -amount

    # 할부 회차는 **회신 여부**로만 판단한다. 규격이 "일시불일 경우 미회신"이라 못박았으므로
    # 키가 없으면 일시불이다. `pay_type` 은 신용/체크 구분이라 여기서 쓸 값이 아니다 (D-023).
    installment = 0
    raw_cnt = _optional(item.get("total_install_cnt"))
    if raw_cnt is not None:
        if not raw_cnt.isdigit():
            raise IngestError(f"할부 회차를 해석할 수 없다: {raw_cnt!r}")
        installment = int(raw_cnt)

    pay_type = _optional(item.get("pay_type"))
    card_type = None
    if pay_type is not None:
        card_type = _CARD_TYPE_BY_PAY_TYPE.get(pay_type)
        if card_type is None:
            raise IngestError(f"알 수 없는 pay_type: {pay_type!r}")

    # 취소·정정 행은 `approved_dtime` = 원 승인일시, `trans_dtime` = 사건(취소·정정) 일시다.
    # `tx_datetime` 은 "이 행의 사건 시각"이어야 상계·기간 집계가 맞으므로 있으면 그것을 쓴다.
    approved_at = parse_dtime(str(item["approved_dtime"]))
    trans_at = _optional(item.get("trans_dtime"))
    event_at = parse_dtime(trans_at) if trans_at else None
    use_trans = event_at is not None and tx_type in (TxType.CANCEL, TxType.CORRECTION)

    return RawTx(
        provider_id=provider_id,
        customer_id=customer_id,
        approval_no=str(item["approved_num"]),
        tx_datetime=event_at if use_trans else approved_at,
        original_datetime=approved_at if use_trans else None,
        merchant_name_raw=str(item["merchant_name"]),
        amount=amount,
        tx_type=tx_type,
        installment_months=installment,
        card_type=card_type,
        # 조건부 미회신 — **키가 없는 것과 빈 문자열을 같게 다룬다.** 빈 문자열을 값으로
        # 받아들이면 "regno 가 있다"고 착각해 REGNO 매칭 단계가 헛돈다 (D-012).
        # 규격 형태(하이픈 포함)를 그대로 보존한다 — 매칭 시 `preprocess.regno_key()` 를 쓴다.
        merchant_regno=_optional(item.get("merchant_regno")),
    )


def _optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_bundle(
    payload: dict,
    *,
    provider_id: str,
    customer_id: str,
    source: str = "<memory>",
    strict: bool = True,
    report: IngestReport | None = None,
) -> list[RawTx]:
    """CD03 응답 전문 하나 → `list[RawTx]`.

    실제 API 연동 시 이 함수에 HTTP 응답 JSON 을 그대로 넣으면 된다.
    """
    rsp_code = str(payload.get("rsp_code", ""))
    if rsp_code != RSP_CODE_OK:
        raise IngestError(
            f"{source}: 정상 응답이 아니다 (rsp_code={rsp_code!r} "
            f"msg={payload.get('rsp_msg')!r})"
        )

    items = payload.get("approved_list")
    if not isinstance(items, list):
        raise IngestError(f"{source}: approved_list 가 없거나 배열이 아니다")

    # `next_page` 가 있으면 **페이지가 더 있다** (규격: 마지막 페이지면 미회신 — D-023).
    # 한 페이지만 읽고 끝내면 거래의 일부만 보고 지표를 내게 된다. 생성기는 페이지네이션을
    # 만들지 않지만, 실연동에서 이 경로를 조용히 지나가지 않도록 여기서 막는다.
    if _optional(payload.get("next_page")) is not None:
        raise IngestError(
            f"{source}: next_page 가 있다 — 다음 페이지가 남아 있는데 한 페이지만 읽으려 한다. "
            "호출 측에서 페이지를 모두 받아 합친 뒤 넣어야 한다"
        )

    # 규격은 건수도 문자열로 준다. 실제 배열 길이와 어긋나면 전송 누락이므로 반드시 막는다 —
    # 여기서 놓치면 "데이터가 원래 그만큼이었다"고 착각한 채 지표를 낸다.
    declared = payload.get("approved_cnt")
    if declared is not None and parse_amount(str(declared)) != len(items):
        raise IngestError(
            f"{source}: approved_cnt={declared} 인데 항목은 {len(items)}개다 (전송 누락 의심)"
        )

    out: list[RawTx] = []
    for idx, item in enumerate(items):
        try:
            out.append(to_raw_tx(item, provider_id=provider_id, customer_id=customer_id))
        except (IngestError, ValidationError) as exc:
            if strict:
                raise IngestError(f"{source}[{idx}]: {exc}") from exc
            if report is not None:
                key = str(item.get("approved_num", idx)) if isinstance(item, dict) else str(idx)
                report.skipped.append((source, key, str(exc)))
                report.tx_skipped += 1

    if report is not None:
        report.bundles += 1
        report.tx_loaded += len(out)
    return out


def split_bundle_name(path: Path) -> tuple[str, str]:
    """`CARD_A_SYNTH_0000_CD03.json` → `("CARD_A", "SYNTH_0000")`.

    제공기관 id 에 `_` 가 들어가므로(`CARD_A`) 앞에서 자르면 안 된다. 고객 id 도 `SYNTH_0000`
    처럼 `_` 를 포함한다. 규약이 `{provider}_{customer}_CD03.json` 이고 **양쪽 다 두 토큰**이라는
    사실에 의존한다 — 생성기와 이 함수가 같은 규약을 공유하므로 형태가 바뀌면 여기가 깨진다.
    """
    name = path.name
    if not name.endswith(BUNDLE_SUFFIX):
        raise IngestError(f"번들 파일명 규약에 맞지 않는다: {name}")
    stem = name[: -len(BUNDLE_SUFFIX)]
    parts = stem.split("_")
    if len(parts) < 4:
        raise IngestError(f"번들 파일명에서 제공기관·고객을 분리할 수 없다: {name}")
    return "_".join(parts[:2]), "_".join(parts[2:])


def load_dataset(
    bundle_dir: Path,
    *,
    strict: bool = True,
) -> tuple[list[RawTx], IngestReport]:
    """`data/synthetic/bundle/` 전체를 읽는다.

    파일 순서를 **정렬**해 고정한다 — 시드가 같아도 디렉터리 순회 순서가 다르면 거래 순서가
    달라지고, 그러면 지표가 미세하게 흔들려 재현성 주장이 약해진다.
    """
    paths = sorted(bundle_dir.glob(f"*{BUNDLE_SUFFIX}"))
    if not paths:
        raise IngestError(
            f"{bundle_dir} 에 번들이 없다. `python -m txlens.generator --seed 42` 를 먼저 실행한다"
        )

    report = IngestReport()
    txs: list[RawTx] = []
    for path in paths:
        provider_id, customer_id = split_bundle_name(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        txs.extend(
            load_bundle(
                payload,
                provider_id=provider_id,
                customer_id=customer_id,
                source=path.name,
                strict=strict,
                report=report,
            )
        )
    return txs, report
