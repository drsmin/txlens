"""Layer 1 — 표준API 봉투 직렬화.

필드명·형식은 금융 마이데이터 표준API **국내 승인내역 조회(카드-008 / CD03, v2 2025-09-30)** 를
따른다. 임의로 지은 이름이 아니라 실제 규격을 쓰는 이유는 두 가지다: ① 실제 연동으로 교체할
지점이 명확해지고 ② 발표에서 "실제 규격 형태로 생성했다"고 말할 근거가 된다.

**규격 원문과 필드 단위로 대조했다** — 그 과정에서 `pay_type` 의미(신용/체크를 일시불/할부로
오해), `total_install_cnt` 미회신 규칙, 규격에 없는 `merchant_category`, `merchant_regno` 하이픈을
바로잡았다. 아직 규격과 다른 부분(`next_page` 페이지네이션 · `trans_dtime`/`modified_amt` ·
카드 단위 조회 · 정렬 방향)은 **의도된 단순화로 설계에 열거**했다 — 모르는 게 아니라 안 한 것이다.

규격의 관행을 그대로 지킨다 — **금액·건수는 문자열**(`N(n)` 의 예시 자체가 `"999"` 다),
일시는 `YYYYMMDDHHMMSS`(`DTIME`), 결제상태는 `status` 코드. 편의상 정수로 바꾸지 않는다.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from txlens.ground_truth import GroundTruth
from txlens.schemas import CardType, RawTx, TxType

#: `status` = 결제상태. 규격 코드셋 `card_approval_status`.
STATUS_APPROVED = "01"
STATUS_CANCELLED = "02"
STATUS_CORRECTED = "03"      # 정정 — `modified_amt` 가 따라온다
STATUS_UNAPPROVED = "04"     # 무승인 매입

#: `pay_type` = **신용/체크** 구분이다. 규격 코드셋 `card_pay_type`.
#: ⚠️ 할부 여부가 아니다 — 그건 `total_install_cnt` 의 회신 여부로 표현된다.
PAY_TYPE_CREDIT = "01"
PAY_TYPE_CHECK = "02"

_STATUS_BY_TX_TYPE = {
    TxType.APPROVE: STATUS_APPROVED,
    TxType.CANCEL: STATUS_CANCELLED,
    TxType.CORRECTION: STATUS_CORRECTED,
    TxType.UNAPPROVED: STATUS_UNAPPROVED,
}


def to_approved_item(tx: RawTx) -> dict:
    """거래 1건 → CD03 `approved_list` 항목."""
    item = {
        "approved_num": tx.approval_no,
        "approved_dtime": tx.tx_datetime.strftime("%Y%m%d%H%M%S"),
        "approved_amt": str(abs(tx.amount)),
        "status": _STATUS_BY_TX_TYPE[tx.tx_type],
        "pay_type": PAY_TYPE_CHECK if tx.card_type is CardType.CHECK else PAY_TYPE_CREDIT,
        "merchant_name": tx.merchant_name_raw,
    }
    # 규격상 필수가 아닌 필드는 **없을 때 키 자체를 넣지 않는다** (조건부 미회신).
    # None 을 채워 보내면 전처리가 "값이 있다"고 착각한다 — 제공기관 편차를 재현하는 핵심 지점.
    #
    # `total_install_cnt` 는 규격이 **"일시불일 경우 미회신"** 이라고 못박았다.
    # `"0"` 을 채워 보내면 규격 위반이고, 수신 측이 "할부 정보가 왔다"고 착각한다.
    if tx.installment_months:
        item["total_install_cnt"] = str(tx.installment_months)
    if tx.merchant_regno is not None:
        item["merchant_regno"] = tx.merchant_regno
    return item


def to_bundles(txs: list[RawTx]) -> dict[tuple[str, str], dict]:
    """(provider_id, customer_id) → CD03 응답 전문."""
    grouped: dict[tuple[str, str], list[RawTx]] = defaultdict(list)
    for tx in txs:
        grouped[(tx.provider_id, tx.customer_id)].append(tx)

    out: dict[tuple[str, str], dict] = {}
    for key, items in grouped.items():
        items.sort(key=lambda t: t.tx_datetime)
        out[key] = {
            "rsp_code": "00000",
            "rsp_msg": "정상처리되었습니다",
            "approved_cnt": str(len(items)),
            "approved_list": [to_approved_item(t) for t in items],
        }
    return out


def write_dataset(txs: list[RawTx], gt: GroundTruth, out_dir: Path) -> dict[str, int]:
    """`data/synthetic/` 에 번들과 정답셋을 쓴다.

    번들은 `.gitignore` 로 제외된다 — 시드로 재현 가능한 산출물이기 때문이다.
    정답셋도 재현 가능하지만 평가의 기준이므로 함께 재생성한다.
    """
    bundle_dir = out_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundles = to_bundles(txs)
    for (provider, customer), payload in bundles.items():
        path = bundle_dir / f"{provider}_{customer}_CD03.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "ground_truth.json").write_text(
        gt.model_dump_json(indent=2), encoding="utf-8"
    )

    manifest = {
        "seed": gt.seed,
        "customers": gt.customers,
        "period": {"start": gt.period_start, "end": gt.period_end},
        "tx_total": gt.tx_total,
        "distinct_raw_names": gt.distinct_raw_names,
        "bundles": len(bundles),
        "note": "시드 고정으로 재현 가능. 번들은 커밋하지 않는다 (README 참고).",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"bundles": len(bundles), "tx": gt.tx_total}
