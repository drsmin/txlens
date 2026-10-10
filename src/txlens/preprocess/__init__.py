"""파이프라인 2단계 — 결정론적 정규화 (blueprint §3.3 ①).

`RawTx` → `NormalizedTx`. LLM 을 쓰지 않고, 규칙별로 순수 함수이며 `tests/test_preprocess.py` 가
규칙 단위로 회귀를 잠근다. **여기서 정확도의 절반이 결정된다** — 뒤 단계(룰 매칭·LLM)가
아무리 좋아도 매칭 키가 지저분하면 같은 가맹점이 갈라진다.

원칙: **버리지 않고 분리해 보존한다.** 지점·결제채널·PG 여부는 필드로 남아 뒤 단계가 쓴다.
"""

from __future__ import annotations

from txlens.preprocess.identifiers import REGNO_DIGITS, regno_key
from txlens.preprocess.names import (
    NameParts,
    canonical_form,
    normalize_name,
    split_branch,
    strip_channel,
    strip_corporate,
)
from txlens.preprocess.pairing import (
    REASON_CANCEL_ROW,
    REASON_CANCELLED,
    REASON_CORRECTED,
    REASON_PARTIAL_CANCEL_ROW,
    has_installment_rows,
    pair_cancellations,
)
from txlens.schemas import NormalizedTx, RawTx

__all__ = [
    "REASON_CANCELLED",
    "REASON_CANCEL_ROW",
    "REASON_CORRECTED",
    "REASON_PARTIAL_CANCEL_ROW",
    "REGNO_DIGITS",
    "NameParts",
    "canonical_form",
    "has_installment_rows",
    "normalize_name",
    "pair_cancellations",
    "preprocess",
    "preprocess_one",
    "regno_key",
    "split_branch",
    "strip_channel",
    "strip_corporate",
]

#: 정규화 결과가 빈 문자열이 된 경우. 원문이 기호뿐인 등 매칭 키를 만들 수 없다.
REASON_EMPTY_NAME = "정규화_불가"


def preprocess_one(
    tx: RawTx,
    *,
    cancel_of: str | None = None,
    excluded_reason: str | None = None,
) -> NormalizedTx:
    """거래 1건을 정규화한다. 취소 페어링 결과는 호출자가 주입한다.

    페어링은 거래 **집합**을 봐야 알 수 있으므로 한 건만 보고 판단하지 않는다 — 이 함수를
    순수하게 유지해 단위 테스트가 쉽도록, 집합 판단은 `preprocess()` 가 맡는다.
    """
    parts = normalize_name(tx.merchant_name_raw)

    reason = excluded_reason
    if reason is None and parts.is_empty:
        reason = REASON_EMPTY_NAME

    return NormalizedTx(
        raw=tx,
        # 빈 문자열은 매칭 키로 쓸 수 없다. 원문을 남겨 사람이 추적할 수 있게 한다
        # (`excluded_reason` 이 붙어 분류 대상에서는 빠진다).
        normalized_name=parts.normalized_name or tx.merchant_name_raw.strip(),
        branch=parts.branch,
        payment_channel=parts.payment_channel,
        pg_masked=parts.pg_masked,
        is_cancelled=excluded_reason == REASON_CANCELLED,
        cancel_of=cancel_of,
        excluded_reason=reason,
    )


def preprocess(txs: list[RawTx]) -> list[NormalizedTx]:
    """거래 집합을 정규화한다. 입력 순서를 유지한다 (재현성 — `ingest` 와 같은 이유).

    취소 페어링만 집합 판단이고 나머지는 건 단위다.
    """
    pairing = pair_cancellations(txs)
    out: list[NormalizedTx] = []
    for tx in txs:
        cancel_of, reason = pairing.get(id(tx), (None, None))
        out.append(preprocess_one(tx, cancel_of=cancel_of, excluded_reason=reason))
    return out
