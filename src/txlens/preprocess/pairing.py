"""승인취소 페어링 · 할부 처리 (blueprint §3.3 ①).

취소된 거래를 분류 대상에 남겨두면 두 번 세어진다 — 미분류율의 분모가 부풀고, 2부의
이상지출 탐지는 "쓰지 않은 돈"을 급증으로 잡는다. 그래서 여기서 상계하고 `excluded_reason` 을
붙여 **왜 빠졌는지 설명 가능하게** 남긴다 (조용히 버리지 않는다).
"""

from __future__ import annotations

from collections import defaultdict

from txlens.schemas import RawTx, TxType

#: `excluded_reason` 값. 문자열을 흩뿌리지 않고 여기 모아 지표 집계에서 그대로 센다.
REASON_CANCELLED = "취소_상계"
REASON_CANCEL_ROW = "취소_거래"
REASON_PARTIAL_CANCEL_ROW = "부분취소_거래"
#: 정정(`status=03`)된 원거래. 실지출은 정정 행의 `modified_amt` 이므로 원거래를 뺀다 (D-023).
REASON_CORRECTED = "정정_상계"


def pair_cancellations(txs: list[RawTx]) -> dict[int, tuple[str | None, str | None]]:
    """취소 페어를 찾아 `id(tx) → (cancel_of, excluded_reason)` 로 돌려준다.

    페어 키는 **(제공기관, 고객, 승인번호)** 다. 승인번호만으로 묶으면 기관 간 번호 충돌 시
    엉뚱한 거래가 상계된다 — 승인번호는 기관 내에서만 유일하다.

    - **전액취소**(합계 0): 원거래를 `is_cancelled` 로 빼고 취소행도 뺀다.
    - **부분취소**(합계 ≠ 0): 원거래는 **남긴다.** 순액이 실제 지출이므로 분류 대상이다.
      취소행만 빼서 이중 계산을 막는다.
    - **정정**(`status=03`): 규격은 원 승인 행과 정정 행을 **둘 다** 회신한다. 실지출은 정정 행의
      `modified_amt` 이므로 **원거래를 빼고 정정 행을 남긴다** — 취소와 반대 방향이다 (D-023).

    반환을 `id()` 키로 준 이유: `RawTx` 는 frozen 이고 같은 값의 거래가 여럿 있을 수 있어
    값으로는 개별 행을 지목할 수 없다. 호출자가 같은 리스트를 순회하는 동안에만 유효하다.
    """
    groups: dict[tuple[str, str, str], list[RawTx]] = defaultdict(list)
    for tx in txs:
        groups[(tx.provider_id, tx.customer_id, tx.approval_no)].append(tx)

    out: dict[int, tuple[str | None, str | None]] = {}
    for (_, _, approval_no), items in groups.items():
        cancels = [t for t in items if t.tx_type is TxType.CANCEL]
        corrections = [t for t in items if t.tx_type is TxType.CORRECTION]
        # 무승인매입(`UNAPPROVED`)은 승인이 없었을 뿐 실지출이라 원거래와 같게 다룬다.
        approvals = [t for t in items if t.tx_type in (TxType.APPROVE, TxType.UNAPPROVED)]

        # 정정이 있으면 원 승인 행은 대체된 금액이므로 뺀다. 정정 행 자체는 분류 대상으로 남긴다.
        if corrections:
            for t in approvals:
                out[id(t)] = (None, REASON_CORRECTED)

        if not cancels or not approvals:
            # 짝 없는 취소행(원거래가 조회 기간 밖)은 분류 대상이 아니지만 상계할 원거래도 없다.
            for t in cancels:
                out[id(t)] = (approval_no, REASON_CANCEL_ROW)
            continue

        net = sum(t.amount for t in items)
        full = net == 0
        for t in cancels:
            out[id(t)] = (
                approval_no,
                REASON_CANCEL_ROW if full else REASON_PARTIAL_CANCEL_ROW,
            )
        if full:
            for t in approvals:
                out[id(t)] = (None, REASON_CANCELLED)
    return out


def has_installment_rows(txs: list[RawTx]) -> bool:
    """같은 승인번호로 회차 청구 행이 여러 개 오는가 — 할부 집계가 필요한 형태인가.

    **현재 데이터에는 집계할 것이 없다.** 생성기는 할부를 `total_install_cnt` 가 붙은 **승인 1건**
    으로 내보내고(`generator/build.py`) 회차별 청구 행을 만들지 않는다. blueprint §3.3 ① 의
    "할부: 회차 거래를 원거래 1건으로 집계"에 해당하는 입력이 아직 데이터에 없다는 뜻이다.

    없는 입력에 맞춰 집계 로직을 쓰면 **테스트할 수 없는 코드**가 되므로 지금은 만들지 않고,
    이 함수로 "그런 데이터가 들어왔는지"만 감시한다. True 가 되는 날 집계를 구현한다.
    회차 행이 없는 할부 거래는 **정상 분류 대상**이다 — 실제 지출이기 때문이다.
    """
    seen: dict[tuple[str, str, str], int] = defaultdict(int)
    for tx in txs:
        if tx.installment_months and tx.tx_type is TxType.APPROVE:
            seen[(tx.provider_id, tx.customer_id, tx.approval_no)] += 1
    return any(count > 1 for count in seen.values())
