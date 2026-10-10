"""탐지기 3종이 공유하는 입력 정리 + 통계 도구.

## 탐지의 단위는 거래가 아니라 (고객 × 기간) 이다

1부는 거래 한 건을 보고 답한다. 2부는 한 고객의 6개월치를 모아 놓고서야 "이건 매달 나가는
돈이다"라고 말할 수 있다. 그래서 여기서 `ClassifiedTx` 를 **집계하기 좋은 납작한 행**으로
한 번 펴 두고, 세 탐지기가 같은 행을 본다 — 같은 입력이어야 셋의 결과를 나란히 놓을 수 있다.

## 취소·정정된 거래는 여기서 빠진다

전처리가 이미 `excluded_reason` 을 붙여 두었다(`preprocess/pairing.py`). 그걸 무시하고 전부
집계하면 이상지출 탐지가 **쓰지 않은 돈을 급증으로 잡는다.** 빼는 판단을 탐지기마다 반복하지
않도록 입구에서 한 번만 한다.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime

from txlens.schemas import CategoryL1, ClassifiedTx

#: MAD 를 정규분포의 표준편차 자리에 놓기 위한 상수 (1 / Φ⁻¹(0.75)).
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class SpendRow:
    """탐지기가 보는 거래 한 줄. `ClassifiedTx` 에서 필요한 것만 꺼낸 것."""

    customer_id: str
    approval_no: str
    at: datetime
    amount: int
    merchant_canonical: str
    category_l1: CategoryL1

    @property
    def month(self) -> str:
        return self.at.strftime("%Y-%m")


def spend_rows(classified: list[ClassifiedTx]) -> list[SpendRow]:
    """분류 결과 → 탐지 입력. 시각 오름차순으로 정렬해 돌려준다.

    **음수·0원 거래는 빼지 않고 걸러낸다** — 취소행은 이미 `is_classifiable` 에서 빠지지만,
    부분취소가 남긴 원거래처럼 금액이 양수인 행은 그대로 지출로 센다(순액이 실지출이다).
    """
    rows = [
        SpendRow(
            customer_id=tx.normalized.raw.customer_id,
            approval_no=tx.normalized.raw.approval_no,
            at=tx.normalized.raw.tx_datetime,
            amount=tx.normalized.raw.amount,
            merchant_canonical=tx.merchant_canonical,
            category_l1=tx.category_l1,
        )
        for tx in classified
        if tx.normalized.is_classifiable and tx.normalized.raw.amount > 0
    ]
    rows.sort(key=lambda r: (r.customer_id, r.at))
    return rows


def robust_z(value: float, sample: list[float]) -> float:
    """MAD 기반 robust z-score.

    평균·표준편차를 쓰지 않는 이유는 **찾으려는 이상치가 그 통계를 오염시키기** 때문이다 —
    한 달치 급증이 평균을 끌어올려 자기 자신을 정상으로 만든다.

    MAD 가 0 이면(표본이 거의 같은 값이거나 표본이 2~3개뿐이면) 나눌 수 없다. 이때 0 을
    돌려주면 "정상"이라고 단정하는 셈이고, 무한대를 돌려주면 표본 3개짜리 카테고리가 전부
    알림이 된다. 그래서 **중앙값 대비 배수**로 물러서서 잰다 — 척도가 없을 때의 차선책임을
    호출부가 알 수 있도록 값의 성질(z 스케일)은 유지한다.
    """
    if len(sample) < 2:
        return 0.0
    median = statistics.median(sample)
    mad = statistics.median([abs(x - median) for x in sample])
    if mad > 0:
        return (value - median) / (mad * MAD_TO_SIGMA)
    if median <= 0:
        return 0.0
    return (value / median - 1.0) * 3.0


def coefficient_of_variation(sample: list[float]) -> float:
    """변동계수 = 표준편차 / 평균. 금액 단위가 달라도 같은 자로 비교하기 위해 쓴다."""
    if len(sample) < 2:
        return 0.0
    mean = statistics.fmean(sample)
    if mean <= 0:
        return 0.0
    return statistics.stdev(sample) / mean
