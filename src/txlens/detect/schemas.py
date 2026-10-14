"""2부 탐지 결과의 계약 (설계 §3.6).

## 알림에는 반드시 근거 문장이 붙는다

`evidence` 는 장식이 아니다. 이 프로젝트의 출력은 차단이 아니라 **"확인해 보세요"** 이고, 판단은
고객이 한다(설계 §3.6). 근거 없이 "이상합니다"만 띄우면 고객이 확인할 방법이 없어 알림이
소음이 된다. 그래서 탐지기는 판정과 함께 **왜 그렇게 봤는지**를 같은 객체에 담아 돌려준다.

## 승인번호를 항상 들고 다닌다

평가가 대표명이나 카테고리로 정답을 맞추면, 분류 품질이 달라질 때 **탐지가 맞았는지 이름이
맞았는지**를 구분할 수 없다. 승인번호는 분류와 무관하게 고정된 값이라 이 둘을 분리해 준다 —
§5.2 전이 실험이 성립하는 이유다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from txlens.schemas import CategoryL1

#: 구독 주기 판정 — 중앙값 간격이 이 범위 안에 들어와야 한다 (설계 §3.6 ①).
PeriodKind = Literal["weekly", "monthly", "yearly"]


class Subscription(BaseModel):
    """정기 결제로 판정된 (고객 × 가맹점) 하나."""

    customer_id: str
    merchant_canonical: str
    category_l1: CategoryL1
    period_kind: PeriodKind
    #: 실제 관측된 결제 간격의 중앙값(일). 규정 주기(30/7/365)가 아니라 **관측값**이다.
    period_days: int
    occurrences: int
    median_amount: int
    #: 금액 변동계수. 구독 판정의 핵심 조건이고, 오탐 함정(매달 결제되지만 금액이 들쭉날쭉한
    #: 가맹점)이 걸러지는 자리다.
    amount_cv: float
    first_at: datetime
    last_at: datetime
    #: 마지막 결제 + 관측 주기. 고객 화면의 "다음 결제일"이다.
    next_expected: datetime
    #: 마지막 결제 후 주기가 여러 번 지났다 — "안 쓰는 구독" 후보.
    unused: bool = False
    #: 금액이 계단식으로 올랐다.
    price_increased: bool = False
    #: 월 환산 고정지출(원). 주기가 달라도 같은 자로 더할 수 있게 월로 환산한다.
    monthly_cost: int = 0
    approval_nos: list[str] = Field(default_factory=list)
    evidence: str = ""


class SpendingAlert(BaseModel):
    """카테고리 기반 이상지출 알림 하나.

    **분류에 완전히 의존한다** — 카테고리가 틀리면 이 알림은 엉뚱한 고객에게 엉뚱한 카테고리로
    간다. §5.2 의 주 소재다.
    """

    customer_id: str
    kind: Literal["category_surge", "single_large"]
    category_l1: CategoryL1
    month: str = Field(description="YYYY-MM")
    amount: int
    #: MAD 기반 robust z-score. 평균·표준편차를 쓰지 않는 이유는 이상치 자체가 통계를
    #: 오염시키기 때문이다 (설계 §3.6 ②).
    robust_z: float = 0.0
    approval_nos: list[str] = Field(default_factory=list)
    evidence: str = ""


class TxAlert(BaseModel):
    """거래 형태 기반 알림 하나. 카테고리를 보지 않는다 — 금액·시각·가맹점만으로 선다."""

    customer_id: str
    kind: Literal["duplicate_charge", "micro_burst"]
    merchant_canonical: str
    amount: int = Field(description="중복결제는 1건 금액, 소액다발은 합계")
    at: datetime
    approval_nos: list[str] = Field(default_factory=list)
    evidence: str = ""


class Detections(BaseModel):
    """탐지 3종의 결과 묶음."""

    subscriptions: list[Subscription] = Field(default_factory=list)
    spending_alerts: list[SpendingAlert] = Field(default_factory=list)
    tx_alerts: list[TxAlert] = Field(default_factory=list)

    @property
    def total_alerts(self) -> int:
        """고객에게 실제로 뜨는 알림 수. 구독 목록은 알림이 아니라 조회 화면이라 세지 않는다."""
        return len(self.spending_alerts) + len(self.tx_alerts)

    def summary(self) -> str:
        unused = sum(1 for s in self.subscriptions if s.unused)
        raised = sum(1 for s in self.subscriptions if s.price_increased)
        return (
            f"구독 {len(self.subscriptions)}건 (미사용 의심 {unused} · 가격인상 {raised}) · "
            f"이상지출 {len(self.spending_alerts)}건 · 이상거래 {len(self.tx_alerts)}건"
        )
