"""정답셋 (ground truth).

1부(분류)와 2부(탐지)의 정답을 **생성 시점에 기록한다.** 사후에 라벨링하는 게 아니라 심으면서
적으므로 정답이 정의상 옳다 — 2부의 정답셋을 공짜로 얻는 이유(blueprint §3.6).

> **심지 않은 패턴은 평가할 수 없다.** 그래서 무엇을 심었는지가 곧 평가 범위다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from txlens.schemas import CategoryL1, CategoryL2


class TxLabel(BaseModel):
    """거래 1건의 정답. 키는 승인번호."""

    canonical: str = Field(description="정답 대표 가맹점명. PG 마스킹이면 원 브랜드를 적지 않는다")
    category_l1: CategoryL1
    category_l2: CategoryL2
    provider_id: str
    pg_masked: bool = False
    #: 오염 형태 — 어느 표기가 어려웠는지 사후 분석용
    style: str = "brand_plain"


class SubscriptionTruth(BaseModel):
    customer_id: str
    canonical: str
    period_days: int
    amount: int
    occurrences: int
    approval_nos: list[str]
    price_increased: bool = False
    unused_months: int = Field(default=0, description="마지막 결제 후 미사용 의심 개월")
    #: 탐지 규칙("최소 3회")으로 원리상 찾을 수 있는가. 연 구독은 6개월 창에서 1회만 발생하므로
    #: False 가 된다. **심었지만 탐지 불가한 것을 recall 분모에 넣으면 점수가 구조적으로 깎인다** —
    #: 평가는 detectable 만으로 recall 을 재고, 나머지는 "원리적 한계"로 따로 보고한다.
    detectable: bool = True


class SpendingAnomalyTruth(BaseModel):
    customer_id: str
    kind: str = Field(description="category_surge | single_large | new_category")
    category_l1: CategoryL1
    month: str = Field(description="YYYY-MM")
    approval_nos: list[str]


class TxAnomalyTruth(BaseModel):
    customer_id: str
    kind: str = Field(description="duplicate_charge | micro_burst")
    canonical: str
    approval_nos: list[str]


class DecoyTruth(BaseModel):
    """오탐 함정 — **정답이 '이상 아님'** 인 것들.

    recall 만 재면 "전부 이상"이라고 답하는 탐지기가 만점을 받는다. precision 을 재려면
    "이상처럼 보이지만 정상인 것"이 데이터에 있어야 한다(blueprint §4).
    """

    customer_id: str
    kind: str = Field(description="subscription_like | normal_consecutive")
    canonical: str
    approval_nos: list[str]
    note: str = ""


class GroundTruth(BaseModel):
    seed: int
    customers: int
    period_start: str
    period_end: str
    tx_total: int
    distinct_raw_names: int

    tx_labels: dict[str, TxLabel] = Field(default_factory=dict)
    subscriptions: list[SubscriptionTruth] = Field(default_factory=list)
    spending_anomalies: list[SpendingAnomalyTruth] = Field(default_factory=list)
    tx_anomalies: list[TxAnomalyTruth] = Field(default_factory=list)
    decoys: list[DecoyTruth] = Field(default_factory=list)

    def summary(self) -> str:
        return (
            f"거래 {self.tx_total}건 · 원본 표기 {self.distinct_raw_names}개 · "
            f"구독 {len(self.subscriptions)} · 이상지출 {len(self.spending_anomalies)} · "
            f"이상거래 {len(self.tx_anomalies)} · 오탐함정 {len(self.decoys)}"
        )
