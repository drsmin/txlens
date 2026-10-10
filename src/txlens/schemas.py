"""파이프라인 각 단계의 계약(contract).

거래는 ingest → preprocess → rule_match → (llm_classify) → (human_review) → finalize 를 지나며
RawTx → NormalizedTx → ClassifiedTx 로 승격된다. 각 단계가 무엇을 보장하는지를 여기에 못 박아,
노드 구현이 서로의 내부를 추측하지 않게 한다.

카테고리 Enum 은 config/taxonomy.yaml 과 **동기화되어야 한다.** 런타임 생성 대신 여기에 명시하는
이유는 정적 타입 검사와 IDE 완성을 얻기 위해서고, 어긋남은 tests/test_taxonomy_sync.py 가 잡는다.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── 카테고리 ────────────────────────────────────────────────────────────────
class CategoryL1(str, Enum):
    FOOD = "food"
    CAFE = "cafe"
    TRANSPORT = "transport"
    HOUSING = "housing"
    HEALTH = "health"
    SHOPPING = "shopping"
    CULTURE = "culture"
    EDUCATION = "education"
    TAX_INSURANCE = "tax_insurance"
    FINANCE = "finance"
    #: 실패가 아니라 정당한 답. 정보가 없으면 억지로 추측하지 않는다.
    UNCLASSIFIED = "unclassified"


class CategoryL2(str, Enum):
    FOOD_RESTAURANT = "food.restaurant"
    FOOD_DELIVERY = "food.delivery"
    FOOD_GROCERY = "food.grocery"
    FOOD_CONVENIENCE = "food.convenience"

    CAFE_COFFEE = "cafe.coffee"
    CAFE_BAKERY = "cafe.bakery"
    CAFE_ALCOHOL = "cafe.alcohol"

    TRANSPORT_TRANSIT = "transport.transit"
    TRANSPORT_TAXI = "transport.taxi"
    TRANSPORT_FUEL = "transport.fuel"
    TRANSPORT_PARKING = "transport.parking"
    TRANSPORT_VEHICLE = "transport.vehicle"

    HOUSING_RENT = "housing.rent"
    HOUSING_UTILITY = "housing.utility"
    HOUSING_TELECOM = "housing.telecom"

    HEALTH_CLINIC = "health.clinic"
    HEALTH_PHARMACY = "health.pharmacy"
    HEALTH_FITNESS = "health.fitness"
    HEALTH_BEAUTY = "health.beauty"

    SHOPPING_FASHION = "shopping.fashion"
    SHOPPING_DIGITAL = "shopping.digital"
    SHOPPING_LIVING = "shopping.living"
    SHOPPING_ONLINE = "shopping.online"

    CULTURE_CONTENT = "culture.content"
    CULTURE_TRAVEL = "culture.travel"
    CULTURE_HOBBY = "culture.hobby"
    CULTURE_BOOK = "culture.book"

    EDUCATION_TUITION = "education.tuition"
    EDUCATION_MATERIAL = "education.material"
    EDUCATION_CERTIFICATION = "education.certification"

    TAX_INSURANCE_INSURANCE = "tax_insurance.insurance"
    TAX_INSURANCE_TAX = "tax_insurance.tax"
    TAX_INSURANCE_PENSION = "tax_insurance.pension"

    FINANCE_TRANSFER = "finance.transfer"
    FINANCE_FEE = "finance.fee"
    FINANCE_INTEREST = "finance.interest"
    FINANCE_LOAN = "finance.loan"
    FINANCE_CASH = "finance.cash"

    UNCLASSIFIED_PG_MASKED = "unclassified.pg_masked"
    UNCLASSIFIED_INSUFFICIENT = "unclassified.insufficient"

    @property
    def l1(self) -> CategoryL1:
        """L2 는 `<l1>.<leaf>` 규약이므로 상위를 계산할 수 있다."""
        return CategoryL1(self.value.split(".", 1)[0])


class _CategoryConsistency(BaseModel):
    """`category_l2` 가 `category_l1` 에 속하는지 보장하는 믹스인.

    LLM 이 `l1=food, l2=cafe.coffee` 같은 조합을 내는 일이 실제로 있다. 스키마에서 막지 않으면
    지표 집계가 조용히 틀어진다.
    """

    category_l1: CategoryL1
    category_l2: CategoryL2

    @model_validator(mode="after")
    def _l2_belongs_to_l1(self):
        if self.category_l2.l1 is not self.category_l1:
            raise ValueError(
                f"category_l2={self.category_l2.value} 는 "
                f"category_l1={self.category_l1.value} 에 속하지 않는다"
            )
        return self


# ── 열거형 ──────────────────────────────────────────────────────────────────
class TxType(str, Enum):
    """`status`(결제상태) 4종에 대응한다 — 카드-008 `card_approval_status` (D-023).

    규격 코드: 01 승인 · 02 승인취소 · 03 정정 · 04 무승인매입. 앞의 둘만 다루면 실데이터에서
    나머지 둘을 만나 죽는다. `CORRECTION` 은 원거래를 대체하고(`modified_amt` 가 실지출),
    `UNAPPROVED` 는 승인 없이 매입된 실지출이라 **분류·탐지 대상이다**.
    """

    APPROVE = "approve"
    CANCEL = "cancel"
    CORRECTION = "correction"
    UNAPPROVED = "unapproved"


class CardType(str, Enum):
    """`pay_type` = **신용/체크** 구분이다 (카드-008 `card_pay_type` — D-023).

    ⚠️ 할부 여부가 아니다. 할부는 `total_install_cnt` 의 **회신 여부**로만 판단한다.
    """

    CREDIT = "credit"
    CHECK = "check"


class PaymentChannel(str, Enum):
    """가맹점명 앞에 붙는 결제 채널. 버리지 않고 분리해 보존한다 (blueprint §3.3 ①)."""

    NAVERPAY = "naverpay"
    KAKAOPAY = "kakaopay"
    TOSSPAY = "tosspay"
    PAYCO = "payco"
    SSGPAY = "ssgpay"
    #: 배달앱은 채널이면서 카테고리 힌트(food.delivery)다. 별도 취급한다.
    BAEMIN = "baemin"
    COUPANG_EATS = "coupang_eats"
    YOGIYO = "yogiyo"
    #: PG사 경유 — 실가맹점명이 소실된 경우. 추측하지 않는다.
    PG = "pg"


class MatchStage(str, Enum):
    """어느 단계가 이 거래를 분류했는가. 단계별 기여도를 지표로 보기 위해 기록한다."""

    #: 단계 0 — PG 경유로 실가맹점이 소실됐다. **미분류가 정답인 케이스**라 `UNMATCHED` 와
    #: 반드시 구분한다. 섞으면 미분류율이 "룰이 못 잡은 것"과 "원리상 알 수 없는 것"을 합산해
    #: 개선 여지가 있는 것처럼 보인다 (실제로는 사전을 아무리 키워도 줄지 않는다).
    PG_MASKED = "pg_masked"
    EXACT = "exact"
    #: 사업자등록번호 일치. 신뢰도가 제공기관에 달려 있어 보조 신호다 (D-012).
    REGNO = "regno"
    ALIAS = "alias"
    REGEX = "regex"
    FUZZY = "fuzzy"
    LLM = "llm"
    HUMAN = "human"
    UNMATCHED = "unmatched"


class Evidence(str, Enum):
    """LLM 이 무엇에 근거해 분류했는가. `GUESS` 는 상위 모델로 escalate 하는 트리거다."""

    BRAND_KNOWN = "brand_known"
    KEYWORD = "keyword"
    PROVIDER_CODE = "provider_code"
    GUESS = "guess"


# ── 단계별 모델 ─────────────────────────────────────────────────────────────
class RawTx(BaseModel):
    """정보제공자에게서 받은 원본 거래. 표준API 거래내역 응답 형태를 모사한다.

    **합성 데이터 전용이다** (D-005). 실제 고객 데이터는 어떤 경로로도 여기 들어오지 않는다.
    """

    model_config = ConfigDict(frozen=True)

    provider_id: str = Field(
        description="정보제공자 식별자 (카드사 등). 표기 관행이 기관마다 크게 다르므로 "
        "지표를 이 축으로 쪼개 본다 — D-012"
    )
    customer_id: str = Field(description="합성 고객 식별자 — 실제 식별정보가 아니다")
    approval_no: str = Field(description="승인번호. 취소 거래 페어링 키")
    #: **tz 없는 KST 로컬 시각.** 표준API 는 거래일시를 오프셋 없이 주고, 이 프로젝트의 데이터는
    #: 전부 한국 단일 시간대다(한국은 DST 도 없다). tz-aware 로 올리면 구독 주기 계산 등 모든
    #: 날짜 연산에 잡음만 늘고 얻는 게 없어 naive 로 통일한다 — D-009.
    tx_datetime: datetime
    merchant_name_raw: str = Field(description="정규화 전 가맹점명. 자유 텍스트")
    amount: int = Field(description="거래금액(원). 취소는 음수")
    currency: str = "KRW"
    tx_type: TxType = TxType.APPROVE
    #: 전체 할부회차. 규격은 **일시불이면 이 필드를 미회신**하므로 0 = 일시불이다 (D-023).
    installment_months: int = Field(default=0, ge=0, description="0 = 일시불")
    #: 신용/체크. 규격 필수 필드라 보존하지만 분류·탐지에는 쓰지 않는다 (D-023).
    card_type: CardType | None = None
    #: 취소·정정 행의 **원거래 승인일시**(`approved_dtime`).
    #: 규격은 취소·정정 행에도 원 승인일시를 주고 사건 시각은 `trans_dtime` 으로 준다.
    #: 그래서 `tx_datetime` = 사건 시각(취소가 일어난 때)이고, 원 승인 시각은 여기 남긴다.
    original_datetime: datetime | None = None
    #: 가맹점 사업자등록번호. **정규화의 주 키로 쓰지 않는다** (D-012).
    #: 채워지는 정도가 제공기관마다 크게 달라 신뢰할 수 없고, PG 경유 시엔 PG사 번호가 오며,
    #: 같은 브랜드가 점포별로 다른 번호를 갖는다 — 즉 "같은 점포"는 잡아도 "같은 브랜드"는 못 잡는다.
    #: 브랜드 단위 집계(카테고리·구독)에는 여전히 이름 정규화가 필요하다.
    #: 있고 사전에 등록된 경우에만 보조 신호로 쓴다(MatchStage.REGNO).
    #: **규격 형태(`aN(12)` — `123-45-67890`, 하이픈 포함)를 그대로 보존한다.** 매칭 키로 쓸 때는
    #: `preprocess.regno_key()` 로 숫자만 남긴다 — 기관마다 하이픈 유무가 다르다 (D-023).
    merchant_regno: str | None = None


class NormalizedTx(BaseModel):
    """결정론적 전처리 결과. LLM 이전에 여기서 정확도의 절반이 결정된다 (blueprint §3.3 ①)."""

    raw: RawTx
    normalized_name: str = Field(description="NFKC·법인표기·지점·채널 제거 후의 매칭 키")
    branch: str | None = Field(default=None, description="분리해 보존한 지점명")
    payment_channel: PaymentChannel | None = None
    pg_masked: bool = Field(
        default=False, description="PG사 경유로 실가맹점 불명. 추측 대신 미분류로 간다"
    )
    is_cancelled: bool = Field(default=False, description="취소로 상계된 원거래")
    cancel_of: str | None = Field(default=None, description="이 거래가 취소하는 원거래 승인번호")
    excluded_reason: str | None = Field(
        default=None, description="분류 대상에서 제외된 이유 (취소 상계·할부 회차 등)"
    )

    @property
    def is_classifiable(self) -> bool:
        return self.excluded_reason is None and not self.is_cancelled


class CancelPairing(BaseModel):
    """`pair_cancellations` 노드가 `normalize` 노드에게 넘기는 표식 한 줄.

    순수 함수 `preprocess.pair_cancellations()` 는 `id(tx)` 를 키로 돌려준다 — 같은 값의 거래가
    여럿 있어 값으로는 행을 지목할 수 없기 때문이다. 그런데 **상태에 담기는 값은 체크포인터가
    직렬화한다.** `id()` 는 한 프로세스 안에서만 유효한 수라 저장했다가 되살리면 아무 행도
    가리키지 않는다. 그래서 노드 경계를 넘을 때는 `raw` 리스트의 **인덱스**로 바꿔 싣는다 —
    그 순서는 `ingest` 가 고정한다.

    노드를 쪼개면 그 사이를 지나는 값이 계약이 된다. 계약이 되는 순간 "프로세스 안에서만 맞는
    키"는 못 쓴다 — 쪼개기 전에는 드러나지 않던 제약이다 (D-031).
    """

    index: int = Field(ge=0, description="`raw` 리스트에서 이 표식이 가리키는 행의 위치")
    cancel_of: str | None = Field(default=None, description="이 거래가 취소하는 원거래 승인번호")
    excluded_reason: str | None = Field(default=None, description="분류 대상에서 빠지는 이유")


class LLMTarget(BaseModel):
    """LLM 에 물어볼 대상 하나. **거래가 아니라 고유 표기가 단위다.**

    표기 단위인 이유는 비용이 아니라 설계다 (blueprint §1.1) — 한 사람의 거래 목록을 보내면 소비
    패턴이 외부로 나간다. 표기 하나만 보내면 "이 문자열이 어느 브랜드인가"라는 질문이 되고, 누가
    언제 얼마를 썼는지는 경계 안에 남는다.
    """

    normalized_name: str
    #: 이 키로 정규화된 **원본 표기 중 대표 1개.** 정규화 키는 채널·지점·법인표기가 이미 떨어져
    #: 있어서 `get_brand_hint` 에게 줄 것이 남지 않는다. 원본을 하나 들고 가면 Agent 가 전처리가
    #: 무엇을 떼어냈는지 조회할 수 있다. 거래 목록이 아니라 **표기 1개**라 전송 단위는 그대로다.
    sample_raw_name: str = ""
    #: 이 표기가 덮는 거래 건수. **프롬프트에 넣지 않는다** — 검토 큐 정렬에만 쓴다
    #: (`ReviewItem.priority`). 모델에게는 "많이 쓰인 가맹점"이라는 편향만 준다.
    tx_count: int = Field(default=0, ge=0)
    payment_channel: PaymentChannel | None = None
    branch: str | None = None


class LLMVerdict(_CategoryConsistency):
    """LLM 분류 결과. structured output 으로 강제되는 형태 (blueprint §3.3 ③).

    거래 단위가 아니라 **고유 가맹점명(distinct) 단위**로 만들어진다 — 개인의 소비 패턴을
    외부로 보내지 않기 위한 설계이며(blueprint §1.1), 캐시 키도 이 단위다.
    """

    merchant_canonical: str = Field(description="대표 가맹점명")
    brand: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="사람 검토용 근거. HITL 화면에 그대로 노출된다")
    evidence: Evidence


class ClassifiedTx(_CategoryConsistency):
    """분류가 확정된 거래."""

    normalized: NormalizedTx
    merchant_canonical: str
    confidence: float = Field(ge=0.0, le=1.0)
    match_stage: MatchStage
    source_verdict: LLMVerdict | None = Field(
        default=None, description="LLM 을 거친 경우의 원본 판정 (감사 추적)"
    )


class RuleProposal(_CategoryConsistency):
    """룰 사전 승격 후보. 이 프로젝트의 심장 (blueprint §3.3 ⑤).

    사람이 교정한 결과와 고신뢰 LLM 결과가 여기로 모여 사전에 승격되고, 다음 라운드부터는
    같은 문자열이 exact/alias 단계에서 즉시 잡힌다 — LLM 호출이 라운드마다 줄어드는 이유다.
    """

    pattern_type: Literal["exact", "alias", "regex"]
    pattern: str = Field(description="매칭 패턴. exact/alias 는 정규화된 문자열, regex 는 정규식")
    merchant_canonical: str
    origin: Literal["human", "llm_high_confidence"]
    source_name: str = Field(description="이 제안을 만든 정규화 가맹점명")
    covers_tx_count: int = Field(ge=0, description="승격 시 즉시 영향받는 거래 건수")


class ReviewItem(BaseModel):
    """HITL 검토 큐의 한 줄.

    `priority` 로 정렬해 **사람의 1분이 가장 많은 거래를 고치게** 한다 (blueprint §3.3 ④).
    """

    normalized_name: str
    verdict: LLMVerdict
    tx_count: int = Field(ge=0, description="이 문자열이 덮는 거래 건수")

    @property
    def priority(self) -> float:
        return self.tx_count * (1.0 - self.verdict.confidence)


class HumanCorrection(_CategoryConsistency):
    """사람이 검토 화면에서 확정한 한 줄 — **자동 승격이 못 하는 일을 하는 자리다.**

    자동 승격은 대표명을 사전에서 찾을 수 있을 때만 별칭을 만든다(D-033). 그래서 사전에 없는
    브랜드는 몇 라운드를 돌려도 영원히 미매칭으로 남는다 — 실측에서 자동이 덮은 것이 40건이고
    남은 것이 276건이다. 그 276건을 여는 열쇠가 이 객체이고, 그래서 이것만이 `origin="human"`
    으로 **새 브랜드를 만들 수 있다** (`rules.promote.propose_from_corrections`).

    `verdict` 를 지우지 않고 들고 있는 이유: 사람이 무엇을 보고 무엇으로 바꿨는지가 남아야
    "사람이 고친 것이 실제로 나은가"를 나중에 셀 수 있다. 판정이 없을 수도 있다 — 검토 큐에는
    LLM 이 답을 못 낸 표기도 들어온다.
    """

    normalized_name: str = Field(description="교정 대상 정규화 표기 = 검토 큐의 키")
    merchant_canonical: str = Field(description="사람이 확정한 대표명")
    tx_count: int = Field(default=0, ge=0, description="이 교정이 즉시 덮는 거래 건수")
    #: 사람이 본 원래 판정. 감사 추적이자 "무엇을 고쳤나"의 대조군이다.
    reviewed_verdict: LLMVerdict | None = None
    reviewer: str = Field(default="human", description="검토자 식별자. 감사 추적용")
    note: str = Field(default="", description="검토자가 남긴 메모. 사전 승격 주석에 실린다")
