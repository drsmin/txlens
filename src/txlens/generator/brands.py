"""가맹점 카탈로그 — 합성 데이터의 씨앗.

각 브랜드는 **정답 라벨을 이미 알고 있다**(`l1`/`l2`). 오염된 표기를 만들어도 정답이 따라오므로
골드셋을 사실상 공짜로 얻는다(blueprint §4).

## `corporate` 필드에 대한 주의

법인명은 **정규화 난이도를 재현하기 위한 값**이다. 실제 브랜드 항목의 법인명은 널리 알려진
`스타벅스 ↔ 에스씨케이컴퍼니` 를 제외하면 `(주)브랜드…` 형태로 **합성**했다 — 실제 법인명을
단정하지 않기 위해서다. "브랜드와 법인명이 완전히 달라 사람도 모르는" 최고난도 케이스는
**가상 브랜드**(`is_fictional=True`)로 만들었다. 가상이므로 사실을 잘못 주장할 여지가 없다.

롱테일을 만들기 위해 상위 브랜드(빈도 높음)와 tail 브랜드(빈도 낮음)를 `frequency` 로 구분한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from txlens.schemas import CategoryL1, CategoryL2


@dataclass(frozen=True)
class Brand:
    brand: str                       # 정답 대표 가맹점명 (canonical)
    l1: CategoryL1
    l2: CategoryL2
    lo: int                          # 결제금액 하한(원)
    hi: int                          # 결제금액 상한(원)
    corporate: str | None = None     # 법인명 표기 (§ 위 주의)
    romanized: str | None = None
    has_branch: bool = True          # 지점명이 붙을 수 있는가
    kind: Literal["normal", "subscription"] = "normal"
    period_days: int | None = None   # 구독 주기 (kind == subscription)
    frequency: float = 1.0           # 일상 소비에서 뽑힐 상대 가중치 (tail 은 낮게)
    is_fictional: bool = False       # 가상 브랜드 여부
    late_night: bool = False         # 심야 결제가 자연스러운 업종
    #: 택소노미에 이 업종의 자리가 없다 → **정답 라벨은 미분류다** (D-026).
    #: `l1`/`l2` 를 미분류로 바꾸지 않는 이유: 그 값은 생성기의 **행동 모델**(시각 분포·주말 가중)
    #: 에도 쓰이므로, 바꾸면 데이터셋 전체가 재배치돼 라벨 정정의 효과를 귀속할 수 없게 된다.
    #: 즉 `l1`/`l2` = "어떤 업종처럼 행동하는가", `gold_l1`/`gold_l2` = "정답이 무엇인가".
    no_taxonomy_home: bool = False

    @property
    def is_subscription(self) -> bool:
        return self.kind == "subscription"

    @property
    def gold_l1(self) -> CategoryL1:
        """골드셋에 기록할 L1. 택소노미에 자리가 없으면 미분류가 정답이다."""
        return CategoryL1.UNCLASSIFIED if self.no_taxonomy_home else self.l1

    @property
    def gold_l2(self) -> CategoryL2:
        return CategoryL2.UNCLASSIFIED_INSUFFICIENT if self.no_taxonomy_home else self.l2


def _c(name: str) -> str:
    """기본 법인명 표기 — 실제 법인명을 단정하지 않는 합성 형태."""
    return f"(주){name}"


# ── 일상 소비 브랜드 (head: frequency 높음) ─────────────────────────────────
_HEAD: list[Brand] = [
    Brand("스타벅스", CategoryL1.CAFE, CategoryL2.CAFE_COFFEE, 4500, 9000,
          corporate="에스씨케이컴퍼니", romanized="STARBUCKS", frequency=9.0),
    Brand("GS25", CategoryL1.FOOD, CategoryL2.FOOD_CONVENIENCE, 1500, 18000,
          corporate=_c("지에스리테일"), romanized="GS25", frequency=10.0, late_night=True),
    Brand("CU", CategoryL1.FOOD, CategoryL2.FOOD_CONVENIENCE, 1500, 18000,
          corporate=_c("씨유"), romanized="CU", frequency=8.0, late_night=True),
    Brand("이마트", CategoryL1.FOOD, CategoryL2.FOOD_GROCERY, 18000, 140000,
          corporate=_c("이마트"), romanized="EMART", frequency=4.0),
    Brand("쿠팡", CategoryL1.SHOPPING, CategoryL2.SHOPPING_ONLINE, 8000, 180000,
          corporate=_c("쿠팡"), romanized="COUPANG", has_branch=False, frequency=7.0),
    Brand("배달의민족", CategoryL1.FOOD, CategoryL2.FOOD_DELIVERY, 12000, 45000,
          corporate=_c("우아한형제들"), romanized="BAEMIN", has_branch=False, frequency=7.0),
    Brand("투썸플레이스", CategoryL1.CAFE, CategoryL2.CAFE_BAKERY, 4000, 16000,
          corporate=_c("투썸플레이스"), romanized="TWOSOME", frequency=4.0),
    Brand("김밥천국", CategoryL1.FOOD, CategoryL2.FOOD_RESTAURANT, 6000, 22000,
          corporate=_c("김밥천국"), frequency=4.0),
    Brand("교촌치킨", CategoryL1.FOOD, CategoryL2.FOOD_RESTAURANT, 18000, 42000,
          corporate=_c("교촌에프앤비"), romanized="KYOCHON", frequency=3.0, late_night=True),
    Brand("서울교통공사", CategoryL1.TRANSPORT, CategoryL2.TRANSPORT_TRANSIT, 1400, 2800,
          corporate=_c("서울교통공사"), has_branch=False, frequency=12.0),
    Brand("카카오T", CategoryL1.TRANSPORT, CategoryL2.TRANSPORT_TAXI, 5500, 32000,
          corporate=_c("카카오모빌리티"), romanized="KAKAO T", has_branch=False,
          frequency=4.0, late_night=True),
    Brand("SK에너지", CategoryL1.TRANSPORT, CategoryL2.TRANSPORT_FUEL, 40000, 95000,
          corporate=_c("에스케이에너지"), romanized="SK ENERGY", frequency=2.0),
    Brand("올리브영", CategoryL1.HEALTH, CategoryL2.HEALTH_BEAUTY, 8000, 70000,
          corporate=_c("씨제이올리브영"), romanized="OLIVE YOUNG", frequency=3.0),
    Brand("다이소", CategoryL1.SHOPPING, CategoryL2.SHOPPING_LIVING, 2000, 35000,
          corporate=_c("아성다이소"), romanized="DAISO", frequency=3.0),
    Brand("무신사", CategoryL1.SHOPPING, CategoryL2.SHOPPING_FASHION, 25000, 220000,
          corporate=_c("무신사"), romanized="MUSINSA", has_branch=False, frequency=2.0),
    Brand("메가박스", CategoryL1.CULTURE, CategoryL2.CULTURE_HOBBY, 12000, 32000,
          corporate=_c("메가박스중앙"), romanized="MEGABOX", frequency=2.0),
    Brand("교보문고", CategoryL1.CULTURE, CategoryL2.CULTURE_BOOK, 12000, 60000,
          corporate=_c("교보문고"), romanized="KYOBO", frequency=1.5),
]

# ── tail 브랜드 (frequency 낮음 — 롱테일의 꼬리) ─────────────────────────────
_TAIL: list[Brand] = [
    Brand("동네약국", CategoryL1.HEALTH, CategoryL2.HEALTH_PHARMACY, 3000, 28000,
          corporate=_c("동네약국"), frequency=1.2),
    Brand("연세이비인후과", CategoryL1.HEALTH, CategoryL2.HEALTH_CLINIC, 5000, 75000,
          corporate=_c("연세이비인후과"), frequency=0.8),
    Brand("성수동카페", CategoryL1.CAFE, CategoryL2.CAFE_COFFEE, 5000, 14000,
          corporate=_c("성수동카페"), frequency=0.8),
    Brand("을지로호프", CategoryL1.CAFE, CategoryL2.CAFE_ALCOHOL, 15000, 60000,
          corporate=_c("을지로호프"), frequency=0.9, late_night=True),
    Brand("낙원떡집", CategoryL1.CAFE, CategoryL2.CAFE_BAKERY, 4000, 20000,
          corporate=_c("낙원떡집"), frequency=0.5),
    Brand("한빛주차장", CategoryL1.TRANSPORT, CategoryL2.TRANSPORT_PARKING, 2000, 18000,
          corporate=_c("한빛주차장"), frequency=1.0),
    # 세탁업은 택소노미에 자리가 없다 → **정답은 미분류다** (D-026). `l1`/`l2` 는 그대로 두고
    # `no_taxonomy_home` 만 켠다: 업종 성향(주말 가중·시간대)은 생활 서비스답게 유지해야 하고,
    # 라벨을 직접 바꾸면 `_day_weight`·`_pick_hour` 가 달라져 데이터셋 전체가 재배치된다.
    Brand("우리동네세탁", CategoryL1.SHOPPING, CategoryL2.SHOPPING_LIVING, 6000, 30000,
          corporate=_c("우리동네세탁"), frequency=0.7, no_taxonomy_home=True),
    Brand("영어의달인학원", CategoryL1.EDUCATION, CategoryL2.EDUCATION_TUITION, 150000, 380000,
          corporate=_c("영어의달인"), frequency=0.4),
]

# ── 가상 브랜드: 법인명이 브랜드와 완전히 다른 최고난도 케이스 ───────────────
# 가상이므로 실제 법인명을 잘못 주장할 위험이 없다. 사람도 표기만 보고는 모른다.
_FICTIONAL: list[Brand] = [
    Brand("빈스로드", CategoryL1.CAFE, CategoryL2.CAFE_COFFEE, 4500, 11000,
          corporate="제이디에프컴퍼니", romanized="BEANS ROAD",
          frequency=1.6, is_fictional=True),
    Brand("한끼밥상", CategoryL1.FOOD, CategoryL2.FOOD_RESTAURANT, 9000, 30000,
          corporate="에이치케이푸드시스템", romanized="HANKKI",
          frequency=1.4, is_fictional=True),
    Brand("데일리마트", CategoryL1.FOOD, CategoryL2.FOOD_GROCERY, 12000, 90000,
          corporate="디엠리테일홀딩스", romanized="DAILY MART",
          frequency=1.2, is_fictional=True),
    Brand("스텝짐", CategoryL1.HEALTH, CategoryL2.HEALTH_FITNESS, 60000, 130000,
          corporate="에스지웰니스", romanized="STEP GYM",
          frequency=0.6, is_fictional=True),
]

# ── 구독 브랜드 (주기 결제) ─────────────────────────────────────────────────
_SUBSCRIPTIONS: list[Brand] = [
    Brand("넷플릭스", CategoryL1.CULTURE, CategoryL2.CULTURE_CONTENT, 13500, 17000,
          corporate=_c("넷플릭스서비시스코리아"), romanized="NETFLIX", has_branch=False,
          kind="subscription", period_days=30),
    Brand("유튜브프리미엄", CategoryL1.CULTURE, CategoryL2.CULTURE_CONTENT, 14900, 14900,
          corporate=_c("구글코리아"), romanized="YOUTUBE PREMIUM", has_branch=False,
          kind="subscription", period_days=30),
    Brand("멜론", CategoryL1.CULTURE, CategoryL2.CULTURE_CONTENT, 10900, 10900,
          corporate=_c("멜론"), romanized="MELON", has_branch=False,
          kind="subscription", period_days=30),
    Brand("쿠팡와우", CategoryL1.SHOPPING, CategoryL2.SHOPPING_ONLINE, 7890, 7890,
          corporate=_c("쿠팡"), romanized="COUPANG WOW", has_branch=False,
          kind="subscription", period_days=30),
    Brand("SKT통신요금", CategoryL1.HOUSING, CategoryL2.HOUSING_TELECOM, 55000, 89000,
          corporate=_c("에스케이텔레콤"), romanized="SKT", has_branch=False,
          kind="subscription", period_days=30),
    Brand("스텝짐멤버십", CategoryL1.HEALTH, CategoryL2.HEALTH_FITNESS, 79000, 79000,
          corporate="에스지웰니스", romanized="STEP GYM MEMBERSHIP", has_branch=False,
          kind="subscription", period_days=30, is_fictional=True),
    Brand("클라우드백업", CategoryL1.CULTURE, CategoryL2.CULTURE_CONTENT, 3300, 3300,
          corporate="브이엘소프트", romanized="CLOUD BACKUP", has_branch=False,
          kind="subscription", period_days=365, is_fictional=True),
]

#: 일상 소비에서 뽑는 풀
SPENDING_BRANDS: list[Brand] = [*_HEAD, *_TAIL, *_FICTIONAL]

#: 구독으로 심는 풀
SUBSCRIPTION_BRANDS: list[Brand] = list(_SUBSCRIPTIONS)

#: 전체 (라벨 사전 구축용)
ALL_BRANDS: list[Brand] = [*SPENDING_BRANDS, *SUBSCRIPTION_BRANDS]

#: 지점명 후보 — 지점 접미사 오염에 쓴다.
#: `２호점` 은 전각 숫자다(의도적) — 전처리의 NFKC 정규화가 실제로 동작하는지 시험한다.
BRANCH_NAMES: list[str] = [
    "강남역점", "홍대입구점", "여의도점", "판교점", "성수점", "잠실점", "종로3가점",
    "서울대입구점", "광화문점", "1234호점", "２호점", "역삼중앙점",
]

#: 간편결제·배달앱 채널 프리픽스 (원문에 붙는 형태)
CHANNEL_PREFIXES: list[tuple[str, str]] = [
    ("NAVERPAY*", "naverpay"),
    ("네이버페이*", "naverpay"),
    ("카카오페이_", "kakaopay"),
    ("KAKAOPAY*", "kakaopay"),
    ("TOSSPAY*", "tosspay"),
    ("PAYCO*", "payco"),
    ("SSGPAY*", "ssgpay"),
]

#: PG사 표기 — 실가맹점이 소실된다. 추측하지 않고 미분류로 간다.
PG_NAMES: list[str] = [
    "(주)케이지이니시스", "케이지모빌리언스", "NHN한국사이버결제", "토스페이먼츠",
    "(주)다날", "KCP", "나이스페이먼츠",
]


def brand_by_name(name: str) -> Brand:
    for b in ALL_BRANDS:
        if b.brand == name:
            return b
    raise KeyError(name)
