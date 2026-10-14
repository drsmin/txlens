"""가맹점명 정규화 — 결정론 규칙만. LLM 없음 (설계 §3.3 ①).

**여기서 정확도의 절반이 결정된다.** 규칙은 전부 순수 함수로 두고 규칙별로 테스트를 잠근다.

## 목록을 생성기에서 import 하지 않는 이유

PG사명·간편결제 프리픽스 목록이 `generator/brands.py` 에도 있다. 그걸 import 하면 파이프라인이
정답 생성기에 의존하게 되고, 실제 연동에서는 생성기가 없다. 그래서 **독립적으로 다시 적고**,
`tests/test_preprocess.py` 가 "생성기가 만드는 모든 프리픽스·PG명을 전처리가 처리하는가"를
검사해 드리프트를 잡는다 (`test_taxonomy_sync.py` 와 같은 방식).

이 목록들은 **닫힌 공개 목록**이라 사전에 적어두는 것이 누출이 아니다 — 누출 우려는
롱테일인 *브랜드* 사전에만 적용된다. 반대로 이걸 비워두면 전처리가 실무보다 무능해진다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from txlens.schemas import PaymentChannel

# ── 결제채널 프리픽스 ───────────────────────────────────────────────────────
#: 표기 → 채널. **긴 것부터 검사한다** (`네이버페이*` 와 `NAVERPAY*` 는 겹치지 않지만,
#: 접두 관계가 생기면 짧은 쪽이 먼저 걸려 잘못 자른다).
#: 표기는 NFKC·대문자 정규화 **후**의 형태로 적는다.
CHANNEL_PREFIXES: tuple[tuple[str, PaymentChannel], ...] = (
    ("NAVERPAY*", PaymentChannel.NAVERPAY),
    ("네이버페이*", PaymentChannel.NAVERPAY),
    ("KAKAOPAY*", PaymentChannel.KAKAOPAY),
    ("카카오페이_", PaymentChannel.KAKAOPAY),
    ("카카오페이*", PaymentChannel.KAKAOPAY),
    ("TOSSPAY*", PaymentChannel.TOSSPAY),
    ("PAYCO*", PaymentChannel.PAYCO),
    ("SSGPAY*", PaymentChannel.SSGPAY),
)

#: 배달앱은 **채널이면서 카테고리 힌트**(food.delivery)다. 별도로 둔다 (설계 §3.3 ①).
#: 구분자가 없는 형태로 붙어 오므로, 뒤에 실가맹점명이 남을 때만 자른다 — 아래 주의 참고.
DELIVERY_PREFIXES: tuple[tuple[str, PaymentChannel], ...] = (
    ("배달의민족", PaymentChannel.BAEMIN),
    ("쿠팡이츠", PaymentChannel.COUPANG_EATS),
    ("요기요", PaymentChannel.YOGIYO),
)

#: PG사 경유 — 실가맹점명이 소실된 경우. **추측하지 않는다** (설계 §3.3 ①).
#: 법인 표기를 제거한 형태로 적는다 (`(주)케이지이니시스` → `케이지이니시스`).
PG_NAMES: frozenset[str] = frozenset({
    "케이지이니시스",
    "케이지모빌리언스",
    "NHN한국사이버결제",
    "한국사이버결제",
    "토스페이먼츠",
    "다날",
    "KCP",
    "나이스페이먼츠",
    "이니시스",
    "올더게이트",
})

# ── 법인 표기 ───────────────────────────────────────────────────────────────
#: 앞뒤에 붙는 법인 표기. `㈜` 는 NFKC 가 `(주)` 로 바꿔주므로 여기서 다시 적지 않는다.
_CORP_PREFIX = re.compile(r"^\s*(?:\(주\)|\(유\)|\(재\)|\(사\)|주식회사|유한회사)\s*")
_CORP_SUFFIX = re.compile(
    r"\s*(?:\(주\)|\(유\)|주식회사|유한회사|CO\.?,?\s?LTD\.?|CO\.?\s?LTD\.?|LTD\.?|INC\.?|LLC\.?)\s*$",
    re.IGNORECASE,
)

# ── 지점 접미사 ─────────────────────────────────────────────────────────────
#: `1234호점` — 공백 없이 붙어 와도 자른다. 숫자+호점 조합은 브랜드명일 수 없다.
_BRANCH_NUMBERED = re.compile(r"\s*(\d+\s*호점)$")
#: 공백으로 분리된 마지막 토큰이 `…점`/`…지점`/`…대리점` 인 경우.
#: **공백을 요구하는 것이 핵심이다.** `\S+점$` 로 넓히면 `면세점`·`본점` 이 붙은 브랜드나
#: 이름 자체가 `점` 으로 끝나는 가맹점을 잘라먹는다 — 되돌릴 수 없는 손실이다.
_BRANCH_TOKEN = re.compile(r"\s+(\S*(?:대리점|지점|점))$")

_PUNCT_EDGE = re.compile(r"^[\s*_\-.,·・:;/\\|()\[\]]+|[\s*_\-.,·・:;/\\|()\[\]]+$")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class NameParts:
    """정규화 결과. 버린 것이 아니라 **분리해 보존한 것**들이 함께 온다."""

    normalized_name: str
    branch: str | None = None
    payment_channel: PaymentChannel | None = None
    pg_masked: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.normalized_name


def canonical_form(name: str) -> str:
    """NFKC · 공백 정리 · 대문자 통일. 모든 규칙의 공통 전처리.

    NFKC 가 하는 일: 전각 영숫자(`ＧＳ２５`)→반각, `㈜`→`(주)`, 전각 공백→공백.
    이걸 먼저 하지 않으면 아래 정규식이 전각 표기를 전부 놓친다.
    """
    text = unicodedata.normalize("NFKC", name)
    text = _WS.sub(" ", text).strip()
    return text.upper()


def strip_channel(name: str) -> tuple[str, PaymentChannel | None]:
    """간편결제·배달앱 프리픽스를 분리한다. **버리지 않고 채널로 보존한다.**

    ⚠️ 배달앱은 그 자체가 가맹점명으로도 온다 (`배달의민족` 은 브랜드 카탈로그의 항목이다).
    프리픽스로 잘라 남는 것이 없으면 **자르지 않는다** — 자르면 정상 브랜드가 빈 문자열이 되어
    미분류로 떨어진다. 이게 "채널이면서 브랜드"인 배달앱의 함정이다.
    """
    for prefix, channel in sorted(CHANNEL_PREFIXES, key=lambda p: -len(p[0])):
        if name.startswith(prefix):
            rest = name[len(prefix):].strip()
            # 간편결제 프리픽스는 구분자(`*`/`_`)를 포함하므로 남는 게 없으면 원장이 깨진 것이다.
            # 채널 정보는 살리고 이름은 원문을 둔다.
            return (rest or name), channel

    for prefix, channel in sorted(DELIVERY_PREFIXES, key=lambda p: -len(p[0])):
        if name.startswith(prefix):
            rest = name[len(prefix):].strip()
            rest = _PUNCT_EDGE.sub("", rest)
            if rest:
                return rest, channel
            return name, channel  # 배달앱 자체가 가맹점 — 이름을 지키고 채널만 표시
    return name, None


def strip_corporate(name: str) -> str:
    """법인 표기 제거. `(주)스타벅스커피코리아` → `스타벅스커피코리아`.

    **법인명 자체를 브랜드로 바꾸지는 않는다** — 그건 사전(alias)의 일이다.
    여기서는 표기만 벗긴다.
    """
    prev = None
    text = name
    # `(주)…CO.,LTD` 처럼 앞뒤로 동시에 붙거나 중복된 경우가 있어 고정점까지 반복한다.
    while prev != text:
        prev = text
        text = _CORP_PREFIX.sub("", text)
        text = _CORP_SUFFIX.sub("", text)
    return text.strip()


def split_branch(name: str) -> tuple[str, str | None]:
    """지점 접미사를 분리해 보존한다. `이마트 성수점` → (`이마트`, `성수점`)."""
    m = _BRANCH_NUMBERED.search(name)
    if m:
        return name[: m.start()].strip(), _WS.sub("", m.group(1))
    m = _BRANCH_TOKEN.search(name)
    if m:
        head = name[: m.start()].strip()
        if head:  # 이름 전체가 `…점` 이면 그건 지점이 아니라 가맹점명이다
            return head, m.group(1)
    return name, None


def normalize_name(raw: str) -> NameParts:
    """가맹점명 하나를 정규화한다. 순서가 규칙의 일부다.

    ① 유니코드·공백·대문자 → ② 채널 프리픽스 → ③ 법인 표기 → ④ PG 판정 → ⑤ 지점 접미사

    ②가 ③보다 먼저인 이유: `NAVERPAY*(주)스타벅스커피코리아` 처럼 프리픽스가 법인 표기보다
    앞에 온다. ④가 ③ 뒤인 이유: PG 목록을 법인 표기를 벗긴 형태로 관리하기 때문이다.
    """
    text = canonical_form(raw)
    if not text:
        return NameParts(normalized_name="")

    text, channel = strip_channel(text)
    text = strip_corporate(text)

    # PG 경유 — 실가맹점이 소실됐다. 지점·채널을 더 따질 것이 없고, 억지 추측 대신 표시만 한다.
    if text in PG_NAMES:
        return NameParts(
            normalized_name=text,
            payment_channel=PaymentChannel.PG,
            pg_masked=True,
        )

    text, branch = split_branch(text)
    text = _PUNCT_EDGE.sub("", text)
    text = _WS.sub(" ", text).strip()

    return NameParts(normalized_name=text, branch=branch, payment_channel=channel)
