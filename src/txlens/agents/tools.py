"""분류 Agent 의 도구 3종 (설계 §3.0).

## 공통 원칙 — 사전을 프롬프트에 붓지 않는다

사전 29개 브랜드·택소노미 34개 L2 를 프롬프트에 전부 넣으면 토큰이 늘고, 사전이 커질수록
프롬프트가 함께 커져 **사전 승격이 비용을 줄이는 게 아니라 늘리는** 구조가 된다. 조회로 바꾸면
사전이 v10 이 되어도 프롬프트는 그대로다. Agent 를 쓰는 이유가 여기 있다 — 필요한 것만 찾아본다.

## 그리고 이미 아는 것을 다시 묻지 않는다

결제채널·지점·법인표기는 `preprocess` 가 **결정론으로 이미 분리해 뒀다.** LLM 에게 다시
추론시키면 느리고 비싸고 틀린다. `get_brand_hint` 는 그 결과를 그대로 읽어 준다.

## 도구가 답을 대신 주지 않는다

`lookup_alias` 는 룰 매칭이 **이미 실패한** 이름에만 불린다. 그래서 돌려주는 후보는 전부 유사도
임계값 미달이고, "가장 비슷한 것"이 정답이라는 보장이 없다 — 도구 설명에 그 사실을 적어 둔다.
여기서 후보를 정답처럼 제시하면 앞서 잡은 `STEP GYM` → `STEP GYM MEMBERSHIP` 흡수를
프롬프트 층에서 다시 만들어내는 셈이다.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from rapidfuzz import process

from txlens.preprocess import normalize_name
from txlens.rules import FUZZY_SCORER, RuleDictionary
from txlens.schemas import PaymentChannel
from txlens.taxonomy import Taxonomy

#: 후보 조회의 하한. 사전 매칭은 90 이상에서 이미 실패했으므로 쓸모 있는 구간은 그 아래다.
#: 40 미만은 사람이 봐도 무관한 문자열이라 보여줄수록 오답을 유도한다.
ALIAS_LOOKUP_FLOOR = 55
ALIAS_LOOKUP_LIMIT = 5

#: 배달앱 채널이 주는 카테고리 힌트. 채널이 곧 업종인 유일한 경우다 (설계 §3.3 ①).
DELIVERY_CHANNELS: frozenset[PaymentChannel] = frozenset({
    PaymentChannel.BAEMIN,
    PaymentChannel.COUPANG_EATS,
    PaymentChannel.YOGIYO,
})


def make_tools(*, dictionary: RuleDictionary, taxonomy: Taxonomy) -> list[BaseTool]:
    """도구 3종을 만든다. 사전·택소노미를 **인자로 받는다**.

    모듈 전역에서 사전을 읽으면 베이스라인 (B) 를 만들 수 없다 — B 는 "LLM 만"이므로 도구가
    v1 사전을 들여다보면 그건 이미 하이브리드다. 사전을 주입하게 해 두면 B 는 빈 사전을,
    C 는 v1 을, 다음 라운드는 v2 를 넘겨 **같은 코드로** 조건을 바꿀 수 있다.
    """

    @tool
    def lookup_alias(name: str) -> str:
        """룰 사전에서 이름이 비슷한 등록 브랜드를 찾는다.

        Args:
            name: 조회할 가맹점 표기 (정규화 전/후 모두 가능).

        Returns:
            비슷한 사전 항목 목록. **이미 자동 매칭에 실패한 이름이므로 후보는 전부 확실하지
            않다.** 유사도가 높아도 다른 가맹점일 수 있으니, 근거가 없으면 채택하지 말고
            미분류로 답하라.
        """
        key = normalize_name(name).normalized_name
        if not key:
            return "조회할 수 있는 문자열이 아니다."
        if not dictionary.fuzzy_choices:
            return "사전이 비어 있다. 사전 조회로는 판단할 수 없다."

        hits = process.extract(
            key,
            dictionary.fuzzy_choices,
            scorer=FUZZY_SCORER,
            score_cutoff=ALIAS_LOOKUP_FLOOR,
            limit=ALIAS_LOOKUP_LIMIT,
        )
        if not hits:
            return f"'{key}' 와 비슷한 사전 항목이 없다. 이름만으로 판단해야 한다."

        lines = [f"'{key}' 와 비슷한 사전 항목 (확정 아님 — 유사도순):"]
        for matched_key, score, _ in hits:
            entry = dictionary.entry_for(matched_key)
            if entry is None:  # pragma: no cover - 색인과 후보 목록은 같은 출처다
                continue
            lines.append(
                f"- 사전 키 '{matched_key}' → {entry.canonical} "
                f"({entry.category_l2.value}) · 유사도 {score:.0f}"
            )
        return "\n".join(lines)

    @tool
    def search_taxonomy(keyword: str) -> str:
        """카테고리 택소노미를 키워드로 검색해 쓸 수 있는 `category_l2` 값을 찾는다.

        Args:
            keyword: 업종 관련 낱말 (예: `약국`, `커피`, `통신`). 한글 라벨과 영문 id 를 모두 본다.

        Returns:
            일치하는 L2 카테고리 목록 (`id (라벨)` 형태). 없으면 그렇다고 답한다.
            **여기 없는 값은 쓸 수 없다** — 스키마 검증에서 막힌다.
        """
        needle = keyword.strip().lower()
        if not needle:
            return "검색어가 비어 있다."
        hits = [
            f"- {cid} ({label})"
            for cid, label in taxonomy.labels.items()
            if "." in cid and (needle in label.lower() or needle in cid.lower())
        ]
        if not hits:
            # ⚠️ 여기서 "없으면 미분류로 답하라"고 쓰면 안 된다. 이 검색은 **라벨 문자열의 부분
            # 일치**일 뿐이라 동의어를 못 잡는다 — 실제로 `호프`(주류)를 못 찾은 모델이 도구의
            # 침묵을 "카테고리가 없다"는 증거로 받아들여 잡을 수 있는 것을 미분류로 버렸다.
            return (
                f"'{keyword}' 가 라벨에 그대로 들어 있는 카테고리는 없다. **이것은 카테고리가 "
                "없다는 뜻이 아니다** — 이 검색은 글자 일치만 본다. 시스템 프롬프트의 카테고리 "
                "목록을 다시 보고 업종의 성격이 맞는 곳을 고르고, 성격조차 알 수 없을 때만 "
                "unclassified.insufficient 로 답하라."
            )
        return f"'{keyword}' 검색 결과:\n" + "\n".join(hits)

    @tool
    def get_brand_hint(name: str) -> str:
        """전처리가 이미 분리해 둔 사실을 읽어 준다. **추론이 아니라 조회다.**

        Args:
            name: 원본 가맹점 표기.

        Returns:
            정규화된 매칭 키, 분리된 지점명, 결제채널, PG 경유 여부. 배달앱 채널이면
            카테고리 힌트도 함께 준다.
        """
        parts = normalize_name(name)
        lines = [f"정규화 키: '{parts.normalized_name}'"]
        if parts.branch:
            lines.append(f"지점: {parts.branch} (브랜드 판단에서 제외하라)")
        if parts.payment_channel is not None:
            lines.append(f"결제채널: {parts.payment_channel.value}")
        if parts.payment_channel in DELIVERY_CHANNELS:
            lines.append("배달앱 결제다 → 실가맹점을 몰라도 food.delivery 로 볼 근거가 된다.")
        if parts.pg_masked:
            lines.append(
                "PG사 경유라 실가맹점명이 오지 않았다 → unclassified.pg_masked 가 정답이다. "
                "추측하지 마라."
            )
        if not parts.branch and parts.payment_channel is None and not parts.pg_masked:
            lines.append("분리된 지점·채널 없음. 이름 전체가 가맹점 표기다.")
        return "\n".join(lines)

    return [lookup_alias, search_taxonomy, get_brand_hint]
