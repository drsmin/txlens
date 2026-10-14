"""가맹점명 오염기 — 이 프로젝트가 풀려는 문제를 만드는 곳.

**오염은 균일 랜덤이 아니라 제공기관 프로파일을 따른다.** 실무 정규화는 "B사는 항상
법인명을 준다" 같은 기관별 패턴을 학습해 이용하므로, 균일 랜덤으로 흩뿌리면 그 군집 구조가
사라져 룰 단계가 실제보다 불리하게 평가된다.

오염 단계는 전처리(`preprocess`)가 되돌려야 하는 것들과 정확히 대응한다:
표기 형태 → 지점 접미사 → 채널 프리픽스 → PG 치환 → 오타.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from txlens.generator.brands import (
    BRANCH_NAMES,
    CHANNEL_PREFIXES,
    PG_NAMES,
    Brand,
)
from txlens.generator.profiles import ProviderProfile
from txlens.generator.rng import weighted_pick


@dataclass(frozen=True)
class CorruptedName:
    """오염 결과 + 정답 추적 정보.

    `pg_masked` 가 True 면 원 브랜드를 표기에서 복원할 수 없다 — 정답 라벨도 `미분류` 가 되어야
    한다(억지 추측 금지). 그래서 오염기가 이 플래그를 결정하고 라벨 생성이 그걸 따른다.
    """

    raw: str                      # 제공기관이 보내온 그대로의 문자열
    style: str                    # brand_plain / brand_with_branch / corporate / romanized
    branch: str | None
    channel: str | None           # PaymentChannel 값 (문자열)
    pg_masked: bool
    has_typo: bool


def _apply_typo(rng: random.Random, s: str) -> str:
    """1글자 치환/삭제/중복. 실제 원장에 흔한 형태만 흉내낸다."""
    if len(s) < 3:
        return s
    i = rng.randrange(1, len(s) - 1)
    mode = rng.choice(("drop", "dup", "swap"))
    if mode == "drop":
        return s[:i] + s[i + 1:]
    if mode == "dup":
        return s[:i] + s[i] + s[i:]
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def corrupt(rng: random.Random, brand: Brand, profile: ProviderProfile) -> CorruptedName:
    """브랜드 하나를 제공기관 관행에 맞춰 오염시킨다."""
    # ① PG 경유 — 실가맹점이 소실된다. 다른 오염을 얹을 대상이 없으므로 즉시 반환.
    if rng.random() < profile.pg_passthrough_rate:
        return CorruptedName(
            raw=rng.choice(PG_NAMES), style="pg", branch=None,
            channel="pg", pg_masked=True, has_typo=False,
        )

    # ② 표기 형태 — 기관별 분포를 따른다
    styles, weights = profile.name_style.as_pairs()
    style = weighted_pick(rng, styles, weights)

    if style == "corporate" and brand.corporate:
        base = brand.corporate
    elif style == "romanized" and brand.romanized:
        base = brand.romanized
    else:
        # 지점명이 없는 브랜드(넷플릭스 등)에 지점을 붙이면 비현실적이다
        if style == "brand_with_branch" and not brand.has_branch:
            style = "brand_plain"
        base = brand.brand

    # ③ 지점 접미사 — 분리해 보존해야 하는 대상
    branch: str | None = None
    if style == "brand_with_branch" and brand.has_branch:
        branch = rng.choice(BRANCH_NAMES)
        base = f"{base} {branch}"

    # ④ 채널 프리픽스 — 채널이면서 카테고리 힌트이기도 하다
    channel: str | None = None
    if rng.random() < profile.channel_prefix_rate:
        prefix, channel = weighted_pick(
            rng, CHANNEL_PREFIXES, [1.0] * len(CHANNEL_PREFIXES)
        )
        base = f"{prefix}{base}"

    # ⑤ 오타
    has_typo = rng.random() < profile.typo_rate
    if has_typo:
        base = _apply_typo(rng, base)

    return CorruptedName(
        raw=base, style=style, branch=branch,
        channel=channel, pg_masked=False, has_typo=has_typo,
    )


def make_regno(rng: random.Random, brand: Brand, branch: str | None, *, pg: bool) -> str:
    """사업자등록번호. **규격 형태는 `aN(12)` — 하이픈 포함 `123-45-67890`** (카드-008).

    하이픈 없는 10자리로 내면 실연동에서 REGNO 매칭 키가 어긋난다. 규격대로 내고, 매칭 쪽에서
    `preprocess.regno_key()` 로 숫자만 남긴다 (기관에 따라 하이픈이 없을 수도 있으므로).

    **같은 브랜드라도 지점이 다르면 다른 번호**가 된다 — regno 로는 "같은 점포"만 잡히고
    "같은 브랜드"는 못 잡는다는 그 근거를 데이터로 재현한다.
    PG 경유면 PG사의 번호가 오므로 브랜드와 무관한 값이 된다.
    """
    key = "PG" if pg else f"{brand.brand}|{branch or ''}"
    seeded = random.Random(f"regno:{key}")
    return (
        f"{seeded.randrange(100, 999)}-{seeded.randrange(10, 99)}-{seeded.randrange(10000, 99999)}"
    )
