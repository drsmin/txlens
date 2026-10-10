"""결정론적 난수.

**목적별·고객별로 파생 시드를 쓴다.** 단일 `Random(seed)` 를 순차 소비하면 생성 로직을 조금만
바꿔도(고객 한 명 추가, 패턴 하나 삽입) 그 뒤 모든 난수가 밀려 데이터 전체가 달라진다.
골드셋 라벨이 거래에 붙어 있으므로 그건 라벨 무효화를 뜻한다.

파생 시드는 `blake2b` 로 만든다 — Python 의 `hash()` 는 프로세스마다 소금이 달라 재현되지 않는다.
"""

from __future__ import annotations

import hashlib
import random


def child_rng(seed: int, *parts: str | int) -> random.Random:
    """`seed` 와 `parts` 조합에서 안정적으로 파생된 RNG.

    같은 인자면 프로세스·실행 시점과 무관하게 항상 같은 수열을 준다.

    >>> child_rng(42, "cust", 1).random() == child_rng(42, "cust", 1).random()
    True
    >>> child_rng(42, "cust", 1).random() == child_rng(42, "cust", 2).random()
    False
    """
    key = ":".join(str(p) for p in (seed, *parts)).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def weighted_pick(rng: random.Random, items: list, weights: list[float]):
    """가중 선택. `random.choices` 는 복원추출 리스트를 주므로 단건용 래퍼."""
    return rng.choices(items, weights=weights, k=1)[0]
