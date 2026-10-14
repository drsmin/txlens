"""식별번호 정규화 — 사업자등록번호 (설계 §3.3 ①).

규격은 `merchant_regno` 를 `aN(12)` **하이픈 포함**(`123-45-67890`)으로 규정한다 (카드-008).
그런데 하이픈을 빼고 10자리로 주는 기관이 실제로 있다. 표기를 그대로 매칭 키로 쓰면
같은 사업자가 형식 때문에 갈라진다 — 가맹점명에서 겪는 문제와 정확히 같은 문제다.

그래서 **원본은 규격 형태로 보존하고**(`RawTx.merchant_regno`) 매칭할 때만 이 함수로 키를 만든다.
"""

from __future__ import annotations

import re

_NON_DIGIT = re.compile(r"\D")

#: 사업자등록번호는 10자리다. 길이가 다르면 사업자등록번호가 아니거나 깨진 값이다.
REGNO_DIGITS = 10


def regno_key(regno: str | None) -> str | None:
    """`"123-45-67890"` → `"1234567890"`. 매칭에 쓸 키를 만든다.

    10자리가 아니면 `None` 을 돌려준다 — **틀린 키로 매칭하는 것보다 매칭하지 않는 게 낫다.**
    regno 는 어차피 보조 신호이므로 여기서 버려도 이름 매칭이 남는다. 반대로 깨진 값을
    키로 쓰면 엉뚱한 브랜드에 붙고, 그건 조용히 틀린 분류가 된다.
    """
    if regno is None:
        return None
    digits = _NON_DIGIT.sub("", regno)
    return digits if len(digits) == REGNO_DIGITS else None
