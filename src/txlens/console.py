"""CLI 가 콘솔 코드페이지 때문에 죽지 않게 하는 한 줄 (D-039③ 과 같은 결함, 같은 처방).

Windows 한국어 콘솔은 기본 코드페이지가 cp949 다. 이 프로젝트의 출력은 전부 한국어이고
`■ · — ✓ ⚠️` 같은 글자가 섞여 있어서, `python -m txlens.eval` 은 표를 **한 줄도 못 찍고**
`UnicodeEncodeError` 로 죽는다. 계산은 다 끝난 뒤에 찍다가 죽으므로 몇 초를 버리고 아무것도
얻지 못한다.

같은 결함을 이미 두 번 고쳤다 — `scripts/render-graph.py`(pre-commit 훅이 이유 없이 커밋을
막았다)와 리포트 빌더의 글리프 검사(실패해도 이유를 못 찍었다). 세 번째부터는 각 CLI 에
베끼지 않고 여기 하나를 부른다.

**깨져 보이는 것이 죽는 것보다 낫다** — 그래서 `errors="replace"` 다. 발표장에서 명령 하나가
예외로 끝나는 것과, 글자 몇 개가 `?` 로 보이는 것 중에서는 뒤가 낫다.
"""

from __future__ import annotations

import sys


def use_utf8_output() -> None:
    """표준 출력·오류를 UTF-8 로 고정한다. 고칠 수 없는 스트림은 그냥 둔다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # 파이프로 감싸인 경우 등
            pass
