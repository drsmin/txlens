"""`.gitignore` 가 **실제로** 동작하는지 검사한다.

왜 테스트가 필요한가: `.gitignore` 에서 `#` 은 줄 맨 앞에서만 주석이다. 패턴 뒤에 설명을 붙이면
그 텍스트가 패턴의 일부가 되어 **조용히 아무것도 매치하지 않는다.** 실제로 이 리포에서
`data/synthetic/*.json  # 설명` 형태로 4개 패턴이 죽어 있었고, 4.3MB 재현 가능 데이터가
커밋될 참이었다. 파일을 눈으로 봐서는 정상으로 보이는 종류의 실패라 검사로 고정한다.

`git check-ignore` 로 git 의 실제 판정을 묻는다 — 패턴을 다시 해석하지 않는다.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 반드시 무시되어야 하는 경로 (재현 가능 산출물·비밀정보·로컬 상태)
MUST_IGNORE = [
    "data/synthetic/ground_truth.json",
    "data/synthetic/manifest.json",
    "data/synthetic/bundle/CARD_A_SYNTH_0000_CD03.json",
    "state.db",
    "run.sqlite",
    ".cache/llm/abc.json",
    "reports/eval-2026-08.md",
    ".env",
    ".env.local",
    "secrets.yaml",
    "credentials.json",
    "server.pem",
    "id_rsa.key",
    ".venv/bin/python",
    "src/txlens/__pycache__/x.pyc",
    "src/txlens.egg-info/PKG-INFO",
]

#: 반드시 추적되어야 하는 경로 (손으로 만든 자산·소스)
MUST_TRACK = [
    "data/synthetic/.gitkeep",
    "data/gold/.gitkeep",
    "rules/.gitkeep",
    ".env.example",
    "config/taxonomy.yaml",
    "config/provider-profiles.yaml",
    "src/txlens/schemas.py",
    "scripts/check-docs-budget.sh",
    "requirements.lock.txt",
]


def _ignored(path: str) -> bool:
    """git 이 이 경로를 무시하는가. 종료코드 0 = 무시, 1 = 무시 안 함."""
    r = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=ROOT, capture_output=True, check=False,
    )
    return r.returncode == 0


@pytest.mark.parametrize("path", MUST_IGNORE)
def test_regenerable_and_secret_paths_are_ignored(path):
    assert _ignored(path), (
        f"{path} 가 무시되지 않는다 — .gitignore 패턴 뒤에 주석을 붙이지 않았는지 확인"
    )


@pytest.mark.parametrize("path", MUST_TRACK)
def test_hand_made_assets_are_not_ignored(path):
    assert not _ignored(path), f"{path} 는 커밋되어야 하는데 무시된다"


def test_no_trailing_comments_in_gitignore():
    """후행 주석은 패턴을 무력화한다 — 파일 자체를 검사해 재발을 막는다."""
    offenders = []
    for i, line in enumerate((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in line:
            offenders.append(f"{i}행: {line}")
    assert not offenders, (
        "패턴 뒤 주석은 패턴의 일부가 되어 매치를 없앤다. 설명은 별도 줄로:\n  "
        + "\n  ".join(offenders)
    )
