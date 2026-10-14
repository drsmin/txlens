"""택소노미 드리프트 가드.

카테고리를 YAML 과 Enum 두 곳에 적는 대가로 정적 타입을 얻었다. 그 대가를 감당하는 게 이 테스트다.
한쪽만 고치면 즉시 빨간불이 되어야 한다 — 그러지 않으면 골드셋 라벨이 조용히 무효가 된다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from txlens.schemas import CategoryL1, CategoryL2
from txlens.taxonomy import assert_in_sync, load_taxonomy


def test_yaml_and_enums_in_sync():
    assert_in_sync()


def test_every_l2_belongs_to_a_real_l1():
    """`<l1>.<leaf>` 규약이 깨지면 CategoryL2.l1 이 잘못된 상위를 돌려준다."""
    l1_values = {m.value for m in CategoryL1}
    for leaf in CategoryL2:
        prefix = leaf.value.split(".", 1)[0]
        assert prefix in l1_values, f"{leaf.value} 의 상위 {prefix} 가 L1 에 없다"
        assert leaf.l1.value == prefix


def test_every_l1_has_at_least_one_l2():
    tax = load_taxonomy()
    for top in tax.l1:
        assert top.l2, f"{top.id} 에 L2 가 없다"


def test_unclassified_exists_and_has_reason_codes():
    """미분류는 정당한 답이므로 반드시 존재하고, '못 했다/안 했다'를 구분할 이유 코드를 가진다."""
    assert CategoryL1.UNCLASSIFIED.value == "unclassified"
    reasons = {m.value for m in CategoryL2 if m.l1 is CategoryL1.UNCLASSIFIED}
    assert "unclassified.pg_masked" in reasons
    assert "unclassified.insufficient" in reasons


def test_labels_cover_all_ids():
    tax = load_taxonomy()
    for cid in tax.l1_ids() + tax.l2_ids():
        assert tax.label(cid), f"{cid} 에 라벨이 없다"


def test_prompt_block_lists_every_category():
    """프롬프트 주입 문자열이 카테고리를 누락하면 LLM 이 그 카테고리를 절대 못 고른다."""
    tax = load_taxonomy()
    block = tax.prompt_block()
    for cid in tax.l1_ids() + tax.l2_ids():
        assert cid in block, f"{cid} 가 프롬프트 블록에 없다"


def test_sync_check_actually_fails_on_drift(tmp_path):
    """가드가 실제로 발동하는지 — 통과만 하는 검사는 커버리지처럼 보여서 없는 것보다 나쁘다."""
    broken = tmp_path / "taxonomy.yaml"
    broken.write_text(
        "version: 1\n"
        "l1:\n"
        "  - id: food\n"
        "    label: 식비\n"
        "    l2:\n"
        "      - { id: food.restaurant, label: 외식 }\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="택소노미 불일치"):
        assert_in_sync(broken)


# ── 문서에 적힌 개수도 드리프트한다 ─────────────────────────────────────────
#: `L1 11개` / `L2 40개` 같은 문장을 담고 있는 문서들. 발표 자료가 포함되는 것이 중요하다 —
#: 청중이 보는 숫자다.
_DOCS_WITH_COUNTS = ("README.md",)
_COUNT_PATTERN = re.compile(r"\bL([12])\s+(\d+)개")


def test_문서에_적힌_택소노미_개수가_실제와_같다():
    """발표 자료에 `L2 34개` 로 **네 곳에** 적혀 있었고 실제는 40개였다.

    Enum↔YAML 은 위 테스트가 지켰지만 **산문에 적힌 숫자**는 아무도 안 지켰다. 청중이 보는
    숫자가 틀리는 것을 사람 기억에 맡길 이유가 없다. 개수를 문서에 쓰려면 이 형식으로 쓴다:
    `L1 11개` · `L2 40개`.
    """
    tax = load_taxonomy()
    actual = {"1": len(tax.l1_ids()), "2": len(tax.l2_ids())}
    root = Path(__file__).resolve().parents[1]

    checked = 0
    for rel in _DOCS_WITH_COUNTS:
        path = root / rel
        if not path.exists():  # pragma: no cover - 리포트 파일명은 날짜로 바뀐다
            continue
        for level, written in _COUNT_PATTERN.findall(path.read_text(encoding="utf-8")):
            checked += 1
            assert int(written) == actual[level], (
                f"{rel}: L{level} 을 {written}개로 적었지만 config/taxonomy.yaml 은 "
                f"{actual[level]}개다"
            )
    assert checked > 0, "개수를 적은 문서를 하나도 못 찾았다 — 패턴이나 문서 목록이 낡았다"
