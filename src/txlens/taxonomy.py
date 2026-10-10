"""config/taxonomy.yaml 로더.

두 곳에서 쓴다: ① LLM 프롬프트에 카테고리 목록을 주입할 때(`prompt_block`), ② 라벨·분류 결과가
유효한 카테고리인지 검증할 때. 같은 목록을 코드에 또 적으면 반드시 어긋나므로 YAML 이 단일 출처다.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml
from pydantic import BaseModel

from txlens.schemas import CategoryL1, CategoryL2

#: 리포 루트 기준 기본 경로. src/txlens/taxonomy.py → 부모 3단계가 루트.
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "taxonomy.yaml"


class L2Entry(BaseModel):
    id: str
    label: str


class L1Entry(BaseModel):
    id: str
    label: str
    l2: list[L2Entry]


class Taxonomy(BaseModel):
    version: int
    l1: list[L1Entry]

    # ── 조회 ──
    @functools.cached_property
    def labels(self) -> dict[str, str]:
        """id → 한글 라벨. L1·L2 를 한 사전에 담는다."""
        out: dict[str, str] = {}
        for top in self.l1:
            out[top.id] = top.label
            for leaf in top.l2:
                out[leaf.id] = leaf.label
        return out

    def label(self, category_id: str) -> str:
        return self.labels[category_id]

    def l1_ids(self) -> list[str]:
        return [t.id for t in self.l1]

    def l2_ids(self) -> list[str]:
        return [leaf.id for t in self.l1 for leaf in t.l2]

    # ── LLM 프롬프트 주입 ──
    def prompt_block(self) -> str:
        """분류 프롬프트에 넣을 카테고리 목록.

        택소노미를 바꾸면 프롬프트가 자동으로 따라오게 하려고 문자열을 여기서 만든다.
        프롬프트 템플릿에 카테고리를 직접 적으면 YAML 을 고쳐도 LLM 은 옛 목록을 본다.
        """
        lines: list[str] = []
        for top in self.l1:
            leaves = ", ".join(f"{leaf.id}({leaf.label})" for leaf in top.l2)
            lines.append(f"- {top.id} ({top.label}): {leaves}")
        return "\n".join(lines)


@functools.lru_cache(maxsize=4)
def load_taxonomy(path: Path | str = DEFAULT_PATH) -> Taxonomy:
    """택소노미를 읽어 검증한다. 같은 경로는 캐시된다."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Taxonomy.model_validate(raw)


def assert_in_sync(path: Path | str = DEFAULT_PATH) -> None:
    """YAML 과 schemas.py 의 Enum 이 일치하는지 확인하고, 어긋나면 어긋난 항목을 알린다.

    한쪽만 고치는 실수가 이 프로젝트에서 가장 조용히 번지는 오류다 — 골드셋 라벨이 id 로
    저장되므로, Enum 에서 빠진 카테고리는 라벨을 통째로 무효화한다.
    """
    tax = load_taxonomy(path)
    for name, ids, enum_cls in (
        ("L1", tax.l1_ids(), CategoryL1),
        ("L2", tax.l2_ids(), CategoryL2),
    ):
        yaml_set = set(ids)
        enum_set = {m.value for m in enum_cls}
        if yaml_set != enum_set:
            only_yaml = sorted(yaml_set - enum_set)
            only_enum = sorted(enum_set - yaml_set)
            raise AssertionError(
                f"{name} 택소노미 불일치 — "
                f"YAML 에만: {only_yaml or '없음'} / Enum 에만: {only_enum or '없음'}"
            )
