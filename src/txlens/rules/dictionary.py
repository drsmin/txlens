"""`rules/v{n}.yaml` 로더 — 사전을 검증해 색인으로 올린다.

## 로더가 관용하지 않는 것들

사전의 오류는 **조용히** 지표를 깎는다. 매칭되지 않는 키는 예외를 내지 않고 그냥 미분류를
늘리므로, 사람은 "룰이 원래 이 정도"라고 오해한다. 그래서 사전이 애초에 못 쓰는 형태면
로딩 시점에 실패시킨다:

- **키가 정규화 형태가 아니다** (`(주)스타벅스`) → 실제 매칭 대상(`normalized_name`)과 절대
  같아질 수 없는 키다. 있으나 없으나 같으므로 실패시킨다.
- **키가 중복된다** → 나중 항목이 앞 항목을 덮어써 라벨이 조용히 바뀐다.
- **`l2` 가 `l1` 에 속하지 않는다** → 지표 집계가 어긋난다 (`_CategoryConsistency` 가 막는다).
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from txlens.preprocess import normalize_name, regno_key
from txlens.schemas import CategoryL1, CategoryL2, _CategoryConsistency

#: 리포 루트 기준 기본 경로. src/txlens/rules/dictionary.py → 부모 3단계가 루트.
DEFAULT_DICTIONARY_PATH = Path(__file__).resolve().parents[3] / "rules" / "v1.yaml"


class RuleDictionaryError(Exception):
    """사전을 신뢰할 수 없다. 조용히 넘기지 않고 멈춘다.

    ⚠️ `ValueError` 를 상속하지 않는다(`IngestError` 와 다른 선택이다). pydantic 은 검증기 안에서
    난 `ValueError` 를 `ValidationError` 로 감싸버려, 위에서 공들여 쓴 "왜 이 키가 죽은 키인가"
    설명이 스택 안에 묻힌다. 사전 오류는 사람이 사전을 고쳐야 하는 오류이므로 메시지가 그대로
    보이는 것이 중요하다.
    """


class RuleEntry(_CategoryConsistency):
    """브랜드 하나 = 라벨 하나 + 그 브랜드에 이르는 키들.

    단계별(exact/alias/…)로 파일을 쪼개지 않고 **브랜드 단위로 묶는** 이유: 라벨이 키 옆에
    있어야 사람이 사전을 고칠 때 라벨을 같이 본다. 단계별로 흩어놓으면 `alias` 만 고치고
    라벨은 옛 것이 남는 사고가 난다.
    """

    model_config = ConfigDict(populate_by_name=True)

    canonical: str = Field(description="대표 가맹점명. 지표·집계·화면에 그대로 노출된다")
    #: YAML 에서는 `l1`/`l2` 로 짧게 적는다 (config/taxonomy.yaml 과 같은 표기).
    category_l1: CategoryL1 = Field(alias="l1")
    category_l2: CategoryL2 = Field(alias="l2")

    exact: list[str] = Field(
        default_factory=list,
        description="정규화 완전일치 키. 비워두면 canonical 자신이 키가 된다",
    )
    alias: list[str] = Field(default_factory=list, description="법인명·로마자 등 별칭")
    regex: list[str] = Field(default_factory=list, description="접두 패턴 (정규식)")
    regno: list[str] = Field(default_factory=list, description="사업자등록번호 (하이픈 무관)")

    @property
    def exact_keys(self) -> list[str]:
        """실제 색인에 들어가는 exact 키. `canonical` 을 항상 포함한다.

        사전마다 `exact: [자기이름]` 을 반복해 적게 하면 빠뜨리는 항목이 반드시 생긴다.
        """
        keys = [self.canonical, *self.exact]
        seen: dict[str, None] = {}
        for key in keys:
            seen.setdefault(key, None)
        return list(seen)


class RuleDictionary(BaseModel):
    """검증된 사전. 색인은 `cached_property` 로 한 번만 만든다."""

    version: int
    fuzzy_threshold: int = Field(default=90, ge=0, le=100)
    entries: list[RuleEntry]

    @model_validator(mode="after")
    def _validate_keys(self):
        seen: dict[str, str] = {}  # 키 → 그 키를 이미 쓴 canonical

        def claim(kind: str, key: str, canonical: str) -> None:
            if key in seen:
                raise RuleDictionaryError(
                    f"{kind} 키가 중복된다: {key!r} — {seen[key]!r} 와 {canonical!r} 가 같은 키를 "
                    "쓰면 나중 항목이 앞 항목을 덮어써 라벨이 조용히 바뀐다"
                )
            seen[key] = canonical

        for entry in self.entries:
            for kind, keys in (("exact", entry.exact_keys), ("alias", entry.alias)):
                for key in keys:
                    normalized = normalize_name(key).normalized_name
                    if normalized != key:
                        raise RuleDictionaryError(
                            f"{kind} 키가 정규화 형태가 아니다: {key!r} → {normalized!r} 로 적어야 "
                            "한다. 정규화 전 형태는 매칭 대상과 같아질 수 없어 영원히 죽은 키가 된다"
                        )
                    claim(kind, key, entry.canonical)

            for raw in entry.regno:
                key = regno_key(raw)
                if key is None:
                    raise RuleDictionaryError(
                        f"사업자등록번호가 10자리가 아니다: {raw!r} ({entry.canonical})"
                    )
                claim("regno", key, entry.canonical)

            for pattern in entry.regex:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise RuleDictionaryError(
                        f"정규식을 컴파일할 수 없다: {pattern!r} ({entry.canonical}) — {exc}"
                    ) from exc
        return self

    # ── 색인 ────────────────────────────────────────────────────────────────
    @functools.cached_property
    def exact_index(self) -> dict[str, RuleEntry]:
        return {key: e for e in self.entries for key in e.exact_keys}

    @functools.cached_property
    def alias_index(self) -> dict[str, RuleEntry]:
        return {key: e for e in self.entries for key in e.alias}

    @functools.cached_property
    def regno_index(self) -> dict[str, RuleEntry]:
        """숫자만 남긴 키로 색인한다 — 기관마다 하이픈 유무가 다르다 (D-023)."""
        return {regno_key(raw): e for e in self.entries for raw in e.regno}  # type: ignore[misc]

    @functools.cached_property
    def regex_index(self) -> list[tuple[re.Pattern[str], RuleEntry]]:
        """**파일 순서를 유지한다.** 먼저 적힌 패턴이 이긴다 — 순서가 룰의 일부다."""
        return [
            (re.compile(pattern, re.IGNORECASE), e) for e in self.entries for pattern in e.regex
        ]

    @functools.cached_property
    def fuzzy_choices(self) -> tuple[str, ...]:
        """유사도 후보 = exact + alias 키.

        `regex` 는 패턴이라 유사도를 재는 대상이 아니고, `canonical` 은 exact 에 이미 있다.
        """
        return (*self.exact_index, *self.alias_index)

    def entry_for(self, key: str) -> RuleEntry | None:
        """exact → alias 순으로 키를 찾는다. 유사도 결과를 라벨로 되돌릴 때 쓴다."""
        return self.exact_index.get(key) or self.alias_index.get(key)

    def summary(self) -> str:
        return (
            f"v{self.version} · 브랜드 {len(self.entries)}개 · "
            f"exact {len(self.exact_index)} · alias {len(self.alias_index)} · "
            f"regex {len(self.regex_index)} · regno {len(self.regno_index)}"
        )


@functools.lru_cache(maxsize=4)
def load_dictionary(path: Path | str = DEFAULT_DICTIONARY_PATH) -> RuleDictionary:
    """사전을 읽어 검증한다. 같은 경로는 캐시된다 (`load_taxonomy` 와 같은 방식)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RuleDictionary.model_validate(raw)
