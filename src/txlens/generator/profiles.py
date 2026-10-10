"""`config/provider-profiles.yaml` 로더 (D-012).

제공기관별 표기 관행을 담은 단일 출처. 생성기는 이 프로파일만 보고 오염 강도를 결정하고,
평가는 같은 `provider_id` 축으로 지표를 쪼갠다.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "provider-profiles.yaml"

_TOL = 1e-6


class NameStyle(BaseModel):
    """가맹점명 표기 형태의 분포. 합이 1 이어야 한다."""

    brand_plain: float = Field(ge=0, le=1)
    brand_with_branch: float = Field(ge=0, le=1)
    corporate: float = Field(ge=0, le=1)
    romanized: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _sums_to_one(self):
        total = self.brand_plain + self.brand_with_branch + self.corporate + self.romanized
        if abs(total - 1.0) > _TOL:
            raise ValueError(f"name_style 합이 {total} — 1.0 이어야 한다")
        return self

    def as_pairs(self) -> tuple[list[str], list[float]]:
        return (
            ["brand_plain", "brand_with_branch", "corporate", "romanized"],
            [self.brand_plain, self.brand_with_branch, self.corporate, self.romanized],
        )


class ProviderProfile(BaseModel):
    id: str
    label: str
    weight: float = Field(gt=0, le=1)
    regno_fill_rate: float = Field(ge=0, le=1)
    name_style: NameStyle
    channel_prefix_rate: float = Field(ge=0, le=1)
    pg_passthrough_rate: float = Field(ge=0, le=1)
    typo_rate: float = Field(ge=0, le=1)


class ProviderSet(BaseModel):
    version: int
    providers: list[ProviderProfile]

    @model_validator(mode="after")
    def _weights_sum_to_one(self):
        total = sum(p.weight for p in self.providers)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"weight 합이 {total} — 1.0 이어야 한다")
        ids = [p.id for p in self.providers]
        if len(ids) != len(set(ids)):
            raise ValueError("provider id 가 중복된다")
        return self

    def ids(self) -> list[str]:
        return [p.id for p in self.providers]

    def weights(self) -> list[float]:
        return [p.weight for p in self.providers]

    def by_id(self, provider_id: str) -> ProviderProfile:
        for p in self.providers:
            if p.id == provider_id:
                return p
        raise KeyError(provider_id)


@functools.lru_cache(maxsize=4)
def load_providers(path: Path | str = DEFAULT_PATH) -> ProviderSet:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ProviderSet.model_validate(raw)
