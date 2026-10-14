"""제공기관 프로파일 계약 테스트.

이 파일은 생성기 입력이므로 값이 틀리면 데이터 전체가 조용히 비현실적으로 만들어진다.
확률·비중 같은 수치 제약을 여기서 고정한다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROFILES = Path(__file__).resolve().parents[1] / "config" / "provider-profiles.yaml"


def _load() -> dict:
    return yaml.safe_load(PROFILES.read_text(encoding="utf-8"))


def test_file_exists_and_has_providers():
    d = _load()
    assert d["version"] == 1
    assert len(d["providers"]) >= 3, "편차를 보여주려면 최소 3개 기관은 있어야 한다"


def test_weights_sum_to_one():
    """비중 합이 1이 아니면 거래 배분이 의도와 달라진다."""
    total = sum(p["weight"] for p in _load()["providers"])
    assert abs(total - 1.0) < 1e-9, f"weight 합이 {total} (1.0 이어야 한다)"


def test_name_style_distributions_sum_to_one():
    for p in _load()["providers"]:
        s = sum(p["name_style"].values())
        assert abs(s - 1.0) < 1e-9, f"{p['id']} name_style 합이 {s}"


def test_rates_are_probabilities():
    keys = ("regno_fill_rate", "channel_prefix_rate", "pg_passthrough_rate", "typo_rate")
    for p in _load()["providers"]:
        for k in keys:
            v = p[k]
            assert 0.0 <= v <= 1.0, f"{p['id']}.{k} = {v} 는 확률 범위를 벗어난다"


def test_ids_unique():
    ids = [p["id"] for p in _load()["providers"]]
    assert len(ids) == len(set(ids))


def test_spread_is_actually_wide():
    """편차를 모델링하는 게 목적이므로, 프로파일이 서로 비슷하면 의미가 없다."""
    fills = [p["regno_fill_rate"] for p in _load()["providers"]]
    assert max(fills) - min(fills) >= 0.5, (
        f"regno 채움률 편차가 {max(fills) - min(fills):.2f} — 기관 간 차이가 너무 작다"
    )
    corp = [p["name_style"]["corporate"] for p in _load()["providers"]]
    assert max(corp) - min(corp) >= 0.3, "법인명 표기 비율 편차가 너무 작다"
