"""합성 거래 데이터 생성기 (2층 구조).

- **Layer 1 — 표준API 봉투** (`emit.py`): 필드명·형식은 실제 규격(CD03) 형태.
- **Layer 2 — 의미 있는 값** (`build.py`): 제공기관 프로파일 기반 오염, 현실적 금액·시각,
  탐지 대상 패턴, 오탐 함정, 그리고 **심으면서 기록하는 정답셋**.

`python -m txlens.generator --seed 42` 로 실행한다. 데이터는 커밋하지 않고 시드로 재현한다.
"""

from txlens.generator.build import build
from txlens.generator.emit import to_bundles, write_dataset

__all__ = ["build", "to_bundles", "write_dataset"]
