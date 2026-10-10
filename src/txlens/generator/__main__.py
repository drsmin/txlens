"""`python -m txlens.generator --seed 42`

docs/recovery.md 재개 절차가 이 명령을 호출한다. 데이터는 커밋하지 않고 시드로 재현한다.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from txlens.console import use_utf8_output
from txlens.generator.build import build
from txlens.generator.emit import write_dataset

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "data" / "synthetic"


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()
    ap = argparse.ArgumentParser(description="TxLens 합성 거래 데이터 생성기")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--customers", type=int, default=30)
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--per-month", type=int, default=42, help="고객당 월 일상소비 건수")
    ap.add_argument(
        "--period-end", default="2026-08-31",
        help="기간 종료일(YYYY-MM-DD). 실행 시각에 의존하지 않게 하려고 인자로 받는다",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    txs, gt = build(
        seed=args.seed,
        customers=args.customers,
        months=args.months,
        per_month=args.per_month,
        # tz 없는 KST 로 통일 (D-009). 오프셋을 요구하면 CLI 사용성만 나빠진다.
        period_end=datetime.strptime(args.period_end, "%Y-%m-%d"),  # noqa: DTZ007
    )
    stats = write_dataset(txs, gt, args.out)

    print(f"시드 {gt.seed} · 고객 {gt.customers}명 · {gt.period_start}~{gt.period_end}")
    print(f"  {gt.summary()}")
    print(f"  번들 {stats['bundles']}개 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
