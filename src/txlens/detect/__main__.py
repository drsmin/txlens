"""`python -m txlens.detect` — 2부 탐지를 돌리고 지표를 낸다.

    python -m txlens.detect                          # 룰 사전(v1) 위에서 탐지 + 지표
    python -m txlens.detect --rules rules/v2.yaml    # 승격 사전 위에서
    python -m txlens.detect --show 5                 # 알림 예시를 근거 문장과 함께 본다
    python -m txlens.detect --transfer               # §5.2 전이 실험 (정규화만 · A · 정답)
    python -m txlens.detect --transfer --verdicts data/gold/verdicts-v1-gpt-4o-mini.json

**모델을 호출하지 않는다.** 탐지 3종은 결정론 규칙이고, 전이 실험의 C 줄은 저장된 판정을
재생해서 만든다 — 같은 질문에 두 번 결제하지 않는다(`agents/replay.py`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from txlens.console import use_utf8_output
from txlens.detect import Detections, detect
from txlens.detect.subscription import monthly_fixed_cost
from txlens.eval.detect_metrics import evaluate_detections
from txlens.eval.harness import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_TRUTH_PATH,
    load_truth,
    run_baseline_state,
)
from txlens.eval.transfer import run_transfer, transfer_table
from txlens.rules import load_dictionary


def _show_examples(det: Detections, limit: int) -> None:
    """근거 문장을 그대로 보여준다 — 화면에 나갈 문장이 여기서 읽히지 않으면 고객도 못 읽는다."""
    cost = monthly_fixed_cost(det.subscriptions)
    print("\n■ 구독 (근거 문장)")
    for s in det.subscriptions[:limit]:
        flags = "".join(["  [미사용 의심]" if s.unused else "", "  [가격 인상]" if s.price_increased else ""])
        print(f"  · {s.customer_id} {s.merchant_canonical} {s.median_amount:,}원/{s.period_kind}{flags}")
        print(f"    {s.evidence}")
    if cost:
        top = sorted(cost.items(), key=lambda kv: -kv[1])[:limit]
        joined = " · ".join(f"{cid} {amount:,}원" for cid, amount in top)
        print(f"  월 고정지출 상위: {joined}")

    print("\n■ 이상지출 (근거 문장)")
    for a in det.spending_alerts[:limit]:
        print(f"  · {a.customer_id} [{a.kind}] {a.evidence}")

    print("\n■ 이상거래 (근거 문장)")
    for a in det.tx_alerts[:limit]:
        print(f"  · {a.customer_id} [{a.kind}] {a.evidence}")


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()
    ap = argparse.ArgumentParser(description="TxLens 2부 — 패턴 탐지")
    ap.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_PATH)
    ap.add_argument(
        "--rules", type=Path, default=None,
        help="룰 사전 경로. 생략하면 rules/v1.yaml",
    )
    ap.add_argument(
        "--transfer", action="store_true",
        help="§5.2 전이 실험 — 같은 탐지기를 여러 분류 결과 위에서 돌려 비교한다",
    )
    ap.add_argument(
        "--verdicts", type=Path, default=None,
        help="저장된 LLM 판정. 주면 전이 실험에 C(하이브리드) 줄이 추가된다 (모델 호출 없음)",
    )
    ap.add_argument("--show", type=int, default=0, help="알림 예시를 근거 문장과 함께 N건 출력")
    ap.add_argument("--json", type=Path, default=None, help="지표를 JSON 으로도 쓴다")
    args = ap.parse_args(argv)

    if not args.bundles.exists():
        ap.error(
            f"번들이 없다: {args.bundles} — `python -m txlens.generator --seed 42` 를 먼저 "
            "실행한다 (docs/recovery.md)"
        )

    payload: dict[str, object]
    if args.transfer:
        rows = run_transfer(
            bundle_dir=args.bundles,
            truth_path=args.truth,
            dictionary_path=args.rules,
            verdicts_path=args.verdicts,
        )
        for row in rows:
            print(row.detection.report())
            print()
        print(transfer_table(rows))
        payload = {"transfer": [r.detection.to_dict() for r in rows]}
    else:
        dictionary = load_dictionary(args.rules) if args.rules else load_dictionary()
        _result, _metrics, state = run_baseline_state(
            label="A · 룰만",
            dictionary=dictionary,
            bundle_dir=args.bundles,
            truth_path=args.truth,
        )
        det = detect(state["confirmed"])
        result = evaluate_detections(det, load_truth(args.truth), label=f"사전 v{dictionary.version}")
        print(f"■ 탐지 결과 — {det.summary()}")
        print()
        print(result.report())
        if args.show:
            _show_examples(det, args.show)
        payload = {"detection": result.to_dict()}

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
