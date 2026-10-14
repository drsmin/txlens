"""`python -m txlens.eval` — 베이스라인 지표를 낸다.

    python -m txlens.eval                                    # (A) 룰만 — 키가 필요 없다
    python -m txlens.eval --baseline c --model openai:gpt-4o-mini
    python -m txlens.eval --baseline all --model openai:gpt-4o-mini --llm-limit 30

데이터가 없으면 `python -m txlens.generator --seed 42` 를 먼저 실행한다 (README 참고).
B·C 는 실제 모델을 호출하므로 `.env` 의 `OPENAI_API_KEY` 가 필요하다. 키가 없으면 **조용히 A 로
떨어지지 않고** 무엇이 없어서 못 돌리는지 말하고 멈춘다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from txlens.console import use_utf8_output
from txlens.eval.harness import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_TRUTH_PATH,
    compare_table,
    run_hybrid_baseline,
    run_llm_baseline,
    run_rules_baseline,
)

#: 모델 문자열의 기본값. 지정하지 않으면 이 모델로 B·C 를 돌린다 (설계 §3.3 ③).
DEFAULT_MODEL = "openai:gpt-4o-mini"


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()
    ap = argparse.ArgumentParser(description="TxLens 분류 베이스라인 평가")
    ap.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_PATH)
    ap.add_argument(
        "--baseline", choices=["a", "b", "c", "all"], default="a",
        help="a=룰만 · b=LLM만 · c=하이브리드 · all=셋 다 + 비교표. b/c/all 은 모델 호출이 있다",
    )
    ap.add_argument(
        "--model", default=None,
        help=f"LLM 스펙 (기본 {DEFAULT_MODEL}). langchain 형식: provider:model",
    )
    ap.add_argument(
        "--rules", type=Path, default=None,
        help="룰 사전 경로. 생략하면 rules/v1.yaml — 승격 사전(v2)과 비교할 때 쓴다",
    )
    ap.add_argument(
        "--llm-limit", type=int, default=None,
        help="LLM 에 보낼 고유 표기 수 상한 (비용 통제). 자른 몫은 지표에 함께 보고된다",
    )
    ap.add_argument(
        "--json", type=Path, default=None,
        help="지표를 JSON 으로도 쓴다. (B) 스냅샷만은 커밋한다 — 화면이 되살릴 수 없다",
    )
    args = ap.parse_args(argv)

    load_dotenv()  # .env 의 OPENAI_API_KEY 를 읽는다. 값은 절대 커밋하지 않는다.
    needs_model = args.baseline in {"b", "c", "all"}
    if needs_model and not os.getenv("OPENAI_API_KEY"):
        ap.error(
            "OPENAI_API_KEY 가 없어 베이스라인 B·C 를 돌릴 수 없다. "
            ".env 에 키를 넣어라 (README 참고). --baseline a 는 키 없이 돈다"
        )
    model = args.model or DEFAULT_MODEL

    rows = []
    if args.baseline in {"a", "all"}:
        rows.append((
            run_rules_baseline(
                bundle_dir=args.bundles, truth_path=args.truth, dictionary_path=args.rules
            ),
            None,
        ))
    if args.baseline in {"b", "all"}:
        rows.append(
            run_llm_baseline(
                model,
                bundle_dir=args.bundles,
                truth_path=args.truth,
                llm_target_limit=args.llm_limit,
            )
        )
    if args.baseline in {"c", "all"}:
        rows.append(
            run_hybrid_baseline(
                model,
                bundle_dir=args.bundles,
                truth_path=args.truth,
                dictionary_path=args.rules,
                llm_target_limit=args.llm_limit,
            )
        )

    for result, metrics in rows:
        print(result.report())
        if metrics is not None:
            print()
            print(metrics.report())
        print()
    if len(rows) > 1:
        print(compare_table(rows))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"result": result.to_dict(), "metrics": metrics.model_dump() if metrics else None}
            for result, metrics in rows
        ]
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
