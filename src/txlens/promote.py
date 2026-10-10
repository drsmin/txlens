"""승격 루프 드라이버 — `python -m txlens.promote` (blueprint §3.3 ⑤).

    python -m txlens.promote --rounds 3 --model openai:gpt-4o-mini
    python -m txlens.promote --rounds 3 --dry-run      # 사전을 쓰지 않고 표만 본다

한 라운드 = 파이프라인 한 번 + 고신뢰 판정의 사전 승격. 다음 라운드는 커진 사전으로 같은
데이터를 다시 돌린다. 증명하려는 것은 하나다 — **라운드가 돌면 LLM 에 보낼 표기가 줄어든다.**

## 루프는 여기 있고, 그래프는 DAG 로 둔다

TxLens 에서 되도는 것은 이 루프 하나뿐이고, 그래프 안에는 사이클이 없다. 그래서 폭주를 막는
장치가 **층마다 다른 물건**이다. 발표에서 "무한루프를 어떻게 막느냐"는 질문에 세 줄로 답한다.

| 층 | 되도는 것 | 막는 장치 |
| --- | --- | --- |
| Agent 안 (`llm_classify`) | 도구 ↔ 모델 왕복 | `ToolCallLimit`·`ModelCallLimit` (`run_limit`, `exit_behavior="end"`) |
| 그래프 | 없음 — DAG 다 | 사이클을 만들지 않는 것 자체가 장치다 |
| 라운드 (이 파일) | 승격 → 재실행 | **고정점** + 라운드 상한 |

## 고정점이 진짜 종료 조건이고, 상한은 안전핀이다

상한만 두면 "3라운드 돌았다"는 말밖에 못 한다. 이 루프가 멈추는 이유는 셋이고, 어느 것으로
멈췄는지를 표에 적는다 — 멈춘 이유가 곧 결과다.

1. **승격할 후보가 없다** — 남은 미매칭이 전부 저신뢰·추측·미분류다. 자동으로 더 갈 곳이 없고
   여기서부터가 사람이 볼 자리다(5주차 HITL). 이 루프가 HITL 을 정당화하는 방식이다.
2. **진행이 없다** — 승격을 했는데 LLM 표기 수가 줄지 않았다. 사전이 커졌는데 미매칭이 그대로면
   승격된 키가 실제로는 아무 거래도 덮지 못했다는 뜻이다.
3. **라운드 상한** — 위 둘이 안 걸렸는데도 도는 경우를 자르는 안전핀. 여기서 멈췄다면 **아직
   수렴하지 않은 것**이므로 표에 그렇게 적는다. 수렴과 중단을 같은 말로 보고하지 않는다.

라운드마다 API 비용이 실제로 나가므로 상한의 기본값은 낮게 둔다. `--dry-run` 은 사전 파일만
쓰지 않을 뿐 모델은 부른다 — 비용이 걱정이면 `--llm-limit` 으로 표기 수를 자른다.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from txlens.agents import build_classifier
from txlens.agents.replay import ReplayClassifier, load_verdicts, save_verdicts
from txlens.console import use_utf8_output
from txlens.eval.harness import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_TRUTH_PATH,
    run_baseline_state,
)
from txlens.eval.metrics import EvalResult, GoldKey
from txlens.graph import RunMetrics
from txlens.graph.nodes import DEFAULT_CONFIDENCE_TAU
from txlens.rules import DEFAULT_DICTIONARY_PATH, RuleDictionary, load_dictionary
from txlens.rules.promote import (
    MIN_PROMOTE_CONFIDENCE,
    apply_proposals,
    dump_dictionary,
)
from txlens.schemas import CategoryL1, Evidence, LLMTarget, LLMVerdict

#: 라운드 상한의 기본값. 라운드마다 모델을 부르므로 "일단 크게"가 곧 비용이다.
DEFAULT_MAX_ROUNDS = 3

STOP_NO_CANDIDATES = "고정점 — 승격할 후보가 없다"
STOP_NO_PROGRESS = "고정점 — 승격했지만 LLM 표기가 줄지 않았다"
STOP_MAX_ROUNDS = "라운드 상한 — 아직 수렴하지 않았다"


@dataclass(frozen=True)
class RoundReport:
    """라운드 하나의 결과. **비용과 정확도를 같은 줄에 둔다.**

    승격은 정확도를 올리려는 장치가 아니라 **호출을 줄이는** 장치다. 두 수를 떼어 놓으면
    "호출이 줄었다"는 자랑과 "정확도가 떨어졌다"는 사실이 다른 표에 실려 서로 안 만난다.
    """

    round: int
    result: EvalResult
    metrics: RunMetrics
    #: 이 라운드의 판정에서 사전으로 받아들여진 후보 수 (거른 것은 빠져 있다).
    promoted: int
    #: 승격이 덮는 거래 건수. 표기 수만 보면 꼬리 하나와 대형 브랜드가 같은 무게로 보인다.
    promoted_tx: int
    #: 이 라운드가 남긴 미매칭 고유 표기 **총수** — `--llm-limit` 으로 자르기 **전**의 수다.
    #: 진행 판정을 보낸 수(`metrics.llm_targets`)로 하면 상한이 걸린 순간 그 값이 상한에
    #: 붙박여, 사전이 커지고 있는데도 "진행 없음"으로 멈춘다.
    pending_targets: int
    dictionary_version: int
    dictionary_keys: int
    stop_reason: str | None = None


def run_rounds(
    model: str | BaseChatModel,
    *,
    dictionary: RuleDictionary | None = None,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    truth_path: Path = DEFAULT_TRUTH_PATH,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    min_confidence: float = MIN_PROMOTE_CONFIDENCE,
    tau: float = DEFAULT_CONFIDENCE_TAU,
    llm_target_limit: int | None = None,
    out_dir: Path | None = None,
    replay: Mapping[str, LLMVerdict] | None = None,
    verdicts_out: Path | None = None,
) -> list[RoundReport]:
    """승격 루프를 고정점까지 돌린다. `out_dir` 이 없으면 사전을 파일로 쓰지 않는다.

    라운드마다 사전을 새로 만들어 **다음 라운드에 주입**한다. 전역 상태를 고치지 않으므로
    같은 프로세스에서 v1 베이스라인을 다시 돌려도 오염되지 않는다.

    `replay` 를 주면 모델을 부르지 않고 저장된 판정으로 돈다 (`agents/replay.py`). 문턱을
    바꿔 가며 루프를 다시 도는 일이 **공짜**가 된다. 다만 실호출과 완전히 같지는 않다 —
    라운드 2 의 Agent 는 커진 사전을 도구로 조회하므로 실제로는 다른 답을 낼 수 있다.
    재생으로 고른 문턱은 마지막에 실호출로 한 번 확인한다.
    """
    current = dictionary if dictionary is not None else load_dictionary()
    reports: list[RoundReport] = []
    previous_targets: int | None = None
    collected: dict[str, LLMVerdict] = {}

    for round_no in range(1, max_rounds + 1):
        # 사전이 라운드마다 바뀌므로 분류기도 매번 새로 만든다 — 도구가 옛 사전을 들고 있으면
        # "사전이 커졌는데 LLM 이 그걸 모르는" 상태가 되어 승격 효과가 과소평가된다.
        classifier = (
            ReplayClassifier(replay) if replay is not None else build_classifier(model, dictionary=current)
        )
        result, metrics, state = run_baseline_state(
            label=f"라운드 {round_no} · v{current.version}",
            dictionary=current,
            classifier=classifier,
            bundle_dir=bundle_dir,
            truth_path=truth_path,
            tau=tau,
            llm_target_limit=llm_target_limit,
            min_promote_confidence=min_confidence,
        )

        collected.update(state.get("llm_results", {}))
        promoted_dict, accepted = apply_proposals(current, state.get("rule_proposals", []))
        pending = metrics.llm_targets + metrics.llm_targets_skipped

        if not accepted:
            stop = STOP_NO_CANDIDATES
        elif previous_targets is not None and pending >= previous_targets:
            stop = STOP_NO_PROGRESS
        elif round_no == max_rounds:
            stop = STOP_MAX_ROUNDS
        else:
            stop = None

        reports.append(
            RoundReport(
                round=round_no,
                result=result,
                metrics=metrics,
                promoted=len(accepted),
                promoted_tx=sum(p.covers_tx_count for p in accepted),
                pending_targets=pending,
                dictionary_version=current.version,
                dictionary_keys=len(current.exact_index) + len(current.alias_index),
                stop_reason=stop,
            )
        )
        if stop is not None:
            break

        if out_dir is not None:
            dump_dictionary(
                promoted_dict,
                out_dir / f"v{promoted_dict.version}.yaml",
                parent_version=current.version,
                accepted=accepted,
                model=model if isinstance(model, str) else type(model).__name__,
            )
        previous_targets = pending
        current = promoted_dict

    if verdicts_out is not None and replay is None:
        save_verdicts(
            collected,
            verdicts_out,
            model=model if isinstance(model, str) else type(model).__name__,
        )
    return reports


def rounds_table(reports: list[RoundReport]) -> str:
    """라운드 표 — 이 루프가 발표에서 말하는 전부다.

    맨 왼쪽이 사전 크기, 가운데가 비용(LLM 표기·모델 호출), 오른쪽이 정확도다. 왼쪽이 커질 때
    가운데가 줄고 오른쪽이 유지되면 주장이 선 것이고, 오른쪽이 내려가면 **자동 승격이 오답을
    굳혔다**는 뜻이라 사람 검토가 필요하다는 근거가 된다. 어느 쪽이든 결과다.
    """
    header = (
        f"{'라운드':<8} {'사전':>6} {'키':>6} {'미매칭표기':>9} {'LLM표기':>7} {'모델호출':>7} "
        f"{'거래 L2':>8} {'표기 L2':>8} {'미매칭':>7} {'승격':>6} {'덮는거래':>8}"
    )
    lines = ["■ 승격 루프", "", "  " + header, "  " + "─" * len(header)]
    for r in reports:
        o = r.result.overall
        lines.append(
            f"  {r.round:<8} {('v' + str(r.dictionary_version)):>6} {r.dictionary_keys:>6} "
            f"{r.pending_targets:>9} {r.metrics.llm_targets:>7} {r.metrics.llm_model_calls:>7} "
            f"{o.accuracy_l2:>8.1%} {r.result.distinct.accuracy_l2:>8.1%} "
            f"{o.unmatched_rate:>7.1%} {r.promoted:>6} {r.promoted_tx:>8}"
        )

    stop = reports[-1].stop_reason if reports else None
    lines += ["", f"  · 멈춘 이유: {stop}"]
    # 상한에서 멈추면 그 라운드의 승격은 **버려진다** — 그 사전으로 잰 지표가 없어 파일로 쓰지
    # 않기 때문이다(의도된 동작). 조용히 버리면 호출 비용을 내고 아무 산출물도 못 받는다.
    if stop == STOP_MAX_ROUNDS and reports[-1].promoted:
        lines.append(
            f"  · ⚠️ 상한에서 멈춰 마지막 라운드의 승격 {reports[-1].promoted}건을 "
            f"**파일로 쓰지 않았다** — 그 사전으로 잰 지표가 없기 때문이다. "
            f"이 승격을 받으려면 `--rounds {len(reports) + 1}` 로 한 라운드 더 돌린다 "
            f"(판정이 저장돼 있으면 `--from-verdicts` 로 호출 0)."
        )
    if len(reports) > 1:
        first, last = reports[0], reports[-1]
        if first.pending_targets:
            cut = 1 - last.pending_targets / first.pending_targets
            lines.append(
                f"  · 미매칭 표기 {first.pending_targets} → {last.pending_targets} "
                f"({cut:.1%} 감소) · 모델 호출 {first.metrics.llm_model_calls} → "
                f"{last.metrics.llm_model_calls}"
            )
    lines.append(
        "  · 승격은 정확도를 올리는 장치가 아니라 **호출을 줄이는** 장치다 — 거래 L2 가 함께 "
        "떨어지면 자동 승격이 오답을 굳힌 것이고, 그게 HITL 이 필요한 이유다."
    )
    return "\n".join(lines)


#: 문턱 후보. `MIN_PROMOTE_CONFIDENCE` 를 정하려고 훑는 값이고, 정하고 나면 근거로 남는다.
CALIBRATION_CUTS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def calibration_report(
    targets: Sequence[LLMTarget],
    verdicts: Mapping[str, LLMVerdict],
    gold: Mapping[str, GoldKey],
) -> str:
    """승격 문턱을 **정답셋으로** 정하기 위한 표 (D-033).

    문턱을 감으로 정하면 두 방향으로 틀린다. 높으면 정답이 사전에 못 올라가고(승격 0건), 낮으면
    오답이 검토 없이 굳는다. 우리에겐 골드셋이 있으므로 각 문턱에서 **승격될 것들이 실제로 몇
    퍼센트나 맞는지**를 세어 정할 수 있다.

    정밀도만 보고 정하지 않는다 — 문턱을 1.0 으로 두면 정밀도는 정의되지 않고 승격은 0 이다.
    같은 줄에 **몇 개가·몇 건을 덮으며** 올라가는지를 함께 둔다.
    """
    tx_count = {t.normalized_name: t.tx_count for t in targets}
    rows: list[tuple[str, LLMVerdict, bool, int]] = []
    for name, verdict in verdicts.items():
        answer = gold.get(name)
        if answer is None:
            continue
        rows.append((name, verdict, verdict.category_l2 is answer.category_l2, tx_count.get(name, 0)))

    lines = ["■ 승격 문턱 보정 (정답셋 대조)", "", f"  판정 {len(rows)}개 · 근거별"]
    header = f"    {'근거':<14}{'개수':>6}{'L2 정답':>8}{'정밀도':>8}{'신뢰도 범위':>16}"
    lines += [header, "    " + "─" * (len(header) - 4)]
    for evidence in Evidence:
        group = [r for r in rows if r[1].evidence is evidence]
        if not group:
            continue
        ok = sum(1 for r in group if r[2])
        confs = [r[1].confidence for r in group]
        lines.append(
            f"    {evidence.value:<14}{len(group):>6}{ok:>8}{ok / len(group):>8.1%}"
            f"{f'{min(confs):.2f}~{max(confs):.2f}':>16}"
        )

    lines += ["", "  문턱별 (근거 guess·미분류는 이미 제외한 뒤)"]
    header = f"    {'문턱':<8}{'승격 후보':>10}{'맞음':>7}{'정밀도':>8}{'덮는 거래':>10}{'남는 표기':>10}"
    lines += [header, "    " + "─" * (len(header) - 4)]
    eligible = [
        r
        for r in rows
        if r[1].evidence is not Evidence.GUESS
        and r[1].category_l1 is not CategoryL1.UNCLASSIFIED
    ]
    for cut in CALIBRATION_CUTS:
        group = [r for r in eligible if r[1].confidence >= cut]
        ok = sum(1 for r in group if r[2])
        covered = sum(r[3] for r in group)
        precision = f"{ok / len(group):.1%}" if group else "—"
        lines.append(
            f"    {cut:<8.2f}{len(group):>10}{ok:>7}{precision:>8}{covered:>10}"
            f"{len(rows) - len(group):>10}"
        )

    wrong = [r for r in eligible if not r[2]]
    if wrong:
        lines += ["", "  틀린 판정 (승격되면 오답이 사전에 굳는다)"]
        for name, verdict, _ok, count in sorted(wrong, key=lambda r: -r[1].confidence)[:10]:
            answer = gold[name]
            lines.append(
                f"    conf {verdict.confidence:.2f} · {name} → {verdict.category_l2.value} "
                f"(정답 {answer.category_l2.value} · {count}건)"
            )
    return "\n".join(lines)


def calibrate(
    verdicts: Mapping[str, LLMVerdict],
    *,
    dictionary: RuleDictionary,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    truth_path: Path = DEFAULT_TRUTH_PATH,
) -> str:
    """저장된 판정으로 문턱 보정표를 만든다. **모델을 부르지 않는다.**

    대상 목록과 정답은 결정론 경로에서 얻는다 — 룰만 돌린 뒤(`classifier=None`) 남은 미매칭이
    곧 LLM 이 받았던 표기이고, 정답은 그 키에 속한 거래의 최빈값이다(`gold_by_key`).
    """
    from txlens.eval.harness import load_truth
    from txlens.eval.metrics import gold_by_key
    from txlens.graph import run_pipeline
    from txlens.rules import RuleMatcher

    state = run_pipeline(bundle_dir, matcher=RuleMatcher(dictionary))
    gold = gold_by_key(state["matched"], load_truth(truth_path))
    return calibration_report(state["llm_targets"], verdicts, gold)


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()
    ap = argparse.ArgumentParser(description="TxLens 사전 승격 루프")
    ap.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_PATH)
    ap.add_argument("--model", default="openai:gpt-4o-mini", help="langchain 형식: provider:model")
    ap.add_argument(
        "--rules", type=Path, default=None,
        help=f"출발 사전. 생략하면 {DEFAULT_DICTIONARY_PATH.name}",
    )
    ap.add_argument(
        "--rounds", type=int, default=DEFAULT_MAX_ROUNDS,
        help=f"라운드 상한 (기본 {DEFAULT_MAX_ROUNDS}). 고정점에 닿으면 그 전에 멈춘다",
    )
    ap.add_argument(
        "--min-confidence", type=float, default=MIN_PROMOTE_CONFIDENCE,
        help=f"자동 승격 하한 (기본 {MIN_PROMOTE_CONFIDENCE}). 낮추면 오답이 사전에 굳는다",
    )
    ap.add_argument(
        "--llm-limit", type=int, default=None,
        help="라운드당 LLM 에 보낼 표기 수 상한 (비용 통제). 자른 몫은 지표에 함께 보고된다",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="사전 파일을 쓰지 않는다. 모델은 부르므로 비용은 그대로 든다",
    )
    ap.add_argument(
        "--verdicts-json", type=Path, default=None,
        help="이 실행의 LLM 판정을 JSON 으로 남긴다. 문턱을 다시 고를 때 재결제하지 않으려고 쓴다",
    )
    ap.add_argument(
        "--from-verdicts", type=Path, default=None,
        help="저장된 판정으로 돈다 (모델 호출 0). 문턱만 바꿔 루프를 다시 보는 용도",
    )
    ap.add_argument(
        "--calibrate", action="store_true",
        help="문턱 보정표만 낸다 — 각 문턱에서 승격될 것들이 정답셋과 얼마나 맞는지 (D-033)",
    )
    args = ap.parse_args(argv)

    load_dotenv()
    replay = None
    if args.from_verdicts is not None:
        replay, replay_model = load_verdicts(args.from_verdicts)
        print(f"  판정 재생: {args.from_verdicts} ({len(replay)}개 · {replay_model})\n")
    elif not os.getenv("OPENAI_API_KEY"):
        ap.error(
            "OPENAI_API_KEY 가 없어 승격 루프를 돌릴 수 없다 — 라운드마다 LLM 판정이 필요하다. "
            ".env 에 키를 넣어라 (docs/recovery.md). 저장된 판정이 있으면 --from-verdicts"
        )

    dictionary = load_dictionary(args.rules) if args.rules else load_dictionary()
    out_dir = None if args.dry_run else (args.rules.parent if args.rules else
                                         DEFAULT_DICTIONARY_PATH.parent)

    if args.calibrate:
        if replay is None:
            ap.error("--calibrate 는 --from-verdicts 와 함께 쓴다 — 판정 없이는 잴 것이 없다")
        print(calibrate(replay, dictionary=dictionary, bundle_dir=args.bundles,
                        truth_path=args.truth))
        return 0

    reports = run_rounds(
        args.model,
        dictionary=dictionary,
        bundle_dir=args.bundles,
        truth_path=args.truth,
        max_rounds=args.rounds,
        min_confidence=args.min_confidence,
        llm_target_limit=args.llm_limit,
        out_dir=out_dir,
        replay=replay,
        verdicts_out=args.verdicts_json,
    )

    for report in reports:
        print(report.result.report())
        print()
        print(report.metrics.report())
        print()
    print(rounds_table(reports))
    if args.verdicts_json is not None and replay is None:
        print(f"\n  → {args.verdicts_json} (판정 저장 — 다음엔 --from-verdicts 로 공짜다)")
    if out_dir is not None and len(reports) > 1:
        print(f"\n  → {out_dir}/v{reports[-1].dictionary_version}.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
