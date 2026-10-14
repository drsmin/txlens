"""파이프라인을 끝까지 돌려 지표를 낸다 — 베이스라인을 **한 함수 호출로** 재현하기 위한 층.

베이스라인 비교(A 룰만 / B LLM만 / C 하이브리드)의 신뢰는 "같은 입력·같은 모수"에서 나온다.
그래서 셋이 모두 **같은 그래프**를 돌고(`txlens.graph.run_pipeline`), 바뀌는 것은 주입물 둘
(사전·분류기)뿐이다. 노트북·Streamlit·CLI 가 모두 이 함수를 호출한다 — 로직을 세 곳에 복사하지
않는다.

## (B) 는 사전을 도구에서도 뺀다

"LLM 만"이라면서 `lookup_alias` 도구가 v1 사전을 들여다보면 그건 이미 하이브리드다. 그래서
`build_classifier(dictionary=EMPTY_DICTIONARY)` 로 도구까지 빈 사전을 보게 한다. 이 한 줄이
B/C 비교를 의미 있게 만든다.

## 모델을 인자로 받는다

API 키가 없는 환경에서도 B/C 의 **코드 경로**는 검증되어야 한다. 그래서 모델은 문자열이든
`BaseChatModel` 이든 받아 넘기고, 테스트는 스크립트된 모델을 넣어 같은 함수를 돌린다.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from txlens.agents import MerchantClassifier, build_classifier
from txlens.eval.metrics import EvalResult, evaluate
from txlens.graph import RunMetrics, TxState, run_pipeline
from txlens.graph.nodes import DEFAULT_CONFIDENCE_TAU
from txlens.ground_truth import GroundTruth
from txlens.ingest import IngestReport, load_dataset
from txlens.preprocess import preprocess
from txlens.rules import RuleDictionary, RuleMatcher, load_dictionary
from txlens.rules.promote import MIN_PROMOTE_CONFIDENCE
from txlens.schemas import NormalizedTx

#: 리포 루트 기준 기본 경로. src/txlens/eval/harness.py → 부모 3단계가 루트.
_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE_DIR = _ROOT / "data" / "synthetic" / "bundle"
DEFAULT_TRUTH_PATH = _ROOT / "data" / "synthetic" / "ground_truth.json"

#: 베이스라인 (B) 용 빈 사전. `version=0` 이라 리포트에 "사전 없음"으로 찍힌다.
EMPTY_DICTIONARY = RuleDictionary(version=0, entries=[])


def load_normalized(
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
) -> tuple[list[NormalizedTx], IngestReport]:
    """번들 → `RawTx` → `NormalizedTx`.

    그래프가 같은 일을 하지만, `ingest` 보고서(관용 모드에서 버린 건수)를 보려면 이 경로가
    필요하다. 그래프 상태에는 보고서를 담지 않는다 — 상태에 없는 것을 지표가 요구하지 않도록.
    """
    txs, report = load_dataset(bundle_dir)
    return preprocess(txs), report


def load_truth(truth_path: Path = DEFAULT_TRUTH_PATH) -> GroundTruth:
    return GroundTruth.model_validate(json.loads(truth_path.read_text(encoding="utf-8")))


def run_baseline(
    *,
    label: str,
    dictionary: RuleDictionary,
    classifier: MerchantClassifier | None = None,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    truth_path: Path = DEFAULT_TRUTH_PATH,
    tau: float = DEFAULT_CONFIDENCE_TAU,
    llm_target_limit: int | None = None,
) -> tuple[EvalResult, RunMetrics]:
    """베이스라인 하나를 돌린다. **정확도와 비용을 함께 돌려준다.**

    둘을 같이 내는 이유: 정확도만 보면 "LLM 을 더 부르면 더 낫다"는 결론밖에 안 나온다.
    이 프로젝트의 주장은 "룰로 잡을 것은 룰로 잡고, LLM 은 꼬리에만 쓴다"이므로 호출 수가
    지표의 절반이다.
    """
    result, metrics, _state = run_baseline_state(
        label=label,
        dictionary=dictionary,
        classifier=classifier,
        bundle_dir=bundle_dir,
        truth_path=truth_path,
        tau=tau,
        llm_target_limit=llm_target_limit,
    )
    return result, metrics


def run_baseline_state(
    *,
    label: str,
    dictionary: RuleDictionary,
    classifier: MerchantClassifier | None = None,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    truth_path: Path = DEFAULT_TRUTH_PATH,
    tau: float = DEFAULT_CONFIDENCE_TAU,
    llm_target_limit: int | None = None,
    min_promote_confidence: float = MIN_PROMOTE_CONFIDENCE,
) -> tuple[EvalResult, RunMetrics, TxState]:
    """`run_baseline` 과 같은 실행이되 **최종 상태까지** 돌려준다.

    승격 루프(`txlens.promote`)는 지표가 아니라 상태 안의 `rule_proposals` 를 가져가야 한다.
    그 한 필드 때문에 파이프라인 실행을 한 벌 더 만들면 라운드와 베이스라인이 서로 다른 코드를
    도는 셈이 되어, 두 표의 숫자를 나란히 놓을 수 없다.
    """
    state = run_pipeline(
        bundle_dir,
        matcher=RuleMatcher(dictionary),
        classifier=classifier,
        tau=tau,
        llm_target_limit=llm_target_limit,
        min_promote_confidence=min_promote_confidence,
    )
    result = evaluate(
        state["confirmed"],
        load_truth(truth_path),
        label=label,
        dictionary_version=dictionary.version,
        all_normalized=state["normalized"],
    )
    return result, state["metrics"], state


def run_rules_baseline(
    *,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    truth_path: Path = DEFAULT_TRUTH_PATH,
    dictionary_path: Path | None = None,
    label: str = "A · 룰만",
) -> EvalResult:
    """베이스라인 (A) — 결정론 룰만. LLM 을 호출하지 않는다 (노드가 그래프에 없다)."""
    dictionary = load_dictionary(dictionary_path) if dictionary_path else load_dictionary()
    result, _metrics = run_baseline(label=label, dictionary=dictionary, bundle_dir=bundle_dir,
                                   truth_path=truth_path)
    return result


def run_llm_baseline(
    model: str | BaseChatModel,
    *,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    truth_path: Path = DEFAULT_TRUTH_PATH,
    label: str = "B · LLM만",
    tau: float = DEFAULT_CONFIDENCE_TAU,
    llm_target_limit: int | None = None,
) -> tuple[EvalResult, RunMetrics]:
    """베이스라인 (B) — 사전 없이 LLM 만. **도구도 빈 사전을 본다** (위 § 참고).

    PG 마스킹은 여기서도 단계 0 으로 갈라진다 — 그건 사전이 하는 일이 아니라 전처리가 확인한
    사실이라서다. 이걸 LLM 에 보내면 "실가맹점명이 아닌 문자열"을 억지로 분류시키는 셈이다.
    """
    classifier = build_classifier(model, dictionary=EMPTY_DICTIONARY)
    return run_baseline(
        label=label,
        dictionary=EMPTY_DICTIONARY,
        classifier=classifier,
        bundle_dir=bundle_dir,
        truth_path=truth_path,
        tau=tau,
        llm_target_limit=llm_target_limit,
    )


def run_hybrid_baseline(
    model: str | BaseChatModel,
    *,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    truth_path: Path = DEFAULT_TRUTH_PATH,
    dictionary_path: Path | None = None,
    label: str = "C · 하이브리드",
    tau: float = DEFAULT_CONFIDENCE_TAU,
    llm_target_limit: int | None = None,
) -> tuple[EvalResult, RunMetrics]:
    """베이스라인 (C) — 룰이 먼저, 미매칭만 LLM. 이 프로젝트가 주장하는 구성이다."""
    dictionary = load_dictionary(dictionary_path) if dictionary_path else load_dictionary()
    classifier = build_classifier(model, dictionary=dictionary)
    return run_baseline(
        label=label,
        dictionary=dictionary,
        classifier=classifier,
        bundle_dir=bundle_dir,
        truth_path=truth_path,
        tau=tau,
        llm_target_limit=llm_target_limit,
    )


def compare_table(rows: list[tuple[EvalResult, RunMetrics | None]]) -> str:
    """A/B/C 비교표 (설계 §5.1).

    **거래 기준과 정규화 키 기준을 나란히 둔다.** 하나만 내면 어느 쪽이든 사실을 왜곡한다 —
    건수로는 상위 브랜드가 대부분을 덮고, 표기 수로는 꼬리가 훨씬 길다. 그 격차가 이 프로젝트의
    논거이므로 비교표에서도 붙여 놓는다. 맨 오른쪽의 LLM 표기 수가 비용 축이다.
    """
    header = (
        f"{'베이스라인':<16} {'거래 L2':>8} {'표기 L2':>8} {'커버리지':>8} "
        f"{'미매칭':>7} {'PG':>6} {'LLM표기':>7} {'모델호출':>7}"
    )
    lines = ["■ 베이스라인 비교", "", "  " + header, "  " + "─" * len(header)]
    for result, metrics in rows:
        o = result.overall
        lines.append(
            f"  {result.label:<16} {o.accuracy_l2:>8.1%} {result.distinct.accuracy_l2:>8.1%} "
            f"{o.coverage:>8.1%} {o.unmatched_rate:>7.1%} {o.pg_masked_rate:>6.1%} "
            f"{(metrics.llm_targets if metrics else 0):>7} "
            f"{(metrics.llm_model_calls if metrics else 0):>7}"
        )
    lines += [
        "",
        "  · 커버리지 + 미매칭 + PG마스킹 = 100%. PG마스킹은 **원리적 한계**라 줄지 않는다.",
        "  · LLM 이 '모른다'고 답한 표기는 미매칭에 남는다 — 미분류를 커버리지로 세지 않는다.",
    ]
    skipped = [m for _r, m in rows if m and m.llm_targets_skipped]
    if skipped:
        lines.append(
            f"  · ⚠️ 상한 때문에 보내지 않은 표기가 있다 "
            f"({', '.join(str(m.llm_targets_skipped) for m in skipped)}개) — 부분 측정이다."
        )
    return "\n".join(lines)
