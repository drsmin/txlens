"""§5.2 1부 → 2부 전이 실험 — 둘을 묶는 논거 (D-010 · D-035).

## 무엇을 재는가

"가맹점명을 못 묶으면 구독도 못 찾는다"는 주장을 **말하지 않고 실험으로 확인한다.** 같은 탐지기
3종을 서로 다른 분류 결과 위에서 돌리고, 바뀌는 것은 분류 하나뿐이게 만든다.

| 분류 단계 | 무엇을 보는가 |
| --- | --- |
| 정규화만 | **1부를 끈 상태.** 사전도 LLM 도 없이 전처리만 통과한 이름으로 묶는다 |
| A · 룰만 | 사전이 못 잡은 표기가 그대로 흩어진 상태 |
| C · 하이브리드 | LLM 이 꼬리를 묶어 준 상태 |
| 정답 분류 | **탐지기 자체의 상한** — 여기서 못 잡는 것은 분류를 고쳐도 안 잡힌다 |

세 번째 줄이 없으면 A→C 의 개선폭을 해석할 수 없다. 60%가 80%로 오른 것이 좋은지는 상한이
82%인지 100%인지에 달려 있다.

베이스라인 B(LLM만)는 넣지 않는다. 사전을 빼면 분류만 달라지는 게 아니라 **어떤 표기가 LLM 에
가는지**가 달라져 "분류만 바꿨다"는 전제가 깨진다. B 의 자리는 §5.1 비교표다.

## 정답 분류는 "완벽한 1부" 이지 "완벽한 데이터"가 아니다

PG 경유로 실가맹점명이 오지 않은 거래는 정답도 미분류다(`pg_masked`). 정답 분류 줄에서도 그
거래는 묶이지 않는다 — 분류를 아무리 고쳐도 줄지 않는 몫이라 상한에 반영되는 것이 맞다.
"""

from __future__ import annotations

from pathlib import Path

from txlens.agents.replay import ReplayClassifier, load_verdicts
from txlens.detect import Detections, detect
from txlens.eval.detect_metrics import DetectEvalResult, evaluate_detections
from txlens.eval.harness import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_TRUTH_PATH,
    EMPTY_DICTIONARY,
    load_normalized,
    load_truth,
    run_baseline_state,
)
from txlens.eval.metrics import EvalResult, evaluate
from txlens.ground_truth import GroundTruth
from txlens.rules import load_dictionary
from txlens.rules.match import PG_MASKED_CANONICAL
from txlens.schemas import (
    CategoryL1,
    CategoryL2,
    ClassifiedTx,
    MatchStage,
    NormalizedTx,
)


def oracle_classified(
    normalized: list[NormalizedTx], truth: GroundTruth
) -> list[ClassifiedTx]:
    """정답 라벨을 그대로 분류 결과로 삼는다 — 1부가 완벽했다면 2부가 보게 될 입력.

    라벨이 없는 거래(정답셋에 없는 승인번호)는 **버리지 않고 미매칭으로 남긴다.** 버리면 모수가
    줄어 상한이 실제보다 좋아 보인다.
    """
    out: list[ClassifiedTx] = []
    for tx in normalized:
        label = truth.tx_labels.get(tx.raw.approval_no)
        if label is None:
            out.append(
                ClassifiedTx(
                    normalized=tx,
                    merchant_canonical=tx.normalized_name,
                    category_l1=CategoryL1.UNCLASSIFIED,
                    category_l2=CategoryL2.UNCLASSIFIED_INSUFFICIENT,
                    confidence=0.0,
                    match_stage=MatchStage.UNMATCHED,
                )
            )
            continue
        pg = label.pg_masked
        out.append(
            ClassifiedTx(
                normalized=tx,
                merchant_canonical=PG_MASKED_CANONICAL if pg else label.canonical,
                category_l1=label.category_l1,
                category_l2=label.category_l2,
                confidence=1.0,
                match_stage=MatchStage.PG_MASKED if pg else MatchStage.EXACT,
            )
        )
    return out



class TransferRow:
    """전이 실험 한 줄 — 같은 탐지기, 다른 분류."""

    def __init__(
        self,
        label: str,
        classification: EvalResult | None,
        detection: DetectEvalResult,
        detections: Detections | None = None,
        classified: list[ClassifiedTx] | None = None,
    ):
        self.label = label
        self.classification = classification
        self.detection = detection
        #: 지표를 내기 전의 **원본 탐지 결과**. 화면(S4)이 구독·알림 목록을 그리는 데 쓴다.
        #: 지표만 남기고 버리면 화면이 같은 네 단계를 다시 계산하게 되고, 그 순간 표의 숫자와
        #: 목록이 서로 다른 코드에서 나온다 (D-018 과 같은 이유).
        self.detections = detections
        #: 그 탐지가 실제로 본 분류 결과. 알림률의 **분모**가 여기서 나온다.
        self.classified = classified

    @property
    def unmatched_rate(self) -> float:
        """미매칭 + PG마스킹 = 브랜드를 묶지 못한 비율. 탐지가 실제로 겪는 손실이다."""
        if self.classification is None:
            return 0.0
        o = self.classification.overall
        return o.unmatched_rate + o.pg_masked_rate


def transfer_table(rows: list[TransferRow]) -> str:
    """§5.2 표. **분류 지표와 탐지 지표를 한 줄에 둔다** — 인과를 눈으로 잇게 하려는 배치다."""
    header = (
        f"{'분류 단계':<16} {'미묶음':>7} {'구독 P/R':>15} {'이상지출 P/R':>15} "
        f"{'이상거래 P/R':>15} {'알림/월':>8}"
    )
    lines = ["■ §5.2 분류 → 탐지 전이 실험", "", "  " + header, "  " + "─" * len(header)]
    for row in rows:
        d = row.detection
        lines.append(
            f"  {row.label:<16} {row.unmatched_rate:>7.1%} "
            f"{d.subscription.precision:>6.1%}/{d.subscription.recall:<8.1%} "
            f"{d.spending.precision:>6.1%}/{d.spending.recall:<8.1%} "
            f"{d.tx_pattern.precision:>6.1%}/{d.tx_pattern.recall:<8.1%} "
            f"{d.alerts_per_customer_month:>8.1f}"
        )
    lines += [
        "",
        "  · '미묶음' = 미매칭 + PG마스킹. 대표명이 없어 그룹이 쪼개지는 비율이다.",
        "  · 탐지기는 모든 줄에서 **완전히 같다.** 바뀌는 것은 그 위에 놓인 분류뿐이다.",
        "  · '정답 분류' 는 1부가 완벽할 때의 상한이다 — 그 아래 칸은 탐지기 자체의 한계다.",
        "  · 이상지출은 급증·고액을 합친 값이다. 쪼갠 성적은 각 줄의 탐지 리포트에 있다.",
    ]
    return "\n".join(lines)


def run_transfer(
    *,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    truth_path: Path = DEFAULT_TRUTH_PATH,
    dictionary_path: Path | None = None,
    verdicts_path: Path | None = None,
) -> list[TransferRow]:
    """A · (판정이 있으면) C · 정답 분류 세 줄을 만든다.

    C 는 저장된 판정을 재생해 만든다 — 전이 실험은 **분류 결과가 무엇이었는지**만 필요하고
    그건 이미 파일에 있다. 문턱 실험과 같은 이유로 여기서도 모델을 다시 부르지 않는다.
    """
    truth = load_truth(truth_path)
    dictionary = load_dictionary(dictionary_path) if dictionary_path else load_dictionary()
    rows: list[TransferRow] = []

    result_n, _m, state_n = run_baseline_state(
        label="정규화만", dictionary=EMPTY_DICTIONARY, bundle_dir=bundle_dir, truth_path=truth_path
    )
    rows.append(_row("정규화만", result_n, detect(state_n["confirmed"]), truth, state_n["confirmed"]))

    result_a, _m, state_a = run_baseline_state(
        label="A · 룰만", dictionary=dictionary, bundle_dir=bundle_dir, truth_path=truth_path
    )
    rows.append(_row("A · 룰만", result_a, detect(state_a["confirmed"]), truth, state_a["confirmed"]))

    if verdicts_path is not None:
        verdicts, _model = load_verdicts(verdicts_path)
        result_c, _m, state_c = run_baseline_state(
            label="C · 하이브리드",
            dictionary=dictionary,
            classifier=ReplayClassifier(verdicts),
            bundle_dir=bundle_dir,
            truth_path=truth_path,
        )
        rows.append(_row("C · 하이브리드", result_c, detect(state_c["confirmed"]), truth, state_c["confirmed"]))

    normalized, _report = load_normalized(bundle_dir)
    oracle = oracle_classified(normalized, truth)
    result_o = evaluate(
        oracle, truth, label="정답 분류", dictionary_version=0, all_normalized=normalized
    )
    rows.append(_row("정답 분류(상한)", result_o, detect(oracle), truth, oracle))
    return rows


def _row(
    label: str,
    classification: EvalResult,
    det: Detections,
    truth: GroundTruth,
    classified: list[ClassifiedTx],
) -> TransferRow:
    return TransferRow(
        label,
        classification,
        evaluate_detections(det, truth, label=label),
        detections=det,
        classified=classified,
    )

