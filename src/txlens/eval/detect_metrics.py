"""2부 지표 — 심은 패턴을 얼마나 찾았고 함정에 얼마나 걸렸나 (blueprint §3.6 · §5.2).

## 정답은 승인번호로 맞춘다

대표명이나 카테고리로 맞추면 **탐지가 맞았는지 분류가 맞았는지** 구분할 수 없다. 승인번호는
분류 품질과 무관하게 고정된 값이라 둘을 갈라 준다. 예외는 카테고리 급증 하나다 — 그건 정의상
"어느 카테고리의 어느 달"이 답이라, 카테고리가 틀리면 **틀린 것이 맞다**(D-035).

## recall 의 분모에서 원리적 불가능을 뺀다

6개월 창에서 연 구독은 1회만 나타나 "최소 3회" 규칙으로 못 잡는다. 정답셋이 `detectable=False`
로 표시해 둔 그 17건을 분모에 넣으면 점수가 구조적으로 깎이고, 규칙을 느슨하게 만들라는 잘못된
압력이 생긴다. 빼되 **몇 건을 왜 뺐는지 함께 보고한다** — 조용히 빼면 분모를 숨기는 것이다.

## precision 을 재려면 함정이 있어야 한다

recall 만 재면 "전부 이상"이라 답하는 탐지기가 만점을 받는다. 생성기는 그래서 오탐 함정을
심는다 — 매달 결제되지만 금액이 들쭉날쭉한 가맹점, 짧은 간격 2건이지만 금액이 다른 결제.
정답이 **"이상 아님"** 인 것들이라, 여기 걸린 건수를 따로 세어 보고한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from txlens.detect.schemas import Detections
from txlens.ground_truth import GroundTruth
from txlens.schemas import CategoryL1

#: 구독 하나를 "찾았다"고 인정할 승인번호 겹침 비율. 6회 중 2회만 묶은 그룹은 같은 구독을
#: 찾은 게 아니라 **쪼개진 조각**이므로 과반을 요구한다.
MATCH_OVERLAP = 0.5


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


@dataclass
class DetectorScore:
    """탐지기 하나의 성적."""

    name: str
    #: 정답과 짝지어진 탐지 수.
    matched: int = 0
    #: 탐지한 전체 수 (precision 의 분모).
    detected: int = 0
    #: 원리상 찾을 수 있는 정답 수 (recall 의 분모).
    detectable: int = 0
    #: 심었지만 규칙상 찾을 수 없는 정답 수. 분모에서 뺀 몫을 드러낸다.
    undetectable: int = 0
    #: 오탐 함정에 걸린 탐지 수. `detected - matched` 의 부분집합이다.
    decoy_hits: int = 0

    @property
    def precision(self) -> float:
        return self.matched / self.detected if self.detected else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.detectable if self.detectable else 0.0

    @property
    def f1(self) -> float:
        return _f1(self.precision, self.recall)


@dataclass
class DetectEvalResult:
    label: str
    subscription: DetectorScore = field(default_factory=lambda: DetectorScore("구독"))
    #: 이상지출을 급증·고액으로 쪼개 보고한다. 합치면 성적이 좋은 쪽이 나쁜 쪽을 가려서,
    #: 무엇을 고쳐야 하는지 표에서 읽을 수 없게 된다.
    surge: DetectorScore = field(default_factory=lambda: DetectorScore("이상지출·급증"))
    large: DetectorScore = field(default_factory=lambda: DetectorScore("이상지출·고액"))
    tx_pattern: DetectorScore = field(default_factory=lambda: DetectorScore("이상거래"))
    #: 고객당 월 알림 건수. 알림은 많을수록 좋은 게 아니라 **읽히지 않으면 없는 것과 같다**.
    alerts_per_customer_month: float = 0.0

    @property
    def scores(self) -> list[DetectorScore]:
        return [self.subscription, self.surge, self.large, self.tx_pattern]

    @property
    def spending(self) -> DetectorScore:
        """급증 + 고액을 합친 성적. §5.1 처럼 한 칸에 담아야 할 때만 쓴다."""
        merged = DetectorScore("이상지출")
        for s in (self.surge, self.large):
            merged.matched += s.matched
            merged.detected += s.detected
            merged.detectable += s.detectable
            merged.undetectable += s.undetectable
            merged.decoy_hits += s.decoy_hits
        return merged

    def report(self) -> str:
        header = (
            f"{'탐지기':<14} {'정밀도':>8} {'재현율':>8} {'F1':>8} "
            f"{'탐지':>6} {'정답':>6} {'불가':>6} {'함정':>6}"
        )
        lines = [f"■ 2부 탐지 지표 — {self.label}", "", "  " + header, "  " + "─" * len(header)]
        for s in self.scores:
            lines.append(
                f"  {s.name:<14} {s.precision:>8.1%} {s.recall:>8.1%} {s.f1:>8.1%} "
                f"{s.detected:>6} {s.detectable:>6} {s.undetectable:>6} {s.decoy_hits:>6}"
            )
        lines += [
            "",
            (
                f"  · 고객당 월 알림 {self.alerts_per_customer_month:.1f}건 "
                f"(이상지출 + 이상거래. 구독 목록은 알림이 아니라 조회 화면이다)"
            ),
            (
                "  · '불가' = 심었지만 규칙상 찾을 수 없는 정답 (연 구독 · 미분류 카테고리). "
                "재현율 분모에서 뺀 몫이다."
            ),
            "  · '함정' = 정답이 '이상 아님'인데 알림을 낸 건수. precision 이 도는 자리다.",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "alerts_per_customer_month": round(self.alerts_per_customer_month, 3),
            "detectors": {
                s.name: {
                    "precision": round(s.precision, 4),
                    "recall": round(s.recall, 4),
                    "f1": round(s.f1, 4),
                    "matched": s.matched,
                    "detected": s.detected,
                    "detectable": s.detectable,
                    "undetectable": s.undetectable,
                    "decoy_hits": s.decoy_hits,
                }
                for s in self.scores
            },
        }


def _overlaps(truth: set[str], found: set[str]) -> bool:
    """정답 승인번호의 과반을 덮었는가."""
    if not truth:
        return False
    return len(truth & found) / len(truth) >= MATCH_OVERLAP


def _score_subscriptions(det: Detections, truth: GroundTruth, score: DetectorScore) -> None:
    planted = [(set(s.approval_nos), s.detectable) for s in truth.subscriptions]
    score.detectable = sum(1 for _a, d in planted if d)
    score.undetectable = sum(1 for _a, d in planted if not d)
    score.detected = len(det.subscriptions)

    decoy_sets = [set(d.approval_nos) for d in truth.decoys if d.kind == "subscription_like"]
    claimed: set[int] = set()
    for found in det.subscriptions:
        found_set = set(found.approval_nos)
        hit = next(
            (
                i
                for i, (approvals, detectable) in enumerate(planted)
                if detectable and i not in claimed and _overlaps(approvals, found_set)
            ),
            None,
        )
        if hit is not None:
            claimed.add(hit)
            score.matched += 1
            continue
        # 연 구독처럼 원리상 못 잡는 정답을 우연히 잡은 경우는 오탐이 아니다 — 실제 구독이다.
        if any(_overlaps(a, found_set) for a, d in planted if not d):
            score.detected -= 1
            continue
        if any(_overlaps(d, found_set) for d in decoy_sets):
            score.decoy_hits += 1


def _score_surges(det: Detections, truth: GroundTruth, score: DetectorScore) -> None:
    """급증은 (고객 · 카테고리 · 월) 이 곧 답이다 — 카테고리가 틀리면 틀린 것이 맞다.

    정밀도는 **하한**이다. 생성기는 우연히 생긴 진짜 급증에 라벨을 붙이지 않으므로, 그걸 찾아낸
    알림도 여기서는 오탐으로 세어진다 (`detect/spending.py` 의 § 참고).
    """
    keys = {
        (t.customer_id, t.category_l1, t.month)
        for t in truth.spending_anomalies
        if t.kind == "category_surge" and t.category_l1 is not CategoryL1.UNCLASSIFIED
    }
    score.undetectable = sum(
        1
        for t in truth.spending_anomalies
        if t.kind == "category_surge" and t.category_l1 is CategoryL1.UNCLASSIFIED
    )
    score.detectable = len(keys)
    found = [a for a in det.spending_alerts if a.kind == "category_surge"]
    score.detected = len(found)

    seen: set[tuple[str, CategoryL1, str]] = set()
    for alert in found:
        key = (alert.customer_id, alert.category_l1, alert.month)
        if key in keys and key not in seen:
            seen.add(key)
            score.matched += 1


def _score_large(det: Detections, truth: GroundTruth, score: DetectorScore) -> None:
    """단발 고액은 승인번호가 곧 답이다 — 어느 거래였는지가 정답이다."""
    approvals = {
        no
        for t in truth.spending_anomalies
        if t.kind == "single_large" and t.category_l1 is not CategoryL1.UNCLASSIFIED
        for no in t.approval_nos
    }
    score.undetectable = sum(
        1
        for t in truth.spending_anomalies
        if t.kind == "single_large" and t.category_l1 is CategoryL1.UNCLASSIFIED
    )
    score.detectable = len(approvals)
    found = [a for a in det.spending_alerts if a.kind == "single_large"]
    score.detected = len(found)

    seen: set[str] = set()
    for alert in found:
        hit = next((no for no in alert.approval_nos if no in approvals and no not in seen), None)
        if hit is not None:
            seen.add(hit)
            score.matched += 1


def _score_tx_patterns(det: Detections, truth: GroundTruth, score: DetectorScore) -> None:
    planted = [(t.kind, set(t.approval_nos)) for t in truth.tx_anomalies]
    score.detectable = len(planted)
    score.detected = len(det.tx_alerts)

    decoy_sets = [set(d.approval_nos) for d in truth.decoys if d.kind == "normal_consecutive"]
    claimed: set[int] = set()
    for alert in det.tx_alerts:
        found = set(alert.approval_nos)
        hit = next(
            (
                i
                for i, (kind, approvals) in enumerate(planted)
                if kind == alert.kind and i not in claimed and _overlaps(approvals, found)
            ),
            None,
        )
        if hit is not None:
            claimed.add(hit)
            score.matched += 1
        elif any(_overlaps(d, found) for d in decoy_sets):
            score.decoy_hits += 1


def evaluate_detections(
    det: Detections, truth: GroundTruth, *, label: str = "탐지"
) -> DetectEvalResult:
    """탐지 결과를 정답셋과 대조한다. 정답은 생성기가 심으면서 적은 것이라 정의상 옳다."""
    result = DetectEvalResult(label=label)
    _score_subscriptions(det, truth, result.subscription)
    _score_surges(det, truth, result.surge)
    _score_large(det, truth, result.large)
    _score_tx_patterns(det, truth, result.tx_pattern)

    months = _month_span(truth)
    denominator = truth.customers * months
    result.alerts_per_customer_month = det.total_alerts / denominator if denominator else 0.0
    return result


def _month_span(truth: GroundTruth) -> int:
    start_y, start_m = (int(x) for x in truth.period_start[:7].split("-"))
    end_y, end_m = (int(x) for x in truth.period_end[:7].split("-"))
    return max(1, (end_y - start_y) * 12 + (end_m - start_m) + 1)
