"""분류 지표 — 베이스라인 A/B/C 를 **같은 자로** 재기 위한 계산기 (설계 §5.1).

## 이 모듈이 고집하는 세 가지

**① 모수를 숨기지 않는다.** 정확도의 분모가 무엇인지(전체 거래인가, 분류 대상인가, 정답이 있는
거래인가)를 결과에 함께 담는다. 분모가 조용히 줄면 정확도는 저절로 오른다.

**② 미분류를 두 종류로 나눈다.** `미매칭`(사전이 못 잡음 — 사전·LLM 으로 줄어든다)과
`PG 마스킹`(실가맹점명이 아예 오지 않음 — **무엇을 붙여도 줄지 않는다**)은 성질이 다르다.
합산하면 "개선 여지"를 실제보다 크게 보고하게 된다.

**③ 거래 기준과 정규화 키 기준을 함께 낸다.** 이 프로젝트의 주장이 "롱테일"이므로 둘의 격차가
곧 논거다 — 건수로는 상위 브랜드가 대부분을 덮지만, **표기의 수로는 룰이 손대지 못한 꼬리가
훨씬 길다.** 하나만 내면 어느 쪽이든 사실을 왜곡한다.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

from txlens.ground_truth import GroundTruth
from txlens.schemas import CategoryL2, ClassifiedTx, MatchStage, NormalizedTx

#: 브랜드를 확정한 단계들. `PG_MASKED`·`UNMATCHED` 는 제외 — 커버리지의 정의다.
RESOLVED_STAGES: frozenset[MatchStage] = frozenset({
    MatchStage.EXACT,
    MatchStage.REGNO,
    MatchStage.ALIAS,
    MatchStage.REGEX,
    MatchStage.FUZZY,
    MatchStage.LLM,
    MatchStage.HUMAN,
})


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass
class Slice:
    """한 축(전체 · 제공기관별)의 거래 기준 지표."""

    name: str
    #: 정답 라벨이 있는 거래 수 = 정확도의 분모. 라벨이 없는 거래는 세지 않고 따로 보고한다.
    labeled: int = 0
    unlabeled: int = 0
    correct_l1: int = 0
    correct_l2: int = 0
    by_stage: collections.Counter[str] = field(default_factory=collections.Counter)
    correct_by_stage: collections.Counter[str] = field(default_factory=collections.Counter)

    @property
    def accuracy_l1(self) -> float:
        return _rate(self.correct_l1, self.labeled)

    @property
    def accuracy_l2(self) -> float:
        return _rate(self.correct_l2, self.labeled)

    @property
    def resolved(self) -> int:
        return sum(c for s, c in self.by_stage.items() if MatchStage(s) in RESOLVED_STAGES)

    @property
    def coverage(self) -> float:
        """브랜드를 확정한 비율. **정확도와 다르다** — 확정했지만 틀린 것도 포함한다."""
        return _rate(self.resolved, self.labeled + self.unlabeled)

    @property
    def unmatched_rate(self) -> float:
        """사전이 못 잡은 비율. 사전·LLM·사람이 줄일 수 있는 몫이다."""
        return _rate(self.by_stage[MatchStage.UNMATCHED.value], self.labeled + self.unlabeled)

    @property
    def pg_masked_rate(self) -> float:
        """실가맹점명이 오지 않은 비율. **원리적 한계**로 따로 보고한다 (줄지 않는다)."""
        return _rate(self.by_stage[MatchStage.PG_MASKED.value], self.labeled + self.unlabeled)

    def add(self, *, stage: MatchStage, ok_l1: bool | None, ok_l2: bool | None) -> None:
        self.by_stage[stage.value] += 1
        if ok_l1 is None or ok_l2 is None:
            self.unlabeled += 1
            return
        self.labeled += 1
        self.correct_l1 += int(ok_l1)
        self.correct_l2 += int(ok_l2)
        self.correct_by_stage[stage.value] += int(ok_l2)

    def stage_accuracy(self, stage: str) -> float:
        return _rate(self.correct_by_stage[stage], self.by_stage[stage])

    def to_dict(self) -> dict[str, object]:
        """JSON 으로 낼 형태. **계산된 비율을 함께 담는다.**

        `dataclasses.asdict()` 를 쓰지 않는 이유가 둘 있다.
        ① `asdict` 는 `Counter` 를 **망가뜨린다** — dict 서브클래스를 `type(o)(쌍 이터러블)` 로
           재구성하므로 `Counter({"exact": 5474})` 가 `{("exact", 5474): 1}` 이 된다. 그러면 JSON
           직렬화가 "keys must be str" 로 죽는다 (실제로 전량 실행 끝에서 죽었다).
        ② `accuracy_l2`·`coverage` 는 property 라 `asdict` 가 **빠뜨린다** — 지표 파일에 지표가
           없고 원시 카운터만 남는다. 읽는 쪽이 비율을 다시 계산해야 하고, 그 계산이 어긋나면
           보고서마다 다른 숫자가 나온다.
        """
        return {
            "name": self.name,
            "labeled": self.labeled,
            "unlabeled": self.unlabeled,
            "accuracy_l1": self.accuracy_l1,
            "accuracy_l2": self.accuracy_l2,
            "coverage": self.coverage,
            "unmatched_rate": self.unmatched_rate,
            "pg_masked_rate": self.pg_masked_rate,
            "by_stage": dict(self.by_stage),
            "stage_accuracy": {s: self.stage_accuracy(s) for s in self.by_stage},
        }


@dataclass
class DistinctSlice:
    """정규화 키 기준 지표.

    키마다 가중치가 **같다.** 건수 가중이면 상위 브랜드가 지표를 덮어 롱테일이 안 보인다.
    한 키의 정답·예측은 그 키에 속한 거래의 **최빈값**으로 정한다 — 같은 정규화 키가 서로 다른
    브랜드를 가리키는 경우(오염이 심한 표기)가 실제로 있어, 그런 키는 최빈 정답으로 대표시킨다.
    """

    keys: int = 0
    correct_l2: int = 0
    unmatched: int = 0
    pg_masked: int = 0

    @property
    def accuracy_l2(self) -> float:
        return _rate(self.correct_l2, self.keys)

    @property
    def unmatched_rate(self) -> float:
        return _rate(self.unmatched, self.keys)

    def to_dict(self) -> dict[str, object]:
        return {
            "keys": self.keys,
            "correct_l2": self.correct_l2,
            "accuracy_l2": self.accuracy_l2,
            "unmatched": self.unmatched,
            "unmatched_rate": self.unmatched_rate,
            "pg_masked": self.pg_masked,
        }


@dataclass
class EvalResult:
    """베이스라인 하나의 측정 결과. `label` 로 A/B/C 를 구분한다."""

    label: str
    dictionary_version: int | None
    #: 모수 설명 — 전체 → 분류 대상 → 정답 있음. 어디서 줄었는지 보이게 남긴다.
    tx_total: int
    tx_classified: int
    excluded_reasons: collections.Counter[str] = field(default_factory=collections.Counter)
    overall: Slice = field(default_factory=lambda: Slice(name="전체"))
    distinct: DistinctSlice = field(default_factory=DistinctSlice)
    by_provider: dict[str, Slice] = field(default_factory=dict)
    #: (정답 L1, 예측 L1, 건수) — 예측이 틀린 조합만, 많은 것부터.
    confusions: list[tuple[str, str, int]] = field(default_factory=list)
    #: (정규화 표기, 거래 건수, 정답 대표명) — 사전에 무엇을 넣어야 하는지의 미리보기.
    unmatched_top: list[tuple[str, int, str]] = field(default_factory=list)

    @property
    def tx_excluded(self) -> int:
        return self.tx_total - self.tx_classified

    def to_dict(self) -> dict[str, object]:
        """지표 파일 하나로 베이스라인 하나를 설명할 수 있게 담는다 (`Slice.to_dict` 참고)."""
        return {
            "label": self.label,
            "dictionary_version": self.dictionary_version,
            "tx_total": self.tx_total,
            "tx_classified": self.tx_classified,
            "tx_excluded": self.tx_excluded,
            "excluded_reasons": dict(self.excluded_reasons),
            "overall": self.overall.to_dict(),
            "distinct": self.distinct.to_dict(),
            "by_provider": {name: s.to_dict() for name, s in sorted(self.by_provider.items())},
            "confusions": [
                {"gold_l1": g, "pred_l1": p, "count": c} for g, p, c in self.confusions
            ],
            "unmatched_top": [
                {"normalized_name": n, "tx_count": c, "gold_canonical": k}
                for n, c, k in self.unmatched_top
            ],
        }

    # ── 사람이 읽는 표 ──────────────────────────────────────────────────────
    def report(self) -> str:
        o = self.overall
        version = f"룰 사전 v{self.dictionary_version}" if self.dictionary_version else "사전 없음"
        lines = [
            f"■ 베이스라인 {self.label} ({version})",
            "",
            (
                f"  모수  전체 {self.tx_total}건 → 분류 대상 {self.tx_classified}건 "
                f"(제외 {self.tx_excluded}건) → 정답 있음 {o.labeled}건"
            ),
        ]
        if self.excluded_reasons:
            reasons = " · ".join(f"{r} {c}" for r, c in self.excluded_reasons.most_common())
            lines.append(f"        제외 사유: {reasons}")
        if o.unlabeled:
            lines.append(f"        ⚠️ 정답 라벨이 없는 거래 {o.unlabeled}건 — 정확도 분모에서 제외")

        lines += [
            "",
            "  [거래 기준]",
            f"    정확도 L1 {o.accuracy_l1:6.1%}   L2 {o.accuracy_l2:6.1%}",
            (
                f"    확정 커버리지 {o.coverage:6.1%}   미매칭 {o.unmatched_rate:6.1%}   "
                f"PG마스킹 {o.pg_masked_rate:6.1%} (원리적 한계)"
            ),
            "",
            "  [정규화 키 기준]  — 키마다 가중치가 같다. 롱테일이 보이는 자리",
            (
                f"    정확도 L2 {self.distinct.accuracy_l2:6.1%}   "
                f"미매칭 {self.distinct.unmatched_rate:6.1%}   "
                f"(전체 {self.distinct.keys}개 키 중 미매칭 {self.distinct.unmatched}개)"
            ),
            "",
            "  [단계별 기여]",
            f"    {'단계':<10} {'건수':>7} {'비중':>7} {'L2 정확도':>9}",
        ]
        for stage in MatchStage:
            count = o.by_stage.get(stage.value, 0)
            if not count:
                continue
            lines.append(
                f"    {stage.value:<10} {count:>7} {_rate(count, o.labeled + o.unlabeled):>7.1%} "
                f"{o.stage_accuracy(stage.value):>9.1%}"
            )
        # 미매칭의 정확도가 0 이 아니면 반드시 설명한다. 그 몫은 **정답이 미분류인 거래**이지
        # 룰이 일을 해서 맞은 것이 아니다 — 이 줄이 없으면 "룰이 못 잡았는데도 10% 맞혔다"로
        # 읽히고, 그건 룰의 성과를 없는 것으로 만들어내는 오독이다.
        if o.stage_accuracy(MatchStage.UNMATCHED.value) > 0:
            lines.append(
                "    ↑ unmatched 가 맞은 몫 = **정답이 미분류인 거래** (택소노미에 자리가 없는 업종"
                "). 룰의 성과가 아니다"
            )

        if self.by_provider:
            lines += [
                "",
                "  [제공기관 축 분해]  — 표기 관행이 기관마다 다르다",
                (
                    f"    {'기관':<8} {'건수':>6} {'L2 정확도':>9} {'커버리지':>8} "
                    f"{'미매칭':>7} {'PG':>6}"
                ),
            ]
            for name, s in sorted(self.by_provider.items()):
                lines.append(
                    f"    {name:<8} {s.labeled + s.unlabeled:>6} {s.accuracy_l2:>9.1%} "
                    f"{s.coverage:>8.1%} {s.unmatched_rate:>7.1%} {s.pg_masked_rate:>6.1%}"
                )

        if self.confusions:
            lines += ["", "  [L1 혼동]  정답 → 예측 (건수)"]
            lines += [f"    {gold} → {pred} ({count})" for gold, pred, count in self.confusions]

        if self.unmatched_top:
            lines += ["", "  [미매칭 상위]  사전 승격 후보 미리보기"]
            lines += [
                f"    {count:>4}건  {name}  (정답 {canonical})"
                for name, count, canonical in self.unmatched_top
            ]
        return "\n".join(lines)


def evaluate(
    classified: list[ClassifiedTx],
    truth: GroundTruth,
    *,
    label: str,
    dictionary_version: int | None = None,
    all_normalized: list[NormalizedTx] | None = None,
    confusion_top: int = 8,
    unmatched_top: int = 15,
) -> EvalResult:
    """분류 결과를 정답과 대조한다. 분류기가 무엇이든(룰·LLM·하이브리드) 이 함수를 쓴다.

    `all_normalized` 를 주면 **분류 대상에서 빠진 거래**(취소 상계 등)를 모수 설명에 함께 담는다.
    주지 않으면 분모가 왜 줄었는지 결과만 보고는 알 수 없다.
    """
    total = len(all_normalized) if all_normalized is not None else len(classified)
    result = EvalResult(
        label=label,
        dictionary_version=dictionary_version,
        tx_total=total,
        tx_classified=len(classified),
    )
    if all_normalized is not None:
        for n in all_normalized:
            if not n.is_classifiable:
                result.excluded_reasons[n.excluded_reason or "미기재"] += 1

    confusion: collections.Counter[tuple[str, str]] = collections.Counter()
    unmatched: collections.Counter[str] = collections.Counter()
    unmatched_answer: dict[str, str] = {}
    # 정규화 키 단위 집계 — 키 → (정답 L2 최빈, 예측 L2 최빈, 단계 최빈)
    per_key: dict[str, tuple[collections.Counter, collections.Counter, collections.Counter]] = {}

    for tx in classified:
        raw = tx.normalized.raw
        stage = tx.match_stage
        gold = truth.tx_labels.get(raw.approval_no)
        ok_l1 = None if gold is None else tx.category_l1 is gold.category_l1
        ok_l2 = None if gold is None else tx.category_l2 is gold.category_l2

        result.overall.add(stage=stage, ok_l1=ok_l1, ok_l2=ok_l2)
        provider = result.by_provider.setdefault(raw.provider_id, Slice(name=raw.provider_id))
        provider.add(stage=stage, ok_l1=ok_l1, ok_l2=ok_l2)

        if gold is not None and ok_l1 is False:
            confusion[(gold.category_l1.value, tx.category_l1.value)] += 1
        if stage is MatchStage.UNMATCHED:
            unmatched[tx.normalized.normalized_name] += 1
            if gold is not None:
                unmatched_answer.setdefault(tx.normalized.normalized_name, gold.canonical)

        key = tx.normalized.normalized_name
        golds, preds, stages = per_key.setdefault(
            key, (collections.Counter(), collections.Counter(), collections.Counter())
        )
        if gold is not None:
            golds[gold.category_l2.value] += 1
        preds[tx.category_l2.value] += 1
        stages[stage.value] += 1

    for golds, preds, stages in per_key.values():
        result.distinct.keys += 1
        if golds and preds.most_common(1)[0][0] == golds.most_common(1)[0][0]:
            result.distinct.correct_l2 += 1
        top_stage = stages.most_common(1)[0][0]
        if top_stage == MatchStage.UNMATCHED.value:
            result.distinct.unmatched += 1
        elif top_stage == MatchStage.PG_MASKED.value:
            result.distinct.pg_masked += 1

    result.confusions = [(g, p, c) for (g, p), c in confusion.most_common(confusion_top)]
    result.unmatched_top = [
        (name, count, unmatched_answer.get(name, "?"))
        for name, count in unmatched.most_common(unmatched_top)
    ]
    return result


@dataclass(frozen=True)
class GoldKey:
    """정규화 키 하나의 정답. 거래가 아니라 **키** 단위다 (LLM 이 보는 단위와 같다)."""

    category_l2: CategoryL2
    canonical: str


def gold_by_key(classified: list[ClassifiedTx], truth: GroundTruth) -> dict[str, GoldKey]:
    """정규화 키 → 정답. 그 키에 속한 거래의 **최빈값**으로 대표시킨다.

    `evaluate` 의 키 단위 집계와 같은 규칙이다 — 두 곳이 다른 규칙을 쓰면 "표기 L2 정확도"와
    "승격 정밀도"가 서로 다른 정답을 보고 재게 된다. 같은 정규화 키가 서로 다른 브랜드를
    가리키는 오염 표기가 실제로 있어 최빈값이 필요하다 (`DistinctSlice` 참고).
    """
    per_key: dict[str, tuple[collections.Counter, collections.Counter]] = {}
    for tx in classified:
        gold = truth.tx_labels.get(tx.normalized.raw.approval_no)
        if gold is None:
            continue
        l2s, names = per_key.setdefault(
            tx.normalized.normalized_name, (collections.Counter(), collections.Counter())
        )
        l2s[gold.category_l2] += 1
        names[gold.canonical] += 1
    return {
        key: GoldKey(category_l2=l2s.most_common(1)[0][0], canonical=names.most_common(1)[0][0])
        for key, (l2s, names) in per_key.items()
    }
