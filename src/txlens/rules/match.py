"""5단계 룰 매칭 — 빠르고 확실한 것부터 (설계 §3.3 ②).

    ① exact  — 정규화 완전일치 (dict lookup, O(1))
    ② regno  — 사업자등록번호. 있고 사전에 등록된 경우만. 보조 신호다
    ③ alias  — 법인명·로마자 → 브랜드
    ④ regex  — 접두 패턴 (공백 없이 붙은 지점명 등)
    ⑤ fuzzy  — `rapidfuzz.token_sort_ratio >= 임계값`. 미달은 **미매칭으로 남긴다**

## 순서가 곧 신뢰도다

앞 단계가 이기는 구조라, 확실한 신호가 애매한 신호에 덮이지 않는다. 그리고 `match_stage` 를
남기므로 "어느 단계가 몇 건을 잡았나"를 지표로 볼 수 있다 — 이것이 LLM 을 붙일 근거(또는 붙이지
않을 근거)가 된다. 단계를 지우면 그 논거가 사라진다.

## 못 잡은 것은 못 잡았다고 답한다

임계값 미달을 억지로 가장 비슷한 브랜드에 붙이면 정확도는 오르지만 **틀린 답이 조용히 섞인다**
— 그 위에서 도는 2부(구독·이상지출)가 전부 거짓말이 된다. 미매칭은 `미분류` 라는 정당한
답으로 내보내고, LLM·사람이 이어받는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz, process

from txlens.preprocess import regno_key
from txlens.rules.dictionary import RuleDictionary, RuleEntry, load_dictionary
from txlens.schemas import CategoryL1, CategoryL2, ClassifiedTx, MatchStage, NormalizedTx

#: 단계별 신뢰도. **exact 만 1.0** 이고 뒤로 갈수록 낮다 — HITL 검토 큐가 이 값으로 정렬되므로
#: (`ReviewItem.priority`) 순서가 뒤집히면 사람이 엉뚱한 것부터 고친다.
#: regno 가 alias 보다 낮은 이유: 점포 단위이고 제공기관에 따라 신뢰도가 다르다.
STAGE_CONFIDENCE: dict[MatchStage, float] = {
    MatchStage.EXACT: 1.0,
    MatchStage.ALIAS: 0.95,
    MatchStage.REGNO: 0.9,
    MatchStage.REGEX: 0.85,
}

#: 유사도 신뢰도 상한. 토큰 순서만 다르면 100 점이 나오는데, 그걸 그대로 쓰면 오타를 흡수한
#: 추정이 완전일치와 같은 신뢰도를 갖는다.
FUZZY_CONFIDENCE_CAP = 0.95

#: 유사도 스코어러. **`token_set_ratio` 를 쓰지 않는다** (초안에서 바꿈).
#: `token_set_ratio` 는 한쪽 토큰 집합이 다른 쪽의 **부분집합이면 100 점**을 준다. 그래서
#: `STEP GYM`(사전에 없는 tail 브랜드) → `STEP GYM MEMBERSHIP`(사전의 구독 브랜드) 처럼
#: **모르는 가맹점을 사전의 긴 이름에 흡수시킨다.** 구독 탐지가 그 위에서 돌면 일회성 결제를
#: 구독 결제로 세므로 2부가 조용히 거짓말을 한다.
#: `token_sort_ratio` 는 어순·오타는 그대로 흡수하면서(위 예시들의 점수가 동일하다) 부분집합에는
#: 낮은 점수를 준다. `WRatio` 도 같은 함정이 있어(부분집합에 90점) 쓰지 않는다.
FUZZY_SCORER = fuzz.token_sort_ratio

#: PG 경유로 실가맹점이 소실된 거래의 대표명. 브랜드를 추측하지 않는다.
PG_MASKED_CANONICAL = "미분류(PG경유)"


@dataclass(frozen=True)
class RuleMatcher:
    """사전 하나를 들고 거래를 분류한다.

    사전을 인자로 받는 이유: 승격 루프가 v1 → v2 사전을 만들면 **같은 코드로 두 사전을 돌려**
    비교해야 한다. 사전 경로를 코드가 들고 있으면 그 비교를 못 한다.
    """

    dictionary: RuleDictionary

    @classmethod
    def from_path(cls, path: str | None = None) -> RuleMatcher:
        return cls(dictionary=load_dictionary(path) if path else load_dictionary())

    # ── 단계 ────────────────────────────────────────────────────────────────
    def match(self, tx: NormalizedTx) -> ClassifiedTx:
        """거래 1건을 분류한다. **항상 결과를 반환한다** — 미매칭도 답이다.

        분류 대상이 아닌 거래(취소 상계 등)를 걸러내는 것은 호출자의 몫이다. 여기서 걸러내면
        "몇 건이 왜 빠졌는지"가 이 함수 안에 숨어 지표의 모수를 설명할 수 없게 된다.
        """
        # 단계 0 — PG 경유는 매칭 대상이 아니다. 이름이 PG사명으로 바뀌어 있으므로 사전을
        # 뒤지면 엉뚱한 것에 붙을 수 있다. **미분류가 정답인 케이스**라 신뢰도는 1.0 이다.
        if tx.pg_masked:
            return ClassifiedTx(
                normalized=tx,
                merchant_canonical=PG_MASKED_CANONICAL,
                category_l1=CategoryL1.UNCLASSIFIED,
                category_l2=CategoryL2.UNCLASSIFIED_PG_MASKED,
                confidence=1.0,
                match_stage=MatchStage.PG_MASKED,
            )

        name = tx.normalized_name

        entry = self.dictionary.exact_index.get(name)
        if entry is not None:
            return self._hit(tx, entry, MatchStage.EXACT)

        key = regno_key(tx.raw.merchant_regno)
        if key is not None:
            entry = self.dictionary.regno_index.get(key)
            if entry is not None:
                return self._hit(tx, entry, MatchStage.REGNO)

        entry = self.dictionary.alias_index.get(name)
        if entry is not None:
            return self._hit(tx, entry, MatchStage.ALIAS)

        for pattern, candidate in self.dictionary.regex_index:
            if pattern.search(name):
                return self._hit(tx, candidate, MatchStage.REGEX)

        hit = process.extractOne(
            name,
            self.dictionary.fuzzy_choices,
            scorer=FUZZY_SCORER,
            score_cutoff=self.dictionary.fuzzy_threshold,
        )
        if hit is not None:
            matched_key, score, _ = hit
            entry = self.dictionary.entry_for(matched_key)
            if entry is not None:
                return self._hit(
                    tx, entry, MatchStage.FUZZY, confidence=min(FUZZY_CONFIDENCE_CAP, score / 100)
                )

        return ClassifiedTx(
            normalized=tx,
            # 못 잡은 문자열을 그대로 남긴다 — HITL 화면과 승격 후보가 이 값을 본다.
            merchant_canonical=name,
            category_l1=CategoryL1.UNCLASSIFIED,
            category_l2=CategoryL2.UNCLASSIFIED_INSUFFICIENT,
            confidence=0.0,
            match_stage=MatchStage.UNMATCHED,
        )

    def match_all(self, txs: list[NormalizedTx]) -> list[ClassifiedTx]:
        """입력 순서를 유지한다 (재현성 — `ingest`·`preprocess` 와 같은 이유)."""
        return [self.match(tx) for tx in txs]

    # ── 내부 ────────────────────────────────────────────────────────────────
    def _hit(
        self,
        tx: NormalizedTx,
        entry: RuleEntry,
        stage: MatchStage,
        *,
        confidence: float | None = None,
    ) -> ClassifiedTx:
        return ClassifiedTx(
            normalized=tx,
            merchant_canonical=entry.canonical,
            category_l1=entry.category_l1,
            category_l2=entry.category_l2,
            confidence=STAGE_CONFIDENCE[stage] if confidence is None else confidence,
            match_stage=stage,
        )
