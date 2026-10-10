"""승격 — 판정을 **지식**으로 바꾸는 자리 (blueprint §3.3 ⑤).

`llm_classify` 가 내는 판정은 그 실행에서만 산다. 다음 달 거래를 다시 받으면 같은 표기를 또
물어본다. 승격은 그 판정을 사전에 적어 **다음 라운드의 룰 단계가 잡게** 만든다 — 이 프로젝트가
"LLM 을 덜 쓰는 설계"라고 주장하는 근거가 여기서 수치가 된다.

## 루프는 그래프 안이 아니라 라운드 사이에 있다

`rule_promote` → `rule_match` 로 되돌리는 사이클을 만들지 않는다. 같은 실행에서 되돌아가면
방금 LLM 판정을 얹은 거래가 다음 바퀴에서 EXACT 단계로 잡혀 **단계 분포가 바뀐다** — 라벨은
그대로인데 "어느 단계가 몇 건을 잡았나"가 달라지고, 그 수치가 곧 베이스라인 비교표다. 같은
답을 두 번 얹으면서 지표만 흔드는 바퀴다.

라운드 사이에 두면 라운드 1 은 손대지 않은 채 라운드 2 가 더해진다. 사전이 파일로 남으므로
(`rules/v{n}.yaml`) 버전 비교·롤백·감사 추적도 함께 얻는다.

## 자동 승격은 **새 브랜드를 만들지 않는다** (D-033)

승격에는 두 종류가 있고, 자동화해도 되는 것은 하나뿐이다.

| 종류 | 무엇 | 누가 |
| --- | --- | --- |
| **별칭 승격** | "이 표기는 이미 아는 브랜드다" (`타벅스` → 스타벅스) | 자동 |
| **새 브랜드 승격** | 사전에 없던 브랜드를 만든다 (`을지로호프`) | 사람 (5주차 HITL) |

경계를 이렇게 그은 것은 실측 때문이다. 미매칭 표기 155개를 gpt-4o-mini 로 판정해 정답셋과
대조하니, **카테고리는 대체로 맞고 대표명이 틀렸다** — 모델은 확신이 없으면 입력 표기를 그대로
대표명으로 돌려주므로(프롬프트가 그렇게 시킨다) `타벅스`·`GS2`·`에스씨케이컴니퍼` 같은 오타가
**새 브랜드로 굳는다.** 그러면 스타벅스의 거래가 여러 이름으로 쪼개져 2부(구독·이상지출)의
집계가 무너진다 — 1부의 실수가 2부로 그대로 전이되는 자리다(D-010).

대표명을 사전에서 찾을 수 있을 때만 승격하면 **사전이 판정의 검증자**가 된다. 실측에서 이
규칙은 후보 34개에 대해 카테고리·대표명 **둘 다 100%** 였고, 그것도 신뢰도 문턱과 거의 무관했다
(문턱 0.0 이나 0.85 나 정밀도가 같다). 문턱을 튜닝하는 대신 자동화의 경계를 옮긴 것이다.

라벨도 판정이 아니라 **사전 항목에서 가져온다.** 모델이 `투썸플레이스 → cafe.coffee` 라고 해도
사전이 `cafe.bakery` 로 쓰고 있으면 사전을 따른다 — 그래야 같은 브랜드가 두 라벨로 갈리지 않는다.

## 그 밖에 거르는 것

| 거르는 것 | 이유 |
| --- | --- |
| `category_l1 = unclassified` | "모른다"는 답은 지식이 아니다. 사전에 적으면 미분류가 룰 단계 커버리지로 둔갑한다 |
| `evidence = guess` | 근거가 추측이면 넣지 않는다. 실측 정밀도 9.1% 로 셋 중 가장 나쁘다 |
| `confidence < 0.7` | 앵커가 1차 방어선이고 이 문턱은 2차다. 실측에서는 걸러내는 것이 없었지만, 모델이 바뀌면 앵커만으로 부족할 수 있다 |

거른 것은 버리는 게 아니라 검토 큐에 남아 있다 — 사람이 보는 자리(5주차 `human_review`)가
그쪽이고, `origin="human"` 승격이 거기서 나온다. 실측에서 자동이 덮는 것은 40건이고 사람에게
남는 것은 276건이다. **그 격차가 HITL 이 필요한 이유의 크기다.**
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from txlens.preprocess import normalize_name
from txlens.rules.dictionary import RuleDictionary, RuleEntry
from txlens.schemas import (
    CategoryL1,
    Evidence,
    HumanCorrection,
    LLMTarget,
    LLMVerdict,
    RuleProposal,
)

#: 자동 승격의 신뢰도 하한 — **2차 방어선**이다 (1차는 위 § 의 사전 앵커).
#: 한때 0.95 였다. 그런데 프롬프트의 few-shot 이 keyword 근거에 0.85~0.9 만 보여주므로 모델은
#: 그 척도를 따르고, 0.95 는 **도달 불가능한 문턱**이라 승격이 영원히 0 건이었다 (D-033).
#: τ(0.8, 검토 큐 경계)와 다른 질문에 답한다 — τ 는 "사람이 봐야 하나", 이 값은 "굳혀도 되나".
MIN_PROMOTE_CONFIDENCE = 0.7


def propose_rules(
    targets: Sequence[LLMTarget],
    verdicts: Mapping[str, LLMVerdict],
    *,
    dictionary: RuleDictionary,
    min_confidence: float = MIN_PROMOTE_CONFIDENCE,
) -> list[RuleProposal]:
    """고신뢰 판정을 **별칭 승격** 후보로 바꾼다. 거래 건수 많은 것부터 (targets 순서).

    `dictionary` 가 검증자다 — 모델이 댄 대표명을 사전에서 찾지 못하면 그 판정은 승격하지
    않는다 (위 § 참고). 찾으면 라벨과 대표명을 **사전 항목에서** 가져온다.

    나오는 후보는 언제나 `alias` 다. 키가 대표명 자신이었다면 그 표기는 애초에 exact 단계에서
    잡혀 미매칭 목록에 오지 않았을 것이기 때문이다 — `pattern_type="exact"` 는 사람이 새
    브랜드를 만들 때(5주차) 쓰는 자리로 남는다.
    """
    proposals: list[RuleProposal] = []
    for target in targets:
        verdict = verdicts.get(target.normalized_name)
        if verdict is None:
            continue
        if verdict.confidence < min_confidence:
            continue
        if verdict.category_l1 is CategoryL1.UNCLASSIFIED:
            continue
        if verdict.evidence is Evidence.GUESS:
            continue

        canonical = normalize_name(verdict.merchant_canonical).normalized_name
        entry = dictionary.entry_for(canonical) if canonical else None
        if entry is None:
            continue

        proposals.append(
            RuleProposal(
                pattern_type="alias",
                pattern=target.normalized_name,
                merchant_canonical=entry.canonical,
                category_l1=entry.category_l1,
                category_l2=entry.category_l2,
                origin="llm_high_confidence",
                source_name=target.normalized_name,
                covers_tx_count=target.tx_count,
            )
        )
    return proposals


def propose_from_corrections(
    corrections: Sequence[HumanCorrection],
    *,
    dictionary: RuleDictionary,
) -> list[RuleProposal]:
    """사람 교정을 승격 후보로 바꾼다 — **새 브랜드가 나올 수 있는 유일한 경로다** (D-036②).

    자동 승격(`propose_rules`)은 사전을 검증자로 쓰기 때문에 사전에 없는 브랜드를 영원히 만들지
    못한다. 그것이 안전장치이자 천장이고, 실측에서 그 천장 위에 276건이 남았다. 이 함수가 그
    천장을 넘는 자리이고, 넘을 수 있는 이유는 **판정의 출처가 모델이 아니라 사람**이기 때문이다.

    나오는 후보는 두 갈래다.

    | 대표명이 사전에 | 후보 | 뜻 |
    | --- | --- | --- |
    | 있다 | `alias` | "이 표기는 아는 브랜드다" — 자동 승격이 확신이 없어 못 한 일 |
    | 없다 | `exact` | "새 브랜드를 만든다" — 자동 승격이 **원리적으로** 못 하는 일 |

    **라벨은 사전이 이긴다** (자동 승격과 같은 규칙). 사람이 `쿠팡와우 → 쿠팡` 이라고 고치면
    라벨은 쿠팡 항목의 것을 쓴다. 사람이 쿠팡의 라벨 자체에 이견이 있다면 그것은 이 표기 하나의
    교정이 아니라 사전 수정이고, 여기서 받아주면 같은 브랜드가 두 라벨로 갈린다.

    대표명은 **정규화해서** 담는다. 사전의 `canonical` 은 그 자신이 exact 키가 되므로
    (`RuleEntry.exact_keys`) 정규화 형태가 아니면 사전 로더가 통째로 거부한다 — 사람이
    `(주)을지로호프` 라고 적어도 사전에는 `을지로호프` 로 들어가야 한다.
    """
    proposals: list[RuleProposal] = []
    for correction in corrections:
        canonical = normalize_name(correction.merchant_canonical).normalized_name
        if not canonical or not correction.normalized_name:
            continue

        entry = dictionary.entry_for(canonical)
        if entry is not None:
            proposals.append(
                RuleProposal(
                    pattern_type="alias",
                    pattern=correction.normalized_name,
                    merchant_canonical=entry.canonical,
                    category_l1=entry.category_l1,
                    category_l2=entry.category_l2,
                    origin="human",
                    source_name=correction.normalized_name,
                    covers_tx_count=correction.tx_count,
                )
            )
            continue

        proposals.append(
            RuleProposal(
                pattern_type="exact",
                pattern=correction.normalized_name,
                merchant_canonical=canonical,
                category_l1=correction.category_l1,
                category_l2=correction.category_l2,
                origin="human",
                source_name=correction.normalized_name,
                covers_tx_count=correction.tx_count,
            )
        )
    return proposals


def apply_proposals(
    dictionary: RuleDictionary, proposals: Sequence[RuleProposal]
) -> tuple[RuleDictionary, list[RuleProposal]]:
    """사전에 후보를 얹어 다음 버전을 만든다. **원본은 건드리지 않는다.**

    받아들인 후보만 두 번째 값으로 돌려준다. 조용히 버리지 않는 이유는 라운드 표의 "승격" 칸이
    사전 증가분과 맞아야 하기 때문이다 — 어긋나면 어느 쪽이 맞는지 알 수 없다.

    **새 항목을 만들 수 있는 것은 `origin="human"` 뿐이다.** 대표명이 사전에 없는데 출처가
    모델이면 그 후보는 버린다 (D-033 — 사전이 판정의 검증자다). 정책을 `propose_rules` 한 곳에만
    두지 않고 여기서도 막는 이유는, 다음에 다른 생산자가 붙었을 때(다른 모델·배치 임포트) 조용히
    새 브랜드가 생기는 것을 막기 위해서다. 사람이 만든 것만 통과시키는 이 한 줄이 자동화의
    경계다 (D-036②).

    이미 쓰이고 있는 키도 건너뛴다. 사전 로더는 키 중복을 **에러**로 막는데(나중 항목이 앞
    항목의 라벨을 조용히 덮어쓰는 사고), 승격이 그 에러를 만들면 라운드가 통째로 멈춘다. 새
    브랜드의 `canonical` 은 그 자신이 exact 키가 되므로(`RuleEntry.exact_keys`) 대표명까지
    같은 검사를 통과해야 한다 — 사람이 만든 브랜드 이름이 이미 다른 브랜드의 별칭이면,
    그것은 새 브랜드가 아니라 사전 수정이 필요한 사안이다.
    """
    entries = [entry.model_copy(deep=True) for entry in dictionary.entries]
    by_canonical = {entry.canonical: entry for entry in entries}
    claimed = set(dictionary.exact_index) | set(dictionary.alias_index)

    accepted: list[RuleProposal] = []
    for proposal in proposals:
        if proposal.pattern in claimed:
            continue
        entry = by_canonical.get(proposal.merchant_canonical)

        if entry is None:
            if proposal.origin != "human":
                continue
            canonical = proposal.merchant_canonical
            if canonical in claimed:
                continue
            entry = RuleEntry(
                canonical=canonical,
                l1=proposal.category_l1,
                l2=proposal.category_l2,
            )
            entries.append(entry)
            by_canonical[canonical] = entry
            claimed.add(canonical)

        # 대표명 자신이 패턴이면 더 넣을 것이 없다 — `exact_keys` 가 이미 그것을 색인한다.
        if proposal.pattern != entry.canonical:
            bucket = entry.exact if proposal.pattern_type == "exact" else entry.alias
            bucket.append(proposal.pattern)
            claimed.add(proposal.pattern)
        accepted.append(proposal)

    return (
        RuleDictionary(
            version=dictionary.version + 1,
            fuzzy_threshold=dictionary.fuzzy_threshold,
            entries=entries,
        ),
        accepted,
    )


def dump_dictionary(
    dictionary: RuleDictionary,
    path: Path,
    *,
    parent_version: int,
    accepted: Sequence[RuleProposal],
    model: str,
) -> None:
    """승격된 사전을 파일로 쓴다. **머리말이 감사 추적이다.**

    사전만 남기면 6개월 뒤에 "이 alias 는 누가 왜 넣었나"에 답할 수 없다. 어느 버전에서
    무엇이 몇 건을 덮으려고 들어왔는지를 파일 안에 적어 두면 사전 하나로 롤백 판단이 된다
    (blueprint §3.3 ⑤ — 사전은 버전 관리 대상이다).

    사람이 손으로 쓴 `v1.yaml` 의 주석은 재생성 대상이 아니라 **원본**이므로 이 함수로
    덮어쓰지 않는다 — 호출자가 `v{n+1}` 경로를 준다.
    """
    covered = sum(p.covers_tx_count for p in accepted)
    human = sum(1 for p in accepted if p.origin == "human")
    # 출처를 나눠 적는다. 자동과 사람 승격은 **되돌릴 때 판단이 다르다** — 자동은 모델을 바꾸면
    # 다시 만들 수 있고, 사람이 만든 새 브랜드는 그 사람의 판단이 유일한 출처다.
    origins = f"자동 {len(accepted) - human}건 + 사람 {human}건" if human else f"자동 {len(accepted)}건"
    lines = [
        (
            f"# 룰 사전 v{dictionary.version} — v{parent_version} + 승격 "
            f"{origins} (blueprint §3.3 ⑤)"
        ),
        "#",
        ("# ⚠️ 이 파일은 `python -m txlens.promote` 가 생성한다. 손으로 고치지 말고 "
         "사전을 고치려면"),
        f"#    v{parent_version} 을 고치고 다시 승격시킨다 — 다음 승격이 이 파일을 덮어쓴다.",
        "#",
        f"# 승격 조건: confidence >= {MIN_PROMOTE_CONFIDENCE} · evidence != guess · l1 != unclassified",
        f"# 판정 모델: {model}",
        f"# 덮는 거래: {covered}건",
        "#",
        "# 승격 목록 (표기 → 대표명 · 덮는 거래 수 · [패턴/출처])",
    ]
    lines += [
        f"#   {p.pattern} → {p.merchant_canonical} · {p.covers_tx_count}건 "
        f"[{p.pattern_type}/{p.origin}]"
        for p in accepted
    ]
    body = yaml.safe_dump(
        dictionary.model_dump(mode="json", by_alias=True, exclude_defaults=True),
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )
    path.write_text("\n".join(lines) + "\n\n" + body, encoding="utf-8")
