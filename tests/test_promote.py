"""승격 루프 — 판정을 사전으로 굳히는 층 (설계 §3.3 ⑤).

여기서 막고 싶은 것은 다섯이다.

① **오타가 새 브랜드로 굳는 것.** 모델은 확신이 없으면 입력 표기를 그대로 대표명으로 돌려준다.
   `타벅스` 가 사전에 새 항목으로 들어가면 스타벅스의 거래가 두 이름으로 쪼개지고, 2부(구독·
   이상지출)의 집계가 그 위에서 무너진다. **자동 승격은 새 브랜드를 만들지 않는다.**
② **틀린 판정이 사전에 굳는 것.** 사전은 다음 라운드에 검토 없이 적용된다. 추측·미분류가
   올라가면 그 표기의 모든 거래가 조용히 틀리고, 룰 단계 커버리지로 세어져 지표를 올린다.
③ **승격이 사전을 깨뜨리는 것.** 키가 중복되면 로더가 실패한다 — 라운드 2 가 아니라 라운드 2 의
   `load_dictionary` 에서 죽는다.
④ **루프가 안 멈추는 것.** 라운드마다 실제 비용이 나간다. 멈춘 이유도 함께 남아야 한다 —
   수렴과 중단은 다른 말이다.
⑤ **승격이 아무 일도 안 하는데 했다고 보고되는 것.** 사전이 커졌으면 다음 라운드에 LLM 으로
   보낼 표기가 **실제로** 줄어야 한다. 그게 이 프로젝트가 하는 주장의 전부다.
"""

from __future__ import annotations

import pytest

from txlens.agents.replay import ReplayClassifier, load_verdicts, save_verdicts
from txlens.eval.harness import DEFAULT_BUNDLE_DIR, DEFAULT_TRUTH_PATH
from txlens.eval.metrics import GoldKey
from txlens.graph import build_graph, run_pipeline
from txlens.promote import (
    STOP_MAX_ROUNDS,
    STOP_NO_CANDIDATES,
    calibration_report,
    rounds_table,
    run_rounds,
)
from txlens.rules import RuleMatcher, load_dictionary
from txlens.rules.promote import apply_proposals, dump_dictionary, propose_rules
from txlens.schemas import CategoryL1, CategoryL2, Evidence, LLMTarget, LLMVerdict

pytestmark_data = pytest.mark.skipif(
    not (DEFAULT_BUNDLE_DIR.exists() and DEFAULT_TRUTH_PATH.exists()),
    reason="합성 데이터가 없다 — `python -m txlens.generator --seed 42` (README 참고)",
)

#: v1 사전에 있는 브랜드. 별칭 승격의 착지점이다.
KNOWN = "스타벅스"
#: 사전에 **없는** 브랜드. 자동으로 만들면 안 되는 쪽 (사람이 만든다).
NEW_BRAND = "을지로호프"


def _target(name: str, tx_count: int = 10) -> LLMTarget:
    return LLMTarget(normalized_name=name, sample_raw_name=name, tx_count=tx_count)


def _verdict(canonical: str = KNOWN, **overrides) -> LLMVerdict:
    """판정 하나. 기본값은 **승격 가능한** 형태다 — 거르는 쪽을 테스트가 지정한다."""
    fields = {
        "merchant_canonical": canonical,
        "brand": canonical,
        "category_l1": CategoryL1.CAFE,
        "category_l2": CategoryL2.CAFE_COFFEE,
        "confidence": 0.85,
        "reasoning": "브랜드를 안다",
        "evidence": Evidence.BRAND_KNOWN,
    }
    return LLMVerdict(**{**fields, **overrides})


def _replay_for_top(count: int, canonical: str = KNOWN) -> dict[str, LLMVerdict]:
    """상위 표기 `count` 개를 **아는 브랜드**로 판정한 대본.

    이름을 손으로 적지 않고 파이프라인에서 뽑는다 — 상위 목록은 데이터가 정하는 것이라
    테스트에 베껴 두면 생성기 시드가 바뀔 때 조용히 어긋난다.
    """
    state = run_pipeline(DEFAULT_BUNDLE_DIR)
    return {t.normalized_name: _verdict(canonical) for t in state["llm_targets"][:count]}


# ── ① 자동 승격은 새 브랜드를 만들지 않는다 ─────────────────────────────────
def test_사전에_있는_브랜드로_이을_때만_승격한다():
    """`타벅스 → 스타벅스` 는 올리고, `을지로호프 → 을지로호프` 는 올리지 않는다.

    후자도 **정답**이지만 사전에 없는 브랜드를 만드는 일이라 사람의 몫이다. 실측에서 모델이
    댄 대표명은 카테고리가 맞아도 이름이 틀리는 일이 잦았다(오타를 그대로 돌려준다).
    """
    targets = [_target("타벅스"), _target(NEW_BRAND)]
    verdicts = {"타벅스": _verdict(KNOWN), NEW_BRAND: _verdict(NEW_BRAND)}

    proposals = propose_rules(targets, verdicts, dictionary=load_dictionary())

    assert [p.source_name for p in proposals] == ["타벅스"]
    assert proposals[0].merchant_canonical == KNOWN
    assert proposals[0].pattern_type == "alias", "자동 승격은 언제나 별칭이다"


def test_라벨은_판정이_아니라_사전에서_가져온다():
    """모델이 `cafe.alcohol` 이라 해도 사전이 `cafe.coffee` 면 사전을 따른다.

    안 그러면 같은 브랜드가 두 라벨로 갈린다 — 실측에서 투썸플레이스가 정확히 그랬다.
    """
    dictionary = load_dictionary()
    entry = dictionary.entry_for(KNOWN)
    verdicts = {"타벅스": _verdict(KNOWN, category_l2=CategoryL2.CAFE_ALCOHOL)}

    proposal = propose_rules([_target("타벅스")], verdicts, dictionary=dictionary)[0]

    assert proposal.category_l2 is entry.category_l2
    assert proposal.category_l2 is not CategoryL2.CAFE_ALCOHOL


def test_추측과_미분류와_저신뢰는_승격하지_않는다():
    """셋 다 **다른 이유로** 위험하다. 근거가 추측인 판정은 실측 정밀도가 9.1% 였다."""
    targets = [_target(n) for n in ("추측", "모른다", "저신뢰", "정상")]
    verdicts = {
        "추측": _verdict(KNOWN, evidence=Evidence.GUESS),
        "모른다": _verdict(
            KNOWN,
            category_l1=CategoryL1.UNCLASSIFIED,
            category_l2=CategoryL2.UNCLASSIFIED_INSUFFICIENT,
        ),
        "저신뢰": _verdict(KNOWN, confidence=0.5),
        "정상": _verdict(KNOWN),
    }

    proposals = propose_rules(targets, verdicts, dictionary=load_dictionary())

    assert [p.source_name for p in proposals] == ["정상"]


def test_승격_순서는_거래_건수를_따른다():
    """사람이 위에서부터 검토하든 상한으로 자르든, 같은 노력이 가장 많은 거래를 덮게 한다."""
    targets = [_target("타벅스", 50), _target("스타스", 10)]
    verdicts = {name: _verdict(KNOWN) for name in ("타벅스", "스타스")}

    proposals = propose_rules(targets, verdicts, dictionary=load_dictionary())

    assert [p.covers_tx_count for p in proposals] == [50, 10]


# ── ② 승격된 사전이 성립하는가 ──────────────────────────────────────────────
def test_별칭은_기존_항목에_붙고_새_항목은_생기지_않는다():
    """같은 브랜드를 두 항목으로 만들면 집계가 둘로 쪼개진다."""
    base = load_dictionary()
    proposals = propose_rules(
        [_target("타벅스")], {"타벅스": _verdict(KNOWN)}, dictionary=base
    )

    promoted, accepted = apply_proposals(base, proposals)

    assert len(accepted) == 1
    assert len(promoted.entries) == len(base.entries), "새 항목이 생기면 안 된다"
    assert promoted.alias_index["타벅스"].canonical == KNOWN


def test_대표명이_사전에_없으면_적용_단계에서도_막힌다():
    """정책이 한 곳에만 있으면 다음 생산자(사람 교정·다른 모델)가 붙을 때 새 브랜드가 샌다."""
    base = load_dictionary()
    stray = propose_rules(
        [_target("타벅스")], {"타벅스": _verdict(KNOWN)}, dictionary=base
    )[0].model_copy(update={"merchant_canonical": "사전에없는브랜드"})

    promoted, accepted = apply_proposals(base, [stray])

    assert accepted == []
    assert len(promoted.entries) == len(base.entries)


def test_이미_쓰이는_키는_건너뛴다():
    """키 중복은 로더에서 **에러**다 — 승격이 그 에러를 만들면 다음 라운드가 통째로 멈춘다."""
    base = load_dictionary()
    existing = next(iter(base.alias_index))
    proposals = propose_rules(
        [_target(existing)], {existing: _verdict(KNOWN)}, dictionary=base
    )

    promoted, accepted = apply_proposals(base, proposals)

    assert accepted == []
    assert promoted.alias_index[existing].canonical == base.alias_index[existing].canonical


def test_승격된_사전은_로더_검증을_통과한다(tmp_path):
    """파일로 나갔다가 다시 들어와야 다음 라운드가 쓸 수 있다 — round-trip 이 계약이다."""
    base = load_dictionary()
    proposals = propose_rules(
        [_target("타벅스"), _target("스타스")],
        {name: _verdict(KNOWN) for name in ("타벅스", "스타스")},
        dictionary=base,
    )
    promoted, accepted = apply_proposals(base, proposals)

    path = tmp_path / f"v{promoted.version}.yaml"
    dump_dictionary(
        promoted, path, parent_version=base.version, accepted=accepted, model="scripted"
    )
    reloaded = load_dictionary(path)

    assert reloaded.version == base.version + 1
    assert reloaded.alias_index["타벅스"].canonical == KNOWN
    # 머리말이 감사 추적이다 — 무엇이 왜 들어왔는지 파일만 보고 판단할 수 있어야 한다.
    # 출처까지 적혀야 한다: 자동 승격은 모델을 바꿔 다시 만들 수 있고, 사람 승격은 그럴 수 없다.
    head = path.read_text(encoding="utf-8")
    assert "자동 2건" in head and "타벅스" in head
    assert "[alias/llm_high_confidence]" in head


# ── ③ 그래프에서의 자리 ─────────────────────────────────────────────────────
def test_분류기가_없으면_승격_노드도_없다():
    """(A) 는 판정을 만들지 않는다 — 승격할 것이 없는데 상자만 그리면 그림이 거짓말한다."""
    assert "rule_promote" not in set(build_graph().get_graph().nodes)


def test_승격_노드는_판정_뒤에_있고_그래프에_사이클은_없다():
    """루프는 라운드 사이에 있다. 그래프가 DAG 라는 것이 곧 무한루프 방지 장치다."""
    graph = build_graph(classifier=ReplayClassifier({})).get_graph()
    edges = {(e.source, e.target) for e in graph.edges}

    assert ("llm_classify", "rule_promote") in edges
    assert ("rule_promote", "finalize") in edges
    assert not any(source == "rule_promote" and target == "rule_match" for source, target in edges)


@pytestmark_data
def test_승격_후보가_상태에_남는다():
    """`rule_proposals` 는 이 실행이 남기는 유일한 지식이다 — 생산자가 없으면 필드가 거짓말한다."""
    replay = _replay_for_top(1)
    state = run_pipeline(
        DEFAULT_BUNDLE_DIR, classifier=ReplayClassifier(replay), llm_target_limit=1
    )

    assert [p.source_name for p in state["rule_proposals"]] == list(replay)
    assert state["rule_proposals"][0].merchant_canonical == KNOWN


# ── ④ 승격이 실제로 호출을 줄이는가 ─────────────────────────────────────────
@pytestmark_data
def test_사전이_커지면_다음_라운드에_보낼_표기가_준다():
    """이 프로젝트의 주장 그 자체. LLM 없이 결정론으로만 확인한다.

    **승격한 것보다 더 줄어들 수 있다.** 승격된 별칭이 fuzzy 단계의 후보가 되면서 오타 변형까지
    함께 잡히기 때문이다 — 사전 한 줄이 표기 하나가 아니라 그 주변까지 덮는다. 그래서
    "승격 수 = 감소 수"로 못 박지 않고 하한으로 잰다.
    """
    base = load_dictionary()
    before = run_pipeline(DEFAULT_BUNDLE_DIR, matcher=RuleMatcher(base))
    replay = _replay_for_top(3)
    targets = [t for t in before["llm_targets"] if t.normalized_name in replay]

    promoted, accepted = apply_proposals(
        base, propose_rules(targets, replay, dictionary=base)
    )
    after = run_pipeline(DEFAULT_BUNDLE_DIR, matcher=RuleMatcher(promoted))

    assert len(accepted) == 3
    after_names = {t.normalized_name for t in after["llm_targets"]}
    assert not (set(replay) & after_names), "승격된 표기가 여전히 LLM 대상이다"
    assert len(after_names) <= len(before["llm_targets"]) - 3
    covered = sum(t.tx_count for t in targets)
    assert after["metrics"].rule_resolved >= before["metrics"].rule_resolved + covered


# ── ⑤ 루프가 멈추는가 ───────────────────────────────────────────────────────
@pytestmark_data
def test_승격할_후보가_없으면_라운드1_에서_멈춘다():
    """전부 "모른다"고 답하는 판정. 자동으로 갈 곳이 없다는 것이 **HITL 이 필요한 이유**다."""
    unknown = _verdict(
        KNOWN,
        category_l1=CategoryL1.UNCLASSIFIED,
        category_l2=CategoryL2.UNCLASSIFIED_INSUFFICIENT,
    )
    replay = {name: unknown for name in _replay_for_top(5)}

    reports = run_rounds("openai:gpt-4o-mini", max_rounds=3, llm_target_limit=3, replay=replay)

    assert len(reports) == 1
    assert reports[0].stop_reason == STOP_NO_CANDIDATES
    assert reports[0].promoted == 0


@pytestmark_data
def test_라운드_상한이_안전핀으로_남는다():
    """상한에서 멈춘 것은 **수렴이 아니다** — 표에 그렇게 적혀야 한다."""
    reports = run_rounds(
        "openai:gpt-4o-mini", max_rounds=2, llm_target_limit=3, replay=_replay_for_top(10)
    )

    assert len(reports) == 2
    assert reports[-1].stop_reason == STOP_MAX_ROUNDS
    assert reports[1].dictionary_version == reports[0].dictionary_version + 1
    assert reports[1].dictionary_keys > reports[0].dictionary_keys


@pytestmark_data
def test_상한이_걸려도_진행_판정은_상한_이전의_수로_한다():
    """`--llm-limit` 은 보낸 수를 상한에 붙박는다 — 그 수로 재면 2 라운드에서 헛멈춘다."""
    reports = run_rounds(
        "openai:gpt-4o-mini", max_rounds=2, llm_target_limit=3, replay=_replay_for_top(10)
    )

    assert [r.metrics.llm_targets for r in reports] == [3, 3], "상한이 걸린 전제가 깨졌다"
    assert reports[1].pending_targets <= reports[0].pending_targets - reports[0].promoted


@pytestmark_data
def test_상한에서_버려진_승격을_표가_경고한다(tmp_path):
    """상한에서 멈추면 그 라운드의 승격은 파일로 안 나간다 — **말해주지 않으면 호출만 날린다.**

    실제로 그랬다: `--rounds 1` 로 돌려 승격 7건이 조용히 버려지고 417호출이 산출물 없이
    끝났다. 버리는 것 자체는 의도다(지표가 없는 사전은 쓰지 않는다). 버렸다는 **사실과
    되찾는 방법**이 표에 있어야 한다.
    """
    reports = run_rounds(
        "openai:gpt-4o-mini",
        max_rounds=1,
        llm_target_limit=3,
        replay=_replay_for_top(10),
        out_dir=tmp_path,
    )

    assert reports[-1].stop_reason == STOP_MAX_ROUNDS
    assert reports[-1].promoted, "승격 0건이면 경고할 것이 없다 — 전제가 깨졌다"
    assert not list(tmp_path.iterdir()), "상한 라운드의 사전은 파일로 나가지 않는다"

    table = rounds_table(reports)
    assert "파일로 쓰지 않았다" in table
    assert "--rounds 2" in table, "되찾는 방법(한 라운드 더)이 적혀야 한다"


@pytestmark_data
def test_고정점에서_멈추면_버린_승격_경고가_없다():
    """고정점은 승격 0건이라 버릴 것이 없다 — 경고가 뜨면 거짓 경보다."""
    reports = run_rounds(
        "openai:gpt-4o-mini", max_rounds=3, llm_target_limit=3,
        replay={
            name: _verdict(
                KNOWN,
                category_l1=CategoryL1.UNCLASSIFIED,
                category_l2=CategoryL2.UNCLASSIFIED_INSUFFICIENT,
            )
            for name in _replay_for_top(5)
        },
    )

    assert reports[-1].stop_reason == STOP_NO_CANDIDATES
    assert "파일로 쓰지 않았다" not in rounds_table(reports)


@pytestmark_data
def test_라운드가_사전을_파일로_남긴다(tmp_path):
    """다음 라운드·다음 달의 실행이 읽을 수 있어야 승격이 실행 밖으로 나간 것이다."""
    reports = run_rounds(
        "openai:gpt-4o-mini",
        max_rounds=2,
        llm_target_limit=3,
        replay=_replay_for_top(10),
        out_dir=tmp_path,
    )

    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == [f"v{reports[0].dictionary_version + 1}.yaml"], (
        "마지막 라운드는 사전을 쓰지 않는다 — 그 사전으로 잰 지표가 없기 때문이다"
    )
    reloaded = load_dictionary(tmp_path / written[0])
    keys = len(reloaded.exact_index) + len(reloaded.alias_index)
    # 파일로 나간 사전이 **라운드 2 가 실제로 쓴 그 사전**이어야 한다.
    assert keys == reports[1].dictionary_keys
    assert reloaded.version == reports[1].dictionary_version


# ── ⑥ 판정 재생 — 같은 질문에 두 번 결제하지 않는다 ─────────────────────────
def test_판정을_저장하고_다시_읽으면_같다(tmp_path):
    """문턱을 다시 고르는 일은 판정을 새로 얻는 일이 아니다 — 파일이 그 사실을 지킨다."""
    verdicts = {"타벅스": _verdict(KNOWN)}
    path = tmp_path / "verdicts.json"
    save_verdicts(verdicts, path, model="openai:gpt-4o-mini")

    loaded, model = load_verdicts(path)

    assert model == "openai:gpt-4o-mini"
    assert loaded == verdicts


def test_재생은_대본에_없는_표기를_실패로_센다():
    """조용히 건너뛰면 그 표기가 분모에서 사라져 재생 결과가 실호출보다 좋아 보인다."""
    replay = ReplayClassifier({"타벅스": _verdict(KNOWN)})

    outcome = replay.classify_all([_target("타벅스"), _target("없는표기")])

    assert set(outcome.verdicts) == {"타벅스"}
    assert outcome.failed == ["없는표기"]
    assert replay.cache.calls == 0, "재생은 모델을 부르지 않는다 — 호출 수는 0 이어야 한다"


@pytestmark_data
def test_재생으로_돈_라운드는_모델을_부르지_않는다():
    """재생 실행의 지표에 실호출 수가 섞이면 '이번 실행이 얼마나 들었나'가 거짓이 된다."""
    reports = run_rounds(
        "openai:gpt-4o-mini", max_rounds=2, llm_target_limit=3, replay=_replay_for_top(10)
    )

    assert all(r.metrics.llm_model_calls == 0 for r in reports)
    assert reports[0].promoted > 0, "재생 판정이 승격까지 이어지지 않았다"


# ── ⑦ 문턱을 정답셋으로 정한다 ──────────────────────────────────────────────
def test_보정표는_문턱마다_정밀도와_덮는_거래를_함께_낸다():
    """정밀도만 보면 문턱을 1.0 으로 올리는 것이 언제나 이긴다 — 승격이 0 건이어도 그렇다."""
    targets = [_target("맞는판정", 71), _target("틀린판정", 30)]
    verdicts = {"맞는판정": _verdict(KNOWN), "틀린판정": _verdict(KNOWN)}
    gold = {
        "맞는판정": GoldKey(category_l2=CategoryL2.CAFE_COFFEE, canonical=KNOWN),
        "틀린판정": GoldKey(category_l2=CategoryL2.FOOD_RESTAURANT, canonical="딴것"),
    }

    table = calibration_report(targets, verdicts, gold)

    assert "0.85" in table and "50.0%" in table, "문턱 0.85 의 정밀도 1/2 이 보이지 않는다"
    assert "0.90" in table and "—" in table, "후보가 0 인 문턱은 정밀도를 못 낸다고 적어야 한다"
    # 틀린 판정은 이름으로 남는다 — 개수만 세면 무엇을 굳힐 뻔했는지 물어볼 수 없다.
    assert "틀린판정" in table
