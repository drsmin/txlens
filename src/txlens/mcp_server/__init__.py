"""TxLens 로컬 MCP 서버 — 분류 에이전트의 도구를 **프로세스 밖으로** 내보낸다 (5주차).

## 왜 MCP 인가 — 같은 도구를 사람도 쓴다

`agents/tools.py` 의 도구 3종은 지금 `llm_classify` 안에서만 산다. 그런데 그 도구가 답하는
질문("이 표기는 사전에 뭐랑 비슷한가", "이 업종에 쓸 수 있는 카테고리가 뭔가")은 **사람이
사전을 고칠 때 던지는 질문과 똑같다** — HITL 승격 루프에서 사람이 하는 일이 바로 그것이다.
같은 질문에 답하는 코드가 에이전트 안에만 있으면, 사람은 Streamlit 을 띄우거나 파이썬을 열어야 한다.

MCP 는 그 경계를 표준 프로토콜로 연다. 서버 하나를 stdio 로 띄우면 **에이전트도, 외부 MCP 클라이언트도
같은 도구를 부른다.** 도구를 두 번 구현하지 않는다는 것이 핵심이다.

## 그래서 로직을 복사하지 않는다

이 모듈은 도구를 **다시 만들지 않는다** — `make_tools()` 가 만든 LangChain 도구를 그대로 감싸고,
설명 문자열까지 `BaseTool.description` 에서 가져온다. 노트북이 패키지를 import 하게 한 것과 같은
이유다 — 같은 말을 두 곳에 적으면 한 곳만 고쳐지는 날이 반드시 온다.

## 사전은 인자로 받는다

`make_tools()` 가 사전을 주입받는 이유(베이스라인 B/C 를 같은 코드로 만들기 위해)가 여기서도
그대로 살아 있다. 서버를 `--rules rules/v2.yaml` 로 띄우면 **승격 후 사전**을 조회하는 서버가 된다.
v1 서버와 v2 서버를 동시에 띄워 같은 이름을 양쪽에 물어보는 것이 승격 효과를 보는 가장 싼 방법이다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from txlens.agents.tools import make_tools
from txlens.preprocess import preprocess
from txlens.rules import DEFAULT_DICTIONARY_PATH, RuleDictionary, RuleMatcher, load_dictionary
from txlens.schemas import RawTx
from txlens.taxonomy import Taxonomy, load_taxonomy

#: 조회용 가짜 거래의 고정 시각. 값은 아무래도 좋지만 **고정**이어야 한다 — 같은 이름을 두 번
#: 물어보면 같은 답이 나와야 사람이 사전을 고친 효과를 이 도구로 확인할 수 있다.
PROBE_DATETIME = datetime(2026, 1, 1, 12, 0)  # noqa: DTZ001 — tz 없는 KST

SERVER_NAME = "txlens"

INSTRUCTIONS = """\
TxLens — 마이데이터 거래내역 정규화·분류 프로젝트의 사전/택소노미 조회 서버.

가맹점 원본 표기를 다룰 때 쓴다. `classify_merchant` 가 결정론 파이프라인의 실제 판정을
그대로 돌려주므로, "이 표기가 왜 미분류인가"를 물을 때는 그것부터 부른다. 이름이 사전에
없어 보이면 `lookup_alias` 로 비슷한 항목을, 카테고리를 고를 때는 `search_taxonomy` 를 쓴다.

**이 서버는 사전을 고치지 않는다 — 읽기 전용이다.** 사전 승격은 사람이 검토 화면에서 한다.
"""


def build_server(
    *,
    dictionary: RuleDictionary | None = None,
    taxonomy: Taxonomy | None = None,
) -> FastMCP:
    """도구 5종 + 리소스 2종을 단 서버를 만든다.

    인자를 비우면 기본 사전(`rules/v1.yaml`)·기본 택소노미를 읽는다. 테스트는 작은 사전을
    직접 넣어 파일 없이 돈다.
    """
    dictionary = dictionary if dictionary is not None else load_dictionary()
    taxonomy = taxonomy if taxonomy is not None else load_taxonomy()

    matcher = RuleMatcher(dictionary=dictionary)
    # stdio 는 stdout 이 프로토콜 전용이라 로그는 stderr 로 간다. 기본 INFO 는 호출마다 한 줄씩
    # 쌓여 클라이언트 로그를 덮으므로, 사람이 봐야 하는 것(경고·오류)만 남긴다.
    server = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS, log_level="WARNING")

    # ── ① 에이전트 도구 3종을 그대로 내보낸다 ──────────────────────────────
    # 설명까지 `BaseTool.description` 에서 가져온다. 여기에 설명을 새로 쓰면 에이전트가 읽는
    # 설명과 MCP 클라이언트가 읽는 설명이 갈라진다 — 같은 도구가 두 개가 되는 것이다.
    lc_tools = {t.name: t for t in make_tools(dictionary=dictionary, taxonomy=taxonomy)}

    def lookup_alias(name: str) -> str:
        return str(lc_tools["lookup_alias"].invoke({"name": name}))

    def search_taxonomy(keyword: str) -> str:
        return str(lc_tools["search_taxonomy"].invoke({"keyword": keyword}))

    def get_brand_hint(name: str) -> str:
        return str(lc_tools["get_brand_hint"].invoke({"name": name}))

    for fn, tool_name in (
        (lookup_alias, "lookup_alias"),
        (search_taxonomy, "search_taxonomy"),
        (get_brand_hint, "get_brand_hint"),
    ):
        server.add_tool(fn, name=tool_name, description=lc_tools[tool_name].description)

    # ── ② 서버 고유 도구 — 파이프라인의 실제 판정 ─────────────────────────
    # 에이전트에게는 줄 수 없는 도구다. 에이전트는 룰 매칭이 **이미 실패한** 건만 받으므로
    # 이 도구를 주면 "룰이 뭐라 했나"를 되묻는 순환이 된다. 반대로 사람에게는 이것이 첫 질문이다.
    def classify_merchant(name: str, merchant_regno: str | None = None) -> str:
        """결정론 파이프라인(전처리 → 룰 매칭)이 이 표기를 어떻게 판정하는지 그대로 보여준다.

        Args:
            name: 가맹점 원본 표기. 정규화 전 형태를 그대로 넣는다.
            merchant_regno: 사업자등록번호(선택). 하이픈은 있어도 없어도 된다.

        Returns:
            정규화 키·대표명·카테고리·신뢰도와 **어느 단계에서 붙었는지**(`match_stage`).
            `unmatched` 면 룰로는 못 붙는다는 뜻이고, 그 건이 LLM 분류로 넘어간다.
        """
        raw = RawTx(
            provider_id="mcp",
            customer_id="mcp-probe",
            approval_no="probe",
            tx_datetime=PROBE_DATETIME,
            merchant_name_raw=name,
            amount=0,
            merchant_regno=merchant_regno,
        )
        normalized = preprocess([raw])[0]
        result = matcher.match(normalized)
        lines = [
            f"원본 표기: {name!r}",
            f"정규화 키: {normalized.normalized_name!r}",
        ]
        if normalized.branch:
            lines.append(f"분리된 지점: {normalized.branch}")
        if normalized.payment_channel is not None:
            lines.append(f"결제채널: {normalized.payment_channel.value}")
        if normalized.pg_masked:
            lines.append("PG 경유 — 실가맹점을 알 수 없다. **미분류가 정답이다**")
        lines += [
            f"대표 가맹점명: {result.merchant_canonical}",
            f"카테고리: {result.category_l2.value} ({taxonomy.label(result.category_l2.value)})",
            f"신뢰도: {result.confidence:.2f}",
            f"매칭 단계: {result.match_stage.value}",
        ]
        return "\n".join(lines)

    def dictionary_status() -> str:
        """이 서버가 들고 있는 사전이 무엇인지 — 버전과 크기.

        Returns:
            사전 버전·브랜드 수·키 수·유사도 임계값. 같은 이름을 서로 다른 사전의 서버에
            물어볼 때, 어느 쪽 답인지 확인하는 용도다.
        """
        exact = sum(len(e.exact_keys) for e in dictionary.entries)
        alias = sum(len(e.alias) for e in dictionary.entries)
        regex = sum(len(e.regex) for e in dictionary.entries)
        regno = sum(len(e.regno) for e in dictionary.entries)
        return (
            f"사전 v{dictionary.version} · 브랜드 {len(dictionary.entries)}개\n"
            f"키: exact {exact} · alias {alias} · regex {regex} · regno {regno}\n"
            f"유사도 임계값: {dictionary.fuzzy_threshold} "
            f"(이 아래는 룰이 붙이지 않고 LLM 으로 넘긴다)\n"
            f"택소노미: L1 {len(taxonomy.l1_ids())}개 · L2 {len(taxonomy.l2_ids())}개"
        )

    server.add_tool(classify_merchant)
    server.add_tool(dictionary_status)

    # ── ③ 리소스 — 도구가 아니라 "붙여 읽는 자료" ─────────────────────────
    # 택소노미 전체를 도구로 뱉으면 호출마다 토큰을 쓴다. 변하지 않는 목록은 리소스가 맞다.
    @server.resource(
        "txlens://taxonomy",
        name="카테고리 택소노미",
        description="쓸 수 있는 category_l2 값 전체. 여기 없는 값은 스키마 검증에서 막힌다.",
        mime_type="text/plain",
    )
    def taxonomy_resource() -> str:
        return taxonomy.prompt_block()

    @server.resource(
        "txlens://dictionary",
        name="룰 사전",
        description="현재 서버가 들고 있는 룰 사전의 브랜드 목록과 키.",
        mime_type="text/plain",
    )
    def dictionary_resource() -> str:
        lines = [f"# 룰 사전 v{dictionary.version} — 브랜드 {len(dictionary.entries)}개"]
        for entry in sorted(dictionary.entries, key=lambda e: e.canonical):
            lines.append(f"\n## {entry.canonical} · {entry.category_l2.value}")
            lines.append(f"- exact: {', '.join(entry.exact_keys)}")
            if entry.alias:
                lines.append(f"- alias: {', '.join(entry.alias)}")
            if entry.regex:
                lines.append(f"- regex: {', '.join(entry.regex)}")
            if entry.regno:
                lines.append(f"- regno: {', '.join(entry.regno)}")
        return "\n".join(lines)

    return server


def build_server_from_paths(
    *, dictionary_path: Path | str | None = None, taxonomy_path: Path | str | None = None
) -> FastMCP:
    """경로에서 읽어 서버를 만든다. `__main__` 의 CLI 가 쓰는 얇은 층."""
    return build_server(
        dictionary=load_dictionary(str(dictionary_path)) if dictionary_path else load_dictionary(),
        taxonomy=load_taxonomy(taxonomy_path) if taxonomy_path else load_taxonomy(),
    )


__all__ = [
    "DEFAULT_DICTIONARY_PATH",
    "INSTRUCTIONS",
    "PROBE_DATETIME",
    "SERVER_NAME",
    "build_server",
    "build_server_from_paths",
]
