"""TxLens MCP 서버 — 도구가 프로세스 밖에서도 같은 답을 내는지 (5주차).

여기서 막고 싶은 것은 셋이다.

① **도구가 두 개가 되는 것.** MCP 서버가 도구를 다시 구현하면 에이전트가 쓰는 `lookup_alias` 와
   Claude 가 쓰는 `lookup_alias` 가 갈라진다. 설명 문자열까지 같은 출처에서 오는지 기계로 본다.
② **사전 주입이 서버에서 풀리는 것.** 사전을 인자로 받는 구조(B/C 베이스라인의 전제)가
   MCP 경계를 넘어서도 살아 있어야 v1 서버와 v2 서버를 나란히 띄울 수 있다.
③ **`.mcp.json` 이 낡는 것.** 설정이 가리키는 명령이 실제로 돌지 않으면 발표 당일에 안 붙는다.

MCP 세션은 async 라 `asyncio.run` 으로 감싼다 — pytest-asyncio 를 새로 들이지 않는다.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from txlens.agents import make_tools
from txlens.mcp_server import PROBE_DATETIME, SERVER_NAME, build_server
from txlens.preprocess import preprocess
from txlens.rules import RuleDictionary, RuleMatcher, load_dictionary
from txlens.schemas import RawTx
from txlens.taxonomy import load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dictionary() -> RuleDictionary:
    return load_dictionary()


@pytest.fixture(scope="module")
def server(dictionary):
    return build_server(dictionary=dictionary, taxonomy=load_taxonomy())


def _call(server, tool_name: str, args: dict) -> str:
    """도구를 **실제 MCP 세션으로** 부르고 텍스트를 돌려준다.

    `build_server()` 가 만든 함수를 직접 부르면 프로토콜 층(스키마 생성·직렬화)을 건너뛴다.
    거기서 깨지면 셀프테스트는 통과하는데 클라이언트만 못 붙는다 — 가장 늦게 발견되는 고장이다.
    """

    async def run() -> str:
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(tool_name, args)
            assert not result.isError, result.content
            return "\n".join(c.text for c in result.content if c.type == "text")

    return asyncio.run(run())


def test_도구_다섯과_리소스_둘이_붙는다(server):
    async def run():
        async with create_connected_server_and_client_session(server) as session:
            tools = await session.list_tools()
            resources = await session.list_resources()
            return {t.name for t in tools.tools}, {str(r.uri) for r in resources.resources}

    names, uris = asyncio.run(run())
    assert names == {
        "lookup_alias",
        "search_taxonomy",
        "get_brand_hint",
        "classify_merchant",
        "dictionary_status",
    }
    assert uris == {"txlens://taxonomy", "txlens://dictionary"}


def test_설명이_에이전트_도구와_같은_출처다(server, dictionary):
    """① 설명을 새로 쓰면 여기서 깨진다. 같은 도구가 두 개가 되는 것을 막는 계약이다."""
    lc = {t.name: t.description for t in make_tools(dictionary=dictionary, taxonomy=load_taxonomy())}

    async def run():
        async with create_connected_server_and_client_session(server) as session:
            return {t.name: t.description for t in (await session.list_tools()).tools}

    mcp_desc = asyncio.run(run())
    for name, description in lc.items():
        assert mcp_desc[name] == description


def test_도구가_에이전트와_같은_답을_낸다(server, dictionary):
    """프로세스 경계를 넘어도 답이 같아야 한다 — 다르면 조회 결과를 믿을 수 없다."""
    tools = {t.name: t for t in make_tools(dictionary=dictionary, taxonomy=load_taxonomy())}
    for tool_name, args in (
        ("lookup_alias", {"name": "스타벅스코리아"}),
        ("search_taxonomy", {"keyword": "커피"}),
        ("get_brand_hint", {"name": "배달의민족_김밥천국 역삼점"}),
    ):
        assert _call(server, tool_name, args) == tools[tool_name].invoke(args)


def test_classify_merchant_는_룰_매칭_결과를_그대로_보여준다(server, dictionary):
    """서버가 판정을 다시 계산하지 않는다는 확인. 숫자가 갈리면 사람이 서버를 못 믿는다."""
    name = "(주)스타벅스코리아 강남점"
    raw = RawTx(
        provider_id="mcp",
        customer_id="mcp-probe",
        approval_no="probe",
        tx_datetime=PROBE_DATETIME,
        merchant_name_raw=name,
        amount=0,
    )
    expected = RuleMatcher(dictionary=dictionary).match(preprocess([raw])[0])

    text = _call(server, "classify_merchant", {"name": name})
    assert expected.merchant_canonical in text
    assert expected.category_l2.value in text
    assert expected.match_stage.value in text


def test_PG_경유는_미분류가_정답이라고_말해준다(server):
    """D-012 — 사람이 "왜 미분류냐"고 물을 때 이 서버가 이유를 준다."""
    text = _call(server, "classify_merchant", {"name": "(주)케이지이니시스"})
    assert "PG" in text and "미분류" in text


def test_사전을_바꿔_띄우면_답이_달라진다():
    """② 사전 주입이 MCP 경계를 넘어서도 산다 — v1 서버와 v2 서버를 나란히 띄울 수 있다."""
    empty = build_server(dictionary=RuleDictionary(version=0, entries=[]), taxonomy=load_taxonomy())
    v1 = build_server(dictionary=load_dictionary(), taxonomy=load_taxonomy())

    assert "사전이 비어 있다" in _call(empty, "lookup_alias", {"name": "스타벅스"})
    assert "v0" in _call(empty, "dictionary_status", {})
    assert "v1" in _call(v1, "dictionary_status", {})


def test_리소스로_택소노미와_사전을_읽는다(server):
    async def run():
        async with create_connected_server_and_client_session(server) as session:
            taxonomy = await session.read_resource("txlens://taxonomy")
            dictionary = await session.read_resource("txlens://dictionary")
            return taxonomy.contents[0].text, dictionary.contents[0].text

    taxonomy_text, dictionary_text = asyncio.run(run())
    assert "cafe.coffee" in taxonomy_text
    assert "스타벅스" in dictionary_text


def test_mcp_json_이_가리키는_명령이_실제로_돈다():
    """③ 설정이 낡으면 발표 당일 Claude Code 가 서버에 못 붙는다. 여기서 먼저 깨지게 둔다."""
    config = json.loads((REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    entry = config["mcpServers"][SERVER_NAME]

    rules = REPO_ROOT / entry["args"][entry["args"].index("--rules") + 1]
    assert rules.exists(), f"{rules} 가 없다 — .mcp.json 이 낡았다"

    # 설정의 인터프리터 경로(.venv/bin/python)는 머신마다 다를 수 있으므로, 명령이 도는지는
    # **지금 테스트를 돌리는 인터프리터**로 확인한다. 검증 대상은 모듈 경로와 인자다.
    proc = subprocess.run(
        [sys.executable, *entry["args"][:-2], "--rules", str(rules), "--selftest"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "도구 5개" in proc.stdout


# ── MCP 경유 도구가 in-process 도구와 같은 답을 내는가 ────────────────────────
# 여기부터는 **진짜 서브프로세스**를 띄운다 (stdio). 위의 in-memory 세션이 못 잡는 것 —
# 모듈 import 실패·인자 오류·프레임 깨짐 — 이 여기서 잡힌다. 느린 대신 이것만이 `.mcp.json`
# 이 실제로 붙는다는 증거다.

#: 두 경로에 똑같이 던지는 질문. 도구마다 인자 이름이 달라 쌍으로 들고 다닌다.
PARITY_CALLS = (
    ("lookup_alias", {"name": "스타벅스코리아"}),
    ("search_taxonomy", {"keyword": "커피"}),
    ("get_brand_hint", {"name": "배달의민족_김밥천국 역삼점"}),
)


def _mcp_text(payload) -> str:
    """MCP 도구의 반환을 텍스트로 편다.

    **in-process 도구는 `str` 을 주고 MCP 도구는 content 블록 리스트를 준다.** 전달 방식이
    바뀌면 값의 모양도 바뀐다는 뜻이라, 이 경로를 기본으로 삼으려면 노드가 이 층을 흡수해야
    한다 (`mcp_server/client.py` 의 제약과 같은 이유로 지금은 열어만 둔다).
    """
    if isinstance(payload, str):
        return payload
    return "\n".join(block["text"] for block in payload if block.get("type") == "text")


def test_MCP_경유_도구가_in_process_와_같은_답을_낸다(dictionary):
    from txlens.mcp_server.client import load_tools

    async def run() -> dict[str, object]:
        by_name = {t.name: t for t in await load_tools(use_current_interpreter=True)}
        return {name: await by_name[name].ainvoke(args) for name, args in PARITY_CALLS}

    over_mcp = asyncio.run(run())
    in_process = {t.name: t for t in make_tools(dictionary=dictionary, taxonomy=load_taxonomy())}

    # 모양은 다르다 — 그 사실 자체를 계약으로 박아 둔다. 언젠가 어댑터가 `str` 을 주게 되면
    # 여기가 깨지고, 그때 `_mcp_text` 를 지우면 된다.
    assert not isinstance(over_mcp["lookup_alias"], str)

    for name, args in PARITY_CALLS:
        assert _mcp_text(over_mcp[name]) == in_process[name].invoke(args)
