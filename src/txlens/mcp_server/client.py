"""MCP 경유로 도구를 받아오는 쪽 — `.mcp.json` 을 **런처의 단일 출처**로 삼는다.

## 왜 클라이언트까지 저장소 안에 두나

서버만 만들어 두면 "Claude Code 에서는 되던데요"로 끝난다. 같은 서버에 **우리 코드로도** 붙어
보면 두 가지가 기계로 확인된다: ① 프로토콜 층이 실제로 왕복한다 ② MCP 로 받은 도구가
in-process 도구와 **같은 답을 낸다**. 두 번째가 이 프로젝트에서 중요하다 — 도구를 프로세스
밖으로 내보낸 뒤에도 지표가 그대로여야 A/B/C 비교(D-025②)가 무너지지 않는다.

## `.mcp.json` 을 여기서도 읽는 이유

서버 실행 명령을 클라이언트에 또 적으면, Claude Code 가 붙는 서버와 우리가 붙는 서버가
조용히 달라진다(사전 버전이 다른 서버에 붙는 것이 가장 흔한 사고다). 런처는 한 곳에만 적는다.

## 남은 제약 — MCP 도구는 async 다

`langchain_mcp_adapters` 가 돌려주는 도구는 `ainvoke` 로만 돈다. 지금 `llm_classify` 노드는
동기라, 이 경로를 기본으로 삼으려면 그래프 전체를 async 로 올려야 한다. **그래서 기본은
in-process 그대로 두고 이 경로는 열어만 둔다** — 베이스라인 수치가 전달 방식 때문에 움직이는
일은 만들지 않는다 (D-048 이 보여준 대로 수는 쉽게 움직인다).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from txlens.mcp_server import SERVER_NAME

#: 저장소 루트. src/txlens/mcp_server/client.py → 부모 3단계.
REPO_ROOT = Path(__file__).resolve().parents[3]
MCP_CONFIG_PATH = REPO_ROOT / ".mcp.json"


def load_connection(
    *, config_path: Path | str = MCP_CONFIG_PATH, use_current_interpreter: bool = False
) -> dict[str, Any]:
    """`.mcp.json` 에서 stdio 연결 설정을 읽는다.

    Args:
        config_path: 설정 파일 경로.
        use_current_interpreter: 설정의 인터프리터 대신 **지금 돌고 있는 파이썬**을 쓴다.
            `.venv/bin/python` 은 머신마다 다를 수 있어, 테스트는 이쪽을 켠다.

    Returns:
        `MultiServerMCPClient` 에 그대로 넘길 수 있는 연결 사전.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    entry = config["mcpServers"][SERVER_NAME]

    command = sys.executable if use_current_interpreter else entry["command"]
    if not use_current_interpreter and not Path(command).is_absolute():
        command = str(REPO_ROOT / command)

    return {
        SERVER_NAME: {
            "command": command,
            "args": list(entry["args"]),
            "cwd": str(REPO_ROOT),
            "transport": "stdio",
        }
    }


def build_client(**kwargs: Any) -> MultiServerMCPClient:
    """`.mcp.json` 이 가리키는 서버에 붙는 클라이언트."""
    return MultiServerMCPClient(load_connection(**kwargs))


async def load_tools(**kwargs: Any) -> list[BaseTool]:
    """MCP 서버의 도구를 LangChain 도구로 받아온다.

    ⚠️ 돌려받은 도구는 **`ainvoke` 로만** 돈다. `create_agent` 에 그대로 넘길 수는 있지만
    그래프도 async 로 돌려야 한다 (모듈 문서 참고).
    """
    return await build_client(**kwargs).get_tools()
