"""`python -m txlens.mcp_server` — TxLens MCP 서버를 stdio 로 띄운다.

    python -m txlens.mcp_server                        # 룰 사전 v1
    python -m txlens.mcp_server --rules rules/v2.yaml  # 승격 사전
    python -m txlens.mcp_server --selftest             # 띄우지 않고 목록만 찍는다

**stdout 은 프로토콜 전용이다.** MCP stdio 는 stdout 으로 JSON-RPC 를 주고받으므로 여기서
`print()` 를 하면 클라이언트가 프레임을 못 읽고 서버가 죽는다. 사람이 볼 말은 전부 stderr 로
보내고, `--selftest` 만 예외다 (그때는 서버를 띄우지 않는다).

`--rules` 이름은 `python -m txlens.detect` 와 맞췄다 — 같은 뜻의 옵션이 모듈마다 다른 이름을
가지면 쓰는 사람이 매번 `--help` 를 연다.
"""

from __future__ import annotations

import argparse
import sys

from txlens.console import use_utf8_output
from txlens.mcp_server import build_server_from_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m txlens.mcp_server",
        description="TxLens 로컬 MCP 서버 (stdio) — 사전·택소노미·룰 판정 조회",
    )
    parser.add_argument("--rules", default=None, help="룰 사전 YAML 경로 (기본: rules/v1.yaml)")
    parser.add_argument("--taxonomy", default=None, help="택소노미 YAML 경로 (기본: config/taxonomy.yaml)")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="서버를 띄우지 않고 등록된 도구·리소스 목록만 찍는다 (설정이 맞는지 확인용)",
    )
    args = parser.parse_args(argv)

    server = build_server_from_paths(dictionary_path=args.rules, taxonomy_path=args.taxonomy)

    if args.selftest:
        use_utf8_output()
        import asyncio

        tools = asyncio.run(server.list_tools())
        resources = asyncio.run(server.list_resources())
        print(f"서버 이름: {server.name}")
        print(f"도구 {len(tools)}개")
        for tool in tools:
            first_line = (tool.description or "").strip().splitlines()[0]
            print(f"  · {tool.name} — {first_line}")
        print(f"리소스 {len(resources)}개")
        for res in resources:
            print(f"  · {res.uri} — {res.name}")
        return 0

    # 서버가 어느 사전을 물고 떴는지는 반드시 남긴다 — v1 서버와 v2 서버를 같이 띄우면
    # 답이 다른 것이 정상이고, 그때 로그가 없으면 어느 쪽 답인지 알 수 없다.
    print(f"[txlens-mcp] 사전 {args.rules or 'rules/v1.yaml'} 로 stdio 대기", file=sys.stderr)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
