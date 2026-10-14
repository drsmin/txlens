"""제출 노트북이 실행 프로그램과 어긋나지 않는다 (부록 H).

노트북은 **출력이 저장된 채로** 제출된다. 채점자는 대개 실행하지 않고 읽는다 — 그러니 지켜야 할
것은 "돌리면 맞다"가 아니라 **"저장된 수가 지금 패키지가 내는 수와 같다"**이다. 패키지를 고치고
노트북을 다시 실행하지 않으면 제출본만 낡는다. 그 드리프트를 사람 기억에 맡기지 않는다.

`nbformat` 없이 JSON 으로 읽는다 — 이 검사가 노트북 도구(dev 의존성)에 묶이지 않게 한다.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from txlens.agents.replay import ReplayClassifier, load_verdicts
from txlens.eval.harness import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_TRUTH_PATH,
    compare_table,
    run_baseline,
)
from txlens.rules import load_dictionary

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "txlens_demo.ipynb"
VERDICTS = ROOT / "data" / "gold" / "verdicts-v1-gpt-4o-mini.json"


def _code_cells() -> list[dict]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _stdout(cell: dict) -> str:
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            text = o["text"]
            out.append("".join(text) if isinstance(text, list) else text)
    return "".join(out)


def test_노트북_셀에_로직_정의가_없다():
    """셀에 `def`/`class` 가 생기는 순간 진실이 둘이 된다 — 로직은 `src/txlens/` 에만."""
    offenders = []
    for i, cell in enumerate(_code_cells()):
        # `%pip` 같은 매직 줄은 파이썬이 아니다
        code = "\n".join(ln for ln in _source(cell).splitlines() if not ln.lstrip().startswith(("%", "!")))
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                offenders.append(f"코드 셀 {i}: {type(node).__name__} (줄 {node.lineno})")
    assert not offenders, "노트북에 로직이 정의돼 있다 — 패키지로 옮긴다:\n" + "\n".join(offenders)


def test_저장된_출력에_에러가_없고_전부_실행됐다():
    """에러 출력이나 실행 안 된 셀이 저장된 채 제출되면 읽는 사람에겐 그게 결과다."""
    for i, cell in enumerate(_code_cells()):
        assert cell.get("execution_count"), f"코드 셀 {i} 가 실행되지 않은 채 저장됐다"
        errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        assert not errors, f"코드 셀 {i} 에 에러가 저장됐다: {errors[0].get('ename')}"


def test_그래프_그림이_코드와_같다고_저장돼_있다():
    """노트북 2절은 커밋된 그림과 컴파일한 그래프를 대조한 결과를 찍는다 — 그 결과가 True 여야 한다."""
    marker = "커밋된 그림과 지금 그래프가 같은가:"
    lines = [ln for c in _code_cells() for ln in _stdout(c).splitlines() if marker in ln]
    assert lines, "그래프 대조 출력을 못 찾았다 — 노트북 2절이 바뀌었다"
    assert lines[0].strip().endswith("True"), lines[0]


@pytest.mark.skipif(
    not DEFAULT_BUNDLE_DIR.exists() or not DEFAULT_TRUTH_PATH.exists(),
    reason="합성 데이터가 없다 — `python -m txlens.generator --seed 42` (README 참고)",
)
def test_저장된_베이스라인_표가_지금_패키지가_내는_표와_같다():
    """노트북 4절의 A·C 표를 지금 다시 계산해 **글자 그대로** 대조한다 (재생이라 API 0 · 결정론).

    다르면 패키지가 바뀐 뒤 노트북을 다시 실행하지 않은 것이다:
    `jupyter execute --inplace notebooks/txlens_demo.ipynb`
    """
    cells = [c for c in _code_cells() if "compare_table(" in _source(c)]
    assert len(cells) == 1, "베이스라인 표를 찍는 셀을 하나로 특정하지 못했다"
    saved = _stdout(cells[0])

    dictionary = load_dictionary()
    verdicts, _model = load_verdicts(VERDICTS)
    res_a, met_a = run_baseline(label="A · 룰만", dictionary=dictionary)
    res_c, met_c = run_baseline(
        label="C · 하이브리드(재생)", dictionary=dictionary, classifier=ReplayClassifier(verdicts)
    )
    expected = compare_table([(res_a, met_a), (res_c, met_c)])
    assert expected in saved, (
        "노트북에 저장된 베이스라인 표가 지금 패키지의 결과와 다르다 — 노트북을 다시 실행한다\n"
        f"--- 저장된 출력 ---\n{saved}\n--- 지금 계산 ---\n{expected}"
    )
