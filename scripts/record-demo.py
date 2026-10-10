#!/usr/bin/env python3
"""데모 영상을 뜬다 — Streamlit 화면을 브라우저로 조작하며 녹화한다 (D-053).

왜 사람이 화면 녹화기로 뜨지 않는가: 녹화는 **다시 뜰 수 있어야** 라이브보다 안전하다(D-053).
손으로 뜨면 클릭 순서·대기 시간이 매번 달라 수치를 고친 뒤 같은 영상을 다시 만들 수 없다.
이 스크립트는 같은 장면을 같은 순서로 다시 뜬다. 화면은 재생 대본으로 돌므로 **API 는 0** 이다.

장면 (약 2분 반):
    0. 표지                — 무엇을 보여 주는가
    1. Home                — 준비 상태 ✅ 셋
    2. S1 검토 큐          — 그래프가 interrupt() 에서 멈춤 → 사람 교정 3줄 → 재개 → 승격 → 라운드 2
    3. S4 인사이트         — §5.2 네 단계 토글: 찾은 구독 43 → 57 → 57 → 69
    4. S2 지표 대시보드    — 같은 수를 그 자리에서 다시 잰다
오른쪽 칸의 실행 로그(D-054)가 **그래프가 실제로 도는 증거**다. 로그 타임스탬프는 그래프의
실제 시간이고, 읽을 시간은 그래프 밖(장면 사이 대기)에서만 준다 — 그래프에 지연을 넣으면
`[0.31s]` 가 성능을 거짓말한다 (D-053).

사용:
    streamlit run app/Home.py --server.port 8501 --server.headless true   # 먼저 띄운다
    python scripts/record-demo.py                     # → docs/reports/media/txlens-demo.mp4
    python scripts/record-demo.py --out /tmp/x.mp4 --browser /path/to/chrome

필요한 것: `playwright`(requirements-dev) · mp4 변환용 `imageio-ffmpeg`(없으면 webm 만 남긴다).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "reports" / "media" / "txlens-demo.mp4"
W, H = 1920, 1080

# S1 에서 고칠 세 줄 — 큐 상위 3개(덮는 거래 65+30+27 = 122건). 정답은 생성기의 브랜드 표다.
#   우리동네세탁: 세탁업은 택소노미에 자리가 없어 정답이 **미분류**다(D-026) — LLM 답을 그대로 승인
#   제이디에프컴퍼니·디엠리테일홀딩스: 법인명과 브랜드가 전혀 다른 가상 브랜드 — 사람만 안다
CORRECTIONS = [
    (0, None, None),
    (1, "빈스로드", "카페"),
    (2, "데일리마트", "식료품"),
]

CAPTION_JS = """
([text, sub]) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);' +
      'z-index:999999;background:rgba(20,24,33,.88);color:#fff;padding:14px 28px;' +
      'border-radius:12px;font:600 26px "NanumBarunGothic",sans-serif;max-width:1500px;' +
      'text-align:center;box-shadow:0 6px 24px rgba(0,0,0,.25);transition:opacity .3s';
    document.body.appendChild(el);
  }
  el.innerHTML = text + (sub ? '<div style="font-weight:400;font-size:20px;opacity:.85;' +
    'margin-top:6px">' + sub + '</div>' : '');
  el.style.opacity = text ? '1' : '0';
}
"""

CARD_JS = """
([title, lines]) => {
  let el = document.getElementById('demo-card');
  if (!title) { if (el) el.remove(); return; }
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-card';
    el.style.cssText = 'position:fixed;inset:0;z-index:1000000;background:#141821;color:#fff;' +
      'display:flex;flex-direction:column;justify-content:center;align-items:center;' +
      'font-family:"NanumBarunGothic",sans-serif';
    document.body.appendChild(el);
  }
  el.innerHTML = '<div style="font-size:64px;font-weight:700;margin-bottom:28px">' + title +
    '</div>' + lines.map(l => '<div style="font-size:30px;opacity:.85;margin:8px">' + l +
    '</div>').join('');
}
"""

HIGHLIGHT_JS = """
([marker, on]) => {
  // 제목 글자로 열을 찾는다 — 열 순서로 찾으면 안쪽 지표 줄의 열이 먼저 걸린다
  const cols = [...document.querySelectorAll('[data-testid="stColumn"]')]
    .filter(c => c.innerText.includes(marker));
  const el = cols[cols.length - 1];
  if (!el) return false;
  el.style.transition = 'box-shadow .3s';
  el.style.boxShadow = on ? '0 0 0 4px #ff4b4b, 0 0 24px rgba(255,75,75,.45)' : '';
  el.style.borderRadius = '10px';
  return true;
}
"""


class Director:
    """장면 진행을 한곳에 모은다 — 대기 시간이 흩어지면 영상 길이를 가늠할 수 없다."""

    def __init__(self, page: Page, base: str) -> None:
        self.page, self.base = page, base

    def caption(self, text: str, sub: str = "") -> None:
        self.page.evaluate(CAPTION_JS, [text, sub])

    def card(self, title: str, lines: list[str] | None = None) -> None:
        self.page.evaluate(CARD_JS, [title, lines or []])

    def hold(self, seconds: float) -> None:
        self.page.wait_for_timeout(int(seconds * 1000))

    def settle(self, timeout: float = 30) -> None:
        """Streamlit 재실행이 끝날 때까지 — 오른쪽 위 'Running' 표시가 사라지면 끝이다."""
        self.page.wait_for_timeout(400)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.page.locator('[data-testid="stStatusWidget"]').count() == 0:
                break
            self.page.wait_for_timeout(200)
        self.page.wait_for_timeout(300)

    def goto(self, path: str) -> None:
        self.page.goto(f"{self.base}/{path}")
        self.page.wait_for_selector('[data-testid="stMain"]')
        self.settle()

    def collapse_sidebar(self) -> None:
        # 사이드바를 접어야 로그 칸이 70자 줄을 자르지 않는다 (D-054 의 6:4 가 1920px 에서 성립)
        # 접힌 상태는 페이지를 옮겨도 남는다 — 펼쳐져 있을 때만 누른다
        if self.page.locator('[data-testid="stSidebar"][aria-expanded="true"]').count() == 0:
            return
        self.page.mouse.move(150, 30)
        self.page.wait_for_timeout(300)
        self.page.locator('[data-testid="stSidebarCollapseButton"] button').first.click()
        self.page.wait_for_timeout(800)

    def scroll_to_text(self, text: str, block: str = "start") -> None:
        self.page.get_by_text(text, exact=False).first.evaluate(
            f"e => e.scrollIntoView({{behavior: 'smooth', block: '{block}'}})"
        )
        self.page.wait_for_timeout(1200)

    def scroll_top(self) -> None:
        self.page.evaluate(
            "() => document.querySelector('[data-testid=\"stMain\"]')"
            ".scrollTo({top: 0, behavior: 'smooth'})"
        )
        self.page.wait_for_timeout(1000)

    def highlight_log(self, on: bool) -> None:
        # 접힌 지난 라운드까지 포함해 로그 칸 전체를 두른다
        self.page.evaluate(HIGHLIGHT_JS, ["실행 로그 — 그래프가", on])


def edit_queue(d: Director) -> None:
    """검토 표(캔버스 그리드)를 사람처럼 고친다. 좌표는 그리드 상자에서 계산한다."""
    page = d.page
    grid = page.locator('[data-testid="stDataFrame"]').first
    box = grid.bounding_box()
    assert box is not None, "검토 표가 없다 — 큐가 비었거나 화면이 바뀌었다"
    x0, y0 = box["x"], box["y"]
    row_y = lambda i: y0 + 36 + 35 * i + 17  # 헤더 36px · 행 35px
    x_apply, x_name, x_cat = x0 + 38, x0 + 450, x0 + 720

    page.mouse.click(x_name, row_y(0))  # 첫 클릭은 그리드에 포커스만 준다
    page.wait_for_timeout(300)
    for row, canonical, category in CORRECTIONS:
        y = row_y(row)
        if canonical:
            page.mouse.click(x_name, y)
            page.wait_for_timeout(300)
            page.keyboard.press("Enter")
            page.wait_for_timeout(300)
            page.keyboard.press("Control+A")
            page.keyboard.type(canonical, delay=120)
            page.wait_for_timeout(300)
            page.keyboard.press("Enter")
            page.wait_for_timeout(600)
        if category:
            page.mouse.dblclick(x_cat, y)
            page.wait_for_timeout(600)
            page.keyboard.type(category, delay=120)
            page.wait_for_timeout(700)
            page.keyboard.press("Enter")
            page.wait_for_timeout(600)
        page.mouse.click(x_apply, y)
        page.wait_for_timeout(900)


def record(base: str, out: Path, browser: str | None) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="txlens-rec-"))
    with sync_playwright() as p:
        launch = {"executable_path": browser} if browser else {}
        b = p.chromium.launch(**launch)

        # 예열 — 녹화 밖에서 S2·S4 를 한 번 열어 st.cache 를 채운다. 안 하면 영상에 7초짜리
        # 빈 화면이 들어간다. 캐시는 세션 사이에 공유된다.
        warm = b.new_page(viewport={"width": W, "height": H})
        for path in ("인사이트", "지표_대시보드"):
            warm.goto(f"{base}/{path}")
            warm.wait_for_selector('[data-testid="stMain"]')
            Director(warm, base).settle(timeout=90)
        warm.close()

        ctx = b.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(tmp),
            record_video_size={"width": W, "height": H},
            locale="ko-KR",
        )
        page = ctx.new_page()
        t_start = time.monotonic()
        d = Director(page, base)

        # 0. 표지 ─────────────────────────────────────────────────────────────
        d.goto("")
        d.card("TxLens", [
            "마이데이터 거래내역 인텔리전스 에이전트",
            "LangGraph 8노드 · 룰(head) + LLM(tail) + 사람 교정 → 사전 승격",
            "화면은 저장된 판정으로 돈다 — API 호출 0",
        ])
        t_cut = time.monotonic() - t_start  # 이 앞(빈 페이지 로딩)은 잘라낸다
        d.hold(5)
        d.card("")

        # 1. Home ─────────────────────────────────────────────────────────────
        d.caption("Home — 실행 준비 상태", "합성 번들 · 룰 사전 v1 · 재생 대본이 모두 ✅ 여야 화면을 넘긴다")
        d.hold(5)

        # 2. S1 검토 큐 ───────────────────────────────────────────────────────
        d.goto("검토_큐")
        d.collapse_sidebar()
        d.highlight_log(True)
        d.caption(
            "S1 검토 큐 — LangGraph 가 8노드를 돌고 interrupt() 에서 사람을 기다린다",
            "오른쪽 = 실제 그래프 실행 로그: ingest → … → llm_classify → interrupt",
        )
        d.hold(8)
        d.highlight_log(False)
        d.caption(
            "신뢰도가 낮은 64개 표기 중 '덮는 거래 × 불확실성' 상위 20개가 큐에 올라왔다",
            "LLM 은 법인명 '제이디에프컴퍼니'가 무엇인지 모른다 — 사람은 안다",
        )
        d.hold(5)
        d.caption("사람이 3줄을 교정한다", "제이디에프컴퍼니 → 빈스로드(카페) · 디엠리테일홀딩스 → 데일리마트(식료품) · 세탁은 미분류 승인")
        edit_queue(d)
        d.hold(1.5)

        d.caption("교정 제출 → 그래프가 멈춘 자리에서 재개 → rule_promote → finalize")
        page.get_by_role("button", name="교정 제출").click()
        d.highlight_log(True)
        d.settle()
        d.hold(3)
        d.highlight_log(False)
        d.scroll_to_text("승격 결과")
        d.caption(
            "교정 3줄이 사전에 들어갔다 — 덮는 거래 122건이 즉시 반영",
            "자동 승격은 사전에 있는 브랜드의 별칭만 — 새 브랜드는 사람만 만든다 (D-033)",
        )
        d.hold(7)

        d.caption("커진 사전으로 그래프를 다시 돌린다 — 라운드 2")
        page.get_by_role("button", name="승격된 사전으로 다시 돌린다").click()
        d.settle()
        d.scroll_top()
        d.highlight_log(True)
        d.caption(
            "라운드 2 — collect_targets: LLM 에 보낼 표기 155 → 115",
            "사람 교정 3줄 + 자동 승격(별칭) 34개가 사전에 들어간 만큼 LLM 몫이 줄었다 · 지난 라운드 로그는 접혀 남는다",
        )
        d.hold(8)
        d.highlight_log(False)
        d.scroll_to_text("라운드별 — LLM 에 보낸 표기 수")
        d.caption("라운드별 LLM 표기 수 — 사전이 자랄수록 LLM 몫이 준다")
        d.hold(5)

        # 3. S4 인사이트 ──────────────────────────────────────────────────────
        d.goto("인사이트")
        d.collapse_sidebar()
        d.caption(
            "S4 인사이트 — 같은 탐지기를 네 가지 분류 결과 위에서 돌린다 (§5.2)",
            "탐지기는 네 줄에서 완전히 같다 · 바뀌는 것은 그 위의 분류뿐",
        )
        d.hold(4)
        for label, sub in [
            ("정규화만", "가맹점명을 못 묶으면 결제가 여러 표기로 쪼개져 '최소 3회' 조건을 못 채운다"),
            ("A · 룰만", "룰 사전이 표기를 묶자 구독이 보이기 시작한다"),
            ("C · 하이브리드", "A 와 같다 — 구독 브랜드는 이미 룰 사전에 있다. 남은 격차는 오귀속"),
            ("정답 분류(상한)", "1부가 완벽할 때의 상한 — 그 격차(쿠팡와우→쿠팡)를 사람이 고친다"),
        ]:
            page.locator('[data-testid="stRadio"] label').filter(has_text=label).first.click()
            d.settle()
            d.caption(f"분류 단계: {label} — 찾은 구독 수를 보라", sub)
            d.hold(5)
        d.scroll_to_text("구독 — 매달 나가는 돈")
        d.caption("구독 목록 — 결제 횟수 · 주기 · 금액 변동이 근거 문장으로 붙는다", "LLM 없이 결정론 통계 규칙으로 탐지한다")
        d.hold(5)

        # 4. S2 지표 대시보드 ─────────────────────────────────────────────────
        d.goto("지표_대시보드")
        d.collapse_sidebar()
        d.caption("S2 지표 대시보드 — 수치를 그 자리에서 다시 잰다", "실호출이 필요한 (B) 한 줄만 스냅샷 · 나머지는 지금 계산")
        d.hold(5)
        d.scroll_to_text("§5.1 베이스라인 3종")
        d.caption(
            "베이스라인 3종 — B(LLM만)가 A(룰만)보다 낮다",
            "사전 없이 LLM 만 쓰면 커버리지는 올라도 자신 있게 틀린다 · C 하이브리드 98.9%",
        )
        d.hold(7)
        d.scroll_to_text("§5.2 분류 → 탐지 전이")
        d.caption("§5.2 — 가맹점명을 못 묶으면 구독도 못 찾는다", "구독 재현율 56.2% → 74.0% → 상한 90.4%")
        d.hold(6)
        d.scroll_to_text("승격 라운드")
        d.caption("승격 라운드 — 사전이 자라며 LLM 몫이 준다")
        d.hold(5)

        d.caption("")
        d.card("TxLens", [
            "1부 정규화·분류 → 2부 패턴 탐지를 하나의 인과 사슬로",
            "LangGraph 8노드 · HITL interrupt · 사전 승격 루프 · 380 tests",
        ])
        d.hold(4)
        video = page.video
        ctx.close()
        b.close()
        webm = Path(video.path()) if video else None

    assert webm and webm.exists(), "녹화 파일이 없다"
    out.parent.mkdir(parents=True, exist_ok=True)
    return encode(webm, out, skip=max(t_cut - 0.3, 0))


def encode(webm: Path, out: Path, *, skip: float) -> Path:
    try:
        import imageio_ffmpeg
    except ImportError:
        target = out.with_suffix(".webm")
        shutil.copy(webm, target)
        print(f"  ! imageio-ffmpeg 가 없어 webm 으로 남긴다 → {target}")
        return target
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-ss", f"{skip:.2f}", "-i", str(webm),
        # 발표 기기 호환이 최우선이다 — H.264 + yuv420p 는 어떤 플레이어도 연다 (D-053 재생 위험)
        "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
        "-r", "30", "-movflags", "+faststart", "-an", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="TxLens 데모 영상 녹화")
    ap.add_argument("--base", default="http://localhost:8501")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--browser", default=None, help="Chromium 실행 파일 (기본: Playwright 내장)")
    args = ap.parse_args()
    out = record(args.base, args.out, args.browser)
    print(f"  → {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}"
          f"  {out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
