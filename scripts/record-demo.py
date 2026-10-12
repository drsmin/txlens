#!/usr/bin/env python3
"""데모 영상을 뜬다 — Streamlit 화면을 브라우저로 조작하며 녹화한다 (D-053 · D-059 · D-060).

왜 사람이 화면 녹화기로 뜨지 않는가: 녹화는 **다시 뜰 수 있어야** 라이브보다 안전하다(D-053).
손으로 뜨면 클릭 순서·대기 시간이 매번 달라 수치를 고친 뒤 같은 영상을 다시 만들 수 없다.
이 스크립트는 같은 장면을 같은 순서로 다시 뜬다. 화면은 재생 대본으로 돌므로 **API 는 0** 이다.

**사람이 조작한 것처럼 보이게 한다** (D-060). 첫 판은 커서가 없고 클릭이 순간이동하며 대기가
매번 같아 기계가 만든 영상으로 읽혔다. 그래서 ① 커서를 그리고 곡선·가감속으로 옮긴다
② 타이핑·대기에 변동을 준다(시드 고정 — 다시 떠도 같은 영상) ③ 스크롤은 휠로 ④ 자막은
왼쪽 아래 한 줄 ⑤ 강조 테두리 대신 커서가 로그를 짚는다. 소리는 넣지 않는다 — "실제로 이렇게
돈다"만 보여 주면 된다(사용자 결정).

장면 (약 2분 반):
    Home → S1 검토 큐(interrupt 에서 멈춤 → 교정 3줄 → 재개 → 승격 → 라운드 2)
         → S4 인사이트(§5.2 네 단계 토글) → S2 지표 대시보드
오른쪽 칸의 실행 로그(D-054)가 그래프가 실제로 도는 증거다. 로그 타임스탬프는 그래프의 실제
시간이고, 읽을 시간은 그래프 밖(장면 사이 대기)에서만 준다 — 그래프에 지연을 넣으면 `[0.31s]`
가 성능을 거짓말한다 (D-053).

사용:
    streamlit run app/Home.py --server.port 8501 --server.headless true   # 먼저 띄운다
    python scripts/record-demo.py                     # → docs/reports/media/txlens-demo.mp4
    python scripts/record-demo.py --out /tmp/x.mp4 --browser /path/to/chrome

필요한 것: `playwright`(requirements-dev) · mp4 변환용 `imageio-ffmpeg`(없으면 webm 만 남긴다).
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

# 녹화만 브라우저를 쓴다 — `--still-only` 는 영상 파일만 읽으므로 playwright 없이도 돌아야
# 한다(발표 기기에서 스틸·재생 조건만 다시 확인할 때).
if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "reports" / "media" / "txlens-demo.mp4"
W, H = 1920, 1080

# 덱에 들어갈 스틸 (D-053 — PDF 는 영상을 실어 나르지 못하므로 그 자리에 한 장을 둔다).
# 32초 지점: S1 검토 큐 · 오른쪽에 8노드 실행 로그 · 커서가 분류 칸에 있고 자막이 교정 3줄을
# 말한다 — D-059 의 계기("랭그래프가 작동하는 로그")가 한 장에 다 들어가는 유일한 프레임이다.
# 손으로 캡처하지 않는다: 영상을 입력으로 삼으므로 영상을 다시 뜨지 않으면 같은 그림이 나온다.
STILL_AT = 32.0
DEFAULT_STILL = ROOT / "assets" / "diagrams" / "demo-still-s1.png"
SEED = 7  # 변동은 주되 다시 떠도 같은 영상이 나오게

# S1 에서 고칠 세 줄 — 큐 상위 3개(덮는 거래 65+30+27 = 122건). 정답은 생성기의 브랜드 표다.
#   우리동네세탁: 세탁업은 택소노미에 자리가 없어 정답이 **미분류**다(D-026) — LLM 답을 그대로 승인
#   제이디에프컴퍼니·디엠리테일홀딩스: 법인명과 브랜드가 전혀 다른 가상 브랜드 — 사람만 안다
CORRECTIONS = [
    (0, None, None),
    (1, "빈스로드", "카페"),
    (2, "데일리마트", "식료품"),
]

# 페이지가 바뀔 때마다 다시 심는다(`add_init_script`) — Streamlit 재실행은 DOM 을 갈지 않지만
# `goto` 는 간다. 커서는 실제 마우스 이벤트를 따라가므로 캔버스 그리드 위에서도 움직인다.
CURSOR_INIT = """
(() => {
  const mk = () => {
    if (document.getElementById('demo-cursor')) return;
    const c = document.createElement('div');
    c.id = 'demo-cursor';
    c.innerHTML = '<svg width="28" height="28" viewBox="0 0 24 24"><path d="M4 2l7 19 2.6-7.6L21 11z"' +
      ' fill="#1b1f27" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    c.style.cssText = 'position:fixed;left:-50px;top:-50px;z-index:2147483647;pointer-events:none;' +
      'transform:translate(-4px,-2px);filter:drop-shadow(0 1px 2px rgba(0,0,0,.35))';
    const r = document.createElement('div');
    r.id = 'demo-ripple';
    r.style.cssText = 'position:fixed;z-index:2147483646;pointer-events:none;width:36px;height:36px;' +
      'margin:-18px 0 0 -18px;border-radius:50%;background:rgba(255,75,75,.35);opacity:0';
    document.body.appendChild(r);
    document.body.appendChild(c);
    document.addEventListener('mousemove', e => {
      c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px';
    }, true);
    document.addEventListener('mousedown', e => {
      r.style.transition = 'none'; r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px';
      r.style.transform = 'scale(.3)'; r.style.opacity = '1';
      requestAnimationFrame(() => requestAnimationFrame(() => {
        r.style.transition = 'transform .4s ease-out, opacity .5s ease-out';
        r.style.transform = 'scale(1.2)'; r.style.opacity = '0';
      }));
    }, true);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mk);
  else mk();
})();
"""

CAPTION_JS = """
(text) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:28px;bottom:24px;z-index:999999;' +
      'background:rgba(27,31,39,.86);color:#fff;padding:9px 16px;border-radius:8px;' +
      'font:500 21px "NanumBarunGothic",sans-serif;max-width:1300px;transition:opacity .25s';
    document.body.appendChild(el);
  }
  el.textContent = text;
  el.style.opacity = text ? '1' : '0';
}
"""

CARD_JS = """
([title, line]) => {
  let el = document.getElementById('demo-card');
  if (!title) { if (el) el.remove(); return; }
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-card';
    el.style.cssText = 'position:fixed;inset:0;z-index:1000000;background:#1b1f27;color:#fff;' +
      'display:flex;flex-direction:column;justify-content:center;align-items:center;' +
      'font-family:"NanumBarunGothic",sans-serif';
    document.body.appendChild(el);
  }
  el.innerHTML = '<div style="font-size:56px;font-weight:700;margin-bottom:18px">' + title +
    '</div><div style="font-size:26px;opacity:.8">' + line + '</div>';
}
"""


class Director:
    """장면 진행과 '사람 손'을 한곳에 모은다 — 대기·움직임이 흩어지면 영상 길이를 가늠할 수 없다."""

    def __init__(self, page: Page, base: str, rnd: random.Random | None = None) -> None:
        self.page, self.base = page, base
        self.rnd = rnd or random.Random(SEED)
        self.pos = (W * 0.55, H * 0.45)

    # ── 화면 위 장치 ─────────────────────────────────────────────────────────
    def caption(self, text: str) -> None:
        self.page.evaluate(CAPTION_JS, text)

    def card(self, title: str, line: str = "") -> None:
        self.page.evaluate(CARD_JS, [title, line])

    # ── 시간 ─────────────────────────────────────────────────────────────────
    def hold(self, seconds: float) -> None:
        """읽을 시간. 매번 똑같으면 기계처럼 보여 ±15% 흔든다."""
        self.page.wait_for_timeout(int(seconds * 1000 * self.rnd.uniform(0.88, 1.15)))

    def beat(self, lo: int = 150, hi: int = 380) -> None:
        self.page.wait_for_timeout(self.rnd.randint(lo, hi))

    def settle(self, timeout: float = 30) -> None:
        """Streamlit 재실행이 끝날 때까지 — 오른쪽 위 'Running' 표시가 사라지면 끝이다."""
        self.page.wait_for_timeout(400)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.page.locator('[data-testid="stStatusWidget"]').count() == 0:
                break
            self.page.wait_for_timeout(200)
        self.page.wait_for_timeout(300)

    # ── 손 ───────────────────────────────────────────────────────────────────
    def move_to(self, x: float, y: float) -> None:
        """곡선 + 가감속으로 옮긴다. 직선 등속이면 로봇처럼 보인다."""
        x0, y0 = self.pos
        dist = math.hypot(x - x0, y - y0)
        if dist < 2:
            return
        n = max(10, min(45, int(dist / 22)))
        bend = self.rnd.uniform(-0.18, 0.18) * dist
        nx, ny = -(y - y0) / dist, (x - x0) / dist  # 진행 방향의 수직
        cx, cy = (x0 + x) / 2 + nx * bend, (y0 + y) / 2 + ny * bend
        for i in range(1, n + 1):
            t = i / n
            t = t * t * (3 - 2 * t)  # ease-in-out
            bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x
            by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y
            self.page.mouse.move(bx, by)
            self.page.wait_for_timeout(self.rnd.randint(7, 14))
        self.pos = (x, y)

    def click_xy(self, x: float, y: float, *, double: bool = False) -> None:
        self.move_to(x, y)
        self.beat(120, 260)
        if double:
            self.page.mouse.dblclick(x, y)
        else:
            self.page.mouse.click(x, y)
        self.beat(180, 360)

    def point(self, loc: Locator, fx: float = 0.5, fy: float = 0.5) -> tuple[float, float]:
        box = loc.bounding_box()
        assert box is not None, "가리킬 요소가 화면에 없다"
        x = box["x"] + box["width"] * min(0.9, max(0.1, fx + self.rnd.uniform(-0.08, 0.08)))
        y = box["y"] + box["height"] * min(0.9, max(0.1, fy + self.rnd.uniform(-0.12, 0.12)))
        return x, y

    def click(self, loc: Locator) -> None:
        loc.scroll_into_view_if_needed()
        self.click_xy(*self.point(loc))

    def type_text(self, text: str) -> None:
        for ch in text:
            self.page.keyboard.type(ch)
            self.page.wait_for_timeout(self.rnd.randint(70, 190))
        self.beat(200, 420)

    def wheel_to(self, loc: Locator, top: float = 170) -> None:
        """휠로 굴려 요소를 화면 위쪽에 둔다 — `scrollIntoView` 는 순간이동이라 사람 손이 아니다."""
        self.move_to(W * self.rnd.uniform(0.28, 0.42), H * self.rnd.uniform(0.45, 0.6))
        for _ in range(60):
            box = loc.bounding_box()
            if box is None:
                break
            delta = box["y"] - top
            if abs(delta) < 40:
                break
            step = max(-360, min(360, delta)) * self.rnd.uniform(0.75, 1.0)
            self.page.mouse.wheel(0, step)
            self.page.wait_for_timeout(self.rnd.randint(55, 110))
        self.beat(300, 600)

    def wheel_top(self) -> None:
        self.move_to(W * 0.35, H * 0.5)
        for _ in range(60):
            top = self.page.evaluate(
                "() => document.querySelector('[data-testid=\"stMain\"]').scrollTop"
            )
            if top <= 0:
                break
            self.page.mouse.wheel(0, -max(120, min(360, top)) * self.rnd.uniform(0.75, 1.0))
            self.page.wait_for_timeout(self.rnd.randint(55, 110))
        self.beat(300, 600)

    # ── 화면 단위 ────────────────────────────────────────────────────────────
    def goto(self, path: str) -> None:
        self.page.goto(f"{self.base}/{path}")
        self.page.wait_for_selector('[data-testid="stMain"]')
        self.settle()

    def collapse_sidebar(self) -> None:
        # 사이드바를 접어야 로그 칸이 70자 줄을 자르지 않는다 (D-054 의 6:4 가 1920px 에서 성립)
        # 접힌 상태는 페이지를 옮겨도 남는다 — 펼쳐져 있을 때만 누른다
        if self.page.locator('[data-testid="stSidebar"][aria-expanded="true"]').count() == 0:
            return
        self.move_to(160, 34)  # 머리 쪽에 올라가야 접기 단추가 나타난다
        self.beat(300, 500)
        self.click(self.page.locator('[data-testid="stSidebarCollapseButton"] button').first)
        self.page.wait_for_timeout(700)

    def trace_log(self) -> None:
        """커서로 실행 로그를 위에서 아래로 짚어 내려간다 — 사람이 화면을 가리키며 보여 주듯."""
        code = self.page.locator('[data-testid="stCode"]').first
        box = code.bounding_box()
        if box is None:
            return
        x = box["x"] + 70
        self.move_to(x, box["y"] + 22)
        self.beat(400, 700)
        lines = max(3, int((box["height"] - 30) / 22))
        for i in range(1, lines):
            self.move_to(x + self.rnd.uniform(-6, 30), box["y"] + 22 + i * 22.5)
            self.page.wait_for_timeout(self.rnd.randint(260, 520))
        self.beat(400, 800)

    def metric(self, label: str, *, last: bool = False) -> str:
        """화면에 지금 떠 있는 지표 값을 읽는다 — 자막의 수를 손으로 적지 않는다."""
        ms = self.page.locator('[data-testid="stMetric"]').filter(has_text=label)
        m = ms.last if last else ms.first
        return m.locator('[data-testid="stMetricValue"]').inner_text().strip()


def edit_queue(d: Director) -> None:
    """검토 표(캔버스 그리드)를 사람처럼 고친다. 좌표는 그리드 상자에서 계산한다."""
    page = d.page
    box = page.locator('[data-testid="stDataFrame"]').first.bounding_box()
    assert box is not None, "검토 표가 없다 — 큐가 비었거나 화면이 바뀌었다"
    x0, y0 = box["x"], box["y"]

    def row_y(i: int) -> float:  # 헤더 36px · 행 35px
        return y0 + 36 + 35 * i + 17

    x_apply, x_name, x_cat = x0 + 38, x0 + 450, x0 + 720

    d.click_xy(x_name + 20, row_y(0))  # 첫 클릭은 그리드에 포커스만 준다
    for row, canonical, category in CORRECTIONS:
        y = row_y(row)
        if canonical:
            d.click_xy(x_name + d.rnd.uniform(-30, 40), y)
            page.keyboard.press("Enter")
            d.beat(250, 450)
            page.keyboard.press("Control+A")
            d.type_text(canonical)
            page.keyboard.press("Enter")
            d.beat(400, 700)
        if category:
            d.click_xy(x_cat + d.rnd.uniform(-40, 60), y, double=True)
            d.beat(400, 650)
            d.type_text(category)
            d.beat(300, 500)
            page.keyboard.press("Enter")
            d.beat(400, 700)
        d.click_xy(x_apply, y)
        d.beat(500, 900)


def record(base: str, out: Path, browser: str | None) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="txlens-rec-"))
    from playwright.sync_api import sync_playwright

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
        ctx.add_init_script(CURSOR_INIT)
        page = ctx.new_page()
        t_start = time.monotonic()
        d = Director(page, base)

        # 표지 ────────────────────────────────────────────────────────────────
        d.goto("")
        d.card("TxLens 데모", "실제 화면 동작 · 저장된 LLM 판정을 재생 (API 호출 없음)")
        t_cut = time.monotonic() - t_start  # 이 앞(빈 페이지 로딩)은 잘라낸다
        d.hold(3)
        d.card("")

        # Home ────────────────────────────────────────────────────────────────
        d.caption("Home · 실행 준비 상태 확인 (데이터 · 룰 사전 · 재생 대본)")
        d.move_to(W * 0.3, H * 0.33)
        d.hold(3.5)

        # S1 검토 큐 ──────────────────────────────────────────────────────────
        d.click(page.get_by_role("link", name="검토 큐").first)
        d.settle()
        d.collapse_sidebar()
        d.caption("S1 검토 큐 · 그래프 8노드 실행 → interrupt()에서 사람 대기 (오른쪽이 실행 로그)")
        d.trace_log()
        d.hold(1.5)
        d.caption("확신 낮은 표기 중 영향 큰 순서로 20개")
        d.move_to(W * 0.25, H * 0.45)
        d.hold(3)
        d.caption("교정 3줄: 제이디에프컴퍼니 → 빈스로드 · 디엠리테일홀딩스 → 데일리마트 · 세탁은 미분류 그대로")
        edit_queue(d)
        d.hold(1)

        d.caption("교정 제출 → 멈췄던 자리에서 그래프 재개")
        d.click(page.get_by_role("button", name="교정 제출"))
        d.settle()
        d.hold(1.5)
        d.wheel_to(page.get_by_text("승격 결과", exact=False).first)
        d.caption(f"사전 {d.metric('룰 사전', last=True)} · 거래 {d.metric('사람 교정이 덮는 거래')} 바로 반영")
        d.move_to(W * 0.36, H * 0.17)
        d.hold(5)

        d.caption("커진 사전으로 다시 실행 (라운드 2)")
        d.click(page.get_by_role("button", name="승격된 사전으로 다시 돌린다"))
        d.settle()
        d.wheel_top()
        d.caption("라운드 2 · collect_targets: LLM에 보낼 표기 155 → 115")
        d.trace_log()
        d.hold(2)
        d.wheel_to(page.get_by_text("라운드별 — LLM 에 보낸 표기 수", exact=False).first)
        d.caption("라운드별 LLM 표기 수")
        d.hold(3.5)

        # S4 인사이트 ─────────────────────────────────────────────────────────
        d.goto("인사이트")
        d.collapse_sidebar()
        d.caption("S4 인사이트 · 탐지기는 그대로, 밑에 깔린 분류 단계만 바꿔 보기")
        d.hold(3)
        for label in ("정규화만", "A · 룰만", "C · 하이브리드", "정답 분류(상한)"):
            d.click(page.locator('[data-testid="stRadio"] label').filter(has_text=label).first)
            d.settle()
            d.caption(f"분류 단계 {label} → 찾은 구독 {d.metric('찾은 구독')}")
            d.move_to(*d.point(page.locator('[data-testid="stMetric"]').filter(has_text="찾은 구독").first))
            d.hold(3.5)
        d.wheel_to(page.get_by_text("구독 — 매달 나가는 돈", exact=False).first)
        d.caption("구독 목록 · 결제 횟수·주기·금액 변동이 근거로 붙음 (LLM 없이 통계 규칙)")
        d.hold(4)

        # S2 지표 대시보드 ────────────────────────────────────────────────────
        d.goto("지표_대시보드")
        d.collapse_sidebar()
        d.caption("S2 지표 대시보드 · 수치를 화면에서 바로 다시 계산")
        d.hold(3.5)
        d.wheel_to(page.get_by_text("§5.1 베이스라인 3종", exact=False).first)
        d.caption("베이스라인 3종 · LLM만 쓴 B가 룰만 쓴 A보다 낮음")
        d.hold(5)
        d.wheel_to(page.get_by_text("§5.2 분류 → 탐지 전이", exact=False).first)
        d.caption("가맹점명을 못 묶으면 구독도 못 찾음 · 재현율 56% → 74% → 90%")
        d.hold(5)
        d.wheel_to(page.get_by_text("승격 라운드", exact=False).first)
        d.caption("승격 라운드가 돌수록 LLM 몫이 줄어듦")
        d.hold(4)
        d.caption("")
        d.hold(1)

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


def ffmpeg() -> str:
    """정적 ffmpeg 경로. 없으면 왜 없는지 말하고 끝낸다 — 조용히 넘기면 낡은 스틸이 덱에 남는다."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - 환경 문제
        raise SystemExit("imageio-ffmpeg 가 없다: pip install -r requirements-dev.txt") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def still(video: Path, out: Path, *, at: float = STILL_AT) -> Path:
    """영상에서 덱에 넣을 한 장을 뽑는다 (녹화를 다시 뜨지 않는다)."""
    assert video.exists(), f"영상이 없다: {video}"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", str(video),
         "-frames:v", "1", str(out)],
        check=True,
    )
    return out


def playback_check(video: Path) -> bool:
    """발표 기기에서 열리는 조건을 확인한다 (D-053 — 남은 위험이 재생 환경 하나다).

    h264 + yuv420p 면 어떤 플레이어도 열고, 오디오 스트림이 없어야 소리를 안 찾는다.
    """
    probe = subprocess.run(
        # ffmpeg 는 입력만 주면 스트림 정보를 찍고 1 로 끝난다 — 그게 정상이다
        [ffmpeg(), "-hide_banner", "-i", str(video)],
        capture_output=True, text=True, check=False,
    ).stderr
    ok = True
    for want, label in (("Video: h264", "H.264"), ("yuv420p", "yuv420p")):
        hit = want in probe
        ok &= hit
        print(f"  {'✓' if hit else '✗'} {label}")
    silent = "Audio:" not in probe
    ok &= silent
    print(f"  {'✓' if silent else '✗'} 오디오 스트림 없음 (무음)")
    for line in probe.splitlines():
        if "Duration" in line:
            print(f"  · {line.strip().split(',')[0]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="TxLens 데모 영상 녹화")
    ap.add_argument("--base", default="http://localhost:8501")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--browser", default=None, help="Chromium 실행 파일 (기본: Playwright 내장)")
    ap.add_argument("--still", type=Path, default=DEFAULT_STILL,
                    help=f"덱에 넣을 스틸 경로 (기본: {DEFAULT_STILL.relative_to(ROOT)})")
    ap.add_argument("--still-only", action="store_true",
                    help="녹화하지 않고 기존 영상에서 스틸만 다시 뽑고 재생 조건을 검사한다")
    args = ap.parse_args()

    if args.still_only:
        png = still(args.out, args.still)
        print(f"  → {png.relative_to(ROOT)}  {png.stat().st_size / 1e6:.1f} MB  (t={STILL_AT:.0f}s)")
        print("재생 조건 (D-053):")
        return 0 if playback_check(args.out) else 1

    out = record(args.base, args.out, args.browser)
    print(f"  → {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}"
          f"  {out.stat().st_size / 1e6:.1f} MB")
    if out.suffix == ".mp4":
        png = still(out, args.still)
        print(f"  → {png.relative_to(ROOT)}  (스틸 t={STILL_AT:.0f}s · 덱이 이 파일을 참조한다)")
        print("재생 조건 (D-053):")
        playback_check(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
