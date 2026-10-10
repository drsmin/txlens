# TxLens — 마이데이터 거래내역 인텔리전스 에이전트

가맹점명이 자유 텍스트로 들어오는 마이데이터 거래내역을 **정규화·분류**하고, 그 위에서
**구독·이상지출·이상거래패턴**을 탐지하는 LangGraph 에이전트.
「KB AI 엘리트 교육」 6주 과정 최종 제출물 (최종 발표 2026-10-17).

**핵심 주장: "가맹점명을 못 묶으면 구독도 못 찾는다."** 1부(정규화·분류)와 2부(패턴 탐지)는
하나의 인과 사슬이고, 같은 탐지기를 서로 다른 분류 결과 위에서 돌려 그 사슬을 잰다(§5.2).

## 제출물

| 무엇 | 위치 |
| --- | --- |
| 발표 자료 (PDF) | [`deliverables/TxLens-final.pdf`](deliverables/) |
| 데모 영상 (2분 25초 · 1080p) | [`deliverables/txlens-demo.mp4`](deliverables/) — 화면 + 오른쪽 칸의 LangGraph 실행 로그 |
| 제출 노트북 | [`notebooks/txlens_demo.ipynb`](notebooks/txlens_demo.ipynb) — 실행 결과가 저장돼 있다 |
| 소스 | `src/txlens/` (패키지) · `app/` (Streamlit 화면) · `tests/` (380 tests) |

**노트북은 얇은 오케스트레이터다.** 로직은 전부 `src/txlens/` 에 있고 노트북은 그것을 import 해서
부르기만 한다 — 노트북의 결과가 곧 실제 프로그램(CLI·화면)의 결과다. 그래서 패키지를 함께 낸다.

## 구조 한눈에

```
ingest → pair_cancellations → normalize → rule_match → collect_targets
       → (미매칭이 있으면) llm_classify → rule_promote → finalize
```

- **1부 정규화·분류** — 룰 사전(head)이 먼저 잡고, 못 푼 꼬리만 LLM Agent(`llm_classify` 안의
  LangChain 1.0 `create_agent`)에 보낸다. 고신뢰 판정은 `rule_promote` 가 사전 별칭으로 **승격**해
  다음 라운드의 LLM 몫을 줄인다. 새 브랜드는 사람만 만든다(HITL — `interrupt()` 로 멈추는 검토 큐).
- **2부 패턴 탐지** — 그래프 밖의 결정론 통계 규칙(LLM 미사용): 구독 · 이상지출 · 이상거래.
- **분류 체계** — L1 11개 · L2 40개 (`config/taxonomy.yaml`, 코드 Enum 과 테스트로 동기).

| 베이스라인 (같은 그래프, 사전·분류기만 다르다) | 거래 L2 | 키 L2 | LLM 호출 |
| --- | --- | --- | --- |
| A · 룰만 | 93.0% | 42.0% | 0 |
| B · LLM만 | 85.6% | 66.7% | 1,026 |
| C · 하이브리드 | **98.9%** | **86.7%** | 580 |

B 가 A 보다 낮다 — 사전 없이 LLM 만 쓰면 커버리지는 올라도 **자신 있게 틀린다.**
§5.2 전이 실험: 같은 구독 탐지기의 재현율이 정규화만 56.2% → 룰 74.0% → 정답 분류 90.4%.

## 실행

Python 3.11 이상.

```bash
pip install -r requirements.txt && pip install -e .
python -m txlens.generator --seed 42      # 합성 데이터 (거래 8,969건 · 정답 동시 생성)
pytest                                    # 380 tests
```

**API 키 없이 도는 것** — 저장된 LLM 판정(`data/gold/verdicts-*.json`)을 재생한다.

```bash
pip install ipykernel && jupyter execute notebooks/txlens_demo.ipynb   # 노트북 (Jupyter·VS Code 로 열어도 된다)
streamlit run app/Home.py                                # 화면: Home · S1 검토 큐 · S2 지표 · S4 인사이트
python -m txlens.eval                                    # 베이스라인 A
python -m txlens.detect --transfer --verdicts data/gold/verdicts-v1-gpt-4o-mini.json   # §5.2
python -m txlens.promote --from-verdicts data/gold/verdicts-v1-gpt-4o-mini.json --dry-run
```

**실제 모델을 부르는 것** — `cp .env.example .env` 후 `OPENAI_API_KEY` 를 넣는다.

```bash
python -m txlens.eval --baseline all     # A/B/C 실측
python -m txlens.promote --rounds 3      # 승격 루프 실호출
```

로컬 MCP 서버(`python -m txlens.mcp_server`, 설정 `.mcp.json`)로 분류 도구 5종을 외부 에이전트에도
내보낸다.

## 디렉터리

```
src/txlens/      패키지 — graph(LangGraph 8노드) · agents · rules · detect · eval · promote · generator · mcp_server
app/             Streamlit 화면 (재생 대본으로 돈다 — API 0)
notebooks/       제출 노트북
tests/           380 tests
config/          택소노미 · 모델 프로필
rules/           룰 사전 v1~v5 (승격 라운드마다 한 판)
data/gold/       저장된 LLM 판정 · 실호출 스냅샷 (합성 데이터 자체는 시드로 재생성)
assets/diagrams/ build_graph() 가 컴파일한 그래프에서 뽑은 그림
scripts/         데모 영상 녹화(record-demo.py) · 그래프 그림(render-graph.py)
deliverables/    발표 PDF · 데모 영상
```
