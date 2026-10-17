# G2-01 — verified paper facts (research notes, NOT submission text)

Purpose: raw, checked facts so that you (the student) can write the Phase 1 literature
review in your own words. The course forbids AI-written text, so nothing here is
phrased for submission. Every row says how it was checked and what is still open.
Verification date: 2026-10-11. Sources were fetched from the venue/arXiv pages named below.
I only saw abstracts / venue pages, NOT the full papers. You must read each paper yourself
before summarising it (limitations in particular).

## Candidate base paper

**Who&When** — Zhang, Yin, Zhang, Liu, Han, Zhang, Li, Wang, Wang, Chen, Wu.
"Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems."
ICML 2025, PMLR 267:76583–76599. arXiv 2505.00212. https://proceedings.mlr.press/v267/zhang25cq.html
- Dataset Who&When: 184 annotated failure tasks (two subsets: Algorithm-Generated via CaptainAgent, Hand-Crafted e.g. Magentic-One); queries from GAIA and AssistantBench; labels = responsible agent, decisive error step, explanation. HF: Kevin355/Who_and_When (parquet files).
- Abstract numbers: best method 53.5% agent-level, 14.2% step-level; some methods below random.
- Three methods: all-at-once, step-by-step, binary search.
- Venue rank: ICML is listed as A* by CORE to my knowledge, BUT I could not open the CORE portal page for ICML in this session (robots block). **You must open https://portal.core.edu.au/conf-ranks/ , search "ICML", and screenshot the CORE2023 rank as proof.**
- Why it is a good base: every 2025–26 attribution paper below evaluates on its dataset, so comparison is direct.
- Fallback if ranking proof fails: use AgenTracer (ICLR 2026, ICLR is CORE A*, proof fetched) as base.

## Reviewed papers (venue evidence)

| # | Paper | Venue / evidence | Notes from abstract/venue page only |
|---|-------|------------------|-------------------------------------|
| 1 | Who&When (above) | ICML 2025 — proceedings page | base candidate |
| 2 | AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems? Zhang, Wang, Chen, Zhou, Wang, Yan | ICLR 2026 — proceedings.iclr.cc PDF read | counterfactual replay + programmatic fault injection -> TracerTraj-2.5K (>2,000 trajectories); AgenTracer-8B (Qwen3-8B, GRPO). Who&When handcrafted w/ GT: agent-level 69.10%, step-level 20.68% (vs Claude-Sonnet-4 17.24%); automated step-level 42.86%. Own limits visible: relies on DeepSeek-R1 analyser; step accuracy still modest. |
| 3 | DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems. Ma, Zhang, Yang, Kang, Lin, Rajmohan, Zhang | ICLR 2026 poster — iclr.cc page confirmed | intervention-based validation of hypotheses; flips 18–28% of failed trials (Magentic-One on GAIA/AssistantBench), validates/refutes 30–60% of hypotheses, 49% recovery on GSMPlus/AG2 |
| 4 | Aegis: Automated Error Generation and Attribution for MAS. Kong et al. | ICLR 2026 poster — iclr.cc page confirmed | awesome-list says 9,533 trajectories; I did NOT read the paper — check numbers yourself |
| 5 | Why Do Multi-Agent LLM Systems Fail? (MAST). Cemri et al. | NeurIPS 2025 spotlight poster — neurips.cc/virtual/2025/poster/121528 (track main vs D&B not shown; check) | MAST-Data 1,600+ annotated traces, 7 frameworks; taxonomy 14 failure modes / 3 categories; kappa 0.88 |
| 6 | Seeing the Whole Elephant (TraceElephant). Chen, Wang, Mu, Wang, Liu, Feng, Wang | ACL 2026 long, pp. 19888–19905, DOI 10.18653/v1/2026.acl-long.912 — ACL Anthology confirmed | full execution traces (inputs+context) vs output-only: up to 76.5% better attribution. **Directly supports an observability gap.** |
| 7 | Process-Level Trajectory Evaluation for Environment Configuration in SE Agents (EnConda-Bench). Kuang et al. | ICLR 2026 — proceedings PDF read | 4,201 READMEs, 323 repos, 9,471 injected errors; best Pass@1 22.9; authors' limits: Python only, LLM-generated errors |
| 8 | AgentRx: Diagnosing AI Agent Failures from Execution Trajectories. Barke, Goyal, Khare, Singh, Nath, Bansal | arXiv comments field says EMNLP Findings 2026 — **not verified on ACL Anthology** | 170 failed trajectories / 11 settings; constraint synthesis + step checks + LLM judge; +75% step localisation (avg) vs prior work. Closest prior work to a constraint/verifier approach. |
| 9 | DiLLS: Interactive Diagnosis of LLM-based MAS via Layered Summary of Agent Behaviors. Sheng et al. | DOI 10.1145/3772318.3790815 (ACM); "CHI 2026" comes from a third-party list — **verify** | layered summaries (activities/actions/operations); evidence is a user study, no faithfulness metric reported in abstract |
| 10 | ECHO: Where Did It All Go Wrong? Banerjee, Nair, Borogovac | NeurIPS 2025 **workshop** (Evaluating the Evolving LLM Lifecycle) — workshop poster, weaker venue | hierarchical context + consensus voting; abstract gives no numbers |

If the supervisor wants stronger venues, replace #9/#10 using the pool below.

## Preprints / unverified-venue pool (arXiv only until you confirm; Phase 1 sheet says arXiv counts only with supervisor approval)

- TRAIL (2505.08638) Deshpande et al., 2025. 148 traces / 841 labelled errors (latter from a third-party list); best model Gemini-2.5-pro 11%.
- A2P Abduct-Act-Predict (2509.10401), West et al.: Who&When step accuracy 47.46% (algorithm-generated) and 29.31% (hand-crafted) vs 16.67% / 12.07% baselines. Venue not stated.
- CDC-MAS (2509.08682), Ma et al.: up to 36.2% step-level on Who&When+TRAIL (self-reported).
- FAMAS (2509.13782), Ge et al.: spectrum analysis; third-party list says ESEC/FSE 2026 — verify.
- ECHO, FALAT (2606.00765), From Flat Logs to Causal Graphs (2602.23701), AgentDebugX, TelemetrySuffBench (2608.07899), Aligning Agents via Planning (ACL 2026 long 1062 — fetched title only), Evaluating and Advancing Process Reward Models for ... (ACL 2026 Findings 602 — title only).

## Ranking proof status

| Venue | Rank | Proof fetched? |
|-------|------|----------------|
| ICLR | CORE A* (CORE2023, CORE2021) | YES — portal.core.edu.au/conf-ranks/2273 |
| ACL | CORE A* (2023…) | YES — portal.core.edu.au/conf-ranks/196 |
| ICML | (A* per my knowledge) | NO — portal blocked; take screenshot yourself |
| NeurIPS | (A* per my knowledge) | NO — same |
| EMNLP | not checked | NO |

## Candidate research gaps (my engineering proposal — edit with your supervisor)

1. **Step-level localisation stays weak.** Evidence: Who&When 14.2% step (best of 3 methods); TRAIL best 11%; A2P reports <17% prior; AgenTracer 20.68% on handcrafted. LLM-judge methods dominate; deterministic evidence-linking/replay verification is not compared. Measure: step accuracy / top-k on controlled fault-injected benchmark (+ Who&When on your laptop).
2. **Observability.** Evidence: TraceElephant (full vs partial traces, up to 76.5%). Who&When logs contain outputs only. Open: *which* trace fields (tool inputs, dependency links, memory ops, plan) carry the gain, and does a lightweight schema recover it? Measure: field-group ablation of step accuracy.
3. **Summaries are judged by readability/user study, not faithfulness + debugging utility.** Evidence: DiLLS (user study; no faithfulness metric in abstract); project brief explicitly asks for faithfulness and debugging utility. Measure: critical-event recall at fixed budget, faithfulness (every claim resolves to a real event whose content matches), downstream localisation from summary only.

Read AgentRx and AgenTracer in full before you commit gap 1: AgentRx is the closest competitor.
