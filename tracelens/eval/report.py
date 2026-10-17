"""Self-contained HTML trace viewer: raw log next to the critical-trace summary (no server, no internet)."""
from __future__ import annotations

import html
import json
from typing import Dict, List, Optional

from ..extract.evidence import EvidenceVerifier, ROLE
from ..extract.summarize import CriticalTraceSummarizer
from ..schema import Trace, trace_tokens
from ..world import World

_CSS = """
:root{--bg:#fff;--fg:#1d2330;--mut:#667085;--line:#e4e7ec;--card:#f8f9fb;--hot:#fff1e0;--hotb:#e07b00;--ok:#e6f4ea;--acc:#2457c5}
@media(prefers-color-scheme:dark){:root{--bg:#12151c;--fg:#e8eaf0;--mut:#9aa3b2;--line:#2a2f3b;--card:#1a1e28;--hot:#3a2a10;--hotb:#ffb04d;--ok:#16301f;--acc:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:16px 20px;border-bottom:1px solid var(--line)}h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mut)}
nav{display:flex;flex-wrap:wrap;gap:6px;padding:12px 20px;border-bottom:1px solid var(--line)}
nav button{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:5px 10px;border-radius:6px;cursor:pointer}
nav button.on{border-color:var(--acc);color:var(--acc);font-weight:600}
main{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px 20px}@media(max-width:900px){main{grid-template-columns:1fr}}
section{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
h2{font-size:14px;margin:0 0 8px}.row{display:grid;grid-template-columns:34px 1fr;gap:8px;padding:5px 6px;border-radius:5px;border-left:3px solid transparent}
.row .id{color:var(--mut);font-variant-numeric:tabular-nums}.row code{white-space:pre-wrap;word-break:break-word;font-size:12px}
.row .k{color:var(--mut);font-size:11px}.hot{background:var(--hot);border-left-color:var(--hotb)}.tag{display:inline-block;background:var(--acc);color:#fff;border-radius:4px;padding:0 6px;font-size:11px;margin-right:6px}
.truth{background:var(--ok);padding:6px 8px;border-radius:6px;margin-bottom:8px}.small{font-size:12px;color:var(--mut)}
"""

_JS = """
const D=JSON.parse(document.getElementById('data').textContent);
function show(i){document.querySelectorAll('nav button').forEach((b,j)=>b.classList.toggle('on',i===j));
const d=D[i];document.getElementById('meta').innerHTML=d.meta;
document.getElementById('raw').innerHTML=d.raw;document.getElementById('sum').innerHTML=d.sum;}
const nav=document.querySelector('nav');D.forEach((d,i)=>{const b=document.createElement('button');b.textContent=d.name;b.onclick=()=>show(i);nav.appendChild(b)});show(0);
"""


def _row(e, hot: bool, tag: str = "", quote: str = "") -> str:
    body = html.escape(quote) if quote else html.escape(json.dumps({"content": e.content, **({"inputs": e.inputs} if e.inputs else {})}, ensure_ascii=False))
    if len(body) > 700:
        body = body[:700] + "…"
    cls = "row hot" if hot else "row"
    t = f'<span class="tag">{html.escape(tag)}</span>' if tag else ""
    return f'<div class="{cls}"><div class="id">{e.id}</div><div><div class="k">{t}{html.escape(e.kind)} · {html.escape(e.actor)}</div><code>{body}</code></div></div>'


def render_viewer(traces: List[Trace], world: World, path: str, budget: int = 6) -> str:
    V = EvidenceVerifier(world)
    S = CriticalTraceSummarizer(V)
    data: List[Dict[str, str]] = []
    for t in traces:
        v = t.view()
        summ = S.summarize(v, budget)
        att = V.attribute(v)
        crit = t.labels.get("fault", {}) or {}
        ce = crit.get("critical_event")
        flagged = {i.event_id: i.role for i in summ.items if i.role in ROLE.values()}
        raw = "".join(_row(e, e.id in flagged) for e in v.events)
        sm = "".join(_row(v.events[i.event_id], i.role in ROLE.values(), i.role, f"{i.quote}" + (f"  — {i.note}" if i.note else "")) for i in summ.items)
        sm += f'<p class="small">{summ.tokens()} tokens vs {trace_tokens(v)} raw ({100*summ.tokens()/max(1,trace_tokens(v)):.0f}%). Each line quotes a real event.</p>'
        truth = (f'<div class="truth"><b>Ground truth</b> (hidden from extractors): fault <b>{html.escape(str(crit.get("kind")))}</b> at event <b>{ce}</b>. '
                 f'Verifier top-1: <b>{att.top}</b> {"✓" if att.top == ce else "✗"}</div>') if crit else '<div class="truth"><b>Clean run</b> (no injected fault)</div>'
        meta = (f'{truth}<b>Task:</b> {html.escape(t.question)}<br><b>Answer:</b> {html.escape(str(t.outcome.get("answer")))} '
                f'— {"correct" if t.outcome.get("success") else "wrong"} (gold: {html.escape(str(t.labels.get("gold")))})')
        data.append({"name": (crit.get("kind") or "clean"), "meta": meta, "raw": raw, "sum": sm})
    payload = json.dumps(data).replace("</", "<\\/")
    page = (f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>TraceLens viewer</title><style>{_CSS}</style><header><h1>TraceLens — raw log vs critical-trace summary</h1>'
            f'<div class="sub">Generated offline. Highlighted rows are events flagged by the evidence verifier.</div></header><nav></nav>'
            f'<div style="padding:12px 20px" id="meta"></div><main><section><h2>Raw log</h2><div id="raw"></div></section>'
            f'<section><h2>Critical-trace summary</h2><div id="sum"></div></section></main>'
            f'<script id="data" type="application/json">{payload}</script><script>{_JS}</script></html>')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return path
