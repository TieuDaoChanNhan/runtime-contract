import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from codeact_runtime.codeact.events import SystemPromptEvent
from codeact_runtime.llm import TokenUsage

# uses your Event union types: StartEvent | StepEvent | FinishEvent | ErrorEvent


class HtmlTraceListener:
    """
    Listener that writes a nice, human-readable HTML trace.

    Works with your current architecture:
        async with AsyncQueuedEventBus([HtmlTraceListener("trace.html"), ...]) as bus:
            ...
            await bus.emit(StartEvent(...))
            await bus.emit(StepEvent(...))
            ...
    """

    _CODE_FENCE_RE = re.compile(
        r"```(?P<lang>[a-zA-Z0-9_+\-]*)\n(?P<code>.*?)```", re.DOTALL
    )

    def __init__(
        self,
        out_path: str | Path,
        *,
        title: str = "CodeAct-Runtime Trace",
        flush_every: int = 1,
    ):
        self.out_path = Path(out_path)
        self.title = title
        self.flush_every = flush_every

        self._fp = None
        self._started_at: Optional[datetime] = None
        self._task: str
        self._num_steps = 0
        self._finish_reason: Optional[str] = None
        self._errors: list[str] = []
        self._token_usage: Optional[TokenUsage] = None

    async def __aenter__(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.out_path.open("w", encoding="utf-8")
        self._started_at = datetime.now()
        # write a skeleton; fill task later when StartEvent arrives
        self._fp.write(self._html_preamble(task="(pending StartEvent)"))
        self._fp.flush()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # If we never saw FinishEvent, still close cleanly
        if self._fp:
            self._fp.write(self._html_footer())
            self._fp.flush()
            self._fp.close()
            self._fp = None

    async def handle(self, event) -> None:
        if not self._fp:
            return

        # Start
        if event.__class__.__name__ == "StartEvent":
            self._task = getattr(event, "task", "")
            # patch header by writing a new header section (simplest)
            self._fp.write(self._render_banner(kind="start", text="Episode started"))
            self._fp.write(self._render_task_block(self._task))
            self._maybe_flush()
            return

        # System prompt
        if isinstance(event, SystemPromptEvent):
            self._fp.write(
                self._render_banner(kind="start", text="System prompt captured")
            )
            self._fp.write(self._render_system_prompts(event.prompts))
            self._maybe_flush()
            return

        # Step
        if event.__class__.__name__ == "StepEvent":
            self._num_steps += 1
            self._fp.write(self._render_step(event))
            self._maybe_flush()
            return

        # Finish
        if event.__class__.__name__ == "FinishEvent":
            self._finish_reason = getattr(event, "reason", "unknown")
            finish_usage = getattr(event, "token_usage", None)
            if finish_usage:
                self._token_usage = finish_usage
            self._fp.write(
                self._render_banner(
                    kind="finish", text=f"Finished: {self._finish_reason}"
                )
            )
            self._maybe_flush(force=True)
            return

        # Error
        if event.__class__.__name__ == "ErrorEvent":
            err = getattr(event, "error", "")
            self._errors.append(err)
            self._fp.write(self._render_banner(kind="error", text=err))
            self._maybe_flush(force=True)
            return

        # Unknown event type: still log
        self._fp.write(
            self._render_banner(kind="warn", text=f"Unknown event: {event!r}")
        )
        self._maybe_flush()

    # ---------------------------
    # Rendering
    # ---------------------------

    def _html_preamble(self, task: str) -> str:
        started = (
            html.escape(self._started_at.isoformat(sep=" ", timespec="seconds"))
            if self._started_at
            else ""
        )
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{html.escape(self.title)}</title>
<style>
  :root {{
    --bg: #0b1020;
    --panel: rgba(17,26,51,0.55);
    --panel2: rgba(15,23,48,0.70);
    --text: #e8eefc;
    --muted: #a9b6de;
    --border: rgba(255,255,255,0.08);
    --codebg: #0b0f1c;
    --accent: #7aa2ff;
    --good: #4ade80;
    --bad: #fb7185;
    --warn: #fbbf24;
  }}
  body {{
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
    background: radial-gradient(1200px 800px at 20% 0%, #172554 0%, var(--bg) 45%);
    color: var(--text);
  }}
  header {{
    position: sticky; top: 0; z-index: 10;
    backdrop-filter: blur(8px);
    background: rgba(11,16,32,0.88);
    border-bottom: 1px solid var(--border);
    padding: 14px 18px;
  }}
  .title {{
    display:flex; justify-content:space-between; align-items:baseline; gap:10px; flex-wrap:wrap;
  }}
  .title h1 {{ font-size: 16px; margin: 0; letter-spacing: 0.2px; }}
  .meta {{ color: var(--muted); font-size: 12px; }}
  main {{ max-width: 1100px; margin: 18px auto 80px auto; padding: 0 16px; }}
  .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; margin: 12px 0 18px; }}
  button {{
    background: rgba(122,162,255,0.14);
    color: var(--text);
    border: 1px solid rgba(122,162,255,0.25);
    padding: 8px 10px;
    border-radius: 10px;
    cursor: pointer;
    font-size: 12px;
  }}
  button:hover {{ background: rgba(122,162,255,0.22); }}
  .pill {{
    font-size: 11px; padding: 2px 8px; border-radius: 999px;
    border: 1px solid var(--border); color: var(--muted);
  }}
  .banner {{
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 10px 12px;
    margin: 10px 0 12px;
    background: var(--panel2);
  }}
  .banner.good {{ border-color: rgba(74,222,128,0.35); }}
  .banner.bad {{ border-color: rgba(251,113,133,0.45); }}
  .banner.warn {{ border-color: rgba(251,191,36,0.45); }}
  .task {{
    margin-top: 10px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    background: rgba(17,26,51,0.75);
    border-radius: 12px;
    white-space: pre-wrap;
  }}
  .turn {{
    border: 1px solid var(--border);
    background: var(--panel);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 12px;
  }}
  .turn-header {{
    display:flex; justify-content:space-between; gap:10px;
    padding: 10px 12px;
    background: var(--panel2);
    border-bottom: 1px solid var(--border);
  }}
  .turn-header a {{ color: var(--accent); text-decoration: none; }}
  .turn-body {{ padding: 12px; display:grid; grid-template-columns: 1fr; gap: 10px; }}
  .block {{
    border: 1px solid var(--border);
    border-radius: 12px;
    background: rgba(15,23,48,0.40);
    overflow: hidden;
  }}
  .block-title {{
    display:flex; justify-content:space-between; align-items:center; gap:8px;
    padding: 8px 10px;
    font-size: 12px;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    background: rgba(15,23,48,0.55);
  }}
  .block-content {{ padding: 10px; }}
  pre {{
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New";
    font-size: 12.5px;
    line-height: 1.45;
  }}
  pre.code {{
    white-space: pre;
    overflow-x: auto;
    background: var(--codebg);
    padding: 10px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
  }}
  details > summary {{
    cursor: pointer;
    list-style: none;
    user-select: none;
  }}
  details > summary::-webkit-details-marker {{ display:none; }}
  .kv {{
    display:grid; grid-template-columns: 180px 1fr;
    gap: 6px 10px;
    font-size: 12px;
    color: var(--muted);
  }}
  .kv div:nth-child(odd) {{ color: var(--muted); }}
  .footer {{ color: var(--muted); font-size: 12px; padding: 18px 4px; }}
</style>
<script>
  function toggleAll(open) {{
    document.querySelectorAll('details').forEach(d => d.open = open);
  }}
  async function copyToClipboard(id) {{
    const el = document.getElementById(id);
    if (!el) return;
    const text = el.innerText;
    try {{
      await navigator.clipboard.writeText(text);
      const btn = document.querySelector('[data-copy-btn="'+id+'"]');
      if (btn) {{
        const old = btn.innerText;
        btn.innerText = "Copied";
        setTimeout(() => btn.innerText = old, 900);
      }}
    }} catch (e) {{
      alert("Copy failed: " + e);
    }}
  }}
</script>
</head>
<body>
<header>
  <div class="title">
    <h1>{html.escape(self.title)}</h1>
    <div class="meta">Started: {started} · File: {html.escape(str(self.out_path))}</div>
  </div>
  <div class="kv" style="margin-top:10px;">
    <div>Task</div><div id="task-text">{html.escape(task)}</div>
    <div>Steps</div><div id="step-count">0</div>
    <div>Status</div><div id="status-text">running</div>
    <div>Total tokens</div><div id="token-total">0</div>
  </div>
  <div class="toolbar" style="margin-top:10px;">
    <button onclick="toggleAll(true)">Expand all</button>
    <button onclick="toggleAll(false)">Collapse all</button>
  </div>
</header>
<main>
"""

    def _html_footer(self) -> str:
        finished_at = datetime.now()
        dur = ""
        if self._started_at:
            dur = f"{(finished_at - self._started_at).total_seconds():.2f}s"
        status = html.escape(
            self._finish_reason or ("error" if self._errors else "running")
        )
        total_tokens = (
            str(self._token_usage.total_tokens) if self._token_usage else "unknown"
        )
        return f"""
  <div class="footer">
    <div><span class="pill">Steps: {self._num_steps}</span> <span class="pill">Duration: {html.escape(dur)}</span> <span class="pill">Status: {status}</span> <span class="pill">Tokens: {total_tokens}</span></div>
  </div>
<script>
  // Patch header summary at end (best-effort)
  document.getElementById("step-count").innerText = "{self._num_steps}";
  document.getElementById("status-text").innerText = "{status}";
  document.getElementById("token-total").innerText = "{total_tokens}";
</script>
</main>
</body>
</html>
"""

    def _render_banner(self, *, kind: str, text: str) -> str:
        klass = "warn"
        if kind == "start":
            klass = "good"
        elif kind == "finish":
            klass = "good"
        elif kind == "error":
            klass = "bad"
        elif kind == "warn":
            klass = "warn"
        return f"""
<div class="banner {klass}">
  <pre>{html.escape(text)}</pre>
</div>
"""

    def _render_task_block(self, task: str) -> str:
        return f"""
<div class="block">
  <div class="block-title"><div>Task</div><div class="pill">prompt</div></div>
  <div class="block-content"><pre>{html.escape(task)}</pre></div>
</div>
"""

    def _render_system_prompts(self, prompts: list[str]) -> str:
        if not prompts:
            return """
<div class="block">
  <div class="block-title"><div>System prompt</div><div class="pill">prompt</div></div>
  <div class="block-content"><pre>(none)</pre></div>
</div>
"""

        rendered = []
        for idx, prompt in enumerate(prompts, start=1):
            rendered.append(
                f"""
<details>
  <summary class="pill">system message {idx}</summary>
  <pre class="code">{html.escape(prompt)}</pre>
</details>
"""
            )

        return f"""
<div class="block">
  <div class="block-title"><div>System prompt</div><div class="pill">prompt</div></div>
  <div class="block-content">
    {"".join(rendered)}
  </div>
</div>
"""

    def _render_step(self, ev) -> str:
        turn = getattr(ev, "turn", -1)
        assistant_text = getattr(ev, "assistant_text", "")
        code = getattr(ev, "code", None)
        interp = getattr(ev, "interpreter_result", None)
        token_usage = getattr(ev, "token_usage", None)

        if token_usage:
            self._token_usage = (
                token_usage
                if self._token_usage is None
                else self._token_usage + token_usage
            )

        turn_id = f"turn-{turn}"
        assistant_html = self._render_text_with_code_fences(
            assistant_text, base_id=turn_id
        )

        # Interpreter result: pretty-print JSON if provided
        interp_html = self._render_interpreter_result(interp)

        # If `code` is separate, show it too (often redundant with fences, but useful for sanity)
        code_html = ""
        if False:  # code is not None:
            code_id = f"{turn_id}-extracted-code"
            code_html = f"""
<div class="block">
  <div class="block-title">
    <div>Extracted code</div>
    <div style="display:flex; gap:8px; align-items:center;">
      <span class="pill">python</span>
      <button data-copy-btn="{code_id}" onclick="copyToClipboard('{code_id}')">Copy</button>
    </div>
  </div>
  <div class="block-content">
    <pre class="code" id="{code_id}"><code>{html.escape(code)}</code></pre>
  </div>
</div>
"""

        return f"""
<section class="turn" id="{turn_id}">
  <div class="turn-header">
    <div><a href="#{turn_id}">Turn {turn}</a></div>
    <div style="display:flex; gap:8px; align-items:center;">
      <div class="pill">#{turn}</div>
      {self._render_token_pill(token_usage)}
    </div>
  </div>
  <div class="turn-body">
    <div class="block">
      <div class="block-title"><div>Assistant</div><div class="pill">model output</div></div>
      <div class="block-content">{assistant_html}</div>
    </div>
    {code_html}
    <div class="block">
      <div class="block-title"><div>Interpreter result</div><div class="pill">observation</div></div>
      <div class="block-content">{interp_html}</div>
    </div>
  </div>
</section>
"""

    def _render_token_pill(self, token_usage: Optional[TokenUsage]) -> str:
        if not token_usage:
            return ""
        return (
            f'<div class="pill">tokens: '
            f"p:{token_usage.prompt_tokens}/c:{token_usage.completion_tokens}/"
            f"t:{token_usage.total_tokens}</div>"
        )

    def _render_interpreter_result(self, interp: Optional[dict[str, Any]]) -> str:
        if interp is None:
            return "<pre>(none)</pre>"

        # If it already contains 'output'/'stdout', show prominently
        out = interp.get("output") if isinstance(interp, dict) else None
        err = interp.get("error") if isinstance(interp, dict) else None

        chunks = []

        if out:
            chunks.append(f"""
<details open>
  <summary class="pill">stdout</summary>
  <pre class="code">{html.escape(str(out))}</pre>
</details>
""")

        if err:
            chunks.append(f"""
<details open>
  <summary class="pill" style="border-color: rgba(251,113,133,0.45);">error</summary>
  <pre class="code">{html.escape(str(err))}</pre>
</details>
""")

        # Always include full JSON (collapsible)
        pretty = self._pretty_json(interp)
        chunks.append(f"""
<details>
  <summary class="pill">full result (json)</summary>
  <pre class="code">{html.escape(pretty)}</pre>
</details>
""")

        return "\n".join(chunks)

    def _pretty_json(self, obj: Any) -> str:
        try:
            return json.dumps(
                obj, indent=2, ensure_ascii=False, sort_keys=True, default=str
            )
        except Exception:
            return str(obj)

    def _render_text_with_code_fences(self, text: str, *, base_id: str) -> str:
        parts = []
        last = 0
        block_idx = 0

        for m in self._CODE_FENCE_RE.finditer(text):
            if m.start() > last:
                chunk = text[last : m.start()]
                if chunk.strip():
                    parts.append(f"<pre>{html.escape(chunk)}</pre>")

            lang = (m.group("lang") or "python").strip()
            code = (m.group("code") or "").strip()
            code_id = f"{base_id}-fence-{block_idx}"
            block_idx += 1

            parts.append(
                f"""
<details open>
  <summary class="pill">code fence ({html.escape(lang)})</summary>
  <div style="display:flex; justify-content:flex-end; margin:6px 0;">
    <button data-copy-btn="{code_id}" onclick="copyToClipboard('{code_id}')">Copy</button>
  </div>
  <pre class="code" id="{code_id}"><code>{html.escape(code)}</code></pre>
</details>
"""
            )
            last = m.end()

        if last < len(text):
            chunk = text[last:]
            if chunk.strip():
                parts.append(f"<pre>{html.escape(chunk)}</pre>")

        return "\n".join(parts) if parts else "<pre>(empty)</pre>"

    def _maybe_flush(self, *, force: bool = False):
        if not self._fp:
            return
        if force or (
            self.flush_every > 0 and (self._num_steps % self.flush_every == 0)
        ):
            self._fp.flush()
