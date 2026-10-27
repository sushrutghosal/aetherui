"""
AetherUI (vision-to-mcp) — dynamic JIT protocol compiler + self-healing web runtime.

Paste any URL:
  Phase 1  COMPILE   Playwright loads the page, Set-of-Marks badges every interactive element, Nemotron (VL)
                     looks at the screenshot + DOM index and proposes MCP tools. Every proposed step is then
                     VALIDATED against the live mark index (bad marks dropped, selectors pinned, enums copied
                     from real <select> options). If the model fails, a DOM heuristic compiler builds tools instead.
  Phase 2  RUN       An agent (or the chat box) calls a tool. The runtime replays the steps, re-locates any
                     element that moved (text/label/placeholder, then Nemotron visual re-grounding), detects
                     blocking modals and heals them visually, then extracts a structured answer.
  PROOF              Every run returns a screenshot timeline (loaded -> each step -> blocker -> healed -> final)
                     plus the text that changed on the page and any backend JSON the page fetched.

Every LLM call has a deterministic fallback, so a model hiccup degrades the demo instead of crashing it.
"""
import asyncio
import base64
import inspect
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

from fastmcp import FastMCP
import builtins


def print(*args, **kwargs):  # logs go to stderr so the stdio MCP channel (stdout) stays clean
    kwargs.setdefault("file", sys.stderr)
    builtins.print(*args, **kwargs)

from openai import AsyncOpenAI
from playwright.async_api import Locator, Page, Response, async_playwright
from pydantic import BaseModel, Field

# =====================================================================
# 1. CONFIG  (put settings in a .env file next to this script; real environment variables still win)
# =====================================================================
def _load_dotenv(path: str):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.replace("export ", "").strip(), v.strip().strip("'\"")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/")
# Vision model id — confirm the exact id on your account with check_nebius.py and export NEBIUS_MODEL.
NEBIUS_MODEL = os.getenv("NEBIUS_MODEL", "nvidia/Nemotron-3.5-Super-VL")
# Routing model; defaults to the vision model so only one id has to be right.
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL") or NEBIUS_MODEL
LLM_TIMEOUT_S = float(os.getenv("AETHER_LLM_TIMEOUT", "60"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.getenv("AETHER_RUNS_DIR", os.path.join(BASE_DIR, "runs"))
MANIFEST_DIR = os.getenv("AETHER_MANIFEST_DIR", os.path.join(BASE_DIR, "manifests"))
CURRENT_DATE_CONTEXT = datetime.now().strftime("%A, %B %d, %Y")
VIEWPORT = {"width": 1200, "height": 780}

nebius_client = AsyncOpenAI(
    base_url=NEBIUS_BASE_URL,
    api_key=os.environ.get("NEBIUS_API_KEY", "missing-key"),
    timeout=LLM_TIMEOUT_S,
    max_retries=1,
)

# =====================================================================
# 2. SCHEMAS
# =====================================================================
ACTIONS = ("click", "fill", "select", "press_enter", "scroll_down", "wait")


class ExecutionStep(BaseModel):
    action: Literal["click", "fill", "select", "press_enter", "scroll_down", "wait"]
    mark_id: int = 0
    css_fallback: str = ""
    target_text: str = ""
    value_template: Optional[str] = None


class CompiledMCPTool(BaseModel):
    name: str
    method: Literal["GET", "POST"] = "GET"
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})
    execution_steps: List[ExecutionStep] = Field(default_factory=list)
    visual_output_keys: List[str] = Field(default_factory=list)


class DiscoveredLink(BaseModel):
    text: str
    href: str


class MCPManifest(BaseModel):
    target_url: str
    app_title: str
    compiler: str = ""
    compiler_notes: str = ""
    discovered_links: List[DiscoveredLink] = Field(default_factory=list)
    below_fold_summary: str = ""
    page_context_snapshot: str = ""
    tools: List[CompiledMCPTool]


# =====================================================================
# 3. LLM HELPERS
# =====================================================================
def parse_json_obj(raw_text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", str(raw_text), flags=re.DOTALL).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


TEXT_ONLY_MODELS: set = set()  # models that rejected image input; we send them the DOM/page text only


def _strip_images(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    parts = [p for p in content if not (isinstance(p, dict) and p.get("type") == "image_url")]
    parts.append({"type": "text", "text": "(No screenshot available: rely on the mark index and page text above.)"})
    return "\n".join(p["text"] for p in parts if isinstance(p, dict) and p.get("type") == "text")


async def llm_json(model: str, system: str, user_content: Any, max_tokens: int = 2048) -> Tuple[Optional[Dict[str, Any]], str]:
    """Returns (parsed_json_or_None, error_message). Never raises.
    - text-only model: retried without the screenshot (DOM + page text still describe the page)
    - unparseable / empty output (e.g. reasoning used up the budget): retried once with more room and a JSON-only reminder"""
    has_image = isinstance(user_content, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in user_content)
    if has_image and model in TEXT_ONLY_MODELS:
        user_content = _strip_images(user_content)
    sys_prompt, budget, parse_retried, last_err = system, max_tokens, False, ""
    for _ in range(3):
        try:
            resp = await nebius_client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=budget,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}],
            )
        except Exception as e:
            msg = str(e)
            if has_image and model not in TEXT_ONLY_MODELS and re.search(r"image|vision|multimodal|not support", msg, re.I):
                print(f"[!] {model} rejected the screenshot -> switching to text-only mode (DOM + page text)")
                TEXT_ONLY_MODELS.add(model)
                user_content = _strip_images(user_content)
                continue
            print(f"[!] LLM call failed ({model}): {e}")
            return None, f"{type(e).__name__}: {msg[:200]}"
        parsed = parse_json_obj(resp.choices[0].message.content)
        if parsed is not None:
            return parsed, ""
        last_err = "model returned no parseable JSON"
        if parse_retried:
            break
        parse_retried = True
        budget = min(budget * 2, 12000)
        sys_prompt = system + "\n\nIMPORTANT: Output ONLY the JSON object. No reasoning, no prose, no markdown."
        print(f"[!] {model} returned no JSON -> retrying with a JSON-only reminder")
    return None, last_err


async def check_nebius() -> str:
    """One quick call at startup so a bad key / URL / model id is obvious immediately."""
    try:
        ids = [m.id for m in (await nebius_client.models.list()).data]
    except Exception as e:
        return f"NOT REACHABLE at {NEBIUS_BASE_URL}: {type(e).__name__}: {str(e)[:150]}"
    missing = [m for m in {NEBIUS_MODEL, ORCHESTRATOR_MODEL} if m not in ids]
    if missing:
        close = [i for i in ids if any(w in i.lower() for w in ("nemotron", "vl"))][:8]
        return f"connected, but model id not found: {missing}. Available NVIDIA/VL ids: {close}"
    return f"OK ({NEBIUS_BASE_URL}, model {NEBIUS_MODEL})"


def model_mode(model: str) -> str:
    return f"{model} (text-only: DOM + page text)" if model in TEXT_ONLY_MODELS else f"{model} (vision)"


def img_part(b64: str, mime: str = "image/png") -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def snake(s: str, fallback: str = "value") -> str:
    out = re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:40]
    return out or fallback


# =====================================================================
# 4. PAGE INSTRUMENTATION (Set-of-Marks, blockers, tables)
# =====================================================================
SOM_INJECTION_JS = """
() => {
    document.querySelectorAll('.v2mcp-badge').forEach(el => el.remove());
    document.querySelectorAll('[data-v2mcp-id]').forEach(el => el.removeAttribute('data-v2mcp-id'));

    const cleanText = (document.body.innerText || '')
        .split('\\n').map(l => l.trim()).filter(l => l.length > 2).slice(0, 160).join('\\n');

    const selectors = 'button:not(.v2mcp-ignore), a:not(.v2mcp-ignore), input:not(.v2mcp-ignore), select:not(.v2mcp-ignore), textarea:not(.v2mcp-ignore), [role="button"], [role="combobox"], [role="searchbox"]';
    const elements = Array.from(document.querySelectorAll(selectors)).filter(el => {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        const type = (el.getAttribute('type') || '').toLowerCase();
        return rect.width > 8 && rect.height > 8 && style.visibility !== 'hidden' && style.display !== 'none' &&
               style.opacity !== '0' && type !== 'hidden';
    });

    function stableSelector(el, markId) {
        const tag = el.tagName.toLowerCase();
        if (el.id && document.querySelectorAll('#' + CSS.escape(el.id)).length === 1) return `#${CSS.escape(el.id)}`;
        const name = el.getAttribute('name');
        if (name && document.querySelectorAll(`${tag}[name="${CSS.escape(name)}"]`).length === 1) return `${tag}[name="${CSS.escape(name)}"]`;
        const aria = el.getAttribute('aria-label');
        if (aria && aria.length < 45 && document.querySelectorAll(`${tag}[aria-label="${CSS.escape(aria)}"]`).length === 1) return `${tag}[aria-label="${CSS.escape(aria)}"]`;
        const ph = el.getAttribute('placeholder');
        if (ph && ph.length < 45 && document.querySelectorAll(`${tag}[placeholder="${CSS.escape(ph)}"]`).length === 1) return `${tag}[placeholder="${CSS.escape(ph)}"]`;
        return '';  // no stable selector; runtime re-locates by text/label or vision
    }

    function labelFor(el) {
        if (el.labels && el.labels.length) return el.labels[0].innerText.trim();
        const prev = el.previousElementSibling;
        if (prev && prev.tagName === 'LABEL') return prev.innerText.trim();
        const row = el.closest('.row, tr, li, div');
        const lab = row ? row.querySelector('label') : null;
        return lab ? lab.innerText.trim() : '';
    }

    const groupRoots = [];
    function groupOf(el) {
        const root = el.closest('form, fieldset, .panel, section, [role="search"], [role="form"]');
        if (!root) return 0;
        let i = groupRoots.indexOf(root);
        if (i < 0) { groupRoots.push(root); i = groupRoots.length - 1; }
        return i + 1;
    }

    const score = el => {
        if (el.closest('#chaos-modal, #aether-chaos-backdrop, [aria-modal="true"], [role="alertdialog"]')) return -1;
        const t = el.tagName.toLowerCase();
        if (t === 'input' || t === 'select' || t === 'textarea' || t === 'button') return 0;
        return 1;
    };
    const prioritized = elements.sort((a, b) => score(a) - score(b)).slice(0, 40);

    const markMap = {};
    const links = [];
    prioritized.forEach((el, idx) => {
        const markId = idx + 1;
        el.setAttribute('data-v2mcp-id', String(markId));
        const rect = el.getBoundingClientRect();
        if (rect.top >= 0 && rect.top <= window.innerHeight && rect.left >= 0) {
            const badge = document.createElement('div');
            badge.className = 'v2mcp-badge';
            badge.textContent = `[${markId}]`;
            Object.assign(badge.style, {
                position: 'fixed', left: `${Math.max(0, rect.left - 3)}px`, top: `${Math.max(0, rect.top - 11)}px`,
                backgroundColor: '#ff0033', color: '#ffffff', fontSize: '10px', fontWeight: 'bold',
                fontFamily: 'monospace', padding: '0px 3px', borderRadius: '3px', zIndex: '2147483647',
                pointerEvents: 'none', boxShadow: '0 1px 2px rgba(0,0,0,0.6)'
            });
            document.body.appendChild(badge);
        }
        const tag = el.tagName.toLowerCase();
        const options = tag === 'select' ? Array.from(el.options).map(o => o.value) : [];
        const text = (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') ||
            el.getAttribute('title') || el.getAttribute('name') || '').replace(/\\s+/g, ' ').slice(0, 75).trim();
        if (tag === 'a' && el.href && !el.href.startsWith('javascript:')) links.push({ text: text || el.href, href: el.href });
        markMap[markId] = {
            mark_id: markId, tag, selector: stableSelector(el, markId), type: (el.getAttribute('type') || '').toLowerCase(),
            name: el.getAttribute('name') || '', placeholder: el.getAttribute('placeholder') || '',
            aria: el.getAttribute('aria-label') || '', label: labelFor(el).slice(0, 60),
            options, text, group: groupOf(el), top: Math.round(rect.top), in_viewport: rect.top >= 0 && rect.top <= window.innerHeight
        };
    });
    return { markMap, links: links.slice(0, 15), cleanText: cleanText.slice(0, 5000) };
}
"""

REMOVE_BADGES_JS = "() => document.querySelectorAll('.v2mcp-badge').forEach(el => el.remove())"

TEXT_LINES_JS = """
() => (document.body.innerText || '').split('\\n').map(l => l.trim()).filter(l => l.length > 2).slice(0, 400)
"""

TABLES_JS = """
() => {
    const out = [];
    for (const t of document.querySelectorAll('table')) {
        const r = t.getBoundingClientRect();
        if (r.width < 50 || r.height < 20) continue;
        const headers = Array.from(t.querySelectorAll('thead th, tr:first-child th')).map(th => th.innerText.trim());
        if (headers.length < 2) continue;
        const rows = [];
        for (const tr of t.querySelectorAll('tbody tr, tr')) {
            const cells = Array.from(tr.querySelectorAll('td')).map(td => td.innerText.trim());
            if (cells.length !== headers.length) continue;
            const row = {};
            headers.forEach((h, i) => row[h || `col_${i}`] = cells[i]);
            rows.push(row);
            if (rows.length >= 50) break;
        }
        if (rows.length) out.push({ headers, rows });
        if (out.length >= 3) break;
    }
    return out;
}
"""

INJECT_CHAOS_JS = """
() => {
    if (document.getElementById('aether-chaos-backdrop')) return;
    const ov = document.createElement('div');
    ov.id = 'aether-chaos-backdrop';
    Object.assign(ov.style, { position: 'fixed', inset: '0', background: 'rgba(0,0,0,0.45)', zIndex: '2147483000',
        display: 'flex', alignItems: 'center', justifyContent: 'center' });
    ov.innerHTML = '<div id="aether-chaos-modal" style="background:#ffffcc;border:3px solid red;padding:22px;font-family:Tahoma,sans-serif;color:#111;box-shadow:5px 5px 0 #000;max-width:420px">' +
        '<p style="margin:0 0 12px 0"><b>WARNING [ERR-902]:</b> Session compliance re-acknowledgement required before content can be accessed.</p>' +
        '<button id="aether-chaos-ack" style="border:2px outset #fff;background:#d4d0c8;padding:6px 12px;font-weight:bold;cursor:pointer">Acknowledge &amp; Continue</button></div>';
    document.body.appendChild(ov);
    document.getElementById('aether-chaos-ack').addEventListener('click', () => ov.remove());
}
"""

# Known blocker selectors first, then a generic "fixed/high-z element with a button covering the viewport centre".
DETECT_BLOCKER_JS = """
() => {
    const visible = el => {
        const st = getComputedStyle(el); const r = el.getBoundingClientRect();
        return st.display !== 'none' && st.visibility !== 'hidden' && r.width > 50 && r.height > 30;
    };
    const known = ['#chaos-modal', '#aether-chaos-backdrop', '[aria-modal="true"]', '[role="alertdialog"]'];
    for (const s of known) {
        for (const el of document.querySelectorAll(s)) {
            if (visible(el) && el.querySelector('button, [role="button"]')) {
                if (!el.id) el.id = 'aether-blocker-' + Math.floor(Math.random() * 1e6);
                return { selector: '#' + CSS.escape(el.id), text: (el.innerText || '').slice(0, 200) };
            }
        }
    }
    const cx = innerWidth / 2, cy = innerHeight / 2;
    for (const el of document.elementsFromPoint(cx, cy)) {
        let node = el;
        while (node && node !== document.body) {
            const st = getComputedStyle(node);
            const r = node.getBoundingClientRect();
            if ((st.position === 'fixed' || st.position === 'sticky') && parseInt(st.zIndex || '0') >= 100 &&
                r.top > 40 && r.width > 200 && node.querySelector('button, [role="button"]') && visible(node)) {
                if (!node.id) node.id = 'aether-blocker-' + Math.floor(Math.random() * 1e6);
                return { selector: '#' + CSS.escape(node.id), text: (node.innerText || '').slice(0, 200) };
            }
            node = node.parentElement;
        }
    }
    return null;
}
"""

MARK_INSIDE_JS = """
([sel, id]) => {
    const b = document.querySelector(sel);
    const e = document.querySelector(`[data-v2mcp-id="${id}"]`);
    return !!(b && e && b.contains(e));
}
"""

SELECT_MATCH_JS = """
(el, target) => {
    const norm = s => String(s).toUpperCase().replace(/[\\s_\\-]+/g, '');
    const t = norm(target);
    for (const o of el.options) if (norm(o.value) === t || norm(o.text) === t) return o.value;
    for (const o of el.options) if (norm(o.value).includes(t) || norm(o.text).includes(t)) return o.value;
    return null;
}
"""


async def annotate_and_capture(page: Page, save_path: Optional[str] = None) -> Tuple[str, Dict[int, Dict[str, Any]], List[Dict[str, str]], str]:
    res = await page.evaluate(SOM_INJECTION_JS)
    shot = await page.screenshot(type="png", full_page=False)
    if save_path:
        with open(save_path, "wb") as f:
            f.write(shot)
    await page.evaluate(REMOVE_BADGES_JS)  # data-v2mcp-id attrs stay so marks remain clickable
    mark_map = {int(k): v for k, v in res["markMap"].items()}
    return base64.b64encode(shot).decode("utf-8"), mark_map, res["links"], res.get("cleanText", "")


async def settle_page(page: Page, wait_ms: int = 700):
    for state, timeout in (("domcontentloaded", 4000), ("networkidle", 2500)):
        try:
            await page.wait_for_load_state(state, timeout=timeout)
        except Exception:
            pass
    await page.wait_for_timeout(wait_ms)


async def open_browser(p):
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        viewport=VIEWPORT,
        locale="en-US",
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
    )
    return browser, await context.new_page()


def mark_brief(mark_map: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = ("mark_id", "tag", "type", "text", "label", "placeholder", "aria", "name", "options", "group", "in_viewport")
    return [{k: v[k] for k in keep if v.get(k) not in ("", [], None)} for v in mark_map.values()]


# =====================================================================
# 5. PHASE 1: COMPILE
# =====================================================================
COMPILER_SYSTEM_PROMPT = """You are the AetherUI Protocol Compiler. You turn a human web page into MCP tools an AI agent can call.
You receive a screenshot with red Set-of-Marks badges ([1], [2], ...), the mark index (one entry per interactive
element, with tag/label/placeholder/options/group), and the page text.

Produce 1-5 tools covering what a user would actually DO on this page:
- search boxes -> a GET tool: fill the input with {{query}} then press_enter on it (or click its search button)
- filter/query panels -> GET tool: select/fill the filter, then click the query button
- forms that change data -> POST tool: fill/select every field, then click the submit button
- top navigation (sections/tabs) -> GET tool "open_section" with ONE click step whose value_template is "{{section}}"
  and an enum of the exact link texts
Rules:
- Only use mark_id values that exist in the mark index. Never invent marks.
- Every fill/select step needs value_template "{{arg_name}}" and arg_name must be in input_schema.properties.
- For <select> elements copy their options into the property's "enum".
- Optional fields (not needed to complete the action) must NOT be in "required".
- Ignore login, sign-up, cookie, ads and footer links.
- Descriptions are what an AI agent reads to decide whether to call the tool, so be exact:
  * a write form keyed by an ID: say it "creates the record if the ID is new, otherwise updates it" and name the
    fields it can set; say plainly that fields not on the form cannot be set.
  * a read/search tool: say what records it returns (the site's own data, e.g. "returns matching videos with
    title, channel, duration and views").
Return ONLY JSON:
{"app_title": "...", "notes": "one sentence on what the page is",
 "tools": [{"name": "snake_case", "method": "GET"|"POST", "description": "...",
   "input_schema": {"type": "object", "properties": {"arg": {"type": "string", "description": "..."}}, "required": ["arg"]},
   "execution_steps": [{"action": "fill"|"select"|"click"|"press_enter", "mark_id": 1, "value_template": "{{arg}}"}],
   "visual_output_keys": ["what the result contains"]}]}"""


def _unique(name: str, used: set) -> str:
    base, i = name, 2
    while name in used:
        name, i = f"{base}_{i}", i + 1
    used.add(name)
    return name


def validate_compiled_tools(raw_tools: Any, mark_map: Dict[int, Dict[str, Any]]) -> Tuple[List[CompiledMCPTool], List[str]]:
    """Ground every model-proposed step against the live mark index. Drops anything that does not exist."""
    tools: List[CompiledMCPTool] = []
    notes: List[str] = []
    used: set = set()
    by_selector = {v["selector"]: v for v in mark_map.values() if v.get("selector")}
    for rt in (raw_tools or [])[:6]:
        if not isinstance(rt, dict):
            continue
        schema = rt.get("input_schema") if isinstance(rt.get("input_schema"), dict) else {}
        props: Dict[str, Any] = dict(schema.get("properties") or {})
        required = [r for r in (schema.get("required") or []) if isinstance(r, str)]
        steps: List[ExecutionStep] = []
        for st in rt.get("execution_steps") or []:
            if not isinstance(st, dict) or st.get("action") not in ACTIONS:
                continue
            action = st["action"]
            mid = st.get("mark_id")
            el = mark_map.get(int(mid)) if str(mid).strip().isdigit() else None
            el = el or by_selector.get(st.get("css_fallback", ""))
            if el is None:
                notes.append(f"dropped step {action} mark={mid} in {rt.get('name')} (mark not on page)")
                continue
            vt = st.get("value_template")
            if el["tag"] == "select" and action in ("fill", "click"):
                action = "select"
            if el["tag"] in ("input", "textarea") and action == "click" and vt:
                action = "fill"
            if action in ("fill", "select") and el["tag"] not in ("input", "textarea", "select"):
                notes.append(f"dropped step {action} on <{el['tag']}> mark={mid} in {rt.get('name')} (element is not a field)")
                continue
            if action == "fill" and el["tag"] == "input" and el.get("type") in ("checkbox", "radio", "submit", "button", "file"):
                notes.append(f"dropped fill on {el.get('type')} input mark={mid} in {rt.get('name')}")
                continue
            if action in ("click", "press_enter") and el["tag"] not in ("input", "textarea", "select"):
                target_text = el.get("text") or el.get("aria") or el.get("label") or ""
            else:
                target_text = el.get("label") or el.get("placeholder") or el.get("aria") or el.get("text") or ""
            if action in ("fill", "select") or (action == "click" and vt):
                arg_names = re.findall(r"\{\{(\w+)\}\}", vt or "")
                if not arg_names:
                    arg_names = [snake(target_text)]
                    vt = "{{" + arg_names[0] + "}}"
                for a in arg_names:
                    spec = props.get(a) if isinstance(props.get(a), dict) else {}
                    spec.setdefault("type", "string")
                    spec.setdefault("description", target_text or a)
                    props[a] = spec
                if action == "select" and el.get("options"):
                    props[arg_names[0]]["enum"] = el["options"]
            else:
                vt = None
            steps.append(ExecutionStep(action=action, mark_id=el["mark_id"], css_fallback=el.get("selector", ""),
                                       target_text=target_text, value_template=vt))
        if not steps:
            continue
        # Every argument referenced by a step must exist; required must be a subset of properties.
        required = [r for r in required if r in props]
        tools.append(CompiledMCPTool(
            name=_unique(snake(rt.get("name") or "tool", "tool"), used),
            method="POST" if str(rt.get("method", "")).upper() == "POST" else "GET",
            description=str(rt.get("description") or "Compiled web action")[:300],
            input_schema={"type": "object", "properties": props, "required": required},
            execution_steps=steps,
            visual_output_keys=[str(k) for k in (rt.get("visual_output_keys") or [])][:8],
        ))
    return tools, notes


SUBMIT_WORDS = re.compile(r"submit|execute|save|send|apply|update|query|search|go|run|confirm|create|add|filter|find", re.I)
SEARCH_WORDS = re.compile(r"search|query|find|^q$|keyword", re.I)
TEXT_INPUT_TYPES = ("", "text", "search", "email", "number", "tel", "url")


def heuristic_compile(mark_map: Dict[int, Dict[str, Any]], title: str) -> List[CompiledMCPTool]:
    """Model-free compiler: one tool per form-like group, a search tool, and a section navigator."""
    tools: List[CompiledMCPTool] = []
    used: set = set()
    fields = [m for m in mark_map.values()
              if (m["tag"] in ("input", "textarea") and m["type"] in TEXT_INPUT_TYPES) or m["tag"] == "select"]
    buttons = [m for m in mark_map.values() if m["tag"] == "button" or m["type"] in ("submit", "button")]

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for f in fields:
        groups.setdefault(f["group"], []).append(f)
    for gid, gfields in groups.items():
        gbuttons = [b for b in buttons if b["group"] == gid] if gid else []
        submit = next((b for b in gbuttons if SUBMIT_WORDS.search(b.get("text", ""))), gbuttons[0] if gbuttons else None)
        is_search = len(gfields) == 1 and gfields[0]["tag"] != "select" and SEARCH_WORDS.search(
            " ".join([gfields[0].get(k, "") for k in ("name", "placeholder", "aria", "label")]))
        props: Dict[str, Any] = {}
        steps: List[ExecutionStep] = []
        for f in gfields:
            label = f.get("label") or f.get("placeholder") or f.get("aria") or f.get("name") or f"field_{f['mark_id']}"
            arg = "query" if is_search else _unique(snake(label), set(props))
            props[arg] = {"type": "string", "description": label}
            if f["tag"] == "select" and f.get("options"):
                props[arg]["enum"] = f["options"]
            steps.append(ExecutionStep(action="select" if f["tag"] == "select" else "fill", mark_id=f["mark_id"],
                                       css_fallback=f.get("selector", ""), target_text=label, value_template="{{" + arg + "}}"))
        if submit:
            steps.append(ExecutionStep(action="click", mark_id=submit["mark_id"], css_fallback=submit.get("selector", ""),
                                       target_text=submit.get("text", "")))
        else:
            last = steps[-1]
            steps.append(ExecutionStep(action="press_enter", mark_id=last.mark_id, css_fallback=last.css_fallback, target_text=last.target_text))
        verb = (submit or {}).get("text", "") if submit else ""
        is_post = bool(len(gfields) >= 2 and re.search(r"execute|save|submit|create|update|send|add|confirm", verb, re.I))
        name = "search_site" if is_search else snake(verb or "submit_form", "submit_form")
        tools.append(CompiledMCPTool(
            name=_unique(name, used),
            method="POST" if is_post else "GET",
            description=(f"Search {title}" if is_search else f"Fill and submit the '{verb or 'form'}' panel on {title}"),
            # Search needs its query; a write form needs at least its first field (usually the record id).
            input_schema={"type": "object", "properties": props, "required": list(props)[:1] if (is_search or is_post) else []},
            execution_steps=steps,
            visual_output_keys=["answer", "items"],
        ))

    # Navigation = the largest horizontal row of short links near the top of the page.
    rows: Dict[int, List[Dict[str, Any]]] = {}
    for m in mark_map.values():
        if m["tag"] == "a" and m.get("in_viewport") and 0 <= m["top"] < 250 and 1 < len(m.get("text", "")) <= 24:
            rows.setdefault(round(m["top"] / 8), []).append(m)
    nav = max(rows.values(), key=len) if rows else []
    if len(nav) >= 3:
        texts = list(dict.fromkeys(m["text"] for m in nav))[:12]
        tools.append(CompiledMCPTool(
            name=_unique("open_section", used),
            method="GET",
            description=f"Open a top-level section of {title} and read it",
            input_schema={"type": "object", "properties": {"section": {"type": "string", "enum": texts, "description": "Section link text"}}, "required": ["section"]},
            execution_steps=[ExecutionStep(action="click", target_text="section link", value_template="{{section}}")],
            visual_output_keys=["answer", "items"],
        ))
    return tools


READ_TOOL = CompiledMCPTool(
    name="read_page",
    method="GET",
    description="Read the current page and answer a question about it / extract its visible records.",
    input_schema={"type": "object", "properties": {}, "required": []},
    execution_steps=[],
    visual_output_keys=["answer", "items"],
)


def site_slug(url: str, title: str = "") -> str:
    host = (urlparse(url).hostname or "").lower()
    if host in ("localhost", "") or re.match(r"^[\d.]+$", host):
        first = re.findall(r"[A-Za-z]+", title or "") or re.findall(r"[A-Za-z]+", urlparse(url).path) or ["local"]
        return snake(first[0], "local")
    parts = [p for p in host.split(".") if p not in ("www", "m")]
    return snake(parts[-2] if len(parts) >= 2 else parts[0], "site")


async def compile_url_to_manifest(url: str, output_path: str = "mcp_manifest.json", som_path: str = "debug_compiled_som.png") -> MCPManifest:
    if not re.match(r"^https?://", url or ""):
        raise ValueError("Paste a full URL starting with http:// or https://")
    print(f"[*] Inspecting UI with Playwright: {url}")
    async with async_playwright() as p:
        browser, page = await open_browser(p)
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await settle_page(page, 900)
            b64_img, mark_map, links, clean_text = await annotate_and_capture(page, save_path=som_path)
            title = await page.title()
            await page.evaluate("window.scrollBy(0, 600)")
            await page.wait_for_timeout(350)
            below_fold = await page.evaluate("() => document.body.innerText.slice(-600)")
        finally:
            await browser.close()

    print(f"[*] {len(mark_map)} marks. Compiling tools via Nebius ({NEBIUS_MODEL})...")
    out, err = await llm_json(
        NEBIUS_MODEL,
        COMPILER_SYSTEM_PROMPT,
        [
            {"type": "text", "text": (
                f"URL: {url}\nTitle: {title}\n"
                f"Mark index:\n{json.dumps(mark_brief(mark_map))[:9000]}\n"
                f"Page text:\n{clean_text[:1800]}"
            )},
            img_part(b64_img),
        ],
        max_tokens=4096,
    )
    tools, dropped = validate_compiled_tools((out or {}).get("tools"), mark_map)
    if tools:
        compiler = f"Nemotron {model_mode(NEBIUS_MODEL)} + live-DOM validation"
        notes = str((out or {}).get("notes", "")).strip()[:200]
        if dropped:
            notes += f" | validator dropped {len(dropped)} ungrounded step(s)"
    else:
        tools = heuristic_compile(mark_map, title or url)
        compiler = "DOM heuristic compiler (vision compile " + (f"unavailable: {err[:60]}" if err else "returned no grounded tools") + ")"
        notes = ""
    tools.append(READ_TOOL.model_copy(deep=True))
    slug = site_slug(url, title)
    for t in tools:  # prefix so several compiled sites can share one MCP connector
        if not t.name.startswith(slug + "_"):
            t.name = f"{slug}_{t.name}"[:60]

    manifest = MCPManifest(
        target_url=url,
        app_title=str((out or {}).get("app_title") or title or url)[:120],
        compiler=compiler,
        compiler_notes=notes,
        discovered_links=[DiscoveredLink(**lnk) for lnk in links],
        below_fold_summary=below_fold.strip()[:240],
        page_context_snapshot=clean_text[:2000],
        tools=tools,
    )
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    blob = manifest.model_dump_json(indent=2)
    for path in (output_path, os.path.join(MANIFEST_DIR, f"{slug}.json")):
        with open(path, "w") as f:
            f.write(blob)
    try:
        import shutil
        shutil.copyfile(som_path, os.path.join(MANIFEST_DIR, f"{slug}.png"))
    except Exception:
        pass
    print(f"[+] {compiler}: {[t.name for t in tools]} -> manifests/{slug}.json")
    return manifest


# =====================================================================
# 6. PHASE 2: RUNTIME
# =====================================================================
LOCATE_SYSTEM_PROMPT = """You re-ground a UI action whose element could not be found by selector or text.
Given the screenshot (red Set-of-Marks badges), the mark index and the intended action, pick the ONE mark to act on.
Return ONLY JSON: {"mark_id": <int or null>, "reason": "short"}"""

HEAL_SYSTEM_PROMPT = """You are the AetherUI self-healing runtime. A blocking modal/overlay covers the page.
From the screenshot (red Set-of-Marks badges) and mark index, choose the ONE mark that acknowledges, dismisses or
completes the modal so the user's task can continue. Return ONLY JSON: {"mark_id": <int>, "reason": "short"}"""

EXTRACT_SYSTEM_PROMPT = """You are the AetherUI result extractor. You get the user's goal, the tool that ran, the final
screenshot, page text, HTML tables, text lines that newly appeared after the action, and NETWORK RECORDS: structured
records parsed from the site's own API responses. Network records are authoritative — build "items" from them when
present (keep their field values exactly, rename keys to readable names). Ignore ads, sponsored and promoted entries.
Use page text only when there are no network records. For actions (POST), decide if the action succeeded and quote the evidence.
Return ONLY JSON: {"answer": "1-2 sentences", "items": [{...records relevant to the goal, max 15...}],
"action_succeeded": true|false|null, "evidence": "short quote from the page proving the result"}"""

# Telemetry / ads / player beacons only — the site's real data APIs (e.g. youtubei/v1/search) must stay.
NOISE_URL = re.compile(r"log_event|/log\b|/ping|analytics|telemetry|doubleclick|googlesyndication|googleads|/pagead|ptracking|heartbeat|gen_204|stats/|watchtime|/jnn/|GenerateIT|feedback|/player\b|/qoe", re.I)


def site_key(host: str) -> str:
    host = (host or "").lower()
    if re.match(r"^[\d.]+$", host) or host == "localhost":
        return host
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


EMBEDDED_DATA_JS = """
() => {
    const out = [];
    for (const k of ['ytInitialData', '__INITIAL_STATE__', '__NUXT__', '__APOLLO_STATE__', '__PRELOADED_STATE__']) {
        try { if (window[k] && typeof window[k] === 'object') out.push({ endpoint: 'embedded:' + k, payload: window[k] }); } catch (e) {}
    }
    for (const s of document.querySelectorAll('script#__NEXT_DATA__, script[type="application/json"], script[type="application/ld+json"]')) {
        try { if (s.textContent.length < 3e6) out.push({ endpoint: 'embedded:' + (s.id || s.type), payload: JSON.parse(s.textContent) }); } catch (e) {}
    }
    return out;
}
"""

_TEXT_KEYS = ("simpleText", "text", "content", "label")
_TITLE_KEY = re.compile(r"title|name|headline|label|id$|_id", re.I)


def _flat(v: Any) -> str:
    if isinstance(v, bool) or v is None:
        return ""
    if isinstance(v, (str, int, float)):
        return str(v)
    if isinstance(v, dict):
        for k in _TEXT_KEYS:
            if isinstance(v.get(k), (str, int, float)) and not isinstance(v.get(k), bool):
                return str(v[k])
        if isinstance(v.get("runs"), list):
            return "".join(str(r.get("text", "")) for r in v["runs"] if isinstance(r, dict))
    return ""


def _as_record(d: Dict[str, Any]) -> Dict[str, str]:
    rec: Dict[str, str] = {}
    for k, v in d.items():
        t = _flat(v).strip()
        if not t or len(t) > 180 or t.startswith(("/", "http", "data:")):
            continue
        if " " not in t and len(t) > 40:  # tokens, tracking params, base64 blobs
            continue
        rec[re.sub(r"(Text|Renderer)$", "", k) or k] = t
    return rec


def find_records(blobs: List[Any], limit: int = 20) -> List[Dict[str, str]]:
    """Generic: walk JSON the site fetched, return the dominant family of repeated, human-readable records."""
    found: List[Dict[str, str]] = []

    def walk(o: Any, depth: int = 0):
        if depth > 60 or len(found) > 5000:
            return
        if isinstance(o, dict):
            rec = _as_record(o)
            if len(rec) >= 3:
                found.append(rec)
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, list):
            for v in o:
                walk(v, depth + 1)

    for b in blobs:
        walk(b)
    groups: Dict[frozenset, List[Dict[str, str]]] = {}
    for r in found:
        groups.setdefault(frozenset(r), []).append(r)

    def score(sig: frozenset, rows: List[Dict[str, str]]) -> float:
        human = sum(1 for r in rows for v in r.values() if " " in v) / max(len(rows), 1)
        if human == 0:
            return 0
        titled = 2 if any(_TITLE_KEY.search(k) for k in sig) else 1
        return len(rows) * (1 + human) * min(len(sig), 8) ** 2 * titled  # richer records (title+channel+views+length) beat thin ones (playlist tracks)

    scored = [(score(sig, rows), sig) for sig, rows in groups.items() if len(rows) >= 2]
    scored = [x for x in scored if x[0] > 0]
    if not scored:
        return []
    best = max(scored, key=lambda x: x[0])[1]
    out, seen = [], set()
    for r in found:  # keep original order; merge close variants of the winning record shape
        keys = frozenset(r)
        if len(keys & best) / len(best) >= 0.6:
            key = json.dumps(r, sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append(r)
        if len(out) >= limit:
            break
    return out


def normalize_args(tool: CompiledMCPTool, kwargs: Dict[str, Any]) -> Dict[str, str]:
    kwargs = kwargs or {}
    props = tool.input_schema.get("properties", {})
    required = set(tool.input_schema.get("required", []))
    out: Dict[str, str] = {}
    canon = lambda s: re.sub(r"[\s_\-]+", "", str(s)).upper()
    for key, spec in props.items():
        raw = kwargs.get(key)
        val = "" if raw is None else str(raw).strip()
        enum = spec.get("enum") if isinstance(spec, dict) else None
        if enum and val:
            match = next((e for e in enum if canon(e) == canon(val)), None) or next((e for e in enum if canon(val) in canon(e)), None)
            if match is None:
                raise ValueError(f"'{val}' is not a valid {key} (options: {', '.join(map(str, enum[:12]))})")
            val = match
        if key in required and not val:
            raise ValueError(f"Missing required argument '{key}' for tool {tool.name}")
        out[key] = val
    return out


class Run:
    """Collects the proof timeline for one tool execution."""

    def __init__(self):
        self.dir = os.path.join(RUNS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        os.makedirs(self.dir, exist_ok=True)
        self.shots: List[Dict[str, str]] = []
        self.trace: List[str] = []

    async def snap(self, page: Page, label: str, annotated_png: Optional[bytes] = None):
        path = os.path.join(self.dir, f"{len(self.shots) + 1:02d}.{'png' if annotated_png else 'jpg'}")
        data = annotated_png or await page.screenshot(type="jpeg", quality=65, full_page=False)
        with open(path, "wb") as f:
            f.write(data)
        self.shots.append({"label": label, "path": path})

    def log(self, msg: str):
        self.trace.append(msg)
        print("   " + msg)


async def _visible(loc: Locator) -> bool:
    try:
        return await loc.count() > 0 and await loc.first.is_visible()
    except Exception:
        return False


async def locate(page: Page, step: ExecutionStep, val: str, run: Run, goal: str) -> Optional[Locator]:
    cands: List[Locator] = []
    if step.css_fallback:
        cands.append(page.locator(step.css_fallback))
    if step.action == "click" and val:  # navigation-style click on a link/button whose text is the argument
        cands += [page.get_by_role("link", name=val, exact=True), page.get_by_role("button", name=val, exact=True), page.get_by_text(val, exact=True)]
    t = step.target_text
    if t:
        if step.action in ("fill", "select", "press_enter"):
            cands += [page.get_by_label(t, exact=True), page.get_by_placeholder(t, exact=True)]
        else:
            cands += [page.get_by_role("button", name=t, exact=True), page.get_by_role("link", name=t, exact=True), page.get_by_text(t, exact=True)]
    for c in cands:
        if await _visible(c):
            return c.first
    # Layout drifted: ask Nemotron to re-ground the element visually.
    run.log(f"[Re-ground] '{t or val}' not found by selector/text -> Nemotron visual re-grounding")
    b64, marks, _, _ = await annotate_and_capture(page)
    choice, err = await llm_json(
        NEBIUS_MODEL, LOCATE_SYSTEM_PROMPT,
        [{"type": "text", "text": f"Goal: {goal}\nAction: {step.action} on element described as '{t}' value='{val}'\nMark index:\n{json.dumps(mark_brief(marks))[:6000]}"}, img_part(b64)],
        max_tokens=800,
    )
    mid = (choice or {}).get("mark_id")
    if str(mid).strip().isdigit() and int(mid) in marks:
        run.log(f"[Re-ground] Nemotron chose mark [{mid}] — {str(choice.get('reason', ''))[:80]}")
        return page.locator(f'[data-v2mcp-id="{int(mid)}"]').first
    run.log(f"[Re-ground] failed ({err or 'no valid mark'})")
    return None


async def heal_blockers(page: Page, run: Run, goal: str) -> bool:
    healed = False
    for _ in range(2):
        blocker = await page.evaluate(DETECT_BLOCKER_JS)
        if not blocker:
            break
        run.log(f"[Detect] Blocking UI: \"{blocker['text'][:80].strip()}\"")
        res = await page.evaluate(SOM_INJECTION_JS)
        annotated = await page.screenshot(type="png", full_page=False)
        await page.evaluate(REMOVE_BADGES_JS)
        await run.snap(page, "Blocker detected (Set-of-Marks view)", annotated_png=annotated)
        marks = {int(k): v for k, v in res["markMap"].items()}
        choice, err = await llm_json(
            NEBIUS_MODEL, HEAL_SYSTEM_PROMPT,
            [{"type": "text", "text": f"Task: {goal}\nBlocker text: {blocker['text']}\nMark index:\n{json.dumps(mark_brief(marks))[:6000]}"},
             img_part(base64.b64encode(annotated).decode())],
            max_tokens=800,
        )
        target = None
        mid = (choice or {}).get("mark_id")
        if str(mid).strip().isdigit() and await page.evaluate(MARK_INSIDE_JS, [blocker["selector"], int(mid)]):
            target = page.locator(f'[data-v2mcp-id="{int(mid)}"]').first
            run.log(f"[Self-Heal] Nemotron chose mark [{mid}] — {str(choice.get('reason', ''))[:80]}")
        else:
            run.log("[Self-Heal] " + (f"vision unavailable ({err[:60]})" if err else f"mark {mid} not inside blocker") + " -> clicking the blocker's own button")
            target = page.locator(blocker["selector"]).locator("button, [role='button']").first
        try:
            await target.click(timeout=5000)
            healed = True
            await settle_page(page, 900)
            await run.snap(page, "After self-heal")
        except Exception as e:
            run.log(f"[Heal Warning] {str(e).splitlines()[0][:100]}")
            break
    return healed


async def perform(page: Page, loc: Locator, step: ExecutionStep, val: str, run: Run):
    if step.action == "fill":
        await loc.fill(val, timeout=5000)
    elif step.action == "select":
        opt = await loc.evaluate(SELECT_MATCH_JS, val)
        if opt is None:
            raise ValueError(f"'{val}' is not an option of this dropdown")
        await loc.select_option(opt, timeout=5000)
        await page.wait_for_timeout(300)
    elif step.action == "click":
        await loc.click(timeout=5000)
        await settle_page(page, 800)
    elif step.action == "press_enter":
        await loc.press("Enter")
        await settle_page(page, 1300)


async def execute_visual_tool(
    url: str,
    tool: CompiledMCPTool,
    kwargs: Dict[str, Any],
    simulate_chaos: bool = False,
    user_intent: str = "",
) -> Dict[str, Any]:
    run = Run()
    goal = user_intent or f"{tool.description} with {json.dumps(kwargs)}"
    t0 = time.time()
    try:
        args = normalize_args(tool, kwargs)
    except ValueError as e:
        return {"status": "error", "method": tool.method, "summary": str(e), "execution_trace": [str(e)],
                "data": {}, "proof": [], "page_changes": [], "network_sources": []}

    filled = [s for s in tool.execution_steps if s.value_template and s.action in ("fill", "select")]
    if tool.method == "POST" and filled and not any(
        all(args.get(a, "") for a in re.findall(r"\{\{(\w+)\}\}", s.value_template)) for s in filled
    ):
        msg = f"Refused: {tool.name} is a write action and no field values were provided ({list(args) or 'no args'})."
        return {"status": "error", "method": tool.method, "summary": msg, "execution_trace": [msg],
                "data": {"answer": msg, "items": []}, "proof": [], "page_changes": [], "network_sources": []}

    payloads: List[Dict[str, Any]] = []
    status, healed, data, changes = "success", False, {}, []

    async with async_playwright() as p:
        browser, page = await open_browser(p)

        target_site = site_key(urlparse(url).hostname or "")

        async def on_response(response: Response):
            try:
                if response.status != 200 or NOISE_URL.search(response.url):
                    return
                if site_key(urlparse(response.url).hostname or "") != target_site:
                    return
                ctype = response.headers.get("content-type", "")
                if "json" in ctype and response.request.resource_type in ("xhr", "fetch"):
                    body = await response.json()
                    payloads.append({"endpoint": response.url, "payload": body})
            except Exception:
                pass

        page.on("response", on_response)
        try:
            run.log(f"Navigate -> {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await settle_page(page, 700)
            before_lines = set(await page.evaluate(TEXT_LINES_JS))
            await run.snap(page, "Loaded page")

            native_chaos = simulate_chaos and await page.locator("#chk_chaos").count() > 0
            if native_chaos:
                await page.locator("#chk_chaos").check()
                run.log("Chaos -> armed the page's own ERR-902 modal (fires on submit)")

            step_start = len(payloads)
            for i, step in enumerate(tool.execution_steps, 1):
                if step.action == "scroll_down":
                    await page.evaluate("window.scrollBy(0, 600)")
                    await page.wait_for_timeout(400)
                    continue
                if step.action == "wait":
                    await page.wait_for_timeout(1000)
                    continue
                val = step.value_template or ""
                for k, v in args.items():
                    val = val.replace("{{" + k + "}}", v)
                if step.value_template and (val == "" or "{{" in val):
                    run.log(f"[Step {i}] skip optional '{step.target_text}' (not provided)")
                    continue
                loc = await locate(page, step, val, run, goal)
                if loc is None:
                    raise RuntimeError(f"Could not find element for step {i} ({step.action} '{step.target_text}')")
                desc = f"{step.action.upper()} '{step.target_text or val}'" + (f" = '{val}'" if val and step.action in ('fill', 'select') else "")
                run.log(f"[Step {i}] {desc}")
                try:
                    await perform(page, loc, step, val, run)
                except Exception as e:
                    run.log(f"[Step {i}] blocked: {str(e).splitlines()[0][:90]} -> healing and retrying once")
                    healed = await heal_blockers(page, run, goal) or healed
                    await perform(page, loc, step, val, run)
                if step.action in ("click", "press_enter", "select") or i == len(tool.execution_steps):
                    await run.snap(page, f"Step {i}: {desc}")

            if simulate_chaos and not native_chaos:
                await page.evaluate(INJECT_CHAOS_JS)
                run.log("Chaos -> injected an ERR-902 overlay onto the live page")
            healed = await heal_blockers(page, run, goal) or healed
            if not healed and not any("Blocker" in s["label"] for s in run.shots):
                run.log("[Detect] No blocking UI")

            await settle_page(page, 600)
            after_lines = await page.evaluate(TEXT_LINES_JS)
            changes = [l for l in after_lines if l not in before_lines][:12]
            if changes:
                run.log(f"[Diff] {len(changes)} new line(s) on page, e.g. \"{changes[0][:80]}\"")
            tables = await page.evaluate(TABLES_JS)
            clean_text = "\n".join(after_lines)[:4000]
            final_png = await page.screenshot(type="png", full_page=False)
            await run.snap(page, "Final state (used for extraction)")

            # ---- network-first data: the site's own API responses after our steps, then any, then embedded state
            await page.wait_for_timeout(500)
            fresh = payloads[step_start:] if tool.execution_steps else payloads
            records, source = find_records([p["payload"] for p in fresh]), ""
            if records:
                source = "network API: " + urlparse(fresh[-1]["endpoint"]).path
            if not records and fresh is not payloads:
                records = find_records([p["payload"] for p in payloads])
                source = "network API (page load)" if records else ""
            if not records:
                embedded = await page.evaluate(EMBEDDED_DATA_JS)
                records = find_records([e["payload"] for e in embedded])
                source = ("embedded app state: " + embedded[0]["endpoint"]) if records else ""
            run.log(f"[Network] {len(payloads)} JSON response(s) captured; {len(records)} structured record(s)"
                    + (f" from {source}" if records else " — falling back to page content"))

            ext, err = await llm_json(
                NEBIUS_MODEL, EXTRACT_SYSTEM_PROMPT,
                [{"type": "text", "text": (
                    f"Today: {CURRENT_DATE_CONTEXT}\nGoal: {goal}\nTool: [{tool.method}] {tool.name} args={json.dumps(args)}\n"
                    f"NETWORK RECORDS ({source or 'none'}): {json.dumps(records)[:6000]}\n"
                    f"New lines after action: {json.dumps(changes)}\nTables: {json.dumps(tables)[:2000]}\n"
                    f"Page text:\n{clean_text[:(1500 if records else 3500)]}"
                )}, img_part(base64.b64encode(final_png).decode())],
                max_tokens=6000,
            )
            if ext:
                data = {k: ext.get(k) for k in ("answer", "items", "action_succeeded", "evidence")}
                run.log(f"[Extract] Nemotron: {str(ext.get('answer', ''))[:100]}")
            elif records:
                data = {"answer": f"{len(records)} record(s) from the site's own API.", "items": records[:15],
                        "action_succeeded": None, "evidence": source}
                run.log(f"[Extract] structured records from {source} (model summary unavailable: {err[:50]})")
            else:
                rows = tables[0]["rows"] if tables else []
                data = {
                    "answer": f"{len(rows) or len(after_lines[:25])} entries read from the page (model unavailable: {err[:60]}).",
                    "items": rows or [{"line": l} for l in after_lines[:25]],
                    "action_succeeded": None,
                    "evidence": changes[0] if changes else "",
                }
                run.log("[Extract] page-content fallback (tables / text)")
            data["source"] = source or "page content"

            if tool.method == "POST":
                ok = data.get("action_succeeded")
                if ok is None:
                    ok = any(re.search(r"success|confirm|saved|updated|created|TXN-", l, re.I) for l in changes)
                status = "success" if ok else "error"
                run.log(f"[Verify] action {'CONFIRMED' if ok else 'NOT CONFIRMED'} — {str(data.get('evidence') or '')[:90]}")
        except Exception as e:
            status = "error"
            run.log(f"[Runtime Error] {type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}")
            data = data or {"answer": f"Run failed: {e}", "items": []}
            try:
                await run.snap(page, "State at failure")
            except Exception:
                pass
        finally:
            await browser.close()

    return {
        "status": status,
        "method": tool.method,
        "summary": ("[Self-Healed] " if healed else "") + str(data.get("answer", "")),
        "elapsed_s": round(time.time() - t0, 1),
        "execution_trace": run.trace,
        "data": data,
        "page_changes": changes,
        "proof": run.shots,
        "network_sources": [urlparse(p["endpoint"]).path for p in payloads[-6:]],
    }


# =====================================================================
# 7. NATURAL LANGUAGE ROUTER
# =====================================================================
async def route_natural_language_command(prompt: str, manifest: MCPManifest) -> Tuple[CompiledMCPTool, Dict[str, str], str]:
    spec = [{"name": t.name, "method": t.method, "description": t.description, "input_schema": t.input_schema} for t in manifest.tools]
    system = (
        f"You route a user request to ONE tool compiled from the website '{manifest.app_title}' ({manifest.target_url}). Today is {CURRENT_DATE_CONTEXT}.\n"
        f"Page snapshot:\n{manifest.page_context_snapshot[:1200]}\n\n"
        "Rules: requests that change data -> a POST tool. Copy enum values EXACTLY as listed. Never invent values for optional "
        "fields the user did not give (use \"\"). Search queries should be clean keywords (no dates, no 'search for'). "
        "If no tool fits, use the tool whose name ends in read_page.\n"
        f"Tools:\n{json.dumps(spec, indent=2)}\n"
        'Return ONLY JSON: {"tool_name": "...", "arguments": {...}}'
    )
    parsed, err = await llm_json(ORCHESTRATOR_MODEL, system, prompt, max_tokens=1024)
    tool = next((t for t in manifest.tools if parsed and t.name == parsed.get("tool_name")), None)
    if tool is not None:
        args = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
        label = ORCHESTRATOR_MODEL
    else:
        tool, args = heuristic_route(prompt, manifest)
        label = "keyword router (" + (f"LLM unavailable: {err[:50]}" if err else "LLM picked no valid tool") + ")"
    # Single-argument tools: if the model left the argument empty, use the prompt itself.
    req = tool.input_schema.get("required", [])
    if len(req) == 1 and not str(args.get(req[0], "")).strip():
        args[req[0]] = _clean_query(prompt)
    try:
        return tool, normalize_args(tool, args), label
    except ValueError:
        read = next(t for t in manifest.tools if t.name.endswith("read_page"))
        return read, {}, label + " -> read_page (missing arguments)"


def _clean_query(prompt: str) -> str:
    q = re.sub(r"^(please\s+)?(search|find|look\s+up|get|show)(\s+me)?(\s+(for|a|an|some|the))*\s+", "", prompt.strip(), flags=re.I)
    q = re.split(r"\s+(and|then)\s+(get|list|show|give|tell|return)\b", q, flags=re.I)[0]
    q = re.sub(r"\b(videos?|on youtube|youtube)\b", "", q, flags=re.I)
    return re.sub(r"\s+", " ", q).strip(" .?!\"'") or prompt.strip()


ID_RE = re.compile(r"\b[A-Za-z]{2,}-\d+(?:-[A-Za-z0-9]+)*\b")
PERSON_RE = re.compile(r"\b(?:by|owner|engineer|assign(?:ed)?\s+to)\s+([A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,3})")
WRITE_WORDS = re.compile(r"update|insert|expedite|quarantine|approve|release|create|submit|add|change|set|ship|hold|execute|save|send", re.I)


STOPWORDS = set("a an the and or of to for in on at by with me my all any some is are be it this that from as get give show list tell what which who".split())


def enum_hit(enum: List[Any], pl: str) -> Optional[str]:
    enum = sorted(enum, key=lambda e: -len(str(e)))  # "CRITICAL_HOLD" must beat "ALL" in "show all CRITICAL_HOLD lots"
    for e in enum:
        if str(e).lower().replace("_", " ") in pl or str(e).lower() in pl:
            return e
    for e in enum:  # "quarantine" -> QUARANTINE_HOLD, "expedite" -> EXPEDITE_SHIP
        first = re.split(r"[_\s\-]+", str(e).lower())[0]
        if len(first) > 3 and re.search(r"\b" + re.escape(first), pl):
            return e
    return None


def heuristic_route(prompt: str, manifest: MCPManifest) -> Tuple[CompiledMCPTool, Dict[str, str]]:
    """Model-free fallback: score tools by word overlap/enum hits, then pull IDs, names and enums out of the prompt."""
    pl = prompt.lower()
    words = set(re.findall(r"[a-z0-9]+", pl)) - STOPWORDS

    def score(t: CompiledMCPTool) -> int:
        s = len(words & set(re.findall(r"[a-z0-9]+", (t.name + " " + t.description).lower())))
        for spec in t.input_schema.get("properties", {}).values():
            if enum_hit(spec.get("enum") or [], pl):
                s += 3
        if t.method == "POST":
            s += 4 if (WRITE_WORDS.search(pl) and ID_RE.search(prompt)) else -2
        return s

    ranked = sorted([t for t in manifest.tools if not t.name.endswith("read_page")], key=score, reverse=True)
    tool = ranked[0] if ranked and score(ranked[0]) > 0 else next(t for t in manifest.tools if t.name.endswith("read_page"))
    args: Dict[str, str] = {}
    ident = ID_RE.search(prompt)
    person = PERSON_RE.search(prompt)
    props = tool.input_schema.get("properties", {})
    for key, spec in props.items():
        k = key.lower()
        hit = enum_hit(spec.get("enum") or [], pl)
        if hit:
            args[key] = hit
        elif ident and re.search(r"id|lot|invoice|number|code|sku|order|ticket", k):
            args[key] = ident.group(0)
        elif person and re.search(r"engineer|owner|name|user|author|approv|assign", k):
            args[key] = person.group(1)
        elif re.search(r"query|search|keyword|^q$|term", k) or (len(props) == 1 and not spec.get("enum")):
            args[key] = _clean_query(prompt)
    return tool, args


# =====================================================================
# 8. FASTMCP SERVER — plug a compiled website into Cursor / Claude Desktop
# =====================================================================
TYPE_MAP = {"string": str, "integer": int, "number": float, "boolean": bool}


def load_manifests(path: str) -> List[MCPManifest]:
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json"))
    else:
        files = [path]
    out = []
    for f in files:
        try:
            with open(f) as fh:
                out.append(MCPManifest.model_validate_json(fh.read()))
        except Exception as e:
            print(f"[!] skipping {f}: {e}")
    return out


def build_mcp_server(path: str = MANIFEST_DIR) -> FastMCP:
    """Serve every compiled site (a manifests/ folder) or one manifest file as a single MCP server."""
    manifests = load_manifests(path)
    mcp = FastMCP("AetherUI")
    seen = set()
    for manifest in manifests:
        for tool_def in manifest.tools:
            if tool_def.name in seen:
                continue
            seen.add(tool_def.name)
            props = tool_def.input_schema.get("properties", {})
            required = set(tool_def.input_schema.get("required", []))

            def make_handler(bound: CompiledMCPTool, target_url: str):
                async def _tool(**kwargs) -> str:
                    res = await execute_visual_tool(target_url, bound, kwargs)
                    res["proof"] = [s["path"] for s in res.get("proof", [])]  # file paths, not image bytes
                    return json.dumps(res, indent=2)
                return _tool

            handler = make_handler(tool_def, manifest.target_url)
            handler.__name__ = tool_def.name
            desc = f"[{manifest.app_title}] {tool_def.description}"
            handler.__doc__ = desc
            handler.__signature__ = inspect.Signature([
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY,
                                  default=inspect.Parameter.empty if name in required else "",
                                  annotation=TYPE_MAP.get(spec.get("type", "string"), str))
                for name, spec in props.items()
            ])
            handler.__annotations__ = {n: TYPE_MAP.get(sp.get("type", "string"), str) for n, sp in props.items()}
            handler.__annotations__["return"] = str
            mcp.tool(name=tool_def.name, description=desc)(handler)
    print(f"[+] AetherUI MCP: {len(seen)} tools from {len(manifests)} site(s): {sorted(seen)}")
    return mcp


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("compile", "serve") or (sys.argv[1] == "compile" and len(sys.argv) < 3):
        print("Usage:\n  python3 vision_to_mcp.py compile <url> [mcp_manifest.json]\n  python3 vision_to_mcp.py serve [manifests_dir_or_file]")
        sys.exit(1)
    if sys.argv[1] == "compile":
        asyncio.run(compile_url_to_manifest(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "mcp_manifest.json"))
    else:
        build_mcp_server(sys.argv[2] if len(sys.argv) > 2 else MANIFEST_DIR).run()