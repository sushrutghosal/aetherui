import base64
import json
import os
import random
import re
import shutil
import sys
import time
from typing import Any, Dict

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from vision_to_mcp import (
    MANIFEST_DIR,
    NEBIUS_MODEL,
    check_nebius,
    ORCHESTRATOR_MODEL,
    RUNS_DIR,
    MCPManifest,
    compile_url_to_manifest,
    execute_visual_tool,
    route_natural_language_command,
)

app = FastAPI(title="AetherUI // JIT Protocol Compiler & Self-Healing Web Runtime")
MANIFEST_PATH = "mcp_manifest.json"
SOM_PATH = "debug_compiled_som.png"

# ---------------------------------------------------------------------
# Demo legacy system: FabTrack ERP backend (the page itself has no public API)
# ---------------------------------------------------------------------
INITIAL_WAFER_DB = [
    {"Lot_ID": "LOT-9920-NV", "Node": "2nm-GAA", "Yield": "94.8%", "Status": "READY_SHIP", "Owner": "S. Ghosal"},
    {"Lot_ID": "LOT-4041-az", "Node": "18A-EUV", "Yield": "68.2%", "Status": "CRITICAL_HOLD", "Owner": "A. Rivera"},
    {"Lot_ID": "LOT-8812-kr", "Node": "3nm-FinFET", "Yield": "91.5%", "Status": "IN_LITHOGRAPHY", "Owner": "J. Chen"},
    {"Lot_ID": "LOT-7731-tw", "Node": "2nm-GAA", "Yield": "72.0%", "Status": "CRITICAL_HOLD", "Owner": "S. Ghosal"},
]
WAFER_DB = [dict(r) for r in INITIAL_WAFER_DB]


class WaferMutateReq(BaseModel):
    lot_id: str
    action_type: str
    owner: str = ""


@app.get("/api/internal/wafer_db")
async def internal_wafer_db(status: str = "ALL"):
    records = WAFER_DB if status in ("ALL", "ALL_LOTS", "") else [r for r in WAFER_DB if status.upper() == r["Status"].upper()]
    return {"source": "FabTrack_Internal_SQL_Replica", "filter": status, "count": len(records), "records": records}


@app.post("/api/internal/wafer_db")
async def mutate_internal_wafer_db(req: WaferMutateReq):
    lot_id = req.lot_id.strip() or "UNKNOWN"
    owner = req.owner.strip()
    existing = next((r for r in WAFER_DB if r["Lot_ID"].upper() == lot_id.upper()), None)
    if existing:
        existing["Status"] = req.action_type
        if owner:  # empty owner keeps the current owner instead of wiping it
            existing["Owner"] = owner
        row, mutation = existing, "UPDATED_EXISTING_LOT"
    else:
        row = {"Lot_ID": lot_id, "Node": "PENDING", "Yield": "PENDING", "Status": req.action_type, "Owner": owner or "UNASSIGNED"}
        WAFER_DB.append(row)
        mutation = "INSERTED_NEW_LOT"
    return {
        "source": "FabTrack_Internal_SQL_Replica",
        "mutation": mutation,
        "confirmation_code": f"TXN-{random.randint(100000, 999999)}",
        "updated_owner": row["Owner"],
        "record": row,
    }


@app.get("/legacy_erp.html", response_class=HTMLResponse)
async def serve_legacy_erp():
    with open("legacy_erp.html", "r") as f:
        return f.read()


# ---------------------------------------------------------------------
# AetherUI API
# ---------------------------------------------------------------------
class CompileReq(BaseModel):
    url: str


class ExecuteFormReq(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    simulate_chaos: bool = False


class ExecuteChatReq(BaseModel):
    prompt: str
    simulate_chaos: bool = False


def file_to_data_uri(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("utf-8")


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return None
    with open(MANIFEST_PATH) as f:
        return MCPManifest.model_validate_json(f.read())


def with_images(result: Dict[str, Any]) -> Dict[str, Any]:
    result["proof"] = [{"label": s["label"], "image": file_to_data_uri(s["path"])} for s in result.get("proof", [])]
    return result


def err(msg: str, code: int = 400):
    return JSONResponse({"error": msg}, status_code=code)


@app.post("/api/reset_cache")
async def reset_cache():
    global WAFER_DB
    WAFER_DB = [dict(r) for r in INITIAL_WAFER_DB]
    shutil.rmtree(RUNS_DIR, ignore_errors=True)
    for f in (MANIFEST_PATH, SOM_PATH):  # clears the screen; compiled sites in manifests/ are kept for the MCP connector
        if os.path.exists(f):
            os.remove(f)
    return {"status": "cleared"}


NEBIUS_STATUS = {"status": "checking..."}


@app.on_event("startup")
async def _startup_check():
    NEBIUS_STATUS["status"] = await check_nebius()
    print(f"[Nebius] {NEBIUS_STATUS['status']}")


@app.get("/api/health")
async def api_health():
    return NEBIUS_STATUS


def list_sites():
    sites = []
    if os.path.isdir(MANIFEST_DIR):
        for f in sorted(os.listdir(MANIFEST_DIR)):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(MANIFEST_DIR, f)) as fh:
                        m = MCPManifest.model_validate_json(fh.read())
                    sites.append({"slug": f[:-5], "title": m.app_title, "url": m.target_url, "tools": [t.name for t in m.tools]})
                except Exception:
                    pass
    return sites


@app.get("/api/sites")
async def api_sites():
    return {"sites": list_sites(), "cwd": os.getcwd(), "python": sys.executable, "manifest_dir": MANIFEST_DIR}


class UseReq(BaseModel):
    slug: str


@app.post("/api/use")
async def api_use(req: UseReq):
    src = os.path.join(MANIFEST_DIR, f"{os.path.basename(req.slug)}.json")
    if not os.path.exists(src):
        return err("Unknown site.")
    shutil.copyfile(src, MANIFEST_PATH)
    png = src[:-5] + ".png"
    if os.path.exists(png):
        shutil.copyfile(png, SOM_PATH)
    m = load_manifest()
    return {"manifest": json.loads(m.model_dump_json()), "som_image": file_to_data_uri(SOM_PATH), "cwd": os.getcwd()}


@app.get("/api/manifest")
async def get_current_manifest(response: Response):
    response.headers["Cache-Control"] = "no-store"
    m = load_manifest()
    return {"manifest": json.loads(m.model_dump_json()) if m else None, "som_image": file_to_data_uri(SOM_PATH), "cwd": os.getcwd(), "python": sys.executable}


async def do_compile(url: str) -> Dict[str, Any]:
    t0 = time.time()
    manifest = await compile_url_to_manifest(url, MANIFEST_PATH, som_path=SOM_PATH)
    return {"elapsed_s": round(time.time() - t0, 1), "manifest": json.loads(manifest.model_dump_json()),
            "som_image": file_to_data_uri(SOM_PATH), "cwd": os.getcwd(), "python": sys.executable}


@app.post("/api/compile")
async def api_compile(req: CompileReq):
    try:
        return await do_compile(req.url.strip())
    except Exception as e:
        return err(f"Compile failed: {type(e).__name__}: {str(e)[:300]}")


@app.post("/api/execute_form")
async def api_execute_form(req: ExecuteFormReq):
    manifest = load_manifest()
    if not manifest:
        return err("Compile a website first.")
    tool = next((t for t in manifest.tools if t.name == req.tool_name), None)
    if not tool:
        return err(f"Tool '{req.tool_name}' is not in the compiled manifest.")
    try:
        result = await execute_visual_tool(manifest.target_url, tool, req.arguments, simulate_chaos=req.simulate_chaos,
                                           user_intent=f"{tool.description} with {json.dumps(req.arguments)}")
    except Exception as e:
        return err(f"Execution failed: {type(e).__name__}: {str(e)[:300]}", 500)
    return {"mode": "Direct tool call (agent protocol)", "tool_called": f"[{tool.method}] {tool.name}",
            "arguments": req.arguments, "result": with_images(result)}


@app.post("/api/execute_chat")
async def api_execute_chat(req: ExecuteChatReq):
    prompt = req.prompt.strip()
    compiled = None
    url_match = re.search(r"https?://\S+", prompt)
    try:
        # Pasting a URL into the chat compiles that site first.
        if url_match:
            url = url_match.group(0).rstrip(".,)")
            current = load_manifest()
            if not current or current.target_url != url:
                compiled = await do_compile(url)
            prompt = (prompt[:url_match.start()] + prompt[url_match.end():]).strip(" ,.:-")
        manifest = load_manifest()
        if not manifest:
            return err("Paste a website URL first (in the chat or the compile box).")
        if not prompt:
            return {"compiled": compiled, "result": None, "mode": "compile only"}
        tool, args, router = await route_natural_language_command(prompt, manifest)
        result = await execute_visual_tool(manifest.target_url, tool, args, simulate_chaos=req.simulate_chaos, user_intent=prompt)
    except Exception as e:
        return err(f"{type(e).__name__}: {str(e)[:300]}", 500)
    return {"compiled": compiled, "mode": f"Chat ({router} -> {NEBIUS_MODEL})", "tool_called": f"[{tool.method}] {tool.name}",
            "arguments": args, "result": with_images(result)}


@app.get("/", response_class=HTMLResponse)
async def index(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return INDEX_HTML.replace("__VL_MODEL__", NEBIUS_MODEL).replace("__ROUTER_MODEL__", ORCHESTRATOR_MODEL)


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AetherUI Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .card { background: #111318; border: 1px solid #23262f; border-radius: 14px; }
    .shot img { aspect-ratio: 1200/780; object-fit: cover; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: #2f3340; border-radius: 3px; }
    .stepnum { width: 22px; height: 22px; border-radius: 999px; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; }
  </style>
</head>
<body class="bg-[#0a0b0f] text-zinc-100 min-h-screen">
<div class="max-w-[1400px] mx-auto px-5 py-4 space-y-4">

  <!-- HEADER -->
  <header class="flex flex-wrap items-center justify-between gap-3">
    <div class="flex items-center gap-3">
      <div class="w-9 h-9 rounded-xl bg-indigo-500/20 border border-indigo-400/40 flex items-center justify-center text-indigo-300 font-bold">Æ</div>
      <div>
        <h1 class="text-xl font-semibold tracking-tight">AetherUI <span class="text-zinc-500 font-normal">Studio</span></h1>
        <p class="text-xs text-zinc-400">Turn any website into MCP tools your AI agent can use. No API, no MCP server to write.</p>
      </div>
    </div>
    <div class="flex items-center gap-2">
      <span id="nebius-health" class="mono text-[11px] px-2.5 py-1 rounded-full border border-zinc-700 text-zinc-400">Nebius: checking…</span>
      <span class="mono text-[11px] px-2.5 py-1 rounded-full border border-zinc-700 text-zinc-400 hidden md:inline">model: __VL_MODEL__</span>
      <button onclick="resetCache()" class="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:bg-zinc-800">Reset</button>
    </div>
  </header>

  <!-- HOW IT WORKS -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
    <div class="card px-4 py-3 flex gap-3 items-start">
      <span class="stepnum bg-indigo-500 text-white">1</span>
      <div><div class="font-semibold text-zinc-100">Compile a website</div><div class="text-zinc-400">Paste a URL. Nemotron reads the page and writes a JSON <b class="text-zinc-200">MCP manifest</b> of typed tools. This manifest is the product.</div></div>
    </div>
    <div class="card px-4 py-3 flex gap-3 items-start border-indigo-500/40">
      <span class="stepnum bg-indigo-500 text-white">2</span>
      <div><div class="font-semibold text-zinc-100">Connect your agent <span class="text-indigo-300 font-normal">(recommended)</span></div><div class="text-zinc-400">Claude Code, Claude Desktop or Cursor get every compiled site as tools through one connector.</div></div>
    </div>
    <div class="card px-4 py-3 flex gap-3 items-start">
      <span class="stepnum bg-zinc-700 text-zinc-200">3</span>
      <div><div class="font-semibold text-zinc-100">Try it here <span class="text-zinc-500 font-normal">(optional)</span></div><div class="text-zinc-400">Quick test without an agent. Every call shows screenshots, network data and a trace as proof.</div></div>
    </div>
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">

    <!-- LEFT: COMPILE + CONNECT -->
    <div class="lg:col-span-5 space-y-4">
      <section class="card p-4 space-y-3">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-semibold flex items-center gap-2"><span class="stepnum bg-indigo-500 text-white">1</span> Compile a website</h2>
          <span id="tool-count-badge" class="mono text-[11px] px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400">no site yet</span>
        </div>
        <div class="flex gap-2">
          <input id="url" placeholder="https://your-legacy-app/..." value="http://127.0.0.1:8080/legacy_erp.html" class="mono flex-1 bg-black/40 border border-zinc-700 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-indigo-400" />
          <button onclick="compileUI()" id="btn-compile" class="bg-indigo-500 hover:bg-indigo-400 text-white font-semibold px-4 py-2 rounded-lg text-xs whitespace-nowrap">Compile</button>
        </div>
        <div class="flex flex-wrap items-center gap-1.5 text-xs">
          <span class="text-zinc-500">Examples:</span>
          <button onclick="setUrlAndCompile(location.origin + '/legacy_erp.html')" class="px-2.5 py-1 rounded-full border border-zinc-700 hover:border-indigo-400 text-zinc-300">FabTrack ERP · no API</button>
          <button onclick="setUrlAndCompile('https://www.youtube.com/')" class="px-2.5 py-1 rounded-full border border-zinc-700 hover:border-indigo-400 text-zinc-300">YouTube</button>
        </div>
        <div id="compile-status" class="text-xs mono text-zinc-400"></div>
        <div id="site-chips" class="flex flex-wrap gap-1.5 text-xs mono"></div>

        <div>
          <div class="text-[11px] uppercase tracking-wider text-zinc-500 mb-1.5">MCP manifest · tools for your agent</div>
          <div id="tool-list" class="space-y-1.5 max-h-[300px] overflow-auto pr-1"><div class="text-xs text-zinc-500">Compile a site to see its tools here.</div></div>
        </div>
        <details class="text-xs">
          <summary class="cursor-pointer text-zinc-400 hover:text-zinc-200">View manifest JSON</summary>
          <pre id="manifest-box" class="mono mt-2 bg-black/40 border border-zinc-800 rounded-lg p-3 text-emerald-300 max-h-64 overflow-auto"></pre>
        </details>
        <details class="text-xs">
          <summary class="cursor-pointer text-zinc-400 hover:text-zinc-200">What Nemotron saw (Set-of-Marks screenshot)</summary>
          <img id="img-som" onclick="zoomImg(this.src)" class="mt-2 rounded-lg border border-zinc-800 w-full cursor-zoom-in hidden" />
        </details>
      </section>

      <section class="card p-4 space-y-2 border-indigo-500/40">
        <h2 class="text-sm font-semibold flex items-center gap-2"><span class="stepnum bg-indigo-500 text-white">2</span> Connect your agent <span class="text-[11px] font-normal text-indigo-300">recommended</span></h2>
        <p class="text-xs text-zinc-400">One MCP connector serves every compiled site. Your agent sees typed tools like <span class="mono text-zinc-300">fabtrack_execute_wafer_lot_action(lot_id, action_type)</span> and calls them like any API.</p>
        <button onclick="openMcpModal()" class="w-full bg-indigo-500 hover:bg-indigo-400 text-white font-semibold py-2 rounded-lg text-sm">Connect to Claude Code · Claude Desktop · Cursor</button>
      </section>
    </div>

    <!-- RIGHT: TRY IT HERE + PROOF -->
    <div class="lg:col-span-7 space-y-4">
      <section class="card p-4 space-y-3">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <h2 class="text-sm font-semibold flex items-center gap-2"><span class="stepnum bg-zinc-700 text-zinc-200">3</span> Try it here <span class="text-[11px] font-normal text-zinc-500">optional · quick test without an agent</span></h2>
          <div class="flex rounded-lg bg-black/40 p-0.5 border border-zinc-800 text-xs">
            <button id="tab-chat" onclick="switchMode('chat')">Chat</button>
            <button id="tab-form" onclick="switchMode('form')">Call a tool</button>
          </div>
        </div>

        <div id="panel-chat" class="space-y-2">
          <div id="chat-log" class="min-h-[120px] max-h-[230px] overflow-auto space-y-2 text-sm pr-1">
            <div class="text-zinc-500 text-xs">Ask in plain English. Nemotron picks the right tool and arguments, just like your agent would.</div>
          </div>
          <div class="flex flex-wrap gap-1.5 text-xs" id="prompt-chips"></div>
          <div class="flex gap-2">
            <textarea id="chat-prompt" rows="1" placeholder="e.g. Show me all CRITICAL_HOLD lots" class="flex-1 bg-black/40 border border-zinc-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-400 resize-none"
              onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();executeChat();}"></textarea>
            <button onclick="executeChat()" id="btn-chat" class="bg-zinc-100 hover:bg-white text-black font-semibold px-5 rounded-lg text-sm">Send</button>
          </div>
        </div>

        <div id="panel-form" class="space-y-2 hidden">
          <div class="text-xs text-zinc-400">This is exactly what an agent sends over MCP: a tool name and typed arguments.</div>
          <select id="tool-select" onchange="renderDynamicInputs()" class="mono w-full bg-black/40 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs"></select>
          <div id="dynamic-inputs" class="grid grid-cols-1 md:grid-cols-3 gap-2 text-xs"></div>
          <div class="flex gap-2 items-start">
            <pre id="call-preview" class="mono flex-1 bg-black/40 border border-zinc-800 rounded-lg p-2 text-[11px] text-indigo-200 max-h-28 overflow-auto"></pre>
            <button onclick="executeForm()" id="btn-form" class="bg-zinc-100 hover:bg-white text-black font-semibold px-4 py-2 rounded-lg text-xs">Call tool</button>
          </div>
        </div>

        <label class="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer">
          <input type="checkbox" id="chaos" class="w-3.5 h-3.5 accent-amber-400" />
          Inject a blocking error popup to show self-healing
        </label>
      </section>

      <section class="card p-4 space-y-3">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-semibold">Proof · what happened in the browser</h2>
          <span id="proof-meta" class="mono text-[11px] text-zinc-500 truncate max-w-[55%]">click a screenshot to enlarge</span>
        </div>
        <div id="proof-strip" class="flex gap-3 overflow-x-auto pb-1 text-xs text-zinc-500">Run something to see the screenshot timeline here.</div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <details open class="text-xs">
            <summary class="cursor-pointer text-zinc-400 mb-1">Execution trace</summary>
            <div id="trace-box" class="mono bg-black/40 border border-zinc-800 rounded-lg p-2 text-[11px] text-zinc-300 h-40 overflow-auto"></div>
          </details>
          <details open class="text-xs">
            <summary class="cursor-pointer text-zinc-400 mb-1">Result returned to the agent (JSON)</summary>
            <pre id="result-box" class="mono bg-black/40 border border-zinc-800 rounded-lg p-2 text-[11px] text-emerald-300 h-40 overflow-auto"></pre>
          </details>
        </div>
      </section>
    </div>
  </div>
</div>

<div id="lightbox" onclick="this.classList.add('hidden')" class="fixed inset-0 bg-black/90 z-50 hidden flex items-center justify-center p-8 cursor-zoom-out">
  <img id="lightbox-img" class="max-h-full max-w-full rounded-lg border border-zinc-700" />
</div>

<div id="mcp-modal" class="fixed inset-0 bg-black/80 z-50 hidden flex items-center justify-center p-6">
  <div class="card max-w-3xl w-full max-h-[90vh] overflow-auto p-6 space-y-4">
    <div class="flex items-start justify-between gap-4">
      <div>
        <h3 class="text-lg font-semibold">Connect your agent</h3>
        <p class="text-xs text-zinc-400 mt-1">One connector serves every compiled site. Each tool call runs the self-healing runtime and returns verified data with screenshot proof.</p>
      </div>
      <button onclick="document.getElementById('mcp-modal').classList.add('hidden')" class="text-zinc-400 hover:text-white text-lg">✕</button>
    </div>
    <div class="flex gap-1 text-xs bg-black/40 p-1 rounded-lg border border-zinc-800 w-fit">
      <button data-ctab="code" onclick="renderConnector('code')" class="ctab px-3 py-1 rounded">Claude Code</button>
      <button data-ctab="desktop" onclick="renderConnector('desktop')" class="ctab px-3 py-1 rounded">Claude Desktop</button>
      <button data-ctab="cursor" onclick="renderConnector('cursor')" class="ctab px-3 py-1 rounded">Cursor</button>
    </div>
    <div id="connector-body" class="space-y-4 text-sm"></div>
  </div>
</div>

<script>
let currentManifest = null;
let serverCwd = "";
let serverPython = "python3";
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const PROMPTS = {
  erp: ['Get me all the wafer lots', 'Show me all CRITICAL_HOLD lots and their yield', 'Add wafer lot LOT-1632002 with status READY_SHIP approved by S. Ghosal', 'Quarantine LOT-8812-kr', 'Show me all QUARANTINE_HOLD lots'],
  yt: ['Search Call of Duty Mobile and give me the title, creator and duration of the top videos', 'Search lofi hip hop and list the top 5 videos with views'],
};

function zoomImg(src) { if (!src) return; document.getElementById('lightbox-img').src = src; document.getElementById('lightbox').classList.remove('hidden'); }

async function api(path, body) {
  const res = await fetch(path, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {}) });
  let data = {};
  try { data = await res.json(); } catch (e) { data = { error: 'Server returned ' + res.status }; }
  if (!res.ok && !data.error) data.error = 'Server returned ' + res.status;
  return data;
}

let copyBuf = [];
function copyIdx(i, btn) {
  const t = copyBuf[i];
  const done = () => { const o = btn.innerText; btn.innerText = '✓ Copied'; setTimeout(() => btn.innerText = o, 1200); };
  if (navigator.clipboard && window.isSecureContext) { navigator.clipboard.writeText(t).then(done); return; }
  const ta = document.createElement('textarea'); ta.value = t; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove(); done();
}
function codeBlock(text) {
  copyBuf.push(text);
  return `<div class="relative"><pre class="mono bg-black/50 border border-zinc-800 rounded-lg p-3 pr-20 text-xs text-emerald-300 overflow-auto whitespace-pre-wrap break-all">${esc(text)}</pre>
    <button onclick="copyIdx(${copyBuf.length - 1}, this)" class="absolute top-2 right-2 text-xs px-2 py-1 rounded bg-indigo-500 hover:bg-indigo-400 text-white">Copy</button></div>`;
}
function step(n, title, body) {
  return `<div class="flex gap-3"><span class="stepnum bg-indigo-500 text-white shrink-0">${n}</span>
    <div class="flex-1 space-y-1.5"><div class="font-semibold text-zinc-100">${title}</div>${body}</div></div>`;
}
let sitesCache = [];
const MCP_PROMPTS = [
  'Using the aether tools, show me all CRITICAL_HOLD wafer lots with their yield.',
  'Add wafer lot LOT-1632002 with status READY_SHIP, engineer S. Ghosal, then list all lots.',
  'Quarantine LOT-8812-kr and confirm the change from the lot table.',
  'Search YouTube for Call of Duty Mobile and give me the top 5 videos with channel and duration.',
  'Write report.py that prints every lot on hold. Get the current data with the aether tools first.',
];
function renderConnector(tab) {
  copyBuf = [];
  document.querySelectorAll('.ctab').forEach(b => b.className = 'ctab px-3 py-1 rounded ' + (b.dataset.ctab === tab ? 'bg-indigo-500 text-white font-semibold' : 'text-zinc-400 hover:text-white'));
  const dir = serverCwd + '/manifests';
  const py = serverPython, script = serverCwd + '/vision_to_mcp.py';
  const toolList = sitesCache.length
    ? sitesCache.map(st => `<div class="text-xs"><span class="text-indigo-300">${esc(st.title.slice(0, 40))}</span> <span class="mono text-zinc-500">${esc(st.tools.join(', '))}</span></div>`).join('')
    : '<div class="text-xs text-amber-300">No sites compiled yet. Compile at least one site first.</div>';
  const prompts = MCP_PROMPTS.map(p => codeBlock(p)).join('');
  const cfg = JSON.stringify({ mcpServers: { aether: { command: py, args: [script, 'serve', dir] } } }, null, 2);
  let html = step(1, 'Keep AetherUI running and compile your sites', `<div class="text-xs text-zinc-400">The connector serves everything in <span class="mono">manifests/</span>. Compiled now:</div>${toolList}`);
  if (tab === 'code') {
    html += step(2, 'Register the connector (once)', '<div class="text-xs text-zinc-400">Run <span class="mono text-zinc-200">./connect.sh</span> in the project folder, or paste this. Your key and model come from <span class="mono">.env</span>.</div>' + codeBlock(`claude mcp remove aether 2>/dev/null; claude mcp add --transport stdio aether -- "${py}" "${script}" serve "${dir}"`));
    html += step(3, 'Check it connected', codeBlock('claude mcp list') + '<div class="text-xs text-zinc-400">Expect <span class="mono text-emerald-300">aether: … ✓ Connected</span>.</div>');
    html += step(4, 'Start a NEW Claude Code session and type /mcp', '<div class="text-xs text-zinc-400">You should see the tools above. After compiling another site, start a new session to pick up its tools.</div>');
  } else {
    const where = tab === 'desktop' ? 'Claude Desktop → Settings → Developer → Edit Config' : 'Cursor → Settings → MCP → Add new MCP server (or edit ~/.cursor/mcp.json)';
    html += step(2, 'Add this to the MCP config', `<div class="text-xs text-zinc-400">${where}. Your key and model come from .env.</div>` + codeBlock(cfg));
    html += step(3, 'Restart the app', '<div class="text-xs text-zinc-400">The aether tools appear in the tools menu.</div>');
  }
  html += step(tab === 'code' ? 5 : 4, 'Try these prompts', prompts);
  document.getElementById('connector-body').innerHTML = html;
}
async function refreshSites() {
  const d = await (await fetch('/api/sites')).json();
  sitesCache = d.sites || [];
  serverCwd = d.cwd || serverCwd; serverPython = d.python || serverPython;
  document.getElementById('site-chips').innerHTML = sitesCache.length ? '<span class="text-zinc-500 py-0.5">Compiled sites:</span>' + sitesCache.map(st =>
    `<button onclick="useSite('${esc(st.slug)}')" class="px-2.5 py-0.5 rounded-full border ${currentManifest && currentManifest.target_url === st.url ? 'border-indigo-400 text-indigo-200 bg-indigo-500/15' : 'border-zinc-700 text-zinc-300 hover:border-zinc-500'}">${esc(st.slug)}</button>`).join('') : '';
}
async function useSite(slug) {
  const d = await api('/api/use', { slug });
  if (d.error) return;
  hydrateManifestUI(d.manifest, d.som_image, d.cwd);
}

async function openMcpModal() {
  await refreshSites();
  renderConnector('code');
  document.getElementById('mcp-modal').classList.remove('hidden');
}

function setUrlAndCompile(u) { document.getElementById('url').value = u; compileUI(); }

async function resetCache() {
  await api('/api/reset_cache');
  location.reload();
}

function switchMode(mode) {
  document.getElementById('panel-chat').classList.toggle('hidden', mode !== 'chat');
  document.getElementById('panel-form').classList.toggle('hidden', mode !== 'form');
  const on = 'px-3 py-1 rounded-md bg-zinc-100 text-black font-semibold', off = 'px-3 py-1 rounded-md text-zinc-400 hover:text-white';
  document.getElementById('tab-chat').className = mode === 'chat' ? on : off;
  document.getElementById('tab-form').className = mode === 'form' ? on : off;
}

function hydrateManifestUI(manifest, somImage, cwd, elapsed) {
  currentManifest = manifest;
  if (cwd) serverCwd = cwd;
  document.getElementById('url').value = manifest.target_url;
  document.getElementById('manifest-box').innerText = JSON.stringify(manifest, null, 2);
  document.getElementById('tool-count-badge').innerText = `${manifest.tools.length} tools · ${manifest.app_title.slice(0, 26)}`;
  document.getElementById('compile-status').innerHTML =
    `<span class="text-emerald-400">✓ compiled${elapsed ? ' in ' + elapsed + 's' : ''}</span> <span class="text-zinc-500">· ${esc(manifest.compiler)}</span>` +
    (manifest.compiler_notes ? `<div class="text-zinc-500 mt-0.5 font-sans">${esc(manifest.compiler_notes)}</div>` : '');
  if (somImage) { const im = document.getElementById('img-som'); im.src = somImage; im.classList.remove('hidden'); }

  document.getElementById('tool-list').innerHTML = manifest.tools.map(t => {
    const args = Object.keys((t.input_schema || {}).properties || {});
    const steps = (t.execution_steps || []).map(s => `${s.action}${s.mark_id ? '[' + s.mark_id + ']' : ''}`).join(' → ') || 'read only';
    const color = t.method === 'POST' ? 'text-amber-300 border-amber-400/40 bg-amber-400/10' : 'text-emerald-300 border-emerald-400/40 bg-emerald-400/10';
    return `<div class="bg-black/30 border border-zinc-800 rounded-lg p-2.5 text-xs">
      <div class="mono"><span class="px-1.5 py-px rounded border text-[10px] ${color}">${t.method === 'POST' ? 'WRITE' : 'READ'}</span> <b class="text-zinc-100">${esc(t.name)}</b><span class="text-zinc-500">(${esc(args.join(', '))})</span></div>
      <div class="text-zinc-400 mt-1">${esc(t.description)}</div>
      <div class="mono text-zinc-600 mt-0.5 text-[10px]">${esc(steps)}</div></div>`;
  }).join('');

  const u = manifest.target_url;
  const key = u.includes('legacy_erp') ? 'erp' : u.includes('youtube') ? 'yt' : null;
  document.getElementById('prompt-chips').innerHTML = (key ? PROMPTS[key] : ['What is on this page?']).map(p =>
    `<button onclick="document.getElementById('chat-prompt').value=this.innerText" class="px-2.5 py-1 rounded-full border border-zinc-700 hover:border-indigo-400 text-zinc-300">${esc(p)}</button>`).join('');

  refreshSites();
  document.getElementById('tool-select').innerHTML = manifest.tools.map(t => `<option value="${esc(t.name)}">${t.method === 'POST' ? 'WRITE' : 'READ'} · ${esc(t.name)}</option>`).join('');
  renderDynamicInputs();
}

function renderDynamicInputs() {
  if (!currentManifest) return;
  const tool = currentManifest.tools.find(t => t.name === document.getElementById('tool-select').value) || currentManifest.tools[0];
  const props = (tool.input_schema && tool.input_schema.properties) || {};
  const req = (tool.input_schema && tool.input_schema.required) || [];
  const keys = Object.keys(props);
  const c = document.getElementById('dynamic-inputs');
  c.innerHTML = keys.length ? keys.map(k => {
    const spec = props[k], lab = `<label class="mono text-zinc-400 block mb-1">${esc(k)}${req.includes(k) ? ' *' : ''}</label>`;
    if (spec.enum && spec.enum.length) return `<div>${lab}<select data-arg="${esc(k)}" oninput="previewCall()" class="mono w-full bg-black/40 border border-zinc-700 rounded px-2 py-1.5">${spec.enum.map(v => `<option>${esc(v)}</option>`).join('')}</select></div>`;
    return `<div>${lab}<input data-arg="${esc(k)}" oninput="previewCall()" placeholder="${esc(spec.description || k)}" class="mono w-full bg-black/40 border border-zinc-700 rounded px-2.5 py-1.5" /></div>`;
  }).join('') : `<div class="col-span-3 text-zinc-500">No arguments.</div>`;
  previewCall();
}

function collectArgs() {
  const args = {};
  document.querySelectorAll('#dynamic-inputs [data-arg]').forEach(el => args[el.getAttribute('data-arg')] = el.value);
  return args;
}
function previewCall() {
  document.getElementById('call-preview').innerText = JSON.stringify({ method: 'tools/call', params: { name: document.getElementById('tool-select').value, arguments: collectArgs() } }, null, 2);
}

async function compileUI() {
  const btn = document.getElementById('btn-compile');
  btn.innerText = 'Compiling…'; btn.disabled = true;
  document.getElementById('compile-status').innerHTML = '<span class="text-amber-300">Loading the page, marking elements, asking Nemotron…</span>';
  const data = await api('/api/compile', { url: document.getElementById('url').value.trim() });
  btn.innerText = 'Compile'; btn.disabled = false;
  if (data.error) { document.getElementById('compile-status').innerHTML = `<span class="text-rose-400">${esc(data.error)}</span>`; return; }
  hydrateManifestUI(data.manifest, data.som_image, data.cwd, data.elapsed_s);
}

function addMsg(html, who) {
  const log = document.getElementById('chat-log');
  const cls = who === 'user' ? 'ml-auto bg-indigo-500/15 border-indigo-400/30' : 'bg-black/30 border-zinc-800';
  log.insertAdjacentHTML('beforeend', `<div class="max-w-[94%] w-fit border rounded-lg px-3 py-2 ${cls}">${html}</div>`);
  log.scrollTop = log.scrollHeight;
  return log.lastElementChild;
}

function itemsTable(items) {
  if (!Array.isArray(items) || !items.length || typeof items[0] !== 'object') return '';
  const cols = Object.keys(items[0]).filter(k => k !== 'URL').slice(0, 6);
  return `<div class="overflow-auto mt-2"><table class="text-xs w-full"><tr>${cols.map(c => `<th class="mono text-left text-zinc-500 font-normal pr-3 pb-1">${esc(c)}</th>`).join('')}</tr>` +
    items.slice(0, 15).map(r => `<tr class="border-t border-zinc-800">${cols.map(c => `<td class="pr-3 py-1 align-top">${esc(r[c])}</td>`).join('')}</tr>`).join('') + '</table></div>';
}

function renderAgentReply(data) {
  const r = data.result;
  const ok = r.status === 'success';
  const d = r.data || {};
  return `<div class="mono flex flex-wrap items-center gap-2 text-[11px] mb-1">
      <span class="px-1.5 rounded ${ok ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}">${ok ? '✓ ' + r.status : '✗ ' + r.status}</span>
      <span class="text-indigo-300">${esc(data.tool_called)}</span><span class="text-zinc-500">${esc(JSON.stringify(data.arguments))}</span>
      <span class="text-zinc-500">${r.elapsed_s}s · ${r.proof.length} screenshots</span></div>
    <div>${esc(r.summary)}</div>
    ${d.source ? `<div class="mono text-[11px] mt-1 ${String(d.source).startsWith('page') ? 'text-amber-300' : 'text-sky-300'}">data source: ${esc(d.source)}</div>` : ''}
    ${d.evidence ? `<div class="text-[11px] text-emerald-300/90 mt-1">evidence: “${esc(d.evidence)}”</div>` : ''}
    ${itemsTable(d.items)}`;
}

function updateProof(data) {
  const r = data.result;
  document.getElementById('proof-strip').innerHTML = (r.proof || []).map((s, i) =>
    `<div class="shot shrink-0 w-48"><img src="${s.image}" onclick="zoomImg(this.src)" class="rounded-lg border border-zinc-700 w-full cursor-zoom-in hover:border-indigo-400" />
     <div class="mt-1 text-zinc-300 text-[11px]"><span class="text-indigo-300">${i + 1}.</span> ${esc(s.label)}</div></div>`).join('') || 'No screenshots.';
  document.getElementById('proof-meta').innerText = `${data.mode || ''}`;
  document.getElementById('trace-box').innerHTML = (r.execution_trace || []).map(t =>
    `<div class="${/Heal|Detect|Re-ground|Chaos/.test(t) ? 'text-amber-300' : /Network/.test(t) ? 'text-sky-300' : /Error|NOT CONFIRMED|blocked/.test(t) ? 'text-rose-300' : ''}">→ ${esc(t)}</div>`).join('');
  const copy = JSON.parse(JSON.stringify(r)); copy.proof = (r.proof || []).map(s => s.label);
  document.getElementById('result-box').innerText = JSON.stringify(copy, null, 2);
}

async function executeChat() {
  const box = document.getElementById('chat-prompt');
  const prompt = box.value.trim();
  if (!prompt) return;
  box.value = '';
  addMsg(esc(prompt), 'user');
  const pending = addMsg('<span class="text-amber-300 text-xs">Picking a tool → driving the browser → reading network data → verifying…</span>');
  const btn = document.getElementById('btn-chat'); btn.disabled = true;
  const data = await api('/api/execute_chat', { prompt, simulate_chaos: document.getElementById('chaos').checked });
  btn.disabled = false;
  if (data.error) { pending.innerHTML = `<span class="text-rose-400">${esc(data.error)}</span>`; return; }
  if (data.compiled) hydrateManifestUI(data.compiled.manifest, data.compiled.som_image, data.compiled.cwd, data.compiled.elapsed_s);
  if (!data.result) { pending.innerHTML = `Compiled <b>${esc(currentManifest.app_title)}</b> into ${currentManifest.tools.length} tools. What should I do on it?`; return; }
  pending.innerHTML = renderAgentReply(data);
  updateProof(data);
}

async function executeForm() {
  const btn = document.getElementById('btn-form');
  btn.innerText = 'Running…'; btn.disabled = true;
  const data = await api('/api/execute_form', { tool_name: document.getElementById('tool-select').value, arguments: collectArgs(),
    simulate_chaos: document.getElementById('chaos').checked });
  btn.innerText = 'Call tool'; btn.disabled = false;
  if (data.error) { document.getElementById('result-box').innerText = data.error; return; }
  switchMode('chat');
  addMsg(`<span class="mono text-xs text-indigo-200">agent → tools/call ${esc(data.tool_called)} ${esc(JSON.stringify(data.arguments))}</span>`, 'user');
  addMsg(renderAgentReply(data));
  updateProof(data);
}

switchMode('chat');
fetch('/api/health').then(r => r.json()).then(h => { const el = document.getElementById('nebius-health'); const ok = h.status.startsWith('OK'); el.innerText = ok ? '● Nebius connected' : '● Nebius: ' + h.status; el.title = h.status; el.className = 'mono text-[11px] px-2.5 py-1 rounded-full border max-w-[420px] truncate ' + (ok ? 'border-emerald-500/40 text-emerald-300' : 'border-rose-500/40 text-rose-300'); });
fetch('/api/manifest').then(r => r.json()).then(d => { serverCwd = d.cwd || ''; serverPython = d.python || serverPython; if (d.manifest) hydrateManifestUI(d.manifest, d.som_image, d.cwd); else refreshSites(); });
</script>
</body>
</html>"""