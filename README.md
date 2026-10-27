# AetherUI: use MCP without writing an MCP server

**Paste a website's URL. Get an MCP server your AI agent can use.**

Most of the software companies actually run on (ERPs, fab trackers, billing portals, vendor and supplier sites, internal tools) has a web UI and **no API**. AI agents can't use any of it. Today the options are to wait years for an API, hand-write a brittle Playwright script per site, or let a browser agent click around from scratch on every request (slow, expensive, and hard to trust with writes).

AetherUI compiles the website instead. **NVIDIA Nemotron** (via **Nebius Token Factory**) looks at the page once and writes a JSON **MCP manifest** of typed tools, such as `fabtrack_execute_wafer_lot_action(lot_id, action_type, authorized_engineer)`. Claude Code, Claude Desktop or Cursor then call those tools like any API. Every call reads the site's own network data, heals unexpected popups, verifies writes, and returns screenshot proof.

> The product is the manifest plus one MCP connector. The web app ("Studio") is just where you compile sites and watch the proof.

Built for the **Nebius × NVIDIA Global AI Hackathon** (Best Apps and Agents track).

---

## See it in 30 seconds

```text
You (in Claude Code):  "Quarantine LOT-8812-kr and confirm the change."

Claude ──MCP──▶ fabtrack_execute_wafer_lot_action(lot_id="LOT-8812-kr", action_type="QUARANTINE_HOLD")
                 ├─ fills the legacy form, clicks Execute
                 ├─ an "ERR-902" compliance popup blocks the page → Nemotron picks the right button → healed
                 ├─ reads the ERP's own internal API response (not scraped text)
                 └─ verifies the row changed → returns data + 5 screenshots + trace
Claude:  "Done. LOT-8812-kr is now QUARANTINE_HOLD (TXN-393572), owner J. Chen unchanged."
```

FabTrack ERP (included) is a deliberately old-style portal: **no public API**, obfuscated element IDs, and a surprise compliance popup.

---

## How it works

| Step | What happens | Model |
|---|---|---|
| **1. Compile** (once per site) | Playwright opens the page and badges every interactive element (Set-of-Marks). Nemotron reads the screenshot and element index and proposes MCP tools. A validator then checks every step against the **live page**: steps pointing at elements that don't exist are dropped, dropdown options are copied from the real `<select>`, and selectors are pinned. Output: `manifests/<site>.json`. | Nemotron (vision) |
| **2. Route** (web chat only) | Plain English → one tool plus typed arguments. Over MCP, your own agent does this part. | Nemotron |
| **3. Run** | The runtime replays the steps deterministically. If an element moved, Nemotron re-grounds it visually. If a modal blocks the page, Nemotron chooses the dismiss button, and the choice is only accepted if that button is inside the blocker. | Nemotron (vision), only when needed |
| **4. Read** (network-first) | The site's **own JSON API responses** are captured and parsed into records (e.g. YouTube's internal search API, the ERP's `/api/internal/wafer_db`). Ads and Shorts drop out. Page text is only a fallback, and every result states its `source`. | none |
| **5. Verify and answer** | Nemotron summarizes the records and, for writes, decides whether the action really succeeded, quoting the evidence. Partial success is reported as partial. | Nemotron |
| **6. Prove** | Every call returns a screenshot timeline, the execution trace and the network endpoints used. | none |

Every model call has a deterministic fallback (DOM compiler, keyword router, blocker's own button, page text), so a model outage degrades a run instead of crashing it.

### How we use NVIDIA Nemotron and Nebius Token Factory

- **Model:** `nvidia/Nemotron-3.5-Super-VL` on **Nebius Token Factory** (OpenAI-compatible API) handles all four jobs: compiling tools from screenshots, routing, visual self-healing and verified extraction. The router model can be set separately (`ORCHESTRATOR_MODEL`) to use a smaller Nemotron for fast calls.
- **Why Token Factory helped:** one OpenAI-compatible endpoint for vision and text, so a single client covers everything. We switched model ids by changing `.env` only. At startup the app calls the models endpoint and tells you exactly what's wrong (key, URL or model id).
- **Text-only fallback:** if the configured model rejects images, the app automatically retries with the DOM index and page text instead of the screenshot, and labels the run "text-only".

---

## Quickstart (3 commands, macOS/Linux, Python 3.11+)

You need a **Nebius Token Factory API key**.

```bash
git clone https://github.com/sushrutghosal/aetherui && cd aetherui
bash setup.sh      # once: creates .venv, installs deps + Chromium, asks for your Nebius key
bash run.sh        # starts AetherUI Studio at http://127.0.0.1:8080 (keep it running)
bash connect.sh    # once: adds ONE MCP connector ("aether") to Claude Code
```

Then:
1. **Studio:** the repo ships with two compiled sites, FabTrack ERP and YouTube. Click one under "Compiled sites", or paste any URL and click **Compile**.
2. **Claude Code** (terminal or VS Code): start a **new** session, type `/mcp`, and you'll see tools like `fabtrack_execute_wafer_lot_action` and `youtube_search_videos`. Try:
   - *"Show me all CRITICAL_HOLD wafer lots with their yield."*
   - *"Add wafer lot LOT-1632002 with status READY_SHIP, engineer S. Ghosal, then list all lots."*
   - *"Search YouTube for lofi hip hop and give me the top 5 videos with channel and views."*
3. **No agent?** Use **Try it here** in Studio: chat, or call a tool directly.

- **Claude Desktop and Cursor:** click **Connect your agent** in Studio for the config, with your paths filled in.
- **Settings** live in `.env` (see `.env.example`). Run `.venv/bin/python check_nebius.py` to list the models your key can use.

### CLI
```bash
.venv/bin/python vision_to_mcp.py compile <url>        # writes manifests/<site>.json
.venv/bin/python vision_to_mcp.py serve [manifests/]   # stdio MCP server for all compiled sites
```

---

## What's in the repo

| File | Purpose |
|---|---|
| `vision_to_mcp.py` | Compiler, runtime, network-first reader, verifier, MCP server |
| `demo_ui.py` | Studio web app, plus the FabTrack ERP demo backend |
| `legacy_erp.html` | FabTrack ERP: a legacy portal with no public API |
| `manifests/` | Compiled sites (the product). One JSON file per site |
| `setup.sh` · `run.sh` · `connect.sh` | Install, run, connect |
| `check_nebius.py` | Lists the models your Nebius key can use |

---

## Limitations (stated honestly)

- **A tool can only do what the page's UI can do.** FabTrack's form has no Node or Yield fields, so those can't be set. The agent reports this instead of pretending.
- **Compiles vary slightly between runs** (e.g. an optional `open_section` tool). Validation guarantees every tool is grounded on the real page, never invented. Commit the manifests you've reviewed.
- **Network-first needs JSON in the browser.** Sites that render everything server-side fall back to page text, and the result says so (`source: page content`).
- **YouTube's internal API isn't a public API.** It mixes Shorts, ads and playlist entries, so we keep the richest record type. Results can include Shorts.
- **No login support yet,** and multi-screen workflows (SAP-style transactions) aren't compiled yet.
- **Speed:** each call launches a fresh headless browser, about 6–15 s per call.
- **The FabTrack database is in memory** and resets on restart or **Reset**.
- **Use it only on systems you're authorized to use, and respect each site's terms.** Bot checks and CAPTCHAs are respected, not bypassed.

---

## Roadmap (realistic)

1. **Promote observed network calls to direct tools.** If "Query DB" always calls `GET /api/internal/wafer_db?status=X`, compile that into a direct HTTP tool: about 10× faster, with the browser kept as fallback and proof.
2. **Login sessions.** Log in once in a visible browser, and tools reuse that session.
3. **Compile by demonstration** for multi-screen flows: record a person doing the task once, then generalize it.
4. **Self-updating manifests.** When a site changes, the healer patches the manifest instead of recovering once.
5. **Hosted on Nebius AI Cloud.** Connect with one URL (MCP over HTTP with a token), no install.
6. **Beyond the web.** The same compile-and-validate idea works for desktop apps through their scripting APIs (Unreal, Blender, Resolve) or accessibility trees, with vision only for canvas UIs.

The long-term idea is a ladder: official API if one exists → the site's own network calls → the UI → vision. Each tool uses the best level available and always returns proof.

**Where it can't go:** encrypted or signed requests (the browser path stays necessary there), canvas-only apps (vision only, slower), CAPTCHAs and bot protection (respected), and anything the UI itself doesn't expose.

---

## License

MIT. See [LICENSE](LICENSE).