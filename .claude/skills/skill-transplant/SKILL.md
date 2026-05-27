---
name: skill-transplant
description: Transplant a skill, command, or capability from one agentic-system repo into another via a 5-phase gated protocol (inventory → import → localize → wire → verify). Use when grafting a feature from an upstream repo that has its own dependencies, platform assumptions, or upstream-specific file references that need adapting before the feature will work in the destination repo. Not for copying a self-contained file — use only when the feature has a dependency footprint and the destination repo has its own conventions the import must respect.
---

Transplant a feature — a skill, command, sub-system, or capability — from a source agentic-system repo into a destination repo. The protocol treats the import like an organ transplant: identify the organ (in-scope files) and its vessels (dependencies), check compatibility with the host (destination repo conventions and platform), localize to prevent rejection (dead references, platform paths, language, optional branches), and verify the transplant is working before closing up.

Execute in 5 strict phases. Do not skip, merge, or reorder.

═══════════════════════════════════════════════════════════════
PHASE 1 — INVENTORY (read-only, no file writes yet)
═══════════════════════════════════════════════════════════════

For the source repo:
1.  Fetch the repo tree. Identify ONLY files implementing the target feature.
    Open each candidate and verify it belongs — do not guess from
    filename. Trace imports transitively; stop at third-party/stdlib.
2.  Enumerate runtime dependencies ACTUALLY used by this feature
    (ignore repo-wide deps used elsewhere).
3.  Identify the entry-point signature: CLI args, exported function,
    expected input shape.
4.  Data contract — BOTH directions:
    (a) What shape does the feature CONSUME (template vars, JSON
        schema, function args)?
    (b) Where does that shape COME FROM in the upstream repo
        (file read? CLI arg? prompt? env var?) — this is critical
        for Phase 2.5 localization.
5.  Position-locked assets: flag any file that uses `__dirname`,
    `import.meta.url`, `__file__`, or similar to resolve adjacent
    assets (fonts, templates, configs). These CANNOT be relocated
    without code patches.
6.  Platform assumptions: flag hardcoded paths (`/tmp/`, `/usr/`,
    `~/.config/`), shell commands (`file`, `which`), or OS-specific
    APIs that may not exist on the destination platform.
7.  Optional integration branches: if the feature has sub-flows
    gated on external services (MCP servers, third-party APIs,
    cloud SDKs), identify them as SEPARATE branches with their own
    dependency list. Each will be evaluated for inclusion in 2.5.
8.  If upstream offers multiple paths to the same outcome
    (e.g., HTML render vs Canva render), compare their token cost,
    reliability, setup burden, and recommend a default.
9.  Localization surface: note the human language of docs/comments,
    any currency/date/locale hardcoding.

For the destination repo:
10. List what already exists that the feature will depend on:
    invocation pattern (one-skill-per-slash? sub-commands?),
    available tools/MCP servers, shell utilities, platform (OS,
    shell, Node/Python version), existing data sources that could
    replace upstream input sources.

Deliverable: a structured report with:
    - In-scope files (grouped by role)
    - Out-of-scope (with one-line reason each)
    - Runtime deps (minimal set)
    - Entry-point signature
    - Data contract (input source + output shape)
    - Position-locked assets
    - Platform gotchas
    - Optional branches (each with cost/reliability verdict)
    - Localization surface
    - Destination-repo compatibility check
    - Recommended default path (if multiple)
    - License considerations

STOP and wait for approval before any writes.

═══════════════════════════════════════════════════════════════
PHASE 2 — IMPORT (raw copy, no edits)
═══════════════════════════════════════════════════════════════

1.  Copy in-scope files verbatim into the destination at the paths
    specified. Respect position-locked assets identified in Phase 1.
2.  Add only the identified deps to the destination's manifest
    (package.json / requirements.txt). Add nothing else.
3.  Run post-install steps (e.g., `npx playwright install chromium`).
4.  Smoke-test the imported feature STANDALONE with a minimal input
    before touching any wiring. Do not proceed if the smoke test
    fails.

═══════════════════════════════════════════════════════════════
PHASE 2.5 — LOCALIZE (adapt to destination)
═══════════════════════════════════════════════════════════════

1.  Language: translate imported docs/comments to the destination
    repo's working language.
2.  Dead references: strip references to upstream-only files, dirs,
    or configs that do not exist in the destination
    (e.g., `cv.md`, `config/profile.yml`, upstream dashboards).
    Replace each with the destination equivalent OR with an
    instruction to prompt the user inline.
3.  Platform paths: replace OS-specific paths flagged in Phase 1
    with portable equivalents (e.g., `/tmp/` → `output/` or
    `os.tmpdir()`). Verify the substitution works on the
    destination's actual platform.
4.  Optional branches: DELETE any optional-integration section
    whose backing service is not configured in the destination.
    Do NOT leave dead instructions "for reference" — they bloat
    context and create invocation confusion.
5.  Invocation name: rename the skill/command to match the
    destination repo's invocation convention.
6.  Data sources: if the destination has a native input source
    that replaces an upstream file read (e.g., terminal-pasted
    text vs `cv.md`), rewrite the data-ingestion step to use it.
7.  Re-verify the feature still runs standalone after localization.

═══════════════════════════════════════════════════════════════
PHASE 3 — WIRE (additive only, no refactors)
═══════════════════════════════════════════════════════════════

1.  Read the target skill/function in the destination.
    Understand its current output.
2.  Add a transformation step that converts that output into the
    imported feature's data contract.
3.  Add the invocation of the imported feature as a final step.
4.  Do NOT refactor existing destination code unless strictly
    required for wiring; if required, explain why before doing so.

═══════════════════════════════════════════════════════════════
PHASE 4 — VERIFY & REPORT
═══════════════════════════════════════════════════════════════

1.  Run the skill end-to-end with real destination data.
2.  Confirm the final artifact is produced and valid.
3.  Report:
      - Files added (by path, grouped by role)
      - Files modified (with scope of each change)
      - Files NOT touched (confirm existing behavior preserved)
      - Dead references removed in Phase 2.5 (with line counts)
      - Known limitations / remaining gotchas
      - Any upstream-sync debt (divergences that will make
        future `git pull` of upstream harder)

═══════════════════════════════════════════════════════════════
CONSTRAINTS (apply to all phases)
═══════════════════════════════════════════════════════════════

- Nothing extra, nothing missing. If unsure whether a file belongs,
  ASK before copying.
- No destructive changes to destination-repo files outside the
  wiring step.
- Flag any license incompatibilities immediately (stop if
  blocking).
- If a Phase 1 finding changes the feasibility of the import
  (e.g., a required MCP is not configured, a required dep
  conflicts with an existing one), STOP and surface before
  copying anything.
- Prefer destination-repo idioms over upstream idioms whenever
  they conflict. The goal is an import that looks native, not
  an archive of the upstream repo.

═══════════════════════════════════════════════════════════════
WORKED EXAMPLE — PDF generation transplant
═══════════════════════════════════════════════════════════════

Scenario: transplant PDF generation (Playwright + headless Chromium,
markdown-to-PDF) from `github.com/santifer/career-ops` into a
personal job-search-os repo, wired to the resume-tailor skill.

Phase 1 findings (condensed):
- In-scope: `scripts/render-pdf.js`, `templates/resume.html`,
  `styles/print.css`.
- Out-of-scope: `cv.md` (upstream-only input source — replaced in
  2.5), entire Canva integration branch (no Canva API configured
  in destination), upstream dashboard configs.
- Runtime deps: `playwright` only. Post-install: `npx playwright
  install chromium`.
- Entry-point: `node render-pdf.js <input.md> <output.pdf>`.
- Data contract: consumes markdown string; produces PDF path.
- Position-locked: `templates/` and `styles/` must sit adjacent
  to `render-pdf.js` (resolved via `__dirname`).
- Platform gotcha: upstream writes to `/tmp/resume.pdf` — needs
  remap for Windows destination.
- Optional branches: Canva render path (SKIP — no API key in
  destination), HTML-only preview (SKIP — not needed for skill).
- Localization surface: English docs already, no locale hardcoding.
- License: MIT (compatible).
- Destination compatibility: Node already present, no conflicts.
- Recommended default: HTML render only (simpler, no external API).

Phase 2 actions: copied 3 files into
`skills/generate-pdf/{render-pdf.js, templates/, styles/}`.
Added `playwright` to package.json. Ran chromium post-install.
Smoke test: `node render-pdf.js sample.md sample.pdf` → passed.

Phase 2.5 actions:
- Replaced `cv.md` read with a parameter: skill accepts markdown
  string directly from resume-tailor output (no file read).
- Remapped `/tmp/resume.pdf` → `output/resume.pdf`
  (portable on Windows/Mac/Linux).
- Deleted Canva branch (45 lines): dead instructions.
- Renamed invocation from `career-ops:pdf` to `/generate-pdf`.
- Re-verified standalone: passed.

Phase 3 actions: added a final step to `/resume-tailor` that pipes
its markdown output into `/generate-pdf`. No refactor of existing
resume-tailor logic.

Phase 4 report:
- Added: 3 files under `skills/generate-pdf/`.
- Modified: 1 file (`skills/resume-tailor/SKILL.md`, added final
  PDF step — 8 lines).
- Removed: 45 lines of dead Canva references during 2.5.
- Known limitation: font rendering depends on chromium's default
  fonts (no custom font import).
- Upstream-sync debt: moderate. Future `git pull` of upstream
  won't merge cleanly because of `cv.md` replacement and Canva
  removal. Document this in PROVENANCE.md.
