---
name: session-workflow
description: Orchestrates a complete agent session — orient, plan, implement, verify, document, learn, sync
---

# Session Workflow

## When to Use
Every session. This is the meta-workflow that sequences all other skills.

## Phases

### 1. Orient
Understand the current state before doing anything.
- Read `CLAUDE.md` (auto-loaded)
- Check memory (`MEMORY.md` — auto-loaded)
- `git status` + `git log --oneline -5` — what branch, what's uncommitted, what happened last
- `dvc status` — is the pipeline in sync?
- Ask the user what they want to accomplish if not clear

### 2. Plan
Scope the work before writing code.
- For non-trivial tasks: propose an approach and get confirmation
- For bug fixes: diagnose first, then propose fix
- Identify which skills are relevant (DVC pipeline? Scientific ML? Verification?)
- Estimate blast radius — what files will change, what might break

### 3. Implement
Do the work. Apply domain skills as needed.
- `coding-style` rule — architecture, types, naming
- `scientific-ml` skill — algorithm details, data shapes
- `dvc-pipeline` skill — pipeline changes
- Make changes in logical groups (don't mix bug fixes with features)

### 4. Verify
Run the verification loop before committing. → **Use `verification-loop` skill**
- Import check
- `uv run pytest tests/`
- `dvc status`
- Dead code check (if files were deleted/moved)
- Docs freshness (if docs were changed)

### 5. Commit
Structured, broken-down commits.
- `fix:` for bug fixes
- `refactor:` for cleanup/moves
- `feat:` for new functionality
- `docs:` for documentation-only changes
- Each commit should be independently meaningful

### 6. Document
Update documentation to match code changes. → **Use `documentation` rule**
- `CLAUDE.md` — if architecture or conventions changed
- `docs/feature/` — if experiment results or design changed
- `README.md` — if setup or usage changed
- Fix stale paths, terminology, file references

### 7. Learn
Extract patterns from this session. → **Use `continuous-learning` skill**
- What broke that could break again?
- What non-obvious decision was made?
- Is the pattern already in a rule/skill? If not, add it.
- Update existing rules if they're incomplete

### 8. Sync
Push state to remotes and persist memory.
- `git push` (if approved)
- `git push --tags` (if new tags)
- `dvc push` (if pipeline produced new outputs)
- Update project memory with session summary + remaining tasks

## Phase Skipping
Not every session hits all phases:
- Quick bug fix: Orient → Implement → Verify → Commit
- Research/exploration: Orient → Plan → (no commit)
- Code review: Orient → Verify → Document
- Cleanup: Orient → Implement → Verify → Commit → Learn

## Anti-Patterns
- Implementing before understanding current state (skip Orient → break things)
- Committing without verifying (skip Verify → broken imports ship)
- Finishing without learning (skip Learn → same bugs recur next session)
- Giant single commits (skip structured Commit → unreadable history)
