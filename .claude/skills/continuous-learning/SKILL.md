---
name: continuous-learning
description: Extract patterns from completed work into reusable skills and rules
---

# Continuous Learning

## When to Use
After completing a significant task (fixing a class of bugs, adding a pipeline stage, refactoring). Extract the pattern so future sessions benefit.

## Process

### 1. Identify the Pattern
Ask: "What did I just fix/build that could break again or be needed again?"

Examples from this repo's history:
- `dvc.yaml` deps path mismatch after file moves → rule: always update dvc.yaml when moving files
- Missing typing imports that only crash at runtime → verification skill
- Dual orchestrators (run_all.py vs dvc.yaml) going out of sync → rule: run_all.py must mirror dvc.yaml stages

### 2. Decide Where to Save
| Pattern type | Save to |
|-------------|---------|
| Always-follow rule | `.claude/rules/<topic>.md` |
| Workflow/procedure | `.claude/skills/<name>/SKILL.md` |
| Project fact | `CLAUDE.md` |

### 3. Write It Down
- Rules: lead with the imperative ("Always X", "Never Y"), add brief rationale
- Skills: lead with "When to Use", then steps, then common pitfalls
- Keep it short — if it's more than a page, split it

### 4. Prune Stale Entries
When code changes make a rule/skill obsolete:
- Delete the file
- Update CLAUDE.md if it referenced the deleted item
- Check that no other skill references the deleted rule

## Anti-Patterns
- Don't save debugging steps for a one-off bug (the fix is in the commit)
- Don't duplicate what's in CLAUDE.md
- Don't save code patterns that are obvious from reading the code
- Don't save things that change frequently (branch names, current K value)
