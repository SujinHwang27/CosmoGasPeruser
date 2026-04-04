---
description: "Session lifecycle protocol — how every agent session should flow"
globs: []
alwaysApply: true
---

# Session Protocol

Every session follows the workflow defined in `.claude/skills/session-workflow/SKILL.md`:

**Orient → Plan → Implement → Verify → Commit → Document → Learn → Sync**

Key enforcement points:
- Always check `git status` and memory before starting work
- Always run the verification loop before committing
- Always apply continuous learning before ending a session
- Never skip documentation when code structure changes
- Commit in logical groups with conventional commit messages
