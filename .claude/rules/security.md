---
description: "Security rules for research codebase"
globs: ["*.py", "*.yaml", "*.yml", "*.json", ".env*"]
alwaysApply: true
---

# Security

- Never commit `.env` files or hardcode credentials (AWS keys, API tokens).
- Use `python-dotenv` to load secrets from `.env` at runtime.
- `.env.example` should document required variables without real values.
- DVC remotes configured via `dvc remote modify --local` for credentials.
- MLflow tracking URI should be configurable, not hardcoded.
