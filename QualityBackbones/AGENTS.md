# AGENTS.md

Guidance for agentic coding assistants in this repository.

## 1) Repository Snapshot
- Current root content: `models.txt`.
- No build files detected: `pyproject.toml`, `setup.py`, `setup.cfg`, `Makefile`.
- No test config detected: `pytest.ini`, `tox.ini`.
- No lint/type config detected: `ruff.toml`, `.flake8`, `mypy.ini`.
- No Cursor rules detected: `.cursor/rules/`, `.cursorrules`.
- No Copilot instructions detected: `.github/copilot-instructions.md`.
- Until project files are added, use conservative defaults below.

## 2) Instruction Priority
Apply this precedence when instructions conflict:
1. Active user request.
2. Runtime system/developer instructions.
3. This `AGENTS.md`.
4. New repo-local config files discovered later.

## 3) Environment Conventions
- Preferred stack: Linux + Python.
- Prefer conda env `encoders` when available.
- Fallback to `encoders_heads` if `encoders` is absent.
- Avoid system/global Python.
- Wrapper patterns:
  - `conda run -n encoders <command>`
  - `conda run -n encoders_heads <command>`

## 4) Build / Lint / Test Commands
### 4.1 Current state
- No confirmed runnable build/lint/test pipeline exists in repo files yet.
- If commands fail due to missing config, report exactly what is missing.
- Do not invent toolchains silently; propose minimal additions explicitly.

### 4.2 Standard Python command set (when project is initialized)
```bash
# Install/update tooling and deps
conda run -n encoders python -m pip install -U pip setuptools wheel
conda run -n encoders python -m pip install -r requirements.txt

# Tests
conda run -n encoders pytest -q
conda run -n encoders pytest tests/test_example.py -q
conda run -n encoders pytest tests/test_example.py::test_case_name -q
conda run -n encoders pytest 'tests/test_example.py::test_case_name[param_value]' -q
conda run -n encoders pytest -k "encoder and not slow" -q
conda run -n encoders pytest --lf -q
conda run -n encoders pytest -x -q

# Lint / format / type-check
conda run -n encoders ruff check .
conda run -n encoders ruff format .
conda run -n encoders mypy src
```

## 5) Coding Style Guidelines
These defaults apply until project-specific tooling overrides them.

### 5.1 Imports
- Use absolute imports from project roots.
- Group imports: stdlib, third-party, local.
- Separate groups with one blank line.
- Prefer explicit imports; avoid wildcard imports.
- Avoid import-time side effects.

### 5.2 Formatting and structure
- Follow PEP 8; target line length 100 (hard max 120).
- Prefer `ruff format` for canonical formatting.
- Keep functions focused and short.
- Prefer early returns over deep nesting.
- Remove dead code and stale commented blocks.

### 5.3 Typing
- Add type hints for public functions/methods.
- Always annotate return types.
- Prefer concrete types (`list[str]`, `dict[str, float]`).
- Use `TypedDict` or dataclasses for structured records.
- Avoid `Any`; if unavoidable, keep scope local and document why.

### 5.4 Naming
- `snake_case`: functions, variables, modules.
- `PascalCase`: classes.
- `UPPER_SNAKE_CASE`: constants.
- Use descriptive names; avoid single letters except tiny loop indices.
- Prefix private helpers with `_`.

### 5.5 Error handling
- Never use bare `except:`.
- Catch specific exceptions and preserve context.
- Raise clear, actionable messages.
- Validate external inputs at boundaries (paths, config, schema).
- Fail fast on invalid configuration.

### 5.6 Logging and diagnostics
- Use `logging` for non-interactive diagnostics.
- Include identifiers in logs (model id, dataset id, batch/chunk id).
- Keep loop logging concise; avoid per-item noise unless debugging.

### 5.7 Data and I/O
- Prefer `pathlib.Path` over string concatenation.
- Normalize paths relative to a known root.
- Avoid silent overwrites unless explicitly requested.
- Use atomic writes when practical.

### 5.8 ML / PyTorch specifics
- In inference paths, call `model.eval()`.
- Wrap inference in `torch.inference_mode()`.
- Handle device placement explicitly (`cpu`/`cuda`).
- Set precision intentionally (`float32`, `float16`, `bfloat16`).
- Guard CUDA-only paths with availability checks.

## 6) Testing Guidelines
- Keep tests deterministic (set seeds when randomness exists).
- Keep unit tests fast and isolated.
- Mark expensive tests as `slow` and exclude by default.
- Avoid network access in unit tests.
- For model tests, use tiny fixtures and smoke-style checks.

Suggested layout when tests are added:
- `tests/unit/` for pure logic.
- `tests/integration/` for end-to-end slices.
- `tests/fixtures/` for tiny sample data.

## 7) Agent Workflow Expectations
- Keep changes minimal and task-scoped.
- Update docs/tests when behavior changes.
- Prefer targeted tests first, then broader checks.
- If full validation is not possible, report what was and was not validated.

### 7.1 Chart debugging workflow
- After creating or changing charts, do not treat them as done until the rendered figure files are visually reviewed.
- Judge chart quality from the rendered artifact itself (`.pdf`/`.png`), not from the plotting code.
- Use this loop for chart debugging:
  1. Open the rendered chart and identify only visual problems: overlapping text, clipped labels, too much information, too much explanatory text, weak hierarchy, or an unclear main message.
  2. Modify the smallest relevant plotting/style/layout code that should fix that visual issue.
  3. Re-render the affected chart.
  4. Open the rendered chart again and re-evaluate only from the visual output.
  5. If the chart looks correct, stop for that representative case; otherwise repeat from step 1.
- For families of charts that share the same renderer or layout, debug one representative chart first, then apply the fix to the family and do a broader re-render.
- After representative fixes are accepted, re-render the full affected experiment/family and re-check a small control set of rendered files.
- When available, prefer using a subagent for the visual-audit step so one agent can inspect the rendered artifact while the main agent handles the code changes.

## 8) Cursor / Copilot Rule Integration
- No Cursor rule files are currently present.
- No Copilot instruction files are currently present.
- If they appear later, treat them as mandatory and merge directives here.

## 9) Update Triggers
Update `AGENTS.md` whenever these appear:
- Real build toolchain (`pyproject.toml`, `Makefile`, CI workflows).
- Test runner config, markers, official test targets.
- Lint/type settings with project-specific deviations.
- Cursor/Copilot instruction files.
- Non-Python stack becomes primary.

Until then, this file is a conservative baseline for agent behavior.
