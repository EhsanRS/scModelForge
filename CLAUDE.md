# scModelForge — Claude Code Project Instructions

## Project Overview

Single-cell foundation model pretraining toolkit. Python 3.10+, PyTorch, AnnData/scverse ecosystem.

- **Package**: `scmodelforge` (lowercase), project name `scModelForge`
- **Layout**: src-layout (`src/scmodelforge/`)
- **Build**: hatchling
- **Training**: PyTorch Lightning (`import lightning.pytorch as pl`)
- **Config**: YAML via OmegaConf, dataclasses in `config/schema.py`
- **Linting**: ruff (line-length=120, py310 target, TCH rules enabled)
- **Tests**: pytest, 820+ tests in `tests/`

## Key Conventions

### All Python files must

- Start with `from __future__ import annotations`
- Use `TYPE_CHECKING` blocks for imports only used in type hints (ruff TCH rules)
- Use `collections.abc.Iterator` not `typing.Iterator`
- Follow ruff rules: E, F, W, I, UP, B, SIM, TCH (with E501 ignored)

### Registry pattern (models, tokenizers, benchmarks)

Every component type uses the same registry pattern:
1. Decorator: `@register_model("name")`, `@register_tokenizer("name")`, `@register_benchmark("name")`
2. Base class with required abstract methods
3. Import in `__init__.py` triggers registration (mandatory)
4. Add to `__all__` in `__init__.py`

### Running tests and lint

```bash
.venv/bin/python -m pytest tests/ -v              # Full suite
.venv/bin/python -m pytest tests/test_models/ -v   # Single module
.venv/bin/ruff check src/ tests/                   # Lint
.venv/bin/ruff check --fix src/ tests/             # Auto-fix
```

## File Structure

```
src/scmodelforge/
  _constants.py          # PAD/UNK/MASK/CLS token IDs
  _plugins.py            # Third-party plugin discovery via entry points
  cli.py                 # Click CLI (train, finetune, benchmark, export, push, shard, preprocess)
  config/schema.py       # All @dataclass configs, load_config()
  data/                  # GeneVocab, CellDataset, CellDataLoader, census, orthologs, perturbation, sharding, sampling, gene_selection, streaming, cloud, preprocess
  tokenizers/            # BaseTokenizer, 4 strategies (rank_value, binned, continuous, gene_embedding), masking, registry
  models/                # 3 architectures, components/, registry, hub (save/load/push)
  training/              # Lightning module, data module, pipeline, optimizers, fsdp
  eval/                  # Benchmarks (linear_probe, embedding_quality, perturbation, grn_inference, cz_benchmarks)
  finetuning/            # FineTuneModel, LoRA, heads, pipeline
```

## Adding New Components

See `prompts/` for detailed implementation guides:

| Task | Prompt file |
|------|-------------|
| Analyze a paper's architecture | `prompts/analyze_architecture.md` |
| Implement a new model | `prompts/implement_model.md` |
| Implement a new tokenizer | `prompts/implement_tokenizer.md` |
| Implement a new benchmark | `prompts/implement_benchmark.md` |
| Add a sub-component (embedding, head, attention) | `prompts/implement_component.md` |

### Quick checklist for any new component

1. Create implementation file in the appropriate package
2. Register with decorator (`@register_model`, `@register_tokenizer`, `@register_benchmark`)
3. Import in package `__init__.py` and add to `__all__`
4. Add any new config fields to `config/schema.py` with defaults
5. Write tests following existing patterns in `tests/`
6. Run `ruff check` and `pytest`
7. Update API docs in `docs/api/`
8. **Commit and push** — after every feature, fix, or meaningful addition, commit and push to the feature branch

### Plugin system (third-party components)

Third-party packages can register tokenizers, models, and benchmarks via Python entry points — no need to modify scModelForge source. Entry-point groups:
- `scmodelforge.tokenizers` — tokenizer classes
- `scmodelforge.models` — model classes
- `scmodelforge.benchmarks` — benchmark classes

Plugin discovery is **lazy**: entry points are scanned once on the first call to `get_*()` or `list_*()`. Built-in components (registered via decorators) always take precedence over plugins with the same name. The shared discovery logic lives in `_plugins.py`; each registry has a `_state`/`_ensure_plugins()` pair.

## Workflow

- **Branch-based development**: Create a feature branch for each new stage, feature, or bug-fix batch. Branch naming: `feature/<short-description>` (e.g. `feature/stage-6-mixed-precision`). Commit frequently to the branch, then open a PR to merge into `main` when the work is complete.
- **Always commit and push** after completing a feature, bug fix, or any meaningful addition. Do not let work accumulate uncommitted. Each logical unit of work should be a separate commit pushed to the feature branch.
- **Pull requests**: When the feature branch is ready, create a PR with a summary of changes and a test plan. Do not push directly to `main`.
- **Update CLAUDE.md** after implementing stages, phases, or new features. Keep the file structure, test count, CLI commands, and other documentation current so it accurately reflects the codebase.

## Important Notes

- `lightning` package required (not just `pytorch-lightning`) for `import lightning.pytorch`
- peft `target_modules` uses plain string names (e.g., `"linear1"`), NOT regex
- Security hook warns on files containing "eval" — false positive for our evaluation module
- `docs/apidocs/` is auto-generated by Sphinx autodoc2 — do not edit manually
- Virtual environment is uv-managed at `.venv/`
