# Changelog

All notable changes to scModelForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- Project scaffolding with src-layout package structure.
- Five core module stubs: data, tokenizers, models, training, eval.
- Configuration schema with YAML loading via OmegaConf.
- CLI skeleton with `train` and `benchmark` subcommands.
- CI/CD via GitHub Actions (lint, test across Python 3.10-3.12, typecheck, release).
- Pre-commit hooks (ruff linter/formatter, trailing whitespace, YAML check).
- Sphinx + MyST documentation skeleton.
- Shared test fixtures (mini AnnData, gene vocab, temp h5ad files).
- Example Geneformer-style training configuration.
