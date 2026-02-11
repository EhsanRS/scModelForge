"""Command-line interface for scModelForge."""

from __future__ import annotations

import click

from scmodelforge import __version__


@click.group()
@click.version_option(version=__version__, prog_name="scModelForge")
def main() -> None:
    """scModelForge: Train single-cell foundation models."""


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
def train(config: str) -> None:
    """Train a model from a YAML config."""
    click.echo(f"Training with config: {config}")
    # Will be implemented in Stage 4 (training module)


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--model", required=True, help="Path to model checkpoint or Hub ID.")
def benchmark(config: str, model: str) -> None:
    """Run evaluation benchmarks on a trained model."""
    click.echo(f"Benchmarking model {model} with config: {config}")
    # Will be implemented in Stage 5 (eval module)


if __name__ == "__main__":
    main()
