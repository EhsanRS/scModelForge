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
@click.option("--resume", type=click.Path(exists=True), default=None, help="Resume from checkpoint.")
def train(config: str, resume: str | None) -> None:
    """Train a model from a YAML config."""
    from scmodelforge.config import load_config
    from scmodelforge.training import TrainingPipeline

    cfg = load_config(config)
    if resume:
        cfg.training.resume_from = resume
    TrainingPipeline(cfg).run()


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--model", required=True, type=click.Path(exists=True), help="Model checkpoint path.")
@click.option("--data", required=True, type=click.Path(exists=True), help="Assessment .h5ad file.")
@click.option("--output", default=None, type=click.Path(), help="Save results as JSON.")
def benchmark(config: str, model: str, data: str, output: str | None) -> None:
    """Run evaluation benchmarks on a trained model."""
    import json

    import anndata as ad
    import torch

    from scmodelforge.config import load_config
    from scmodelforge.data.gene_vocab import GeneVocab
    from scmodelforge.eval.harness import EvalHarness
    from scmodelforge.models.registry import get_model
    from scmodelforge.tokenizers.registry import get_tokenizer

    cfg = load_config(config)

    # Load data
    adata = ad.read_h5ad(data)
    click.echo(f"Loaded {adata.n_obs} cells from {data}")

    # Build vocab and tokenizer
    vocab = GeneVocab.from_adata(adata)
    tok_cfg = cfg.tokenizer
    tokenizer = get_tokenizer(
        tok_cfg.strategy,
        gene_vocab=vocab,
        max_len=tok_cfg.max_genes,
        prepend_cls=tok_cfg.prepend_cls,
    )

    # Build model from config
    cfg.model.vocab_size = len(vocab)
    nn_model = get_model(cfg.model.architecture, cfg.model)

    # Load weights from checkpoint
    checkpoint = torch.load(model, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Strip "model." prefix if saved from LightningModule
    cleaned = {}
    for k, v in state_dict.items():
        key = k.removeprefix("model.")
        cleaned[key] = v
    nn_model.load_state_dict(cleaned)
    click.echo("Model weights loaded.")

    # Build harness and run
    harness = EvalHarness.from_config(cfg.eval)
    results = harness.run(
        nn_model,
        {"data": adata},
        tokenizer,
        batch_size=cfg.eval.batch_size,
    )

    # Print results
    for r in results:
        click.echo(r.summary())

    # Optionally save JSON
    if output:
        with open(output, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        click.echo(f"Results saved to {output}")


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--checkpoint", required=True, type=click.Path(exists=True), help="Pretrained checkpoint path.")
def finetune(config: str, checkpoint: str) -> None:
    """Fine-tune a pretrained model on a downstream task."""
    from scmodelforge.config import load_config
    from scmodelforge.config.schema import FinetuneConfig
    from scmodelforge.finetuning import FineTunePipeline

    cfg = load_config(config)
    if cfg.finetune is None:
        cfg.finetune = FinetuneConfig()
    cfg.finetune.checkpoint_path = checkpoint
    FineTunePipeline(cfg).run()


if __name__ == "__main__":
    main()
