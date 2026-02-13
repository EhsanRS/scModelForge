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
@click.option("--model", default=None, type=click.Path(exists=True), help="Model checkpoint path.")
@click.option("--data", required=True, type=str, help="Assessment .h5ad file (local path or cloud URL).")
@click.option("--output", default=None, type=click.Path(), help="Save results as JSON.")
@click.option("--external-model", default=None, type=str, help="External model name (e.g. 'geneformer').")
@click.option("--external-source", default=None, type=str, help="Model path or HF repo (e.g. 'ctheodoris/Geneformer').")
@click.option("--device", default="cpu", type=str, help="Device for inference (e.g. 'cpu', 'cuda').")
@click.option("--isolated", is_flag=True, default=False, help="Run external model in isolated subprocess environment.")
@click.option("--env-dir", default=None, type=click.Path(), help="Environment directory for isolated mode.")
def benchmark(
    config: str,
    model: str | None,
    data: str,
    output: str | None,
    external_model: str | None,
    external_source: str | None,
    device: str,
    isolated: bool,
    env_dir: str | None,
) -> None:
    """Run benchmarks on a trained model (native or external)."""
    import json

    import anndata as ad

    from scmodelforge.config import load_config
    from scmodelforge.eval.harness import EvalHarness

    cfg = load_config(config)

    # Load data
    from scmodelforge.data.cloud import is_cloud_path
    from scmodelforge.data.cloud import read_h5ad as cloud_read_h5ad

    if is_cloud_path(data):
        cloud_cfg = cfg.data.cloud
        adata = cloud_read_h5ad(
            data,
            storage_options=cloud_cfg.storage_options or None,
            cache_dir=cloud_cfg.cache_dir,
        )
    else:
        adata = ad.read_h5ad(data)
    click.echo(f"Loaded {adata.n_obs} cells from {data}")

    # Build harness
    harness = EvalHarness.from_config(cfg.eval)

    if external_model:
        if isolated:
            # Isolated subprocess mode
            from scmodelforge.zoo.isolation import IsolatedAdapter

            adapter_kwargs: dict[str, str] = {"device": device}
            if external_source:
                adapter_kwargs["model_name_or_path"] = external_source
            adapter = IsolatedAdapter(
                external_model,
                env_dir=env_dir,
                **adapter_kwargs,
            )
        else:
            # Direct in-process mode
            from scmodelforge.zoo.registry import get_external_model

            adapter_kwargs = {"device": device}
            if external_source:
                adapter_kwargs["model_name_or_path"] = external_source
            adapter = get_external_model(external_model, **adapter_kwargs)
        click.echo(f"Using external model: {adapter.info.full_name or adapter.info.name}")

        results = harness.run_external(adapter, {"data": adata}, device=device)
    else:
        # Native model path (original behaviour)
        if model is None:
            raise click.ClickException("--model is required when not using --external-model")

        import torch

        from scmodelforge.data.gene_vocab import GeneVocab
        from scmodelforge.models.registry import get_model
        from scmodelforge.tokenizers._utils import build_tokenizer_kwargs
        from scmodelforge.tokenizers.registry import get_tokenizer

        vocab = GeneVocab.from_adata(adata)
        tok_cfg = cfg.tokenizer
        tokenizer = get_tokenizer(tok_cfg.strategy, **build_tokenizer_kwargs(tok_cfg, vocab))

        cfg.model.vocab_size = len(vocab)
        nn_model = get_model(cfg.model.architecture, cfg.model)

        checkpoint = torch.load(model, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = {k.removeprefix("model."): v for k, v in state_dict.items()}
        nn_model.load_state_dict(cleaned)
        click.echo("Model weights loaded.")

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


@main.command(name="export")
@click.option("--checkpoint", required=True, type=click.Path(exists=True), help="Lightning checkpoint path.")
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--output", required=True, type=click.Path(), help="Output directory for HF-format model.")
@click.option("--gene-vocab", default=None, type=click.Path(exists=True), help="Gene vocabulary JSON file.")
@click.option("--no-safetensors", is_flag=True, default=False, help="Use torch format instead of safetensors.")
def export_model(checkpoint: str, config: str, output: str, gene_vocab: str | None, no_safetensors: bool) -> None:
    """Export a Lightning checkpoint to HuggingFace-format directory."""
    import torch

    from scmodelforge.config import load_config
    from scmodelforge.data.gene_vocab import GeneVocab
    from scmodelforge.models.hub import save_pretrained
    from scmodelforge.models.registry import get_model

    cfg = load_config(config)

    # Load vocab if provided
    vocab = GeneVocab.from_file(gene_vocab) if gene_vocab else None
    if vocab is not None:
        cfg.model.vocab_size = len(vocab)
    elif cfg.model.vocab_size is None:
        raise click.ClickException("model.vocab_size must be set in config or provide --gene-vocab")

    # Build model and load checkpoint
    nn_model = get_model(cfg.model.architecture, cfg.model)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {k.removeprefix("model."): v for k, v in state_dict.items()}
    nn_model.load_state_dict(cleaned)

    # Save in HF format
    save_dir = save_pretrained(
        nn_model,
        output,
        model_config=cfg.model,
        tokenizer_config=cfg.tokenizer,
        gene_vocab=vocab,
        safe_serialization=not no_safetensors,
    )
    click.echo(f"Model exported to {save_dir}")


@main.command()
@click.option("--model-dir", required=True, type=click.Path(exists=True), help="HF-format model directory.")
@click.option("--repo-id", required=True, help="HuggingFace Hub repo ID (e.g. user/model-name).")
@click.option("--private", is_flag=True, default=False, help="Create a private repository.")
@click.option("--commit-message", default="Upload scModelForge model", help="Commit message.")
def push(model_dir: str, repo_id: str, private: bool, commit_message: str) -> None:
    """Push an exported model directory to HuggingFace Hub."""
    from scmodelforge.models.hub import push_to_hub

    url = push_to_hub(
        model_dir,
        repo_id,
        private=private,
        commit_message=commit_message,
    )
    click.echo(f"Model pushed to {url}")


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--output-dir", required=True, type=click.Path(), help="Output shard directory.")
@click.option("--shard-size", default=500_000, type=int, help="Max cells per shard.")
@click.option("--gene-vocab", default=None, type=click.Path(exists=True), help="Gene vocabulary JSON file.")
def shard(config: str, output_dir: str, shard_size: int, gene_vocab: str | None) -> None:
    """Convert .h5ad files to memory-mapped shard format."""
    from scmodelforge.config import load_config
    from scmodelforge.data.gene_vocab import GeneVocab
    from scmodelforge.data.sharding import convert_to_shards

    cfg = load_config(config)

    if gene_vocab:
        vocab = GeneVocab.from_file(gene_vocab)
    else:
        # Build vocab from the first data file
        import anndata as ad

        from scmodelforge.data.cloud import is_cloud_path
        from scmodelforge.data.cloud import read_h5ad as cloud_read_h5ad

        if not cfg.data.paths:
            raise click.ClickException("No data paths in config and no --gene-vocab provided")
        first_path = cfg.data.paths[0]
        if is_cloud_path(first_path):
            cloud_cfg = cfg.data.cloud
            first_adata = cloud_read_h5ad(
                first_path,
                storage_options=cloud_cfg.storage_options or None,
                cache_dir=cloud_cfg.cache_dir,
            )
        else:
            first_adata = ad.read_h5ad(first_path)
        vocab = GeneVocab.from_adata(first_adata)

    cloud_cfg = cfg.data.cloud
    out = convert_to_shards(
        sources=cfg.data.paths,
        gene_vocab=vocab,
        output_dir=output_dir,
        shard_size=shard_size,
        storage_options=cloud_cfg.storage_options or None,
    )
    click.echo(f"Shards written to {out}")


@main.command()
@click.option("--config", default=None, type=click.Path(exists=True), help="YAML config (uses data.paths + preprocessing).")
@click.option("--input", "input_path", default=None, type=str, help="Input .h5ad file (overrides config).")
@click.option("--output", required=True, type=click.Path(), help="Output preprocessed .h5ad file.")
@click.option("--hvg", default=None, type=int, help="Select top N highly variable genes.")
def preprocess(config: str | None, input_path: str | None, output: str, hvg: int | None) -> None:
    """Preprocess an .h5ad file (normalize, log1p, HVG) and save to disk."""
    from scmodelforge.data.preprocess import preprocess_h5ad

    if config is not None:
        from scmodelforge.config import load_config

        cfg = load_config(config)
        src = input_path or (cfg.data.paths[0] if cfg.data.paths else None)
        if src is None:
            raise click.ClickException("No input path: provide --input or set data.paths in config.")
        normalize = cfg.data.preprocessing.normalize
        target_sum = cfg.data.preprocessing.target_sum
        log1p = cfg.data.preprocessing.log1p
        hvg_n = hvg if hvg is not None else cfg.data.preprocessing.hvg_selection
    elif input_path is not None:
        src = input_path
        normalize = "library_size"
        target_sum = 1e4
        log1p = True
        hvg_n = hvg
    else:
        raise click.ClickException("Provide --input or --config.")

    click.echo(f"Preprocessing {src} -> {output}")
    preprocess_h5ad(
        input_path=src,
        output_path=output,
        normalize=normalize,
        target_sum=target_sum,
        log1p=log1p,
        hvg_n_top_genes=hvg_n,
    )
    click.echo("Done.")


@main.group()
def zoo() -> None:
    """Manage isolated environments for external pretrained models."""


@zoo.command(name="install")
@click.argument("model_name")
@click.option("--env-dir", default=None, type=click.Path(), help="Base directory for environments.")
@click.option("--python", "python_version", default=None, type=str, help="Python version (e.g. '3.10').")
@click.option("--extra-deps", multiple=True, help="Additional pip packages to install.")
def zoo_install(model_name: str, env_dir: str | None, python_version: str | None, extra_deps: tuple[str, ...]) -> None:
    """Install an isolated environment for MODEL_NAME."""
    from scmodelforge.zoo.isolation import install_env

    click.echo(f"Installing isolated environment for '{model_name}'...")
    install_env(
        model_name,
        env_dir=env_dir,
        python_version=python_version,
        extra_deps=list(extra_deps) if extra_deps else None,
    )
    click.echo(f"Environment for '{model_name}' installed successfully.")


@zoo.command(name="list")
@click.option("--env-dir", default=None, type=click.Path(), help="Base directory for environments.")
def zoo_list(env_dir: str | None) -> None:
    """List installed isolated model environments."""
    from scmodelforge.zoo._env_registry import list_installed_envs

    envs = list_installed_envs(base_dir=env_dir)
    if not envs:
        click.echo("No isolated environments installed.")
        return
    for info in envs:
        status_icon = "+" if info.status == "installed" else "!"
        click.echo(f"  [{status_icon}] {info.model_name}  ({info.status})  {info.env_path}")


@zoo.command(name="remove")
@click.argument("model_name")
@click.option("--env-dir", default=None, type=click.Path(), help="Base directory for environments.")
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def zoo_remove(model_name: str, env_dir: str | None, yes: bool) -> None:
    """Remove an isolated environment for MODEL_NAME."""
    from scmodelforge.zoo._env_registry import remove_env

    if not yes:
        click.confirm(f"Remove isolated environment for '{model_name}'?", abort=True)
    if remove_env(model_name, base_dir=env_dir):
        click.echo(f"Removed environment for '{model_name}'.")
    else:
        click.echo(f"No environment found for '{model_name}'.")


if __name__ == "__main__":
    main()
