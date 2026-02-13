# scverse Ecosystem Integration

How to use scModelForge with scanpy, anndata, scvi-tools, and other scverse tools in your existing single-cell workflows.

## Why integrate with scverse?

scModelForge is built on top of the scverse ecosystem. You don't need to learn a completely new workflow - the toolkit reads AnnData objects natively, works with scanpy preprocessing functions, and produces embeddings that plug directly into your existing analysis pipelines.

If you already use scanpy for clustering, UMAP visualization, and differential expression, you can add foundation model embeddings without changing your workflow. The embeddings from scModelForge often provide richer representations than PCA or basic preprocessing, leading to better clustering, trajectory inference, and batch integration.

## The basic integration pattern

The core workflow is straightforward:

1. Preprocess your data with scanpy (or use scModelForge preprocessing)
2. Train or load a scModelForge model
3. Extract embeddings from the model
4. Store embeddings in `adata.obsm` just like PCA or scVI embeddings
5. Use the embeddings with any scanpy analysis function

From there, all your familiar tools work: `sc.pp.neighbors()`, `sc.tl.umap()`, `sc.tl.leiden()`, `sc.tl.rank_genes_groups()`, and more.

## From scanpy to scModelForge

Here is a typical scanpy preprocessing workflow and where scModelForge fits in:

```python
import scanpy as sc

# Standard scanpy preprocessing
adata = sc.read_h5ad("my_data.h5ad")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Save for scModelForge
adata.write("processed.h5ad")
```

scModelForge reads this processed data directly. You can also skip manual preprocessing and let scModelForge handle it via the `preprocess` CLI:

```bash
scmodelforge preprocess \
  --input raw_data.h5ad \
  --output processed.h5ad \
  --hvg 2000
```

Or specify preprocessing in your training config:

```yaml
data:
  path: raw_data.h5ad
  preprocessing:
    normalize: total
    target_sum: 10000
    log_transform: true
    hvg: 2000
```

Both approaches work. Use the CLI/config preprocessing when you want reproducibility across experiments, or use scanpy directly when you need fine-grained control.

## Using scModelForge embeddings in scanpy

Once you have a trained model, extract embeddings and add them to your AnnData object:

```python
import anndata as ad
import torch
from scmodelforge.eval._utils import extract_embeddings
from scmodelforge.models import load_pretrained_with_vocab
from scmodelforge.tokenizers import get_tokenizer

# Load model and tokenizer
model, gene_vocab = load_pretrained_with_vocab("checkpoints/my_model/")
tokenizer = get_tokenizer("rank_value", gene_vocab=gene_vocab)
model.eval()

# Load your data
adata = ad.read_h5ad("my_data.h5ad")

# Extract embeddings (uses model.encode() for forward pass)
embeddings = extract_embeddings(
    model=model,
    tokenizer=tokenizer,
    adata=adata,
    device="cuda",
    batch_size=256
)

# Store in obsm just like PCA or scVI
adata.obsm["X_scmodelforge"] = embeddings

# Now use with any scanpy tool
sc.pp.neighbors(adata, use_rep="X_scmodelforge")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color=["cell_type", "leiden"])
```

The `extract_embeddings()` function handles batching, device management, and tokenization automatically. It returns a numpy array with shape `(n_cells, embedding_dim)`.

## Comparison with PCA and scVI embeddings

You can compare scModelForge embeddings side-by-side with other representation methods:

```python
import scanpy as sc
import anndata as ad
from scmodelforge.eval._utils import extract_embeddings

adata = ad.read_h5ad("data.h5ad")

# Compute PCA embeddings (standard scanpy)
sc.tl.pca(adata, n_comps=50)
adata.obsm["X_pca"] = adata.obsm["X_pca"]  # Already stored

# Compute scVI embeddings (if scvi-tools installed)
import scvi
scvi.model.SCVI.setup_anndata(adata, layer="counts")
vae = scvi.model.SCVI(adata)
vae.train()
adata.obsm["X_scvi"] = vae.get_latent_representation()

# Compute scModelForge embeddings
adata.obsm["X_scmodelforge"] = extract_embeddings(model, adata, tokenizer)

# Compare UMAPs side-by-side
for rep in ["X_pca", "X_scvi", "X_scmodelforge"]:
    sc.pp.neighbors(adata, use_rep=rep, key_added=rep)
    sc.tl.umap(adata, neighbors_key=rep)

# Visualize all three
sc.pl.umap(adata, color="cell_type", title="PCA")
sc.pl.umap(adata, color="cell_type", title="scVI", neighbors_key="X_scvi")
sc.pl.umap(adata, color="cell_type", title="scModelForge", neighbors_key="X_scmodelforge")
```

Key differences:

- **PCA**: Linear, fast, but limited expressiveness (50 dims)
- **scVI**: Nonlinear VAE, learns batch correction, requires training per dataset (10-30 dims)
- **scModelForge**: Nonlinear transformer, pretrained on large corpus, transfers across datasets (256-512 dims)

scModelForge embeddings are typically higher-dimensional than scVI but capture richer structure from pretraining. You may want to reduce dimensionality for some downstream tools (see tips below).

## Building a gene vocabulary from existing data

When training a new model, you can create a gene vocabulary directly from your AnnData object:

```python
import anndata as ad
from scmodelforge.data import GeneVocab

adata = ad.read_h5ad("training_data.h5ad")

# Build vocab from var_names (should be gene symbols or Ensembl IDs)
vocab = GeneVocab.from_adata(adata)
print(f"Created vocabulary with {len(vocab)} genes")

# Save for reuse
vocab.save("gene_vocab.json")
```

The vocabulary is built from `adata.var_names`. Make sure these are standardized gene identifiers (HUGO symbols or Ensembl IDs). If you have mixed annotations, standardize them first:

```python
import scanpy as sc

# If var_names are Ensembl IDs, optionally map to symbols
if adata.var_names[0].startswith("ENSG"):
    # Use scverse utilities or biomaRt to map IDs
    pass  # Implementation depends on your annotation source

# Ensure uniqueness
adata.var_names_make_unique()
```

## CELLxGENE Census integration

scModelForge integrates directly with CELLxGENE Census for access to millions of standardized cells:

```python
import cellxgene_census

# Query Census for specific cell types and tissues
with cellxgene_census.open_soma() as census:
    adata = cellxgene_census.get_anndata(
        census,
        organism="Homo sapiens",
        obs_value_filter="tissue_general == 'lung' and cell_type in ['alveolar epithelial cell', 'fibroblast']"
    )

# Save locally for training
adata.write("census_lung.h5ad")
```

Or configure Census loading directly in your training config:

```yaml
data:
  census:
    organism: Homo sapiens
    obs_value_filter: "tissue_general == 'brain'"
    var_value_filter: "feature_biotype == 'protein_coding'"
  gene_selection:
    method: most_expressed
    n_genes: 2000
```

The Census data uses standardized gene symbols and cell ontology terms, making it ideal for pretraining foundation models that transfer across datasets.

See the [Data Loading & Preprocessing](data_loading.md) tutorial for more Census examples.

## Batch integration workflow

Foundation model embeddings often provide excellent batch integration without dataset-specific training:

```python
import scanpy as sc
import anndata as ad
from scmodelforge.eval._utils import extract_embeddings

# Load multi-batch dataset
adata = ad.read_h5ad("multi_batch_data.h5ad")  # Has adata.obs["batch"]

# Extract embeddings from pretrained model
adata.obsm["X_scmodelforge"] = extract_embeddings(model, adata, tokenizer)

# Visualize batch mixing
sc.pp.neighbors(adata, use_rep="X_scmodelforge")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["batch", "cell_type"])
```

Assess batch integration quality with scIB metrics:

```python
from scmodelforge.eval import get_benchmark

# scModelForge includes scIB wrapper
benchmark = get_benchmark("embedding_quality",
                          batch_key="batch",
                          label_key="cell_type")
result = benchmark.run(embeddings, adata, "my_dataset")
print(result.metrics)
# {'nmi': 0.82, 'ari': 0.76, 'asw_label': 0.68, 'asw_batch': 0.15, 'overall': 0.71}
```

Low `asw_batch` (batch silhouette) and high `asw_label` (cell type silhouette) indicate good batch mixing with preserved biology.

For dataset-specific batch correction, you can fine-tune the model with a batch integration objective (see [Fine-tuning for Cell Type Annotation](finetuning_cell_type.md)).

## Differential expression on foundation model clusters

Use scanpy's differential expression tools with scModelForge-derived clusters:

```python
import scanpy as sc

# Cluster using scModelForge embeddings
sc.pp.neighbors(adata, use_rep="X_scmodelforge")
sc.tl.leiden(adata, resolution=1.0)

# Find marker genes for each cluster
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)

# Or use a more sophisticated test
sc.tl.rank_genes_groups(adata, "leiden", method="t-test", use_raw=True)
sc.tl.filter_rank_genes_groups(adata, min_fold_change=1.5)
sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, groupby="leiden", cmap="viridis")
```

Markers identified this way often reveal biologically coherent clusters because the foundation model embeddings capture gene expression structure learned from millions of cells.

## Trajectory inference with foundation model embeddings

Use scModelForge embeddings with scanpy's trajectory tools:

```python
import scanpy as sc

# Build neighbor graph from embeddings
sc.pp.neighbors(adata, use_rep="X_scmodelforge")

# Diffusion pseudotime (requires specifying a root cell)
sc.tl.diffmap(adata, n_comps=15)
root_idx = adata.obs.index[adata.obs["cell_type"] == "stem_cell"][0]
adata.uns["iroot"] = root_idx
sc.tl.dpt(adata)

# Visualize trajectory
sc.pl.umap(adata, color=["dpt_pseudotime", "cell_type"], cmap="viridis")

# Or use PAGA for trajectory structure
sc.tl.paga(adata, groups="cell_type")
sc.pl.paga(adata, color="cell_type")
```

For more advanced trajectory analysis, try scvelo or CellRank on top of scModelForge embeddings:

```python
import scvelo as scv

# scvelo works with any embedding in obsm
scv.pp.moments(adata, n_pcs=None, n_neighbors=30, use_rep="X_scmodelforge")
scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding_stream(adata, basis="umap")
```

## Compositional analysis with scCODA

Use scModelForge clusters for compositional analysis:

```python
import scanpy as sc

# Cluster with scModelForge embeddings
sc.pp.neighbors(adata, use_rep="X_scmodelforge")
sc.tl.leiden(adata, key_added="scmodelforge_clusters")

# Now use with sccoda (if installed)
# import sccoda
# sccoda.tl.compositional_analysis(adata, cluster_key="scmodelforge_clusters", ...)
```

## Spatial analysis with Squidpy

If you have spatial transcriptomics data, use scModelForge embeddings with Squidpy:

```python
import squidpy as sq

# Compute neighborhood graph using scModelForge embeddings
sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key="spatial")
sq.gr.nhood_enrichment(adata, cluster_key="leiden")

# Visualize spatial patterns
sq.pl.spatial_scatter(
    adata,
    color="leiden",
    shape=None,
    size=10
)
```

The foundation model embeddings capture cell state, while the spatial graph captures location - together they reveal spatial organization of cell types and states.

## Tips for successful integration

**Store embeddings with clear key names:**

Always use descriptive keys in `adata.obsm`:

```python
adata.obsm["X_scmodelforge"] = embeddings
adata.obsm["X_scmodelforge_finetuned"] = finetuned_embeddings
adata.obsm["X_pca"] = pca_embeddings
```

This prevents confusion when you have multiple representations.

**Handle high dimensionality:**

scModelForge embeddings are typically 256-512 dimensional, much richer than PCA (50 dims). Some scverse tools expect lower dimensions. You can reduce dimensionality if needed:

```python
# Option 1: Run PCA on top of scModelForge embeddings
sc.pp.neighbors(adata, use_rep="X_scmodelforge")
sc.tl.pca(adata, use_rep="X_scmodelforge", n_comps=50)

# Option 2: Use fewer neighbors in kNN graph
sc.pp.neighbors(adata, use_rep="X_scmodelforge", n_neighbors=15)  # Default is 30
```

In practice, most scanpy functions handle 256-512 dimensions without issues.

**Match gene vocabularies:**

Gene names in `adata.var_names` must match the vocabulary used during model training. If they don't match exactly:

```python
from scmodelforge.data import GeneVocab

# Load the model's vocab
vocab = GeneVocab.from_file("model_vocab.json")

# Check overlap
overlap = set(adata.var_names) & set(vocab.genes)
print(f"Overlap: {len(overlap)} / {len(adata.var_names)} genes")

# Filter to common genes
adata = adata[:, adata.var_names.isin(vocab.genes)].copy()
```

If you trained on Ensembl IDs but your new data uses gene symbols (or vice versa), you'll need to standardize first.

**Preserve raw counts for DE:**

When using scModelForge with scanpy DE tools, keep raw counts accessible:

```python
# Before normalization, store raw counts
adata.layers["counts"] = adata.X.copy()

# Then normalize for model input
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Later, for DE analysis:
sc.tl.rank_genes_groups(adata, "leiden", use_raw=False, layer="counts")
```

Or use `adata.raw`:

```python
adata.raw = adata.copy()  # Store raw before preprocessing
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.tl.rank_genes_groups(adata, "leiden", use_raw=True)
```

## Next steps

- [Model Assessment & Benchmarking](evaluation.md) - Quantify embedding quality with scIB and other metrics
- [Data Loading & Preprocessing](data_loading.md) - Advanced data loading including Census and cloud storage
- [Fine-tuning for Cell Type Annotation](finetuning_cell_type.md) - Adapt pretrained models to specific tasks
- [Perturbation Response Prediction](perturbation_prediction.md) - Use embeddings to predict perturbation effects

## Further reading

- [Scanpy documentation](https://scanpy.readthedocs.io/)
- [AnnData documentation](https://anndata.readthedocs.io/)
- [scverse ecosystem](https://scverse.org/)
- [scIB benchmarking paper](https://doi.org/10.1038/s41592-021-01336-8)
