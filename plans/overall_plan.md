
||<p>**scModelForge**</p><p>A Pretraining Toolkit for Single-Cell Foundation Models</p><p>*Democratising the training of cell foundation models*</p>|
| :- | :- |

Project Proposal & Technical Design

Version 0.1  |  February 2026

Status: Draft for Community Feedback

|**License**|Apache 2.0 (planned)|
| :- | :- |
|**Stack**|Python, PyTorch, AnnData/scverse|
|**Target Users**|ML engineers entering bio, comp-bio researchers, biotech teams|

*cellforge — Project Proposal*
# <a name="_toc221706536"></a>**Contents**
[Contents	2](#_toc221706536)

[Executive Summary	3](#_toc221706537)

[Problem Statement	3](#_toc221706538)

[Who is this for?	3](#_toc221706539)

[What is broken today?	3](#_toc221706540)

[Design Principles	4](#_toc221706541)

[Architecture	5](#_toc221706542)

[Module 1: scModelForge.data	5](#_toc221706543)

[Module 2: scModelForge.tokenizers	6](#_toc221706544)

[Module 3: scModelForge.models	6](#_toc221706545)

[Module 4: scModelForge.training	6](#_toc221706546)

[Module 5: scModelForge.eval	7](#_toc221706547)

[Example User Experience	8](#_toc221706548)

[Phased Roadmap	10](#_toc221706549)

[Phase 1: Foundation (Months 1–3)	10](#_toc221706550)

[Phase 2: Breadth (Months 4–6)	10](#_toc221706551)

[Phase 3: Community & Scale (Months 7–12)	11](#_toc221706552)

[Landscape & Positioning	12](#_toc221706553)

[Risks & Mitigations	12](#_toc221706554)

[Immediate Next Steps	14](#_toc221706555)




# <a name="_toc221706537"></a>**Executive Summary**
**scModelForge** is an open-source Python toolkit that makes training and fine-tuning single-cell foundation models accessible to researchers who are not infrastructure specialists. It provides the missing layer between raw biological data (AnnData/CELLxGENE) and production-quality model training — standardised tokenisation strategies, GPU-accelerated data streaming, reference model architectures, and integrated evaluation — in a single, config-driven package built natively on the scverse ecosystem.

In the LLM world, frameworks like Megatron-LM, DeepSpeed, and TorchTitan separate training infrastructure from model innovation, enabling rapid iteration. The single-cell foundation model space currently lacks this separation of concerns. Each group — scGPT, Geneformer, TranscriptFormer, CellFM, scPRINT — builds bespoke data pipelines, tokenisation logic, and training loops from scratch. This duplication slows the field, raises the barrier to entry, and makes fair comparison between approaches difficult.

scModelForge aims to change this. *Train your own single-cell foundation model in 200 lines of config, not 2,000 lines of bespoke code.*
# <a name="_toc221706538"></a>**Problem Statement**
## <a name="_toc221706539"></a>**Who is this for?**
We identify three underserved personas in the current landscape:

- **ML engineers entering biology:** They understand PyTorch and distributed training but face a steep learning curve around scRNA-seq data structures, gene vocabularies, normalisation choices, and the AnnData ecosystem.
- **Computational biology researchers:** They know the biology and use scanpy/scvi-tools daily, but training a transformer from scratch on 50M+ cells requires infrastructure knowledge (distributed training, mixed precision, data sharding) they may not have.
- **Small biotech teams with proprietary data:** They have valuable Perturb-seq or drug-screen datasets and want to fine-tune existing models or train domain-specific ones on their own compute, without rebuilding infrastructure from scratch.
## <a name="_toc221706540"></a>**What is broken today?**
Several concrete pain points exist in the current landscape that scModelForge aims to address:

- **No standard tokenisation interface.** Geneformer rank-orders genes by expression. scGPT bins expression values into discrete tokens. TranscriptFormer uses continuous expression as attention bias with ESM-2 gene embeddings. scFoundation predicts raw values via masking. Each approach is hardcoded into its respective codebase with no shared abstraction.
- **Bespoke data pipelines for every model.** Every group writes its own AnnData-to-batches pipeline. There is no equivalent of HuggingFace Datasets or Mosaic StreamingDataset for cell-by-gene matrices that handles sharding, streaming from object storage, on-the-fly normalisation, and multi-worker loading at the scale of 100M+ cells.
- **Evaluation is fragmented.** pertpy, scIB, PerturBench, cz-benchmarks, and the Arc Virtual Cell Challenge all measure different things with different metrics. Comparing models across papers is unreliable. No framework integrates evaluation into the training loop.
- **Reproducibility crisis.** Training details are often scattered across paper supplements, GitHub issues, and unreleased code. Reproducing published results requires significant reverse-engineering effort.
# <a name="_toc221706541"></a>**Design Principles**
scModelForge is guided by the following principles, each learned from what has worked (and failed) in both the LLM and single-cell communities:

- **scverse-native.** AnnData in, AnnData out. No custom data formats. If it breaks interoperability with scanpy, we do not ship it.
- **PyTorch-first.** The scverse community is PyTorch-native (scvi-tools, scGPT, Geneformer). We build on PyTorch and PyTorch Lightning/Fabric, not JAX. This is a pragmatic adoption decision.
- **Separation of concerns.** Data loading, tokenisation, model architecture, training loop, and evaluation are independent, swappable components. Changing your tokenisation strategy should not require rewriting your model.
- **Config-driven, code-optional.** Common workflows are expressible as YAML/TOML configs. Custom components are standard Python classes that implement documented interfaces.
- **Batteries included, not mandatory.** We ship reference implementations of major architectures (Geneformer-style, scGPT-style, masked autoencoder) and standard tokenisers, but every component is replaceable.
- **Evaluation is a first-class citizen.** Built-in benchmark harnesses run during training via callbacks, not as a separate post-hoc step.


# <a name="_toc221706542"></a>**Architecture**
scModelForge is organised into five core modules, each corresponding to a separable concern in the model training pipeline. The diagram below shows the data flow from raw biological data to a trained model with evaluation metrics:

|<p>**scModelForge Architecture Overview**</p><p>AnnData / CELLxGENE / H5AD files</p><p>↓</p><p>┌───────────────────────────────────┐</p><p>│  scModelForge.data                   │   Streaming DataLoader, sharding, caching</p><p>└───────────────────────────────────┘</p><p>↓</p><p>┌───────────────────────────────────┐</p><p>│  scModelForge.tokenizers             │   Pluggable: rank, binned, continuous, ESM-2</p><p>└───────────────────────────────────┘</p><p>↓</p><p>┌───────────────────────────────────┐</p><p>│  scModelForge.models                 │   Reference architectures + custom</p><p>└───────────────────────────────────┘</p><p>↓</p><p>┌───────────────────────────────────┐</p><p>│  scModelForge.training               │   Config-driven loop, DDP/FSDP, callbacks</p><p>└───────────────────────────────────┘</p><p>↓</p><p>┌───────────────────────────────────┐</p><p>│  scModelForge.eval                   │   Inline benchmarks: scIB, perturbation, GRN</p><p>└───────────────────────────────────┘</p>|
| :-: |

## <a name="_toc221706543"></a>**Module 1: scModelForge.data**
Handles the pipeline from raw biological data to GPU-ready batches. This is the highest-value component because every group currently builds this from scratch.

- **AnnData streaming:** Read directly from .h5ad files (local or remote/S3) with lazy loading. Support for CELLxGENE Census API as a first-class data source. Automatic sharding across workers and nodes.
- **Preprocessing pipeline:** Configurable normalisation (library size, log1p, scran), highly variable gene selection, and batch-aware sampling. Runs on-the-fly or as a cached preprocessing step.
- **Gene vocabulary management:** Unified gene-to-index mapping across datasets with different gene panels. Handles ENSEMBL/symbol conversion, species mapping, and vocabulary intersection logic.
## <a name="_toc221706544"></a>**Module 2: scModelForge.tokenizers**
The most domain-specific component with no LLM analog. Provides a standard interface for converting a cell’s gene expression vector into model input.

|**Strategy**|**Description**|**Used by**|
| :- | :- | :- |
|RankValue|Rank genes by expression, use rank as position encoding|Geneformer|
|BinnedExpression|Discretise expression into N bins, treat as token IDs|scGPT|
|ContinuousProjection|Project raw values via learned linear layer or attention bias|TranscriptFormer, scFoundation|
|GeneEmbedding|Use pretrained gene embeddings (ESM-2, GO, LLM-derived)|TranscriptFormer, scGenePT, UCE|

Each tokenizer implements a common interface: it takes a sparse expression vector and gene list, and returns input\_ids, attention\_mask, and any auxiliary tensors. This abstraction allows direct comparison of tokenisation strategies with identical model architectures.
## <a name="_toc221706545"></a>**Module 3: scModelForge.models**
Reference implementations of the major architecture families found in the literature. These are not intended to replace the original codebases but to provide clean, standardised, and interoperable implementations that work with the scModelForge data and tokenizer modules.

- **TransformerEncoder:** BERT-style masked language model (Geneformer pattern). Configurable depth, heads, hidden dimension.
- **AutoregressiveTransformer:** GPT-style next-token prediction (scGPT pattern). Supports both gene identity and expression value prediction.
- **MaskedAutoencoder:** Predicts raw expression values from masked inputs (scFoundation pattern).
- **Custom:** Any nn.Module that implements the ScModelForgeModel protocol (forward, encode, generate methods).
## <a name="_toc221706546"></a>**Module 4: scModelForge.training**
A config-driven training loop built on PyTorch Lightning Fabric that handles distributed training, mixed precision, gradient accumulation, checkpointing, and logging. Key design decisions:

- Supports DDP and FSDP out of the box. No Megatron dependency — models in this space are not yet large enough to require tensor/pipeline parallelism.
- WandB and TensorBoard logging built in. Tracks both training metrics and biological evaluation metrics on the same dashboard.
- Checkpoint format is compatible with HuggingFace Hub for model sharing.
- Fine-tuning recipes for common scenarios: cell type annotation, perturbation prediction, batch integration.
## <a name="_toc221706547"></a>**Module 5: scModelForge.eval**
Integrated evaluation that runs as callbacks during training and as standalone benchmarking scripts. Wraps existing community benchmarks behind a unified API:

- **Cell embedding quality:** scIB metrics (NMI, ASW, ARI for biology conservation; ASW\_batch, graph connectivity for batch correction).
- **Perturbation prediction:** Pearson correlation, MSE on differentially expressed genes, energy distance (from PerturBench). Includes the critical linear baseline comparison.
- **Gene network inference:** AUROC/AUPRC against known regulatory networks (ENCODE, ChIP-Atlas).
- **Standardised held-out datasets:** Ships with preprocessed evaluation splits (Tabula Sapiens, Norman perturbation data, immune atlas) for consistent comparison.


# <a name="_toc221706548"></a>**Example User Experience**
The following pseudocode illustrates how a researcher would use scModelForge to train a Geneformer-style model on a custom dataset. This is the target experience — not the current state of the codebase.

\# scModelForge\_config.yaml

data:

`  `source: cellxgene\_census

`  `organism: homo\_sapiens

`  `filters:

`    `tissue: [lung, heart, liver]

`    `is\_primary\_data: true

`  `preprocessing:

`    `normalize: library\_size

`    `hvg\_selection: 2000

tokenizer:

`  `strategy: rank\_value      # geneformer-style

`  `max\_genes: 2048

`  `gene\_vocab: human\_protein\_coding

model:

`  `architecture: transformer\_encoder

`  `hidden\_dim: 512

`  `num\_layers: 12

`  `num\_heads: 8

`  `pretraining\_task: masked\_gene\_prediction

`  `mask\_ratio: 0.15

training:

`  `batch\_size: 64

`  `max\_epochs: 10

`  `precision: bf16-mixed

`  `strategy: ddp

`  `num\_gpus: 4

`  `lr: 1e-4

eval:

`  `every\_n\_epochs: 2

`  `benchmarks:

`    `- cell\_type\_annotation:

`        `dataset: tabula\_sapiens

`    `- batch\_integration:

`        `dataset: immune\_atlas

\# Launch training

$ scModelForge train --config scModelForge\_config.yaml

For fine-tuning on proprietary Perturb-seq data, the experience would look similar but with a local .h5ad file as the data source and a perturbation prediction eval benchmark:

\# Fine-tune on your own Perturb-seq data

$ scModelForge finetune \

`    `--base-model scModelForge://geneformer-v2 \

`    `--data ./my\_perturbseq.h5ad \

`    `--task perturbation\_prediction \

`    `--eval perturbbench


# <a name="_toc221706549"></a>**Phased Roadmap**
We propose a three-phase development plan, each phase producing a usable artifact and an opportunity for community feedback.
## <a name="_toc221706550"></a>**Phase 1: Foundation (Months 1–3)**
**Goal:** Prove the core abstraction works. Deliver a minimal but functional toolkit that can train a Geneformer-style model from a CELLxGENE data source with built-in evaluation.

|**Deliverable**|**Details**|**Success Criterion**|
| :- | :- | :- |
|scModelForge.data v0.1|AnnData streaming from local .h5ad files. Basic sharding. Gene vocab builder.|Load and iterate 10M cells at >50k cells/sec on single GPU|
|scModelForge.tokenizers v0.1|RankValue tokenizer (Geneformer-style). Tokenizer base class with documented interface.|Produce identical tokenisation to official Geneformer on test data|
|scModelForge.models v0.1|TransformerEncoder with masked gene prediction. Configurable via YAML.|Reproduce Geneformer cell-type annotation results within 2% on Tabula Sapiens|
|scModelForge.training v0.1|Lightning Fabric loop with DDP, bf16, WandB logging, checkpointing.|Train on 4 GPUs with linear scaling efficiency >85%|
|scModelForge.eval v0.1|Cell-type annotation eval via scIB metrics. Linear baseline for perturbation.|Metrics match published scIB results on reference datasets|

## <a name="_toc221706551"></a>**Phase 2: Breadth (Months 4–6)**
**Goal:** Expand tokenisation strategies and model architectures. Add CELLxGENE Census as a remote data source. Enable fine-tuning workflows.

- Add BinnedExpression tokenizer (scGPT-style) and ContinuousProjection tokenizer (TranscriptFormer-style)
- Add AutoregressiveTransformer and MaskedAutoencoder model architectures
- CELLxGENE Census API integration for remote data streaming
- Fine-tuning API with LoRA/adapter support for efficient downstream adaptation
- Perturbation prediction evaluation via PerturBench integration
- HuggingFace Hub integration for model sharing (push/pull checkpoints)
- Documentation site with tutorials: train-from-scratch, fine-tune, evaluate, add-custom-tokenizer

## <a name="_toc221706552"></a>**Phase 3: Community & Scale (Months 7–12)**
**Goal:** Seek adoption, integrate community feedback, and handle larger-scale training. This is where we engage seriously with the scverse community and CZI/Arc ecosystem.

- FSDP support for models >1B parameters
- GeneEmbedding tokenizer (ESM-2 and LLM-derived gene representations)
- Multi-species support (human + mouse unified gene vocabulary)
- Integration with pertpy for perturbation-specific preprocessing and metadata
- cz-benchmarks compatibility for direct comparison on CZI Virtual Cells Platform leaderboards
- Community contribution guide and plugin system for third-party tokenisers/models
- Explore scverse core package candidacy


# <a name="_toc221706553"></a>**Landscape & Positioning**
scModelForge occupies a specific niche that is not currently well-served by existing tools. The following table clarifies the relationship to adjacent projects:

|**Project**|**What it does**|**Relationship to scModelForge**|
| :- | :- | :- |
|BioNeMo|NVIDIA’s distributed training framework for drug discovery (proteins, molecules, DNA)|Focuses on protein/molecule/drug modalities. Has Geneformer recipe but not single-cell-native. scModelForge is complementary and lighter-weight.|
|scvi-tools|PyTorch library for probabilistic single-cell models (scVI, scANVI, totalVI)|Focused on VAE-based analysis models, not pretraining transformers at scale. scModelForge may wrap scVI as an optional encoder/decoder component.|
|BioLLM|Benchmarking framework for single-cell foundation models|Evaluation only. scModelForge.eval aims to integrate BioLLM’s benchmarks as one evaluation backend.|
|Helical|Unified inference API across scGPT, Geneformer, UCE|Inference-focused. scModelForge is training-focused. Could share a model registry.|
|CZI VCP|CZI’s Virtual Cells Platform with models, data, and benchmarks|Platform/portal. scModelForge is a local toolkit. Goal is to produce models deployable on VCP and evaluated with cz-benchmarks.|
|pertpy|Perturbation analysis framework (Theis lab / scverse)|Analysis-focused post-training. scModelForge.data may use pertpy for preprocessing. scModelForge.eval wraps pertpy metrics.|

**Key positioning statement:** scModelForge is the training layer. It sits between the data/analysis ecosystem (scverse) and the deployment/evaluation ecosystem (CZI VCP, HuggingFace). It is not a replacement for any of these tools — it is the glue that connects raw data to trained models using standardised, reproducible, and accessible patterns.
# <a name="_toc221706554"></a>**Risks & Mitigations**
We should be honest about the risks to this project and plan accordingly:

|**Risk**|**Severity**|**Mitigation**|
| :- | :- | :- |
|CZI/NVIDIA build the same thing faster|High. They have significant engineering resources and NVIDIA backing.|Ship Phase 1 fast. Differentiate through community-first design and scverse integration depth. If they build something better, contribute to it instead.|
|Foundation models prove unnecessary|Medium. If CellFlow-style smaller models keep winning, demand for pretraining toolkits decreases.|Design scModelForge to also support fine-tuning workflows and smaller models. The data pipeline and evaluation modules are valuable regardless.|
|scverse community rejects external tool|Medium. Strong culture of building from within.|Engage early (Phase 1). Use AnnData natively. Follow scverse development best practices from day one. Aim for ecosystem package status.|
|Maintenance burden as side project|High. Academic open-source often suffers from abandonment.|Scope tightly. Phase 1 should be useful standalone. Modular design means individual modules can be maintained independently.|


# <a name="_toc221706555"></a>**Immediate Next Steps**
To move from proposal to execution, we suggest the following concrete steps:

- **Week 1–2: Repository scaffolding.** Set up the monorepo structure, CI/CD (GitHub Actions), pre-commit hooks, pytest infrastructure, and documentation skeleton (Sphinx or mkdocs). Publish to PyPI as a placeholder package.
- **Week 3–4: scModelForge.tokenizers prototype.** Implement the BaseTokenizer interface and the RankValue tokenizer. Validate against official Geneformer tokenisation on a reference dataset. This is the most novel component and the best proof-of-concept.
- **Week 5–6: scModelForge.data v0.1.** Build the AnnData streaming DataLoader with basic sharding. Benchmark throughput against naive approaches.
- **Week 7–8: First end-to-end training run.** Train a small Geneformer-style model on a subset of CELLxGENE data using the toolkit. Evaluate with scIB. Write up results as first blog post / README showcase.
- **Week 9–10: Community engagement.** Share on scverse Discourse, Bioinformatics Twitter/Bluesky, and relevant GitHub discussions. Solicit feedback on the abstraction design before committing to Phase 2.

||*The measure of success for scModelForge is not whether it becomes the dominant tool in the field. It is whether a computational biology PhD student can go from having an interesting hypothesis about cell state representations to having a trained, evaluated model in days rather than months.*|
| :- | :- |

Page 2
