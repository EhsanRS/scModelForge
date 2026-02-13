# HuggingFace Hub Integration

This tutorial covers how to save, load, share, and download scModelForge models using the HuggingFace Hub. If you are familiar with GitHub but new to model sharing, this guide will help you understand how to distribute and reuse pretrained single-cell foundation models.

## Overview

The [HuggingFace Hub](https://huggingface.co/models) is a model repository where researchers share pretrained models with the community. Think of it as GitHub for machine learning models. scModelForge integrates with the Hub so you can:

- **Push**: Share your trained models with the research community
- **Pull**: Download and use community-contributed models
- **Reproduce**: Load exact model configurations and weights for reproducible research
- **Fine-tune**: Use pretrained models as starting points for downstream tasks

When you push a model to the Hub, anyone can load it with a single line of code. This enables easy model sharing and accelerates single-cell research.

## Installation

To use Hub integration, install scModelForge with the optional `hub` dependency:

```bash
pip install "scmodelforge[hub]"
```

This installs two additional packages:

- **huggingface-hub**: API client for uploading and downloading models
- **safetensors**: Efficient and secure model serialization format

If you only need to load models (not push them), these dependencies are optional. The `load_pretrained` functions will work with models saved in PyTorch's native `.pt` format.

## Model Format: Lightning Checkpoints vs Hub Format

scModelForge uses PyTorch Lightning for training, which saves checkpoints with the `.ckpt` extension. These checkpoints contain:

- Model weights
- Optimizer state
- Learning rate scheduler state
- Training step counter
- Random number generator state

This comprehensive state allows you to resume training from any checkpoint. However, for sharing pretrained models, you typically only need the model weights and configuration.

**Hub format** is a lighter distribution format optimized for inference and fine-tuning:

- `config.json`: Model architecture configuration
- `model.safetensors` or `model.pt`: Model weights only
- `gene_vocab.json`: Gene vocabulary (essential for single-cell models)
- `README.md`: Auto-generated model card with metadata

Hub format is 50-80% smaller than Lightning checkpoints and loads faster.

## Exporting a Lightning Checkpoint to Hub Format

After training a model with `scmodelforge train`, you will have Lightning checkpoints in your output directory. To prepare one for sharing, export it to Hub format using the CLI:

```bash
scmodelforge export \
  --checkpoint checkpoints/best.ckpt \
  --output-dir ./my_model
```

This command:

1. Loads the Lightning checkpoint
2. Extracts the model weights (discarding optimizer and training state)
3. Saves the model configuration to `config.json`
4. Saves weights to `model.safetensors` (or `model.pt` if safetensors is unavailable)
5. Saves the gene vocabulary to `gene_vocab.json`
6. Generates a basic model card in `README.md`

After running this command, the `./my_model/` directory will contain:

```
my_model/
├── config.json
├── model.safetensors
├── gene_vocab.json
└── README.md
```

You can inspect these files to verify the export:

```bash
ls -lh ./my_model/
cat ./my_model/config.json
```

The `config.json` file contains human-readable model hyperparameters like layer dimensions, attention heads, and vocabulary size.

## Saving Models with the Python API

For programmatic workflows, use the `save_pretrained` function directly:

```python
from scmodelforge.models.hub import save_pretrained
from scmodelforge.config import load_config
from scmodelforge.data import GeneVocab

# Load your trained model, config, and vocabulary
model = ...  # your trained model instance
config = load_config("./configs/my_config.yaml")
gene_vocab = GeneVocab.from_adata(adata)  # Or GeneVocab.from_file("gene_vocab.json")

# Save to Hub format
save_pretrained(
    model,
    "./my_model",
    model_config=config.model,
    tokenizer_config=config.tokenizer,
    gene_vocab=gene_vocab,
)
```

**Parameters:**

- `model`: The PyTorch model instance (any scModelForge architecture)
- `save_directory`: Path where Hub format files will be written
- `model_config`: The ModelConfig used during training
- `tokenizer_config`: Optional TokenizerConfig (saved alongside model config)
- `gene_vocab`: Optional GeneVocab instance used during training
- `safe_serialization`: Whether to use safetensors format (default: True)

The function will create the directory if it does not exist and overwrite existing files.

### Customizing the Model Card

The auto-generated `README.md` includes basic model metadata, but you should customize it before sharing:

```python
# After saving
with open("./my_model/README.md", "a") as f:
    f.write("""

## Training Data

This model was pretrained on 1.2 million human PBMCs from the CELLxGENE Census,
spanning 15 tissue types and 40 cell types.

## Intended Use

This model is intended for:
- Cell type annotation
- Batch integration
- Gene expression prediction

## Citation

If you use this model, please cite:

[Your paper citation here]
""")
```

A well-documented model card helps users understand when and how to use your model.

## Loading Local Models

Once you have a model in Hub format (either from `export` or `save_pretrained`), you can load it for inference or fine-tuning.

### Loading Model Weights Only

```python
from scmodelforge.models.hub import load_pretrained

model = load_pretrained("./my_model")
print(f"Model type: {model.__class__.__name__}")
```

This loads the model architecture and weights, ready for `model.forward()` or `model.encode()`.

### Loading Model with Vocabulary

For single-cell models, you almost always need the gene vocabulary alongside the model. Use `load_pretrained_with_vocab` to load both:

```python
from scmodelforge.models.hub import load_pretrained_with_vocab

model, gene_vocab = load_pretrained_with_vocab("./my_model")

print(f"Model: {model.__class__.__name__}")
print(f"Vocabulary size: {len(gene_vocab)} genes")
print(f"Example genes: {gene_vocab.genes[:5]}")
```

This is the recommended loading method because it ensures the gene vocabulary matches the model's embedding layer.

### Using the Loaded Model

Once loaded, you can use the model for encoding, prediction, or fine-tuning:

```python
import torch
from scmodelforge.tokenizers import get_tokenizer

# Prepare your data
tokenizer = get_tokenizer("rank_value", gene_vocab=gene_vocab)
cell_data = ...  # your CellData instance
tokenized = tokenizer.tokenize(cell_data)

# Encode with the model
model.eval()
with torch.no_grad():
    output = model.encode(
        gene_ids=tokenized.gene_ids.unsqueeze(0),
        values=tokenized.values.unsqueeze(0)
    )
    embedding = output.pooled_output

print(f"Cell embedding shape: {embedding.shape}")
```

## Pushing to HuggingFace Hub

To share your model with the community, push it to the HuggingFace Hub.

### Step 1: Create a HuggingFace Account

If you do not already have one:

1. Go to [huggingface.co](https://huggingface.co/)
2. Click "Sign up" and create an account
3. Go to Settings > Access Tokens: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Create a new token with "Write" permissions

### Step 2: Login with the CLI

Authenticate your local machine with the Hub:

```bash
huggingface-cli login
```

When prompted, paste your access token. This stores your credentials locally for future uploads.

### Step 3: Push Your Model

#### Using the CLI

```bash
scmodelforge push \
  --model-dir ./my_model \
  --repo-id username/my-scmodel
```

Replace `username` with your HuggingFace username and `my-scmodel` with your chosen model name. The repo ID must follow the format `username/model-name`.

This command:

1. Creates a new model repository on the Hub (if it does not exist)
2. Uploads all files from `./my_model/` to the repository
3. Sets appropriate metadata for model discovery

#### Using the Python API

```python
from scmodelforge.models.hub import push_to_hub

push_to_hub(
    model_directory="./my_model",
    repo_id="username/my-scmodel",
    commit_message="Initial upload of pretrained model",  # optional
    private=False  # set to True for private models
)
```

**Parameters:**

- `model_directory`: Path to local Hub-format model directory
- `repo_id`: HuggingFace repository ID (`username/model-name`)
- `commit_message`: Optional description of this upload
- `private`: Whether the model should be private (default: False)

After pushing, your model will be available at `https://huggingface.co/username/my-scmodel`.

### Updating a Model

To upload a new version of your model (e.g., after additional training), simply run `push_to_hub` again with the same `repo_id`. The Hub maintains version history automatically.

## Loading from HuggingFace Hub

Once a model is on the Hub, anyone can load it with a single line.

### Automatic Hub Detection

The `load_pretrained` and `load_pretrained_with_vocab` functions automatically detect Hub repository IDs:

```python
from scmodelforge.models.hub import load_pretrained_with_vocab

# Load a community model
model, gene_vocab = load_pretrained_with_vocab("username/my-scmodel")
```

The functions distinguish between:

- **Local paths**: Start with `/` or `.` or exist as directories (e.g., `./my_model`, `/home/user/models/model1`)
- **Hub repo IDs**: Format `username/model-name` with exactly one `/` (e.g., `scmodelforge/geneformer-pbmc`)

If you specify `username/my-scmodel` and there is no local directory with that name, the function will automatically download from the Hub.

### Using Hub Models for Downstream Tasks

Hub models are immediately usable for:

**Cell embedding:**

```python
from scmodelforge.eval._utils import extract_embeddings
from scmodelforge.tokenizers import get_tokenizer

model, gene_vocab = load_pretrained_with_vocab("username/my-scmodel")
tokenizer = get_tokenizer("rank_value", gene_vocab=gene_vocab)

# Extract embeddings for all cells in your AnnData
embeddings = extract_embeddings(model, adata, tokenizer)
```

**Fine-tuning** (covered in the next section):

```python
from scmodelforge.finetuning import load_pretrained_backbone

backbone = load_pretrained_backbone("username/my-scmodel")
# Continue with fine-tuning pipeline
```

**Benchmarking:**

```python
from scmodelforge.eval import EvalHarness

harness = EvalHarness.from_config(eval_config)
results = harness.run(model, eval_datasets)
```

### Browsing Available Models

To discover models on the Hub, visit:

- [https://huggingface.co/models?library=scmodelforge](https://huggingface.co/models?library=scmodelforge)

You can filter by task (cell type annotation, batch integration, etc.) and sort by downloads or likes.

## Using Hub Models for Fine-Tuning

Pretrained models from the Hub can serve as backbones for task-specific fine-tuning. The fine-tuning module has built-in support for Hub repo IDs.

### Loading a Pretrained Backbone

```python
from scmodelforge.finetuning import load_pretrained_backbone

# Load from Hub
backbone = load_pretrained_backbone("username/my-scmodel")
print(f"Loaded backbone: {backbone.__class__.__name__}")
```

This function accepts:

- Local Hub-format directories (`./my_model`)
- Local Lightning checkpoints (`./checkpoints/best.ckpt`)
- Hub repository IDs (`username/my-scmodel`)

The function automatically detects the input type and loads appropriately.

### Fine-Tuning with a Hub Model

Here is a complete fine-tuning workflow using a Hub model:

```python
from scmodelforge.finetuning import FineTuneModel, FineTunePipeline
from scmodelforge.finetuning.heads import build_task_head
from scmodelforge.config import load_config

# 1. Load pretrained backbone from Hub
backbone = load_pretrained_backbone("username/my-scmodel")

# 2. Create a task-specific head
task_head = build_task_head(
    head_type="classification",
    input_dim=backbone.config.d_model,
    output_dim=20,  # 20 cell types
    hidden_dim=256,
    dropout=0.1
)

# 3. Combine into fine-tuning model
finetune_model = FineTuneModel(
    backbone=backbone,
    task_head=task_head
)

# 4. Run fine-tuning pipeline
config = load_config("./configs/finetune_celltype.yaml")
pipeline = FineTunePipeline(config)
pipeline.run(finetune_model)
```

Alternatively, use the CLI:

```bash
scmodelforge finetune \
  --config ./configs/finetune_celltype.yaml \
  --checkpoint username/my-scmodel
```

The CLI automatically handles Hub model loading when you provide a repo ID to the `--checkpoint` argument.

### Benefits of Hub-Based Fine-Tuning

- **Reproducibility**: Exact model versions are preserved on the Hub
- **Collaboration**: Teams can share pretrained backbones without copying large files
- **Versioning**: The Hub tracks model updates automatically
- **Efficiency**: Download once and cache locally for future runs

## Model Cards and Documentation

Good model documentation is essential for reproducible research. When sharing a model on the Hub, your `README.md` should include:

### Essential Information

1. **Model description**: What architecture is this? (TransformerEncoder, Autoregressive, etc.)
2. **Training data**: What dataset(s) were used? How many cells? Which species?
3. **Preprocessing**: What normalization and filtering was applied?
4. **Intended use**: What tasks is this model suitable for?
5. **Limitations**: What are known failure modes or inappropriate uses?
6. **Citation**: How should users cite your work?

### Example Model Card

```markdown
# Geneformer-PBMC

## Model Description

A transformer-based single-cell foundation model pretrained on 1.2M human PBMCs
from the CELLxGENE Census. Uses a 12-layer transformer encoder with
rank-value tokenization.

## Training Details

- **Architecture**: TransformerEncoder (12 layers, 512 hidden dim, 8 attention heads)
- **Tokenization**: Rank-value (top 2048 genes per cell)
- **Pretraining objective**: Masked gene prediction (15% masking rate)
- **Training data**: 1.2M cells from CELLxGENE Census (human PBMCs)
- **Gene vocabulary**: 19,264 protein-coding genes

## Intended Use

This model can be used for:

- Cell type annotation (via linear probe or fine-tuning)
- Batch effect correction
- Gene expression imputation
- Cell state analysis

## Limitations

- Trained only on human data (not suitable for other species without fine-tuning)
- Limited to protein-coding genes
- Performance may degrade on rare cell types not well-represented in training data

## Citation

If you use this model, please cite:

[Your paper citation]
```

Edit the auto-generated `README.md` before pushing to include this information.

## Advanced Topics

### Private Models

To share models within a team without making them public:

```python
from scmodelforge.models.hub import push_to_hub

push_to_hub(
    model_directory="./my_model",
    repo_id="username/private-model",
    private=True
)
```

Private models are only accessible to you and users you explicitly grant access to via the HuggingFace Hub interface.

### Model Versioning

The Hub automatically tracks versions using git under the hood. Each `push_to_hub` call creates a new commit. To reference a specific version:

```python
# Load a specific commit
model, vocab = load_pretrained_with_vocab(
    "username/my-scmodel",
    revision="a1b2c3d4"  # commit hash
)
```

This ensures exact reproducibility even as models are updated.

### Safetensors vs PyTorch Format

By default, scModelForge uses the **safetensors** format for model weights because it:

- Loads faster (no pickle deserialization)
- Is more secure (no arbitrary code execution)
- Has built-in shape validation

If safetensors is not installed, the exporter falls back to PyTorch's native `.pt` format. Both formats work with `load_pretrained`, but safetensors is recommended for production use.

### Downloading Without Loading

To download a model to a local cache without loading it:

```python
from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="username/my-scmodel",
    cache_dir="./model_cache"
)
print(f"Model downloaded to: {local_path}")
```

This is useful for:

- Batch downloading many models
- Inspecting model files before loading
- Archiving models offline

## Troubleshooting

### "Repository not found" error

If you see this error when pushing:

```
Repository 'username/my-scmodel' not found
```

Ensure:

1. You are logged in (`huggingface-cli login`)
2. The username in the repo ID matches your HuggingFace username
3. You have not exceeded your quota for model repositories

### "Permission denied" error

When pushing, this means:

- You are not logged in, or
- Your access token lacks "Write" permissions, or
- You are trying to push to someone else's repository

### Gene vocabulary mismatch

If you load a model and get errors during inference:

```
RuntimeError: Index out of range in gene embedding layer
```

This usually means you are using a different gene vocabulary than the model was trained with. Always use `load_pretrained_with_vocab` to ensure the vocabulary matches the model.

### Large model upload times

Uploading large models (>1GB) can take time depending on your internet connection. The `push_to_hub` function shows a progress bar. For very large models, consider:

1. Using a wired connection (not WiFi)
2. Splitting uploads across multiple sessions (the Hub resumes partial uploads)
3. Compressing models with safetensors (typically 10-20% smaller than PyTorch format)

## What's Next

Now that you understand how to share and load models from the Hub, explore these topics:

- **[Fine-Tuning](finetuning_cell_type.md)**: Adapt pretrained models to specific tasks
- **[Benchmarking](evaluation.md)**: Test models on standardized tasks
- **[Model Training](pretraining.md)**: Train your own foundation models

For questions or issues with Hub integration, visit the [scModelForge GitHub discussions](https://github.com/EhsanRS/scModelForge/discussions).
