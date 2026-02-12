"""Lightning DataModule for tokenized cell data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scmodelforge.config.schema import DataConfig, GeneSelectionConfig, SamplingConfig, TokenizerConfig
    from scmodelforge.data.gene_vocab import GeneVocab
    from scmodelforge.data.sampling import WeightedCellSampler
    from scmodelforge.tokenizers.base import BaseTokenizer, MaskedTokenizedCell, TokenizedCell
    from scmodelforge.tokenizers.masking import MaskingStrategy

logger = logging.getLogger(__name__)


class TokenizedCellDataset(Dataset):
    """Wraps a :class:`CellDataset` (or Subset) with tokenization and masking.

    Parameters
    ----------
    dataset
        Underlying dataset whose ``__getitem__`` returns a dict with
        ``expression``, ``gene_indices``, and ``metadata`` keys.
    tokenizer
        Tokenizer to convert raw cell data into model inputs.
    masking
        Optional masking strategy applied after tokenization.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: BaseTokenizer,
        masking: MaskingStrategy | None = None,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.masking = masking

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> TokenizedCell | MaskedTokenizedCell:
        """Tokenize and optionally mask a single cell.

        Returns
        -------
        TokenizedCell or MaskedTokenizedCell
        """
        sample = self.dataset[idx]
        cell = self.tokenizer.tokenize(
            expression=sample["expression"],
            gene_indices=sample["gene_indices"],
            metadata=sample.get("metadata"),
        )
        if self.masking is not None:
            cell = self.masking.apply(cell)
        return cell


class _StreamingTokenizedDataset(IterableDataset):
    """Wraps a :class:`StreamingCellDataset` with tokenization and masking.

    Each yielded cell dict is tokenized and optionally masked in the
    ``__iter__`` loop.
    """

    def __init__(
        self,
        streaming_dataset: IterableDataset,
        tokenizer: BaseTokenizer,
        masking: MaskingStrategy | None = None,
    ) -> None:
        super().__init__()
        self.streaming_dataset = streaming_dataset
        self.tokenizer = tokenizer
        self.masking = masking

    def __iter__(self) -> Iterator[TokenizedCell | MaskedTokenizedCell]:
        for sample in self.streaming_dataset:
            cell = self.tokenizer.tokenize(
                expression=sample["expression"],
                gene_indices=sample["gene_indices"],
                metadata=sample.get("metadata"),
            )
            if self.masking is not None:
                cell = self.masking.apply(cell)
            yield cell


class CellDataModule:
    """Lightning-style DataModule for single-cell pretraining.

    Loads data, builds vocab / tokenizer / masking, splits into train/val,
    and wraps each split with :class:`TokenizedCellDataset`.

    Parameters
    ----------
    data_config
        Data loading and preprocessing configuration.
    tokenizer_config
        Tokenizer strategy and masking configuration.
    training_batch_size
        Batch size for training.
    num_workers
        DataLoader worker count.
    val_split
        Fraction of data reserved for validation.
    seed
        Random seed for reproducible splits.
    adata
        Optional pre-loaded AnnData for testing (skips file loading).
    sampling_config
        Weighted sampling configuration. ``None`` uses default (random).
    gene_selection_config
        Batch-level gene selection configuration. ``None`` uses default (all).
    """

    def __init__(
        self,
        data_config: DataConfig,
        tokenizer_config: TokenizerConfig,
        training_batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.05,
        seed: int = 42,
        adata: object | None = None,
        sampling_config: SamplingConfig | None = None,
        gene_selection_config: GeneSelectionConfig | None = None,
    ) -> None:
        self._data_config = data_config
        self._tokenizer_config = tokenizer_config
        self._batch_size = training_batch_size
        self._num_workers = num_workers
        self._val_split = val_split
        self._seed = seed
        self._adata = adata
        self._sampling_config = sampling_config
        self._gene_selection_config = gene_selection_config

        self._gene_vocab: GeneVocab | None = None
        self._tokenizer: BaseTokenizer | None = None
        self._masking: MaskingStrategy | None = None
        self._train_dataset: Dataset | IterableDataset | None = None
        self._val_dataset: TokenizedCellDataset | None = None
        self._sampler: WeightedCellSampler | None = None
        self._train_collate_fn = None
        self._use_gene_selection = False
        self._streaming = False
        self._is_setup = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gene_vocab(self) -> GeneVocab:
        """Gene vocabulary (available after :meth:`setup`)."""
        if self._gene_vocab is None:
            msg = "Call setup() before accessing gene_vocab."
            raise RuntimeError(msg)
        return self._gene_vocab

    @property
    def tokenizer(self) -> BaseTokenizer:
        """Tokenizer instance (available after :meth:`setup`)."""
        if self._tokenizer is None:
            msg = "Call setup() before accessing tokenizer."
            raise RuntimeError(msg)
        return self._tokenizer

    @property
    def masking(self) -> MaskingStrategy | None:
        """Masking strategy (available after :meth:`setup`)."""
        return self._masking

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Load data, build tokenizer, split into train/val.

        Idempotent — calling multiple times is safe.
        """
        if self._is_setup:
            return

        from scmodelforge.data._utils import load_adata
        from scmodelforge.data.dataset import CellDataset
        from scmodelforge.data.gene_vocab import GeneVocab
        from scmodelforge.data.preprocessing import PreprocessingPipeline
        from scmodelforge.tokenizers.masking import MaskingStrategy
        from scmodelforge.tokenizers.registry import get_tokenizer

        # 1. Load AnnData
        adata = load_adata(self._data_config, adata=self._adata)

        # 2. Build GeneVocab
        self._gene_vocab = GeneVocab.from_adata(adata)

        # 3. Build preprocessing
        pp_cfg = self._data_config.preprocessing
        preprocessing = PreprocessingPipeline(
            normalize=pp_cfg.normalize,
            target_sum=pp_cfg.target_sum,
            log1p=pp_cfg.log1p,
        )

        # 4. Build CellDataset
        # Include obs_keys needed for weighted sampling label extraction
        obs_keys = None
        samp_cfg = self._sampling_config
        if samp_cfg is not None and samp_cfg.strategy == "weighted":
            obs_keys = [samp_cfg.label_key]
        full_dataset = CellDataset(adata, self._gene_vocab, preprocessing, obs_keys=obs_keys)

        # 5. Build tokenizer
        tok_cfg = self._tokenizer_config
        self._tokenizer = get_tokenizer(
            tok_cfg.strategy,
            gene_vocab=self._gene_vocab,
            max_len=tok_cfg.max_genes,
            prepend_cls=tok_cfg.prepend_cls,
        )

        # 6. Build masking
        mask_cfg = tok_cfg.masking
        mask_action_ratio = 1.0 - mask_cfg.random_replace_ratio - mask_cfg.keep_ratio
        self._masking = MaskingStrategy(
            mask_ratio=mask_cfg.mask_ratio,
            mask_action_ratio=mask_action_ratio,
            random_replace_ratio=mask_cfg.random_replace_ratio,
            vocab_size=len(self._gene_vocab),
        )

        # 7. Streaming mode
        if self._data_config.streaming:
            self._streaming = True
            self._setup_streaming(adata, preprocessing)
            return

        # 7b. Train/val split (map-style path)
        n_total = len(full_dataset)
        n_val = max(1, int(n_total * self._val_split))
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(self._seed)
        train_subset, val_subset = random_split(
            full_dataset, [n_train, n_val], generator=generator,
        )

        # 8. Determine gene selection mode
        gs_cfg = self._gene_selection_config
        self._use_gene_selection = gs_cfg is not None and gs_cfg.strategy != "all"

        if self._use_gene_selection:
            # Gene selection mode: train dataset stays raw (pre-tokenization),
            # collation is handled by GeneSelectionCollator
            from scmodelforge.data.gene_selection import GeneSelectionCollator

            self._train_dataset = train_subset
            collator = GeneSelectionCollator(
                tokenizer=self._tokenizer,
                masking=self._masking,
                strategy=gs_cfg.strategy,  # type: ignore[union-attr]
                n_genes=gs_cfg.n_genes,  # type: ignore[union-attr]
            )
            self._train_collate_fn = collator
        else:
            # Standard mode: wrap with TokenizedCellDataset
            self._train_dataset = TokenizedCellDataset(train_subset, self._tokenizer, self._masking)
            self._train_collate_fn = self._tokenizer._collate

        # Val always uses standard TokenizedCellDataset
        self._val_dataset = TokenizedCellDataset(val_subset, self._tokenizer, self._masking)

        # 9. Build weighted sampler if configured
        samp_cfg = self._sampling_config
        if samp_cfg is not None and samp_cfg.strategy == "weighted":
            from scmodelforge.data.sampling import WeightedCellSampler, extract_labels_from_dataset

            labels = extract_labels_from_dataset(train_subset, label_key=samp_cfg.label_key)
            self._sampler = WeightedCellSampler(
                labels=labels,
                replacement=samp_cfg.replacement,
                seed=self._seed,
                curriculum_warmup_epochs=samp_cfg.curriculum_warmup_epochs,
            )
            logger.info(
                "Weighted sampling enabled: %d classes, curriculum_warmup=%d",
                len(self._sampler.class_weights),
                samp_cfg.curriculum_warmup_epochs,
            )

        self._is_setup = True
        logger.info("CellDataModule setup: %d train, %d val cells", n_train, n_val)

    def _setup_streaming(self, adata: Any, preprocessing: Any) -> None:
        """Set up streaming mode: train uses IterableDataset, val uses map-style."""
        from scmodelforge.data.dataset import CellDataset
        from scmodelforge.data.streaming import StreamingCellDataset

        dcfg = self._data_config

        # Streaming train dataset
        streaming_ds = StreamingCellDataset(
            file_paths=dcfg.paths,
            gene_vocab=self._gene_vocab,  # type: ignore[arg-type]
            preprocessing=preprocessing,
            chunk_size=dcfg.streaming_chunk_size,
            shuffle_buffer_size=dcfg.streaming_shuffle_buffer,
            seed=self._seed,
        )
        self._train_dataset = _StreamingTokenizedDataset(
            streaming_ds, self._tokenizer, self._masking,  # type: ignore[arg-type]
        )
        self._train_collate_fn = self._tokenizer._collate  # type: ignore[union-attr]

        # Val uses small map-style dataset for reproducibility
        full_dataset = CellDataset(adata, self._gene_vocab, preprocessing)  # type: ignore[arg-type]
        n_total = len(full_dataset)
        n_val = max(1, int(n_total * self._val_split))
        generator = torch.Generator().manual_seed(self._seed)
        _, val_subset = random_split(
            full_dataset, [n_total - n_val, n_val], generator=generator,
        )
        self._val_dataset = TokenizedCellDataset(
            val_subset, self._tokenizer, self._masking,  # type: ignore[arg-type]
        )

        self._is_setup = True
        logger.info("CellDataModule setup (streaming): %d files, %d val cells", len(dcfg.paths), n_val)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        """Training DataLoader with shuffle or weighted sampler."""
        if self._train_dataset is None:
            msg = "Call setup() before train_dataloader()."
            raise RuntimeError(msg)

        # Streaming datasets handle their own shuffling
        if self._streaming:
            return DataLoader(
                self._train_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                collate_fn=self._train_collate_fn,
                pin_memory=True,
                drop_last=False,
            )

        use_sampler = self._sampler is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=not use_sampler,
            sampler=self._sampler if use_sampler else None,
            num_workers=self._num_workers,
            collate_fn=self._train_collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation DataLoader without shuffle."""
        if self._val_dataset is None:
            msg = "Call setup() before val_dataloader()."
            raise RuntimeError(msg)
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=self._tokenizer._collate,  # type: ignore[union-attr]
            pin_memory=True,
            drop_last=False,
        )
