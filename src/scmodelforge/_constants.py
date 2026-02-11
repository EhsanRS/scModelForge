"""Shared constants used across scModelForge modules."""

# Special token indices (reserved at the start of every gene vocabulary)
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
MASK_TOKEN_ID = 2
CLS_TOKEN_ID = 3

SPECIAL_TOKENS = {
    "<pad>": PAD_TOKEN_ID,
    "<unk>": UNK_TOKEN_ID,
    "<mask>": MASK_TOKEN_ID,
    "<cls>": CLS_TOKEN_ID,
}

NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)

# Default limits
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_HIDDEN_DIM = 512
DEFAULT_NUM_LAYERS = 12
DEFAULT_NUM_HEADS = 8
