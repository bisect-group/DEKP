import re
import sys
from pathlib import Path
from typing import List

import torch
from transformers import T5Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE_DIR = REPO_ROOT / "DEKP"
if str(ORIGINAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_CODE_DIR))

from tokenizer import MolTranBertTokenizer  # noqa: E402

# MoLFormer vocabulary (IBM/molformer, training/bert_vocab.txt), as used by the original DEKP code.
SMILES_VOCAB_PATH = Path(__file__).with_name("bert_vocab.txt")


def load_protein_tokenizer(model_name_or_path: str) -> T5Tokenizer:
    return T5Tokenizer.from_pretrained(model_name_or_path, do_lower_case=False)


def load_smiles_tokenizer() -> MolTranBertTokenizer:
    return MolTranBertTokenizer(str(SMILES_VOCAB_PATH))


def protein_token_ids(tokenizer: T5Tokenizer, sequence: str) -> List[int]:
    """Unpadded ids exactly as in DEKP KcatDataset: rare residues map to X, residues are space-joined."""
    residues = re.sub(r"[UZOB*]", "X", str(sequence).rstrip("*"))
    return tokenizer.encode(" ".join(residues), add_special_tokens=True)


def smiles_token_ids(tokenizer: MolTranBertTokenizer, smiles: str) -> List[int]:
    """Unpadded ids exactly as in DEKP KcatDataset: characters are space-joined before the regex split.

    The regex split is called directly so the ids do not depend on the lower-casing of newer transformers versions.
    """
    tokens = tokenizer._tokenize(" ".join(str(smiles)))
    return [tokenizer.cls_token_id, *tokenizer.convert_tokens_to_ids(tokens), tokenizer.sep_token_id]


def pad_tokens(token_ids: torch.Tensor, max_length: int, pad_id: int) -> torch.Tensor:
    """Pad to max_length, or truncate keeping the final special token (as tokenizer truncation does)."""
    if token_ids.numel() > max_length:
        return torch.cat([token_ids[: max_length - 1], token_ids[-1:]]).long()
    padding = torch.full((max_length - token_ids.numel(),), int(pad_id), dtype=torch.long)
    return torch.cat([token_ids.long(), padding])
