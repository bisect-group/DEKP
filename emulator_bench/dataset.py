from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch

from emulator_bench.bench_tokenizers import pad_tokens
from emulator_bench.common import (
    ligand_cache_path,
    normalize_sequence,
    protein_cache_path,
    structure_cache_path,
)

OPTIONAL_METADATA_KEYS = ["ECNumber", "ec_number", "Type", "type", "uniprot_id", "UniprotID"]


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load cached payload {path}. Rebuild this cache entry or rerun cache_embeddings.py --overwrite."
        ) from exc


class ProteinEmbeddingStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def get(self, sequence: str):
        return _load_pt(protein_cache_path(self.cache_dir, normalize_sequence(sequence)))


class LigandEmbeddingStore:
    def __init__(self, cache_dir: Path, fingerprint: str):
        self.cache_dir = Path(cache_dir)
        self.fingerprint = fingerprint

    def get(self, smiles: str):
        return _load_pt(ligand_cache_path(self.cache_dir, str(smiles).strip(), self.fingerprint))


class StructureEmbeddingStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def get(self, structure_id: str, sequence: str):
        return _load_pt(structure_cache_path(self.cache_dir, structure_id=structure_id, fallback_sequence=sequence))


class CachedDEKPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        protein_store: ProteinEmbeddingStore,
        ligand_store: LigandEmbeddingStore,
        structure_store: StructureEmbeddingStore,
        feature_names: List[str],
        sequence_col: str,
        smiles_col: str,
        protein_id_col: str,
        target_col: Optional[str],
        protein_max_len: int,
        smiles_max_len: int,
        protein_pad_id: int = 0,
        smiles_pad_id: int = 0,
        structure_id_col: Optional[str] = None,
    ):
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.protein_store = protein_store
        self.ligand_store = ligand_store
        self.structure_store = structure_store
        self.feature_names = list(feature_names)
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.protein_id_col = protein_id_col
        self.structure_id_col = structure_id_col or protein_id_col
        self.target_col = target_col
        self.protein_max_len = int(protein_max_len)
        self.smiles_max_len = int(smiles_max_len)
        self.protein_pad_id = int(protein_pad_id)
        self.smiles_pad_id = int(smiles_pad_id)

        frame = self.frame
        self._sequences = [str(value) for value in frame[sequence_col]]
        self._smiles = [str(value) for value in frame[smiles_col]]
        self._protein_ids = [str(value) for value in frame[protein_id_col]]
        self._structure_ids = [
            protein_id if pd.isna(raw) or str(raw).strip().lower() in ("", "nan", "none") else str(raw).strip()
            for protein_id, raw in zip(self._protein_ids, frame[self.structure_id_col])
        ]
        self._targets = frame[target_col].astype(float).tolist() if target_col else None
        self._optional = {key: frame[key].tolist() for key in OPTIONAL_METADATA_KEYS if key in frame.columns}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        sequence = self._sequences[idx]
        smiles = self._smiles[idx]
        protein_id = self._protein_ids[idx]
        structure_id = self._structure_ids[idx]

        protein_payload = self.protein_store.get(sequence)
        ligand_payload = self.ligand_store.get(smiles)
        structure_payload = self.structure_store.get(structure_id, sequence)

        feature_tensors = []
        for feature_name in self.feature_names:
            for payload in (protein_payload, ligand_payload, structure_payload):
                if feature_name in payload:
                    feature_tensors.append(payload[feature_name].float())
                    break
            else:
                raise KeyError(
                    f"Feature `{feature_name}` was requested but is missing from cached protein/ligand/structure payloads."
                )
        feature_tensor = torch.cat(feature_tensors, dim=-1)

        label_value = self._targets[idx] if self._targets is not None else float("nan")
        label_tensor = torch.tensor(label_value, dtype=torch.float32)
        protein_tokens = pad_tokens(protein_payload["token_ids"], self.protein_max_len, self.protein_pad_id)
        smiles_tokens = pad_tokens(ligand_payload["token_ids"], self.smiles_max_len, self.smiles_pad_id)
        graph = structure_payload["graph"]

        metadata = {
            "row_index": int(idx),
            "protein_id": protein_id,
            "structure_id": structure_id,
            "sequence": sequence,
            "smiles": smiles,
            "target": label_value,
        }
        for key, values in self._optional.items():
            metadata[key] = values[idx]

        return graph, protein_tokens, smiles_tokens, feature_tensor, label_tensor, metadata
