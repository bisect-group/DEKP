import math
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch import nn
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from transformers import T5EncoderModel, T5Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RDLogger.DisableLog("rdApp.warning")

DEFAULT_TRFM_WEIGHTS = REPO_ROOT / "emulator_bench" / "trfm_12_23000.pkl"
GRAPH_RADIUS = 10
GRAPH_NEIGHBORS = 20
DEFAULT_TRFM_VOCAB = REPO_ROOT / "DEKP" / "vocab.pkl"


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name in {"TorchVocab", "Vocab", "WordVocab"}:
            return globals()[name]
        return super().find_class(module, name)


class TorchVocab:
    def __init__(self, counter, max_size=None, min_freq=1, specials=None):
        specials = specials or ["<pad>", "<oov>"]
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        self.itos = list(specials)
        for tok in specials:
            counter.pop(tok, None)

        max_size = None if max_size is None else max_size + len(self.itos)
        words_and_frequencies = sorted(counter.items(), key=lambda item: item[0])
        words_and_frequencies.sort(key=lambda item: item[1], reverse=True)
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"], max_size=max_size, min_freq=min_freq)


class WordVocab(Vocab):
    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as handle:
            return _CompatUnpickler(handle).load()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_size,
        )
        self.out = nn.Linear(hidden_size, out_size)

    def encode(self, src, columns):
        """Encode src (T, B). Column b gets pe[b], as the original batched call does (its pe slice runs over the batch axis)."""
        embedded = self.embed(src) + self.pe.pe[0, columns]
        output = embedded
        for idx in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[idx](output, None)
        penultimate = output.detach().cpu().numpy()
        output = self.trfm.encoder.layers[-1](output, None)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)
        output = output.detach().cpu().numpy()
        return np.hstack([np.mean(output, axis=0), np.max(output, axis=0), output[0, :, :], penultimate[0, :, :]])


_TRFM_PAIRS = set(
    "Cl Ca Cu Br Be Ba Bi Si Se Sr Na Ni Rb Ra Xe Li Al As Ag Au Mg Mn Te Zn si se te He +2 +3 +4 -2 -3 -4 Kr Fe".split()
)


def split_smiles_for_trfm(smiles: str) -> str:
    """Same split as `split` in the original utils.py (used by extract_smiles_trfm_features.py)."""
    smiles = str(smiles)
    tokens, i = [], 0
    while i < len(smiles) - 1:
        width = 3 if smiles[i] == "%" else 2 if smiles[i : i + 2] in _TRFM_PAIRS else 1
        tokens.append(smiles[i : i + width])
        i += width
    if i == len(smiles) - 1:
        tokens.append(smiles[i])
    return " ".join(tokens)


def get_trfm_inputs(smiles_values: Sequence[str], vocab: WordVocab, seq_len: int = 220) -> Tuple[torch.Tensor, torch.Tensor]:
    x_id, x_seg = [], []
    for smiles in smiles_values:
        tokens = split_smiles_for_trfm(smiles).split()
        if len(tokens) > seq_len - 2:
            half = (seq_len - 2) // 2
            tokens = tokens[:half] + tokens[-(seq_len - 2 - half) :]
        ids = [vocab.stoi.get(token, vocab.unk_index) for token in tokens]
        ids = [vocab.sos_index] + ids + [vocab.eos_index]
        seg = [1] * len(ids)
        padding = [vocab.pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        seg.extend(padding)
        x_id.append(ids)
        x_seg.append(seg)
    return torch.tensor(x_id, dtype=torch.long), torch.tensor(x_seg, dtype=torch.long)


def load_smiles_transformer(weights_path: str | Path | None = None, vocab_path: str | Path | None = None, device: torch.device | None = None):
    weights_path = Path(weights_path or DEFAULT_TRFM_WEIGHTS)
    vocab_path = Path(vocab_path or DEFAULT_TRFM_VOCAB)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing SMILES Transformer weights: {weights_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing SMILES Transformer vocab: {vocab_path}")
    device = device or torch.device("cpu")
    vocab = WordVocab.load_vocab(str(vocab_path))
    model = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, vocab


def embed_smiles_trfm_batch(smiles_values: Sequence[str], columns: Sequence[int], model: TrfmSeq2seq, vocab: WordVocab, device: torch.device) -> Dict[str, np.ndarray]:
    xid, _ = get_trfm_inputs(smiles_values, vocab=vocab)
    xid = torch.t(xid).to(device)
    with torch.no_grad():
        embeddings = model.encode(xid, torch.as_tensor(list(columns), device=device))
    return {
        smiles: np.asarray(embeddings[idx], dtype=np.float32)
        for idx, smiles in enumerate(smiles_values)
    }


def load_prot_t5(model_name_or_path: str, device: torch.device):
    model = T5EncoderModel.from_pretrained(model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def embed_prot_t5(model, tokenizer, sequence: str, device: torch.device) -> np.ndarray:
    """Mean ProtT5 residue vector as in `Seq_to_vec` of extract_prott5_feature.py.

    Sequences over 1000 residues keep the first and last 500. One sequence per forward pass, fp32.
    The </s> position is dropped. The mean is summed in float64, residue by residue.
    """
    sequence = str(sequence)
    if len(sequence) > 1000:
        sequence = sequence[:500] + sequence[-500:]
    encoding = tokenizer([re.sub(r"[UZOB]", "X", " ".join(sequence))], add_special_tokens=True, padding=True, return_tensors="pt")
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        hidden = model(input_ids=encoding["input_ids"].to(device), attention_mask=attention_mask).last_hidden_state
    residues = hidden[0, : int(attention_mask.sum()) - 1].cpu().numpy().astype(np.float64)
    return residues.cumsum(axis=0)[-1] / len(residues)


def load_pickle(path: Path) -> Dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_legacy_feature_pickles(feature_dir: Path, feature_names: Iterable[str]) -> Dict[str, Dict]:
    feature_dir = Path(feature_dir)
    payload = {}
    for name in feature_names:
        path = feature_dir / f"{name}.pkl"
        if path.exists():
            payload[name] = load_pickle(path)
    graph_path = feature_dir / "pyg_graph.pkl"
    if graph_path.exists():
        payload["graph"] = load_pickle(graph_path)
    return payload


def resolve_legacy_feature(mapping: Dict, key, fallback_key=None):
    if mapping is None:
        return None
    if key in mapping:
        return mapping[key]
    if fallback_key is not None and fallback_key in mapping:
        return mapping[fallback_key]
    return None


def _vectorize_atom(atom):
    atom_type = "CNOS$"
    aa_type = "ACDEFGHIKLMNPQRSTVWY$"
    atom_feature = np.zeros(len(atom_type) + len(aa_type), dtype=np.float32)
    symbol = str(atom.GetSymbol())
    symbol_idx = atom_type.find(symbol if symbol in atom_type else "$")
    atom_feature[max(0, symbol_idx)] = 1.0
    atom_feature[len(atom_type) + len(aa_type) - 1] = 1.0
    return atom_feature


def smiles_to_atom_features(smiles: str):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    atom_indices = []
    atom_features = []
    for atom in molecule.GetAtoms():
        atom_indices.append(atom.GetAtomicNum())
        atom_features.append(_vectorize_atom(atom))
    if not atom_indices:
        atom_indices = [0]
        atom_features = [np.zeros(27, dtype=np.float32)]
    atom_indices = np.asarray(atom_indices, dtype=np.int64)
    atom_features = np.asarray(atom_features, dtype=np.float32)
    return atom_indices, atom_features


def load_structure_graph_builder():
    from DEKP.Encode.extract_pdb_feature import get_geo_feat, parse_pdb

    return parse_pdb, get_geo_feat


def parse_pdb_to_array(pdb_path: Path, atom_type: str = "CA") -> np.ndarray:
    parse_pdb, get_geo_feat = load_structure_graph_builder()
    try:
        pdb_array = parse_pdb(str(pdb_path), atom_type=atom_type, nneighbor=GRAPH_NEIGHBORS, cal_cb=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to build graph from PDB `{pdb_path}`") from exc
    return pdb_array


def build_graph_from_array(pdb_array: np.ndarray, device: str | torch.device = "cpu") -> Data:
    """Same graph as DEKP get_graph_fea defaults: C-alpha radius graph, radius 10, at most 20 neighbours."""
    _, get_geo_feat = load_structure_graph_builder()
    x = torch.tensor(pdb_array, dtype=torch.float32, device=torch.device(device))
    ca_pos = x[:, 1].contiguous()
    edge_index = radius_graph(ca_pos, r=GRAPH_RADIUS, loop=False, max_num_neighbors=GRAPH_NEIGHBORS)
    node_feature, edge_feature = get_geo_feat(x, edge_index)
    return Data(
        x=node_feature.float().cpu(),
        edge_index=edge_index.long().cpu(),
        edge_attr=edge_feature.float().cpu(),
    )


def build_graph_from_pdb(pdb_path: Path, atom_type: str = "CA", device: str | torch.device = "cpu") -> Data:
    return build_graph_from_array(parse_pdb_to_array(pdb_path, atom_type=atom_type), device=device)


def compute_dssp_feature(pdb_path: Path, dssp_executable: str | None = None) -> np.ndarray:
    from Bio.PDB import DSSP, PDBParser

    max_asa = {
        "G": 188,
        "A": 198,
        "V": 220,
        "I": 233,
        "L": 304,
        "F": 272,
        "P": 203,
        "M": 262,
        "W": 317,
        "C": 201,
        "S": 234,
        "T": 215,
        "N": 254,
        "Q": 259,
        "Y": 304,
        "H": 258,
        "D": 236,
        "E": 262,
        "K": 317,
        "R": 319,
    }
    ss_type = "HBEGITS-"
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(pdb_path))
    model = structure[0]
    kwargs = {}
    if dssp_executable:
        kwargs["dssp"] = dssp_executable
    dssp = DSSP(model, str(pdb_path), **kwargs)
    features = []
    for record in dssp.property_list:
        ss_vec = np.zeros(8, dtype=np.float32)
        ss = record[2]
        ss_idx = ss_type.find(ss)
        ss_vec[ss_idx if ss_idx >= 0 else -1] = 1.0
        phi = float(record[4])
        psi = float(record[5])
        asa = float(record[3])
        aa_name = record[1]
        radian = np.array([phi, psi], dtype=np.float32) * (np.pi / 180.0)
        asa = min(float(asa) / max_asa.get(aa_name, 1.0), 1.0)
        feature = np.concatenate([np.sin(radian), np.cos(radian), np.asarray([asa], dtype=np.float32), ss_vec])
        features.append(feature)
    if not features:
        raise ValueError(f"DSSP returned no residues for {pdb_path}")
    return np.asarray(features, dtype=np.float32).mean(axis=0)
