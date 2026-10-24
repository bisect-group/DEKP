"""Equivalence of the emulator_bench wrapper with the authors' original DEKP code (CPU, fp32).

Training: the original loop of DEKP/pretrain.py and the wrapper loop (persistent loader, `_prefetch`) run on
the same toy data. Batch hashes, losses and final weights must be identical.
Graphs: the wrapper graph builder against the original `get_graph_fea`, and chain extraction against the
old `processed_exp_pdb` files.
Latents and Extra Trees: the original `prepare_ds` against the wrapper `extract_latents`.

The wrapper training step lives in `main()`, so the test repeats its statements (fp32, no autocast).
"""

import hashlib
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import ExtraTreesRegressor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "DEKP", ROOT / "DEKP" / "Encode"):
    sys.path.insert(0, str(path))

import extract_pdb_feature as original_graphs  # noqa: E402
import fine_tune  # noqa: E402
import pretrain  # noqa: E402
from emulator_bench import train_single_target_tvt as wrapper  # noqa: E402
from emulator_bench.common import ligand_cache_path, protein_cache_path, set_seed, structure_cache_path  # noqa: E402
from emulator_bench.dataset import CachedDEKPDataset, LigandEmbeddingStore, ProteinEmbeddingStore, StructureEmbeddingStore  # noqa: E402
from emulator_bench.feature_pipeline import build_graph_from_pdb  # noqa: E402
from emulator_bench.modeling import MetaDecoder, graph_collate_fn  # noqa: E402
from emulator_bench.prepare_splits import extract_chain  # noqa: E402

CPU = torch.device("cpu")
ROWS, BATCH, EPOCHS, SEED = 24, 8, 3, 666
PROTEIN_LEN, SMILES_LEN = 40, 30
FEATURE_DIMS = [8, 6]  # trfm, t5
FEATURES = ["trfm", "t5"]
FINGERPRINT = "fp"
INTERMEDIATE = ROOT.parents[1] / "data" / "intermediate"


def _toy_rows() -> list[dict]:
    """Fixed random rows. Sequences and SMILES differ per row, so each row has its own cache entries."""
    gen = torch.Generator().manual_seed(0)
    rows = []
    for i in range(ROWS):
        n_nodes = int(torch.randint(5, 12, (1,), generator=gen))
        n_edges = int(torch.randint(10, 30, (1,), generator=gen))
        graph = Data(
            x=torch.randn(n_nodes, 42, generator=gen),
            edge_index=torch.randint(0, n_nodes, (2, n_edges), generator=gen),
            edge_attr=torch.randn(n_edges, 92, generator=gen),
        )
        protein_len, smiles_len = (int(v) for v in torch.randint(8, 25, (2,), generator=gen))
        rows.append(
            {
                "uniprot_id": f"P{i:05d}",
                "structure_id": f"S{i:05d}",
                "sequence": "A" * (i + 1) + "C",
                "smiles": "C" * (i + 1) + "O",
                "value": float(torch.randn(1, generator=gen)),
                "graph": graph,
                "protein_ids": torch.randint(1, 28, (protein_len,), generator=gen),
                "smiles_ids": torch.randint(1, 30, (smiles_len,), generator=gen),
                "trfm": torch.randn(FEATURE_DIMS[0], generator=gen),
                "t5": torch.randn(FEATURE_DIMS[1], generator=gen),
            }
        )
    return rows


def _pad(ids: torch.Tensor, length: int) -> torch.Tensor:
    return torch.cat([ids, torch.zeros(length - ids.numel(), dtype=torch.long)])


class OriginalToyDataset(Dataset):
    """Items in the layout of the original `KcatDataset.__getitem__`."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        features = torch.cat([row[name] for name in FEATURES]).float()
        return (
            row["graph"],
            _pad(row["protein_ids"], PROTEIN_LEN),
            _pad(row["smiles_ids"], SMILES_LEN),
            features,
            torch.tensor(row["value"], dtype=torch.float32),
            row["uniprot_id"],
            row["sequence"],
            "1.1.1.1",
            "wild",
        )


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    """One CPU thread keeps the toy runs fast and their float sums in a fixed order."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def rows():
    return _toy_rows()


@pytest.fixture(scope="module")
def wrapper_dataset(rows, tmp_path_factory):
    """The real `CachedDEKPDataset` (memory-mapped cache files) over a toy cache."""
    cache = tmp_path_factory.mktemp("cache")
    for row in rows:
        for path, payload in (
            (protein_cache_path(cache, row["sequence"]), {"token_ids": row["protein_ids"], "t5": row["t5"]}),
            (ligand_cache_path(cache, row["smiles"], FINGERPRINT), {"token_ids": row["smiles_ids"], "trfm": row["trfm"]}),
            (structure_cache_path(cache, row["structure_id"], row["sequence"]), {"graph": row["graph"]}),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
    return CachedDEKPDataset(
        pd.DataFrame(rows).drop(columns=["graph", "protein_ids", "smiles_ids", "trfm", "t5"]),
        ProteinEmbeddingStore(cache),
        LigandEmbeddingStore(cache, FINGERPRINT),
        StructureEmbeddingStore(cache),
        feature_names=FEATURES,
        sequence_col="sequence",
        smiles_col="smiles",
        protein_id_col="uniprot_id",
        structure_id_col="structure_id",
        target_col="value",
        protein_max_len=PROTEIN_LEN,
        smiles_max_len=SMILES_LEN,
    )


def _make_model() -> torch.nn.Module:
    return pretrain.MetaDecoder(
        28, 30, feature_dim_list=FEATURE_DIMS, hidden=32, num_layers=3, protein_len=PROTEIN_LEN, smi_len=SMILES_LEN, dropout=0.5, kernel_size=9
    )


def _batch_hash(graph, protein, smiles, features, labels, ids) -> str:
    digest = hashlib.sha256()
    for tensor in (graph.x, graph.edge_index, graph.edge_attr, graph.batch, protein, smiles, features, labels):
        digest.update(tensor.numpy().tobytes())
    digest.update(",".join(ids).encode())
    return digest.hexdigest()


def _same_state(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def run_original(rows):
    """Training statements of DEKP/pretrain.py (`__main__`): shuffled non-persistent loader, AdamW, MSE."""
    set_seed(SEED)
    model = _make_model()
    loader = DataLoader(OriginalToyDataset(rows), batch_size=BATCH, shuffle=True, num_workers=2, collate_fn=pretrain.graph_collate_fn)
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    hashes, losses = [], []
    for _ in range(EPOCHS):
        model.train()
        for graph, protein, smi, feature, label, identifier, _, _, _ in loader:
            y_pred = model(graph, protein, smi, feature)
            model.zero_grad()
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()
            hashes.append(_batch_hash(graph, protein, smi, feature, label, identifier))
            losses.append(loss.item())
    return hashes, losses, model.state_dict()


def run_wrapper(dataset, sync_rng=True):
    """Wrapper statements of `train_single_target_tvt.main()`: persistent workers and `_prefetch`.

    With `sync_rng=False` the loader is iterated directly, without the extra RNG draw of `_sync_rng`.
    """
    set_seed(SEED)
    model = _make_model()
    args = Namespace(num_workers=2, pin_memory=False, persistent_workers=True, prefetch_factor=2)
    loader = wrapper._make_loader(dataset, batch_size=BATCH, shuffle=True, args=args)
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    hashes, losses = [], []
    for _ in range(EPOCHS):
        model.train()
        batches = wrapper._prefetch(loader, CPU) if sync_rng else loader
        for graph, protein, smi, feature, labels, metadata in batches:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cpu", dtype=None, enabled=False):
                outputs = model(graph, protein, smi, feature)
            loss = criterion(outputs.float(), labels)
            loss.backward()
            optimizer.step()
            hashes.append(_batch_hash(graph, protein, smi, feature, labels, [m["protein_id"] for m in metadata]))
            losses.append(loss.item())
    return hashes, losses, model.state_dict()


def test_training_matches_original_loop(rows, wrapper_dataset):
    original = run_original(rows)
    ours = run_wrapper(wrapper_dataset)
    steps = EPOCHS * (ROWS // BATCH)
    assert len(original[0]) == len(ours[0]) == steps
    assert original[0] == ours[0], "batch contents or order differ from the original"
    assert original[1] == ours[1], "per-step losses differ from the original"
    assert _same_state(original[2], ours[2]), "weights after training differ from the original"


def test_persistent_workers_need_the_rng_sync(rows, wrapper_dataset):
    """Persistent workers draw the base seed once. Without `_sync_rng` the order changes from epoch 2 on."""
    per_epoch = ROWS // BATCH
    original = run_original(rows)[0]
    unsynced = run_wrapper(wrapper_dataset, sync_rng=False)[0]
    assert unsynced[:per_epoch] == original[:per_epoch]
    assert unsynced[per_epoch:] != original[per_epoch:]


def _latents_original(rows, model, monkeypatch):
    monkeypatch.setattr(fine_tune, "device", CPU, raising=False)
    loader = DataLoader(OriginalToyDataset(rows), batch_size=BATCH, shuffle=False, collate_fn=fine_tune.graph_collate_fn)
    x, y, *_ = fine_tune.prepare_ds(loader, model)
    return x, y


def _latents_wrapper(dataset, model):
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, collate_fn=graph_collate_fn)
    x, y, _ = wrapper.extract_latents(model, loader, CPU)
    return x, np.array(y)


def test_latents_and_extra_trees_match_original(rows, wrapper_dataset, monkeypatch):
    """Same latents as `fine_tune.prepare_ds`. Extra Trees uses a seed and all cores, which keeps predictions equal.

    The original `ResidualMLP` calls `F.dropout` without the training flag, so dropout (p=0.5) stays active in eval
    mode and latents depend on the global RNG. Both sides are seeded alike before extraction.
    The original `ExtraTreesRegressor()` has no seed, so its own runs differ from each other. The wrapper seeds it.
    """
    set_seed(SEED)
    model = _make_model()
    set_seed(SEED + 1)
    original_x, original_y = _latents_original(rows, model, monkeypatch)
    set_seed(SEED + 1)
    wrapper_x, wrapper_y = _latents_wrapper(wrapper_dataset, model)
    assert np.array_equal(original_x, wrapper_x.astype(original_x.dtype)) and np.allclose(original_y, wrapper_y, atol=0)

    reference = ExtraTreesRegressor(random_state=SEED).fit(original_x, original_y)
    ours = ExtraTreesRegressor(n_jobs=-1, random_state=SEED).fit(wrapper_x, wrapper_y)
    assert np.array_equal(reference.predict(original_x), ours.predict(wrapper_x))


needs_data = pytest.mark.skipif(not (INTERMEDIATE / "processed_exp_pdb").exists(), reason="local structure data missing")


def _single_chain_entries(count=16, max_residues=100):
    """First `count` old processed files (sorted) of at most `max_residues` residues whose mmCIF has one amino-acid chain."""
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    entries = []
    for pdb in sorted((INTERMEDIATE / "processed_exp_pdb").glob("*.pdb")):
        cif = INTERMEDIATE / "mmcif" / f"{pdb.stem}_updated.cif"
        if not cif.exists() or sum(" CA " in line[:20] for line in open(pdb)) > max_residues:
            continue
        atoms = pdbx.get_structure(pdbx.CIFFile.read(cif), model=1, use_author_fields=True)
        chains = set(atoms[struc.filter_amino_acids(atoms)].chain_id)
        if len(chains) == 1:
            entries.append((pdb, cif, str(chains.pop())))
        if len(entries) == count:
            break
    return entries


@pytest.fixture(scope="module")
def entries():
    return _single_chain_entries()


def _same_graph(a: Data, b: Data) -> bool:
    return all(torch.equal(getattr(a, key), getattr(b, key)) for key in ("x", "edge_index", "edge_attr"))


def _original_graph(path: Path, contiguous: bool) -> Data:
    """Original `get_graph_fea`. With `contiguous`, the C-alpha view given to `radius_graph` is copied first."""
    if not contiguous:
        return original_graphs.get_graph_fea(str(path))
    radius_graph = original_graphs.radius_graph
    original_graphs.radius_graph = lambda x, num_workers=None, **kwargs: radius_graph(x.contiguous(), **kwargs)
    try:
        return original_graphs.get_graph_fea(str(path))
    finally:
        original_graphs.radius_graph = radius_graph


@needs_data
def test_graph_builder_equals_original_with_contiguous_input(entries):
    assert len(entries) == 16
    bad = [pdb.stem for pdb, _, _ in entries if not _same_graph(_original_graph(pdb, True), build_graph_from_pdb(pdb))]
    assert not bad, f"graphs differ from the original code: {bad}"


@needs_data
@pytest.mark.xfail(
    reason="With pyg-lib (no torch-cluster) the original passes a strided C-alpha view to radius_graph, which returns "
    "edges longer than the 10 A radius. The wrapper passes a contiguous copy. Results with the original code in this env differ.",
    strict=False,
)
def test_graph_builder_equals_original_unpatched(entries):
    assert all(_same_graph(_original_graph(pdb, False), build_graph_from_pdb(pdb)) for pdb, _, _ in entries)


@needs_data
def test_chain_graphs_against_old_processed_files(entries, tmp_path):
    """Graphs from the extracted chain against the old whole-entry files. Every difference must come from alternate locations."""
    import biotite.structure.io.pdbx as pdbx

    differing = []
    for pdb, cif, chain in entries:
        extract_chain(cif, chain, tmp_path / pdb.name)
        if not _same_graph(build_graph_from_pdb(tmp_path / pdb.name), build_graph_from_pdb(pdb)):
            alt_ids = set(pdbx.CIFFile.read(cif).block["atom_site"]["label_alt_id"].as_array())
            differing.append((pdb.stem, alt_ids - {".", "?"}))
    print(f"{len(entries) - len(differing)} of {len(entries)} graphs match exactly; differing: {differing}")
    assert len(entries) - len(differing) >= 14
    assert all(alt_ids for _, alt_ids in differing), f"differences without alternate locations: {differing}"
