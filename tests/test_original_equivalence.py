"""Equivalence of the emulator_bench wrapper with the authors' original DEKP code (CPU, fp32).

Training: the original loop of DEKP/pretrain.py and the wrapper loop (persistent loader, `_prefetch`) run on
the same toy data. Batch hashes, losses and final weights must be identical.
Graphs: the wrapper graph builder against the original `get_graph_fea`, and chain extraction against the
old `processed_exp_pdb` files.
Latents and Extra Trees: the original `prepare_ds` against the wrapper `extract_latents`.

The wrapper training step lives in `main()`, so the test repeats its statements (fp32, no autocast).
"""

import gc
import hashlib
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
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
from emulator_bench.common import ligand_cache_path, protein_cache_path, resolve_amp_dtype, set_seed, structure_cache_path  # noqa: E402
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
# Calm toy for the GPU test: more rows, larger batch and lower lr keep bf16 rounding noise small, so a 5% weight error stands out.
GPU_ROWS, GPU_BATCH, GPU_EPOCHS, GPU_LR = 96, 48, 3, 1e-4
INTERMEDIATE = ROOT.parents[1] / "data" / "intermediate"


def _toy_rows(count=ROWS) -> list[dict]:
    """Fixed random rows. Sequences and SMILES differ per row, so each row has its own cache entries."""
    gen = torch.Generator().manual_seed(0)
    rows = []
    for i in range(count):
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


def _cached_dataset(rows, cache):
    """The real `CachedDEKPDataset` (memory-mapped cache files) over a toy cache."""
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


@pytest.fixture(scope="module")
def wrapper_dataset(rows, tmp_path_factory):
    return _cached_dataset(rows, tmp_path_factory.mktemp("cache"))


@pytest.fixture(scope="module")
def gpu_dataset(tmp_path_factory):
    return _cached_dataset(_toy_rows(GPU_ROWS), tmp_path_factory.mktemp("gpu_cache"))


def _make_model() -> torch.nn.Module:
    return pretrain.MetaDecoder(
        28, 30, feature_dim_list=FEATURE_DIMS, hidden=32, num_layers=3, protein_len=PROTEIN_LEN, smi_len=SMILES_LEN, dropout=0.5, kernel_size=9
    )


def _batch_hash(graph, protein, smiles, features, labels, ids) -> str:
    digest = hashlib.sha256()
    for tensor in (graph.x, graph.edge_index, graph.edge_attr, graph.batch, protein, smiles, features, labels):
        digest.update(tensor.cpu().numpy().tobytes())
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


def run_wrapper(dataset, sync_rng=True, device=CPU, amp_dtype=None, noise=False, negative_scale=1.0, batch=BATCH, epochs=EPOCHS, lr=1e-3):
    """Wrapper statements of `train_single_target_tvt.main()`: persistent workers and `_prefetch`.

    With `sync_rng=False` the loader is iterated directly, without the extra RNG draw of `_sync_rng`.
    """
    gc.collect()  # a forked loader worker must not inherit unreachable CUDA objects: it aborts when it frees them
    set_seed(SEED)
    model = _make_model().to(device)
    if noise:
        _perturb(model)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(negative_scale)
    args = Namespace(num_workers=2, pin_memory=False, persistent_workers=True, prefetch_factor=2)
    loader = wrapper._make_loader(dataset, batch_size=batch, shuffle=True, args=args)
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    hashes, losses = [], []
    for _ in range(epochs):
        model.train()
        batches = wrapper._prefetch(loader, device) if sync_rng else loader
        for graph, protein, smi, feature, labels, metadata in batches:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
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


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def _mse(model, dataset) -> float:
    """Mean squared error of `model` on all rows in fp32 on the CPU, with a fixed RNG (dropout stays active in eval)."""
    model = model.to(CPU).eval()
    set_seed(SEED + 1)
    errors = []
    with torch.no_grad():
        for graph, protein, smi, feature, labels, _ in DataLoader(dataset, batch_size=BATCH, shuffle=False, collate_fn=graph_collate_fn):
            errors.append((model(graph, protein, smi, feature).float() - labels) ** 2)
    return torch.cat(errors).mean().item()


K = 10  # the production gap may be 10 times the fp32 noise (the bf16 run with initial weights perturbed by one bf16 rounding)
U = 2**-8  # one rounding of the weights at the production precision (bf16)


def _perturb(module, rel=U):
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(rel * p.abs() * torch.randn(p.shape, generator=generator).to(p.device))


@pytest.mark.gpu
@needs_cuda
def test_amp_training_stays_close_to_fp32(gpu_dataset, monkeypatch):
    """Production `--amp` (bf16 autocast, side-stream prefetch, cudnn.benchmark) against fp32 on the same GPU, seed and batches.

    Dropout is removed from all runs: its masks differ between dtypes. The toy has 96 rows, batch 48, lr 1e-4 and 3 epochs (calm: small bf16 noise). The reference is fp32; the noise control is
    the bf16 run with initial weights perturbed by 2**-8 relative. Bounds: max per-step loss gap and final MSE gap each
    within K times their noise floor; the step-0 loss gap within K times its floor. The batches are compared exactly.
    """
    monkeypatch.setattr(F, "dropout", lambda input, p=0.5, training=True, inplace=False: input.clone())
    device = torch.device("cuda")
    amp_dtype = resolve_amp_dtype(device)[0]
    scale = 1.05 if os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1" else 1.0  # scales only the production run's initial weights
    TRAIN = {"batch": GPU_BATCH, "epochs": GPU_EPOCHS, "lr": GPU_LR}
    benchmark = torch.backends.cudnn.benchmark
    runs = [run_wrapper(gpu_dataset, device=device, **TRAIN)]
    try:
        torch.backends.cudnn.benchmark = True  # as in train_single_target_tvt.main on CUDA
        runs += [
            run_wrapper(gpu_dataset, device=device, amp_dtype=amp_dtype, noise=True, **TRAIN),
            run_wrapper(gpu_dataset, device=device, amp_dtype=amp_dtype, **TRAIN),
        ]
        if scale != 1.0:  # the gap uses this run; the floor keeps the unscaled production run
            runs.append(run_wrapper(gpu_dataset, device=device, amp_dtype=amp_dtype, negative_scale=scale, **TRAIN))
    finally:
        torch.backends.cudnn.benchmark = benchmark
    assert all(run[0] == runs[0][0] for run in runs)
    model = _make_model()
    mse = []
    for run in runs:
        model.load_state_dict(run[2])
        mse.append(_mse(model, gpu_dataset))
    ref, noise, clean, prod = (torch.tensor(runs[i][1], dtype=torch.float64) for i in (0, 1, 2, -1))
    floor, gap = (noise - clean).abs().max().item(), (prod - ref).abs().max().item()
    tiny, tiny0 = 1e-7 * ref.abs().max().item(), 1e-7 * ref[0].abs().item()
    gap0, floor0 = (prod[0] - ref[0]).abs().item(), (noise[0] - clean[0]).abs().item()
    final_gap, final_floor = abs(mse[-1] - mse[0]), abs(mse[1] - mse[2])
    print(
        f"dekp bf16: max gap={gap:.3e}, max floor={floor:.3e}, ratio={gap / max(floor, tiny):.2f}, K={K}, "
        f"step0 gap={gap0:.3e}, floor={floor0:.3e}, ratio={gap0 / max(floor0, tiny0):.2f}; final gap={final_gap:.3e}, final floor={final_floor:.3e}"
    )
    assert gap0 <= K * max(floor0, tiny0)
    assert gap <= K * max(floor, tiny)
    assert final_gap <= K * max(final_floor, 1e-7 * mse[0])


@pytest.mark.gpu
@needs_cuda
def test_side_stream_prefetch_yields_the_same_batches(wrapper_dataset):
    """The CUDA copy thread yields the loader batches unchanged and in order."""
    device = torch.device("cuda")
    args = Namespace(num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    gc.collect()  # see run_wrapper
    set_seed(SEED)
    loader = wrapper._make_loader(wrapper_dataset, batch_size=BATCH, shuffle=True, args=args)
    cpu = [(g.x, p, s, f, y) for g, p, s, f, y, _ in wrapper._prefetch(loader, CPU)]
    set_seed(SEED)
    loader = wrapper._make_loader(wrapper_dataset, batch_size=BATCH, shuffle=True, args=args)
    gpu = [(g.x, p, s, f, y) for g, p, s, f, y, _ in wrapper._prefetch(loader, device)]
    assert len(cpu) == len(gpu) == ROWS // BATCH
    assert all(torch.equal(a, b.cpu()) for left, right in zip(cpu, gpu) for a, b in zip(left, right))


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


HAS_LOCAL_STRUCTURES = (INTERMEDIATE / "processed_exp_pdb").exists()
RESIDUES = ("ALA", "GLY", "SER", "LEU")
SIDE_CHAIN = {"ALA": ("CB",), "GLY": (), "SER": ("CB", "OG"), "LEU": ("CB", "CG", "CD1", "CD2")}


def _synthetic_entries(directory: Path, count=16):
    """Random-walk backbones written as an mmCIF file and as a whole-entry PDB file in the old processed layout."""
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    gen = np.random.default_rng(0)
    entries = []
    for index in range(count):
        names, residues, coords = [], [], []
        ca = np.zeros(3)
        for number in range(1, int(gen.integers(20, 41)) + 1):
            ca = ca + 3.8 * (lambda v: v / np.linalg.norm(v))(gen.normal(size=3))
            name = RESIDUES[int(gen.integers(len(RESIDUES)))]
            for atom in ("N", "CA", "C", "O") + SIDE_CHAIN[name]:
                names.append(atom)
                residues.append((name, number))
                coords.append(ca + (0 if atom == "CA" else gen.normal(scale=1.3, size=3)))
        coords = np.round(np.array(coords), 3)
        atoms = struc.AtomArray(len(names))
        atoms.coord = coords.astype(np.float32)
        atoms.chain_id[:] = "A"
        atoms.atom_name = np.array(names)
        atoms.element = np.array([n[0] for n in names])
        atoms.res_name = np.array([r for r, _ in residues])
        atoms.res_id = np.array([i for _, i in residues])
        cif = directory / f"s{index:02d}_updated.cif"
        cif_file = pdbx.CIFFile()
        pdbx.set_structure(cif_file, atoms, data_block="s")
        cif_file.write(cif)
        lines = [
            f"ATOM  {serial:5d} {atom if len(atom) == 4 else f' {atom:<3}'} {res:>3} A{number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            for serial, (atom, (res, number), (x, y, z)) in enumerate(zip(names, residues, coords), start=1)
        ]
        pdb = directory / f"s{index:02d}.pdb"
        pdb.write_text("\n".join(lines + ["TER"]) + "\n")
        entries.append((pdb, cif, "A"))
    return entries


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
def entries(tmp_path_factory):
    """Old processed files of the local data when present, else synthetic structures."""
    if HAS_LOCAL_STRUCTURES:
        return _single_chain_entries()
    return _synthetic_entries(tmp_path_factory.mktemp("structures"))


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


def test_graph_builder_equals_original_with_contiguous_input(entries):
    assert len(entries) == 16
    bad = [pdb.stem for pdb, _, _ in entries if not _same_graph(_original_graph(pdb, True), build_graph_from_pdb(pdb))]
    assert not bad, f"graphs differ from the original code: {bad}"


@pytest.mark.xfail(
    reason="With pyg-lib (no torch-cluster) the original passes a strided C-alpha view to radius_graph, which returns "
    "edges longer than the 10 A radius. The wrapper passes a contiguous copy. Results with the original code in this env differ.",
    strict=False,
)
def test_graph_builder_equals_original_unpatched(entries):
    assert all(_same_graph(_original_graph(pdb, False), build_graph_from_pdb(pdb)) for pdb, _, _ in entries)


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
