import argparse
import gc
import json
import queue
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from torch.optim import AdamW
from torch.utils.data import DataLoader
try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    COMMON_PROTEIN_ID_COLS,
    COMMON_SEQUENCE_COLS,
    COMMON_SMILES_COLS,
    COMMON_TARGET_COLS,
    COMMON_STRUCTURE_ID_COLS,
    DEFAULT_FEATURES,
    SEEDS,
    append_csv_row,
    find_first_existing_column,
    ligand_cache_path,
    load_json,
    protein_cache_path,
    read_table,
    regression_metrics,
    resolve_amp_dtype,
    save_json,
    set_seed,
    structure_cache_path,
    write_csv,
)
from emulator_bench.dataset import CachedDEKPDataset, LigandEmbeddingStore, ProteinEmbeddingStore, StructureEmbeddingStore
from emulator_bench.modeling import MetaDecoder, graph_collate_fn


def _worker_init_fn(worker_id: int) -> None:
    # Each structure cache file is ~4 MB and prefetch_factor batches are in flight per worker.
    # Run Python's cyclic GC more aggressively in workers so that Python-level cycles don't accumulate.
    import gc as _gc
    _gc.set_threshold(100, 5, 2)


def _resolve_columns(frame: pd.DataFrame, manifest: dict, args):
    resolved = manifest.get("resolved_columns", {})
    pdb_record_col = resolved.get("pdb_record_col") or ("pdbs" if "pdbs" in frame.columns else None)
    structure_id_col = pdb_record_col or find_first_existing_column(
        frame, COMMON_STRUCTURE_ID_COLS, explicit=args.structure_id_col or resolved.get("structure_id_col"), required=False
    )
    return {
        "sequence_col": find_first_existing_column(frame, COMMON_SEQUENCE_COLS, explicit=args.sequence_col or resolved.get("sequence_col"), required=True),
        "smiles_col": find_first_existing_column(frame, COMMON_SMILES_COLS, explicit=args.smiles_col or resolved.get("smiles_col"), required=True),
        "protein_id_col": find_first_existing_column(frame, COMMON_PROTEIN_ID_COLS, explicit=args.protein_id_col or resolved.get("protein_id_col"), required=False),
        "structure_id_col": structure_id_col,
        "target_col": find_first_existing_column(frame, COMMON_TARGET_COLS, explicit=args.target_col or resolved.get("target_col"), required=True),
        "pdb_record_col": pdb_record_col,
    }


def _make_loader(dataset, batch_size, shuffle, args):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "collate_fn": graph_collate_fn,
        "drop_last": False,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["worker_init_fn"] = _worker_init_fn
    return DataLoader(**kwargs)


def _move_batch(graph_batch, protein_tokens, smiles_tokens, features, labels, device):
    graph_batch = graph_batch.to(device, non_blocking=True)
    protein_tokens = protein_tokens.to(device, non_blocking=True)
    smiles_tokens = smiles_tokens.to(device, non_blocking=True)
    features = features.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return graph_batch, protein_tokens, smiles_tokens, features, labels


def _sync_rng(loader) -> None:
    """Keep the global RNG sequence of a persistent loader equal to a non-persistent one.

    A persistent loader draws its worker base seed once. A non-persistent loader draws it at every
    iteration. One matching draw before each re-iteration keeps the global RNG and the shuffle order the same.
    """
    if loader.num_workers > 0 and loader.persistent_workers:
        if getattr(loader, "_iterated", False):
            torch.empty((), dtype=torch.int64).random_()
        loader._iterated = True


def _check_persistent_order() -> None:
    """Assert on a toy loader that persistent workers with `_sync_rng` match non-persistent shuffle order and RNG state."""
    state = torch.get_rng_state()
    results = []
    for persistent in (False, True):
        torch.manual_seed(0)
        loader = DataLoader(list(range(20)), batch_size=5, shuffle=True, num_workers=2, persistent_workers=persistent)
        orders = []
        for _ in range(3):
            _sync_rng(loader)
            orders.append([batch.tolist() for batch in loader])
        results.append((orders, torch.get_rng_state()))
    torch.set_rng_state(state)
    assert results[0][0] == results[1][0] and torch.equal(results[0][1], results[1][1]), (
        "Persistent workers change the shuffle order or the global RNG state in this PyTorch version."
    )


def _prefetch(loader, device):
    """Yield (graph, protein_tokens, smiles_tokens, features, labels, metadata) with tensors on `device`.

    On CUDA a thread copies the next batch on a side stream while the consumer works on the current one.
    At most two batches are on the GPU at once.
    """
    _sync_rng(loader)
    if device.type != "cuda":
        for *tensors, metadata in loader:
            yield (*_move_batch(*tensors, device), metadata)
        return

    stream = torch.cuda.Stream(device)
    free_slot = threading.Semaphore(1)
    ready = queue.Queue()

    def produce():
        try:
            for *tensors, metadata in loader:
                free_slot.acquire()
                with torch.cuda.stream(stream):
                    moved = _move_batch(*(tensor.pin_memory() for tensor in tensors), device)
                    copied = torch.cuda.Event()
                    copied.record(stream)
                ready.put((moved, metadata, copied))
            ready.put(None)
        except BaseException as exc:
            ready.put(exc)

    threading.Thread(target=produce, daemon=True).start()
    current = torch.cuda.current_stream(device)
    while (item := ready.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        moved, metadata, copied = item
        free_slot.release()
        current.wait_event(copied)
        moved[0].apply(lambda tensor: tensor.record_stream(current) or tensor)
        for tensor in moved[1:]:
            tensor.record_stream(current)
        yield (*moved, metadata)


def evaluate(model, loader, device):
    model.eval()
    mse_loss = torch.nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_examples = 0
    preds, labels, metadata_rows = [], [], []

    with torch.no_grad():
        for graph_batch, protein_tokens, smiles_tokens, features, label_tensor, metadata in _prefetch(loader, device):
            outputs = model(graph_batch, protein_tokens, smiles_tokens, features)
            loss = mse_loss(outputs, label_tensor)

            batch_size = int(label_tensor.numel())
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            preds.extend(outputs.detach().float().cpu().tolist())
            labels.extend(label_tensor.detach().float().cpu().tolist())
            metadata_rows.extend(metadata)

    metrics = regression_metrics(labels, preds)
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics, preds, labels, metadata_rows


def extract_latents(model, loader, device):
    model.eval()
    latents, labels, metadata_rows = [], [], []
    with torch.no_grad():
        for graph_batch, protein_tokens, smiles_tokens, features, label_tensor, metadata in _prefetch(loader, device):
            model(graph_batch, protein_tokens, smiles_tokens, features)
            latents.append(model.encode().detach().float().cpu())
            labels.extend(label_tensor.detach().float().cpu().tolist())
            metadata_rows.extend(metadata)
    return torch.cat(latents).numpy(), labels, metadata_rows


def _save_checkpoint(path: Path, model, optimizer, epoch: int, args, manifest: dict, feature_dim_list, best_metric: float):
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "cache_manifest": manifest,
        "feature_dim_list": feature_dim_list,
        "best_metric": float(best_metric),
    }
    torch.save(payload, path)


def _write_prediction_csv(path: Path, preds, labels, metadata_rows):
    rows = []
    for pred, label, metadata in zip(preds, labels, metadata_rows):
        row = dict(metadata)
        row["label"] = label
        row["prediction"] = pred
        rows.append(row)
    write_csv(path, rows)


def _resolve_feature_dims(manifest: dict, feature_names, train_ds) -> list[int]:
    feature_dims = dict(manifest.get("feature_dims", {}))
    missing = [name for name in feature_names if name not in feature_dims]
    if missing:
        sample_row = train_ds.frame.iloc[0]
        sequence = str(sample_row[train_ds.sequence_col])
        smiles = str(sample_row[train_ds.smiles_col])
        structure_id = str(sample_row[train_ds.structure_id_col]) if train_ds.structure_id_col in sample_row.index else str(sample_row[train_ds.protein_id_col])
        protein_payload = train_ds.protein_store.get(sequence)
        ligand_payload = train_ds.ligand_store.get(smiles)
        structure_payload = train_ds.structure_store.get(structure_id, sequence)
        for name in missing:
            if name in protein_payload:
                feature_dims[name] = int(torch.as_tensor(protein_payload[name]).numel())
            elif name in ligand_payload:
                feature_dims[name] = int(torch.as_tensor(ligand_payload[name]).numel())
            elif name in structure_payload:
                feature_dims[name] = int(torch.as_tensor(structure_payload[name]).numel())
            else:
                raise KeyError(f"Requested feature `{name}` is missing from both the cache manifest and cached payloads.")
    return [int(feature_dims[name]) for name in feature_names]


def _filter_missing_cache_rows(frame: pd.DataFrame, split_name: str, resolved_columns: dict, cache_dir: Path, manifest: dict) -> pd.DataFrame:
    sequence_col = resolved_columns["sequence_col"]
    protein_id_col = resolved_columns["protein_id_col"]
    structure_id_col = resolved_columns["structure_id_col"]

    keep_mask = []
    missing_protein_ids = []
    missing_ligand_ids = []
    missing_structure_ids = []
    for _, row in frame.iterrows():
        sequence = str(row[sequence_col])
        smiles = str(row[resolved_columns["smiles_col"]])
        protein_id = str(row[protein_id_col])
        structure_value = row[structure_id_col] if structure_id_col in row.index else None
        if structure_value is None or pd.isna(structure_value) or str(structure_value).strip().lower() in ("", "nan", "none"):
            structure_id = protein_id
        else:
            structure_id = str(structure_value).strip()

        protein_path = protein_cache_path(cache_dir, sequence=sequence)
        ligand_path = ligand_cache_path(cache_dir, smiles=smiles, fingerprint=manifest["trfm_fingerprint"])
        structure_path = structure_cache_path(cache_dir, structure_id=structure_id, fallback_sequence=sequence)

        protein_exists = protein_path.exists()
        ligand_exists = ligand_path.exists()
        structure_exists = structure_path.exists()
        keep_mask.append(protein_exists and ligand_exists and structure_exists)
        if not protein_exists:
            missing_protein_ids.append(protein_id)
        if not ligand_exists:
            missing_ligand_ids.append(smiles)
        if not structure_exists:
            missing_structure_ids.append(structure_id)

    total_rows = int(len(frame))
    removed_rows = int(total_rows - sum(keep_mask))
    kept_rows = int(total_rows - removed_rows)
    removed_pct = (100.0 * removed_rows / total_rows) if total_rows else 0.0
    payload = {
        "split": split_name,
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "removed_rows_missing_any_cache": removed_rows,
        "removed_pct_missing_any_cache": round(removed_pct, 4),
        "rows_missing_protein_cache": len(missing_protein_ids),
        "rows_missing_ligand_cache": len(missing_ligand_ids),
        "rows_missing_structure_cache": len(missing_structure_ids),
    }
    if missing_protein_ids:
        payload["missing_protein_examples"] = missing_protein_ids[:10]
    if missing_ligand_ids:
        payload["missing_ligand_examples"] = missing_ligand_ids[:10]
    if missing_structure_ids:
        payload["missing_structure_examples"] = missing_structure_ids[:10]
    print(json.dumps(payload), flush=True)

    return frame.loc[keep_mask].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Train DEKP on explicit train/val/test splits with cached features.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="dekp_run")
    parser.add_argument("--feature_list", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--sequence_col", type=str, default=None)
    parser.add_argument("--smiles_col", type=str, default=None)
    parser.add_argument("--protein_id_col", type=str, default=None)
    parser.add_argument("--structure_id_col", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Mixed precision for the network training (bf16 on Ampere or newer, else fp16). The original trains in fp32. Extra Trees is unaffected.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--compile_model", action="store_true")
    args = parser.parse_args()

    if args.num_workers > 0 and args.persistent_workers:
        _check_persistent_order()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    manifest = load_json(cache_dir / "manifest.json")
    args.feature_list = [item.strip() for item in args.feature_list.split(",") if item.strip()]

    train_df = read_table(Path(args.train_path))
    val_df = read_table(Path(args.val_path))
    test_df = read_table(Path(args.test_path))
    resolved_columns = _resolve_columns(train_df, manifest, args)
    if resolved_columns["protein_id_col"] is None:
        resolved_columns["protein_id_col"] = resolved_columns["sequence_col"]
    if resolved_columns["structure_id_col"] is None:
        resolved_columns["structure_id_col"] = resolved_columns["protein_id_col"]

    original_train_rows = len(train_df)
    original_val_rows = len(val_df)
    original_test_rows = len(test_df)

    # Original DEKP preprocessing (pretrain.py): drop placeholder labels and multi-component SMILES.
    # Applied to train only so every baseline keeps the same val and test rows.
    train_df = train_df[(train_df["value"] != 1e-8) & ~train_df[resolved_columns["smiles_col"]].str.contains(".", regex=False)]
    preprocessed_train = train_df
    train_df = _filter_missing_cache_rows(train_df, "train", resolved_columns, cache_dir, manifest)
    val_df = _filter_missing_cache_rows(val_df, "val", resolved_columns, cache_dir, manifest)
    test_df = _filter_missing_cache_rows(test_df, "test", resolved_columns, cache_dir, manifest)

    filtered_total = len(train_df) + len(val_df) + len(test_df)
    original_total = len(preprocessed_train) + original_val_rows + original_test_rows
    removed_total = int(original_total - filtered_total)
    removed_total_pct = (100.0 * removed_total / original_total) if original_total else 0.0
    print(
        json.dumps(
            {
                "split_filter_summary": {
                    "total_rows": int(original_total),
                    "kept_rows": int(filtered_total),
                    "removed_rows_missing_any_cache": removed_total,
                    "removed_pct_missing_any_cache": round(removed_total_pct, 4),
                }
            }
        ),
        flush=True,
    )
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"After dropping rows with missing cache entries, split sizes are train={len(train_df)}, val={len(val_df)}, test={len(test_df)}."
        )

    # As in DEKP pretrain.py: token padding length and positional-embedding size are the longest raw
    # sequence and SMILES of the run.
    run_frames = (train_df, val_df, test_df)
    protein_max_len = max(int(frame[resolved_columns["sequence_col"]].astype(str).str.len().max()) for frame in run_frames)
    smiles_max_len = max(int(frame[resolved_columns["smiles_col"]].astype(str).str.len().max()) for frame in run_frames)

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp_dtype = resolve_amp_dtype(device)[0] if args.amp else None
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    protein_store = ProteinEmbeddingStore(cache_dir)
    ligand_store = LigandEmbeddingStore(cache_dir, manifest["trfm_fingerprint"])
    structure_store = StructureEmbeddingStore(cache_dir)

    dataset_kwargs = {
        "protein_store": protein_store,
        "ligand_store": ligand_store,
        "structure_store": structure_store,
        "feature_names": args.feature_list,
        "sequence_col": resolved_columns["sequence_col"],
        "smiles_col": resolved_columns["smiles_col"],
        "protein_id_col": resolved_columns["protein_id_col"],
        "structure_id_col": resolved_columns["structure_id_col"],
        "target_col": resolved_columns["target_col"],
        "protein_max_len": protein_max_len,
        "smiles_max_len": smiles_max_len,
        "protein_pad_id": int(manifest["protein_pad_id"]),
        "smiles_pad_id": int(manifest["smiles_pad_id"]),
    }
    train_ds = CachedDEKPDataset(train_df, **dataset_kwargs)
    val_ds = CachedDEKPDataset(val_df, **dataset_kwargs)
    test_ds = CachedDEKPDataset(test_df, **dataset_kwargs)

    train_loader = _make_loader(train_ds, batch_size=args.batch_size, shuffle=True, args=args)
    val_loader = _make_loader(val_ds, batch_size=args.batch_size, shuffle=False, args=args)
    test_loader = _make_loader(test_ds, batch_size=args.batch_size, shuffle=False, args=args)

    feature_dim_list = _resolve_feature_dims(manifest, args.feature_list, train_ds)
    model = MetaDecoder(
        seq_vocab_size=int(manifest["protein_vocab_size"]),
        smi_vocab_size=int(manifest["smiles_vocab_size"]),
        feature_dim_list=feature_dim_list,
        hidden=args.hidden,
        num_layers=args.num_layers,
        protein_len=protein_max_len,
        smi_len=smiles_max_len,
        dropout=args.dropout,
        kernel_size=args.kernel_size,
    ).to(device)
    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    started = time.time()
    log_path = out_dir / "logfile.csv"

    with progress(total=args.epochs, desc="Training", unit="epoch", leave=True) as epoch_bar:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_examples = 0
            batch_count = 0
            iterator = progress(_prefetch(train_loader, device), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=False)
            for graph_batch, protein_tokens, smiles_tokens, features, labels, _ in iterator:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    outputs = model(graph_batch, protein_tokens, smiles_tokens, features)
                loss = criterion(outputs.float(), labels)
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                batch_size = int(labels.numel())
                train_loss_sum += float(loss.item()) * batch_size
                train_examples += batch_size
                batch_count += 1
                if batch_count % 100 == 0:
                    gc.collect()
                iterator.set_postfix(loss=f"{(train_loss_sum / max(1, train_examples)):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            del iterator

            train_loss = train_loss_sum / max(1, train_examples)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_csv_row(log_path, row)

            if epoch % 10 == 0 or epoch == args.epochs:
                _save_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, epoch, args, manifest, feature_dim_list, float("nan"))
            epoch_bar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            epoch_bar.update(1)
            gc.collect()

    # DEKP final predictor: Extra Trees (sklearn defaults, seeded) fitted on the trained network's latent vector.
    nn_val_metrics, _, _, _ = evaluate(model, val_loader, device=device)
    nn_test_metrics, _, _, _ = evaluate(model, test_loader, device=device)
    train_eval_loader = _make_loader(train_ds, batch_size=args.batch_size, shuffle=False, args=args)
    train_x, train_y, _ = extract_latents(model, train_eval_loader, device)
    val_x, val_labels, val_metadata = extract_latents(model, val_loader, device)
    test_x, test_labels, test_metadata = extract_latents(model, test_loader, device)
    trees = ExtraTreesRegressor(n_jobs=-1, random_state=args.seed).fit(train_x, train_y)
    val_preds = trees.predict(val_x).tolist()
    test_preds = trees.predict(test_x).tolist()
    val_metrics = regression_metrics(val_labels, val_preds)
    test_metrics = regression_metrics(test_labels, test_preds)
    final_summary = {
        "task_name": args.task_name,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
        "feature_list": ",".join(args.feature_list),
        "orig_train_rows": original_train_rows,
        "orig_val_rows": original_val_rows,
        "orig_test_rows": original_test_rows,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        **{f"nn_val_{key}": value for key, value in nn_val_metrics.items()},
        **{f"nn_test_{key}": value for key, value in nn_test_metrics.items()},
    }

    write_csv(out_dir / "final_results_val.csv", [val_metrics])
    write_csv(out_dir / "final_results_test.csv", [test_metrics])
    write_csv(out_dir / "final_results_val_nn.csv", [nn_val_metrics])
    write_csv(out_dir / "final_results_test_nn.csv", [nn_test_metrics])
    _write_prediction_csv(out_dir / "pred_label_val.csv", val_preds, val_labels, val_metadata)
    _write_prediction_csv(out_dir / "pred_label_test.csv", test_preds, test_labels, test_metadata)
    write_csv(out_dir / "run_summary.csv", [final_summary])
    save_json(out_dir / "run_summary.json", final_summary)


if __name__ == "__main__":
    main()
