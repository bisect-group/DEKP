import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
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
    DEFAULT_FEATURES,
    DEFAULT_SPLIT_GROUPS,
    SEEDS,
    TASKS,
    apply_manifest_paths_to_jobs,
    discover_split_jobs,
    normalize_threshold_args,
    split_sizes,
    summarize_seed_runs,
    task_paths,
)

CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def maybe_cache_embeddings(args):
    if args.skip_cache:
        return
    cmd = [
        sys.executable,
        str(CACHE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--cache_dir",
        args.cache_dir,
        "--feature_list",
        args.feature_list,
        "--device",
        args.cache_device,
        "--trfm_batch_size",
        str(args.trfm_batch_size),
        "--graph_atom_type",
        args.graph_atom_type,
        "--manifests_dir",
        args.manifests_dir,
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.legacy_feature_dir:
        cmd.extend(["--legacy_feature_dir", args.legacy_feature_dir])
    if args.prot_t5_model:
        cmd.extend(["--prot_t5_model", args.prot_t5_model])
    if args.trfm_weights_path:
        cmd.extend(["--trfm_weights_path", args.trfm_weights_path])
    if args.trfm_vocab_path:
        cmd.extend(["--trfm_vocab_path", args.trfm_vocab_path])
    if args.cache_dtype:
        cmd.extend(["--cache_dtype", args.cache_dtype])
    if args.overwrite_cache:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _run_training(job, seed, args):
    result_root = Path(args.out_dir) / job["split_group"] / job["split_name"] / f"seed_{seed}"
    result_root.mkdir(parents=True, exist_ok=True)
    final_test_path = result_root / "final_results_test.csv"
    if final_test_path.exists() and not args.overwrite_runs:
        return result_root

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_path",
        job["train_path"],
        "--val_path",
        job["val_path"],
        "--test_path",
        job["test_path"],
        "--cache_dir",
        args.cache_dir,
        "--out_dir",
        str(result_root),
        "--task_name",
        f"{job['split_group']}_{job['split_name']}_seed{seed}",
        "--feature_list",
        args.feature_list,
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--hidden",
        str(args.hidden),
        "--num_layers",
        str(args.num_layers),
        "--kernel_size",
        str(args.kernel_size),
        "--dropout",
        str(args.dropout),
        "--device",
        args.device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--cache_items",
        str(args.cache_items),
        "--seed",
        str(seed),
    ]
    for key in ["sequence_col", "smiles_col", "protein_id_col", "structure_id_col", "target_col"]:
        value = getattr(args, key)
        if value:
            cmd.extend([f"--{key}", value])
    if args.persistent_workers:
        cmd.append("--persistent_workers")
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.preload_proteins:
        cmd.append("--preload_proteins")
    if args.preload_ligands:
        cmd.append("--preload_ligands")
    if args.preload_structures:
        cmd.append("--preload_structures")
    if args.compile_model:
        cmd.append("--compile_model")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return result_root


def main():
    parser = argparse.ArgumentParser(description="Run the DEKP benchmark sweep across EMULaToR split families.")
    parser.add_argument("--task", choices=TASKS, default="km")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--manifests_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--feature_list", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--sequence_col", type=str, default=None)
    parser.add_argument("--smiles_col", type=str, default=None)
    parser.add_argument("--protein_id_col", type=str, default=None)
    parser.add_argument("--structure_id_col", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--preload_ligands", action="store_true")
    parser.add_argument("--preload_structures", action="store_true")
    parser.add_argument("--cache_items", type=int, default=512)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--legacy_feature_dir", type=str, default=None)
    parser.add_argument("--prot_t5_model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--trfm_weights_path", type=str, default=None)
    parser.add_argument("--trfm_vocab_path", type=str, default=None)
    parser.add_argument("--trfm_batch_size", type=int, default=256)
    parser.add_argument("--graph_atom_type", type=str, default="CA")
    parser.add_argument("--cache_dtype", choices=["float16", "float32"], default="float32")
    args = parser.parse_args()

    paths = task_paths(args.task)
    for key, value in paths.items():
        if getattr(args, key) is None:
            setattr(args, key, str(value))
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache_embeddings(args)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")
    jobs = apply_manifest_paths_to_jobs(jobs, Path(args.manifests_dir).expanduser(), require=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run_rows = []
    for job in progress(jobs, desc="Split jobs", unit="job"):
        for seed in args.seeds:
            result_root = _run_training(job, seed, args)
            metrics = pd.read_csv(result_root / "final_results_test.csv").iloc[0].to_dict()
            used = pd.read_csv(result_root / "run_summary.csv").iloc[0]
            sizes = split_sizes(Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"]))
            used_sizes = {f"{part}_size": int(used[f"{part}_rows"]) for part in ("train", "val", "test")}
            total = sum(used_sizes.values())
            row = {
                "split_group": job["split_group"],
                "split_name": job["split_name"],
                "difficulty": job["difficulty"],
                "seed": seed,
                "result_dir": str(result_root),
                **used_sizes,
                **{f"{part}_ratio": used_sizes[f"{part}_size"] / total for part in ("train", "val", "test")},
                **{f"orig_{part}_size": sizes[f"{part}_size"] for part in ("train", "val", "test")},
                **metrics,
            }
            run_rows.append(row)

    pd.DataFrame(run_rows).to_csv(Path(args.out_dir) / "dekp_summary_runs.csv", index=False)
    threshold_df = summarize_seed_runs(
        rows=run_rows,
        group_cols=["split_group", "split_name", "difficulty"],
        metric_cols=["rmse", "mse", "mae", "r2", "pearson", "spearman", "loss"],
    )
    threshold_df.to_csv(Path(args.out_dir) / "dekp_summary_thresholds.csv", index=False)
    group_df = summarize_seed_runs(
        rows=run_rows,
        group_cols=["split_group"],
        metric_cols=["rmse", "mse", "mae", "r2", "pearson", "spearman", "loss"],
    )
    group_df.to_csv(Path(args.out_dir) / "dekp_summary_by_split_group.csv", index=False)

    ranked_df = threshold_df.sort_values(
        by="rmse_mean" if "rmse_mean" in threshold_df.columns else threshold_df.columns[0],
        ascending=True,
    )
    ranked_df.to_csv(Path(args.out_dir) / "dekp_summary_ranked.csv", index=False)


if __name__ == "__main__":
    main()
