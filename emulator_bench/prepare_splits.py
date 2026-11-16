import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_SPLIT_GROUPS,
    INTERMEDIATE_DIR,
    KEY_COLUMNS,
    TASKS,
    atomic_json,
    discover_split_jobs,
    manifest_path,
    normalize_sequence,
    normalize_threshold_args,
    read_table,
    require_columns,
    stable_hash,
    task_paths,
    write_table,
)


DEFAULT_MMCIF_DIR = INTERMEDIATE_DIR / "mmcif"
DEFAULT_ALPHAFOLD_PDB_DIR = INTERMEDIATE_DIR / "alphafold"
DEFAULT_ESM_PDB_DIR = INTERMEDIATE_DIR / "esmfold2_pdb"
ESM_PDB_PREFIX = "ESMFold2-"
STRUCTURE_MATCH_COLUMNS = ["smiles", "sequence", "value", "smiles_hash", "log10_value"]
CANDIDATE_COLUMNS = ["pdbs", "pdb_source", "pdb_type", "pdb_chain", "pdb_coverage"]


def _norm_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def mmcif_file(mmcif_dir: Path, pdb_id: str) -> Optional[Path]:
    for name in (f"{pdb_id.upper()}_updated.cif", f"{pdb_id.upper()}.cif"):
        if (Path(mmcif_dir) / name).exists():
            return Path(mmcif_dir) / name
    return None


def _build_alphafold_index(directory: Path) -> Dict[str, str]:
    """Accession to the newest `AF-<accession>-F1-model_v<n>.pdb`."""
    directory = Path(directory).expanduser()
    index: Dict[str, tuple] = {}
    for path in directory.glob("AF-*-F1-model_v*.pdb") if directory.exists() else []:
        accession, _, version_text = path.stem[len("AF-") :].partition("-F1-model_v")
        version = int(version_text) if version_text.isdigit() else -1
        if accession not in index or version > index[accession][0]:
            index[accession] = (version, str(path))
    return {key: path for key, (_version, path) in index.items()}


def _build_esm_index(directory: Path) -> Dict[str, str]:
    """Row key (accession, or accession plus mutations) to `ESMFold2-<key>.pdb`."""
    directory = Path(directory).expanduser()
    if not directory.exists():
        return {}
    return {path.name[len(ESM_PDB_PREFIX) : -len(".pdb")]: str(path) for path in directory.glob(f"{ESM_PDB_PREFIX}*.pdb")}


def extract_chain(cif_path: Path, chain_id: str, out_path: Path) -> None:
    """Write the amino-acid atoms of one author chain as a PDB file that `parse_pdb` reads.

    Residues are numbered 1..n so insertion codes do not merge residues.
    """
    atoms = pdbx.get_structure(pdbx.CIFFile.read(cif_path), model=1, use_author_fields=True)
    atoms = atoms[struc.filter_amino_acids(atoms) & (atoms.chain_id == chain_id)]
    if atoms.array_length() == 0:
        raise ValueError(f"No amino-acid atoms for chain {chain_id} in {cif_path}")
    starts = np.zeros(atoms.array_length(), dtype=int)
    starts[struc.get_residue_starts(atoms)] = 1
    residue_index = np.cumsum(starts)
    lines = []
    for serial, (atom_name, res_name, number, (x, y, z)) in enumerate(
        zip(atoms.atom_name, atoms.res_name, residue_index, atoms.coord), start=1
    ):
        name = atom_name if len(atom_name) == 4 else f" {atom_name:<3}"
        lines.append(f"ATOM  {serial % 100000:5d} {name} {res_name:>3} A{number % 10000:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
    lines.append("TER")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(f"{out_path}.tmp.{os.getpid()}")
    tmp_path.write_text("\n".join(lines) + "\n")
    tmp_path.replace(out_path)


def _extract_chain_job(job: tuple) -> None:
    extract_chain(*job)


def _row_key(row: dict) -> str:
    payload = {column: str(row.get(column, "")) for column in STRUCTURE_MATCH_COLUMNS}
    return stable_hash(json.dumps(payload, sort_keys=True))


def _candidate_lookup(candidate_df_path: Path) -> Dict[str, List[dict]]:
    path = Path(candidate_df_path).expanduser()
    frame = read_table(path)
    require_columns(frame, KEY_COLUMNS + CANDIDATE_COLUMNS, path)
    lookup: Dict[str, List[dict]] = {}
    for row in frame[KEY_COLUMNS + CANDIDATE_COLUMNS].to_dict("records"):
        lookup.setdefault(_row_key(row), []).append(row)
        lookup.setdefault("sequence:" + stable_hash(normalize_sequence(row.get("sequence", ""))), []).append(row)
    return lookup


class StructureResolver:
    """One structure per row: the dataset's best-covering experimental chain, else AlphaFold, else ESMFold2."""

    def __init__(self, args):
        self.mmcif_dir = Path(args.mmcif_dir).expanduser()
        self.chains_dir = Path(args.manifests_dir) / "_chains"
        self.alphafold_index = _build_alphafold_index(args.alphafold_pdb_dir)
        self.esm_index = _build_esm_index(args.esm_pdb_dir)
        self.pending_chains: Dict[Path, tuple] = {}

    def _experimental(self, candidates: List[dict]) -> Optional[dict]:
        best, best_coverage = None, -1.0
        for item in candidates:
            pdb_id, _, accession = _norm_text(item.get("pdbs")).partition(":")
            chain = _norm_text(item.get("pdb_chain"))
            cif_path = mmcif_file(self.mmcif_dir, pdb_id) if pdb_id and chain else None
            coverage = float(item["pdb_coverage"]) if pd.notna(item.get("pdb_coverage")) else 0.0
            if _norm_text(item.get("pdb_type")).lower() == "experimental" and cif_path and coverage > best_coverage:
                best, best_coverage = (pdb_id.upper(), accession, chain, cif_path), coverage
        if best is None:
            return None
        pdb_id, accession, chain, cif_path = best
        out_path = self.chains_dir / f"{pdb_id}_{chain}.pdb"
        if not out_path.exists():
            self.pending_chains[out_path] = (cif_path, chain, out_path)
        return {
            "structure_path": str(out_path),
            "pdbs": f"{pdb_id}:{accession}",
            "pdb_source": "PDBe",
            "pdb_type": "experimental",
            "chain_id": chain,
            "resolved_structure_status": "selected_experimental",
            "resolved_structure_coverage": round(best_coverage, 4),
        }

    def _predicted(self, uniprot_id: str) -> Optional[dict]:
        accession = uniprot_id.split("|", 1)[0].strip()
        if accession in self.alphafold_index:
            return {"structure_path": self.alphafold_index[accession], "pdbs": accession, "pdb_source": "AlphaFold",
                    "pdb_type": "predicted", "chain_id": "A", "resolved_structure_status": "selected_alphafold"}
        key = next((k for k in (uniprot_id.strip(), accession) if k in self.esm_index), None)
        if key is not None:
            return {"structure_path": self.esm_index[key], "pdbs": key, "pdb_source": "ESM",
                    "pdb_type": "predicted", "chain_id": "A", "resolved_structure_status": "selected_esm"}
        return None

    def resolve(self, row: dict, candidate_groups: List[List[dict]]) -> dict:
        for candidates in candidate_groups:
            result = self._experimental(candidates)
            if result:
                return result
        return self._predicted(_norm_text(row.get("uniprot_id"))) or {"structure_path": "", "resolved_structure_status": "unresolved"}

    def extract_pending(self, workers: int) -> None:
        jobs = list(self.pending_chains.values())
        self.pending_chains = {}
        if not jobs:
            return
        with get_context("fork").Pool(min(workers, len(jobs))) as pool:
            for _ in progress(pool.imap_unordered(_extract_chain_job, jobs, chunksize=16), total=len(jobs), desc="Extracting chains", unit="chain"):
                pass


def _sample_id(row: dict) -> str:
    return stable_hash(json.dumps({column: str(row.get(column, "")) for column in KEY_COLUMNS}, sort_keys=True))


def prepare_one_split(path: Path, output_path: Path, candidates: Dict[str, List[dict]], resolver: StructureResolver, args) -> dict:
    frame = read_table(path)
    require_columns(frame, KEY_COLUMNS, path)
    if args.limit_rows:
        frame = frame.head(args.limit_rows).copy()
    rows = []
    failed = 0
    for row in progress(frame.to_dict("records"), desc=f"Preparing {path.parent.name}/{path.name}", unit="row"):
        exact_candidates = candidates.get(_row_key(row), [])
        sequence_candidates = candidates.get("sequence:" + stable_hash(normalize_sequence(row.get("sequence", ""))), [])
        result = resolver.resolve(row, [exact_candidates, sequence_candidates])
        if not result.get("structure_path"):
            failed += 1
            if args.fail_on_unresolved:
                raise RuntimeError(f"Could not resolve structure for row sample_id={_sample_id(row)}")
            continue
        out = dict(row)
        out["sample_id"] = _sample_id(row)
        out["target"] = float(row[args.target_col])
        out.update(result)
        rows.append(out)
    resolver.extract_pending(args.workers)
    out_frame = pd.DataFrame(rows)
    write_table(output_path, out_frame)
    return {
        "input_path": str(path),
        "output_path": str(output_path),
        "rows_in": int(len(frame)),
        "rows_out": int(len(out_frame)),
        "rows_unresolved": int(failed),
        "by_source": out_frame["resolved_structure_status"].value_counts().to_dict() if len(out_frame) else {},
    }


def prepared_manifest_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = read_table(path)
    except Exception:
        return False
    required = {"sequence", "smiles", "structure_path", "chain_id", "target", "pdbs", "pdb_source", "pdb_type"}
    return bool(required.issubset(frame.columns))


def main():
    parser = argparse.ArgumentParser(description="Prepare DEKP EMULaToR split manifests with one resolved structure per row.")
    parser.add_argument("--task", choices=TASKS, default="km")
    parser.add_argument("--base_dir", type=Path, default=None)
    parser.add_argument("--manifests_dir", type=Path, default=None)
    parser.add_argument("--candidate_df_path", type=Path, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--mmcif_dir", type=Path, default=DEFAULT_MMCIF_DIR)
    parser.add_argument("--alphafold_pdb_dir", type=Path, default=DEFAULT_ALPHAFOLD_PDB_DIR)
    parser.add_argument("--esm_pdb_dir", type=Path, default=DEFAULT_ESM_PDB_DIR)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1), help="Processes that extract chains from mmCIF files.")
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail_on_unresolved", action="store_true")
    args = parser.parse_args()

    paths = task_paths(args.task)
    args.base_dir = Path(args.base_dir or paths["base_dir"]).expanduser()
    args.manifests_dir = Path(args.manifests_dir or paths["manifests_dir"]).expanduser()
    args.candidate_df_path = args.candidate_df_path or args.base_dir / f"{args.task}_kinetic_params_3d.parquet"
    args.mmcif_dir = Path(args.mmcif_dir).expanduser()
    args.alphafold_pdb_dir = Path(args.alphafold_pdb_dir).expanduser()
    args.esm_pdb_dir = Path(args.esm_pdb_dir).expanduser()
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)

    jobs = discover_split_jobs(args.base_dir, args.split_groups, args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")
    candidates = _candidate_lookup(args.candidate_df_path)
    resolver = StructureResolver(args)

    summaries = []
    for job in jobs:
        for split in ("train", "val", "test"):
            out_path = manifest_path(args.manifests_dir, job, split)
            if not args.overwrite and prepared_manifest_is_valid(out_path):
                summaries.append({"output_path": str(out_path), "status": "skipped_exists"})
                continue
            summaries.append(prepare_one_split(Path(job[f"{split}_path"]), out_path, candidates, resolver, args))

    summary = {"jobs": len(jobs), "splits": summaries}
    atomic_json(args.manifests_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
