import sys
from argparse import Namespace
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.prepare_splits import StructureResolver, extract_chain  # noqa: E402


def _write_cif(path: Path) -> None:
    """Two chains: A has residues 1-2 (res 2 has an insertion-code twin), B has residue 1."""
    atoms = struc.AtomArray(7)
    atoms.chain_id[:] = ["A", "A", "A", "A", "A", "B", "B"]
    atoms.res_id[:] = [1, 1, 2, 2, 2, 1, 1]
    atoms.ins_code[:] = ["", "", "", "", "A", "", ""]
    atoms.res_name[:] = ["ALA", "ALA", "GLY", "GLY", "GLY", "SER", "SER"]
    atoms.atom_name[:] = ["N", "CA", "N", "CA", "C", "N", "CA"]
    atoms.element[:] = ["N", "C", "N", "C", "C", "N", "C"]
    atoms.coord = np.arange(21, dtype=np.float32).reshape(7, 3)
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atoms, data_block="1ABC")
    cif.write(path)


def _args(tmp_path: Path, mmcif: Path, alphafold: Path, esm: Path) -> Namespace:
    return Namespace(mmcif_dir=mmcif, manifests_dir=tmp_path / "manifests", alphafold_pdb_dir=alphafold, esm_pdb_dir=esm)


def test_extract_chain_keeps_only_the_chain_and_numbers_residues(tmp_path):
    cif = tmp_path / "1ABC_updated.cif"
    _write_cif(cif)
    out = tmp_path / "1ABC_A.pdb"
    extract_chain(cif, "A", out)
    lines = out.read_text().splitlines()
    atoms = [line for line in lines if line.startswith("ATOM")]
    assert len(atoms) == 5
    assert {line[21] for line in atoms} == {"A"}
    assert [int(line[22:26]) for line in atoms] == [1, 1, 2, 2, 3]
    assert [line[13:16].strip() for line in atoms] == ["N", "CA", "N", "CA", "C"]
    assert [float(line[30:38]) for line in atoms][:2] == [0.0, 3.0]


def test_resolver_prefers_best_covering_experimental_chain(tmp_path):
    mmcif, alphafold, esm = (tmp_path / name for name in ("mmcif", "af", "esm"))
    for folder in (mmcif, alphafold, esm):
        folder.mkdir()
    _write_cif(mmcif / "1ABC_updated.cif")
    (alphafold / "AF-P12345-F1-model_v6.pdb").write_text("")
    resolver = StructureResolver(_args(tmp_path, mmcif, alphafold, esm))
    candidates = [
        {"pdbs": "9ZZZ:P12345", "pdb_type": "experimental", "pdb_chain": "A", "pdb_coverage": 0.99},  # no mmCIF file
        {"pdbs": "1ABC:P12345", "pdb_type": "experimental", "pdb_chain": "B", "pdb_coverage": 0.6},
        {"pdbs": "1abc:P12345", "pdb_type": "experimental", "pdb_chain": "A", "pdb_coverage": 0.9},
    ]
    result = resolver.resolve({"uniprot_id": "P12345"}, [candidates])
    assert result["resolved_structure_status"] == "selected_experimental"
    assert (result["pdbs"], result["chain_id"]) == ("1ABC:P12345", "A")
    assert Path(result["structure_path"]) == tmp_path / "manifests" / "_chains" / "1ABC_A.pdb"
    resolver.extract_pending(workers=1)
    assert Path(result["structure_path"]).exists()


def test_resolver_falls_back_to_alphafold_then_esm_by_row_accession(tmp_path):
    mmcif, alphafold, esm = (tmp_path / name for name in ("mmcif", "af", "esm"))
    for folder in (mmcif, alphafold, esm):
        folder.mkdir()
    (alphafold / "AF-P12345-F1-model_v1.pdb").write_text("")
    (alphafold / "AF-P12345-F1-model_v6.pdb").write_text("")
    (esm / "ESMFold2-P12345.pdb").write_text("")
    (esm / "ESMFold2-Q99999|A1B.pdb").write_text("")
    resolver = StructureResolver(_args(tmp_path, mmcif, alphafold, esm))
    experimental_only = [{"pdbs": "1ABC:P12345", "pdb_type": "experimental", "pdb_chain": "A", "pdb_coverage": 0.9}]
    result = resolver.resolve({"uniprot_id": "P12345|X1Y"}, [experimental_only])
    assert result["resolved_structure_status"] == "selected_alphafold"
    assert result["structure_path"].endswith("AF-P12345-F1-model_v6.pdb")
    result = resolver.resolve({"uniprot_id": "Q99999|A1B"}, [[]])
    assert (result["resolved_structure_status"], result["pdbs"]) == ("selected_esm", "Q99999|A1B")
    assert resolver.resolve({"uniprot_id": "Z00000"}, [[]])["resolved_structure_status"] == "unresolved"
