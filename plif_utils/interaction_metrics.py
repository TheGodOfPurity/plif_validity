"""Utilities to score generated ligands against a reference interaction pattern.

This module is designed for the common workflow:
1. load one protein PDB file,
2. load one reference ligand SDF file,
3. iterate over a directory of generated ligand SDF files,
4. compute interaction recovery and PLIF similarity for each generated ligand.

Two backends are provided:
- ProLIF: uses the same preparation logic as the rest of this repository.
- PLIP: optional backend that uses the installed ``plip`` Python package directly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Iterator, Literal, Sequence
import warnings
import xml.etree.ElementTree as ET

import pandas as pd
import prolif as plf
from rdkit import Chem

from .settings import settings
from .system_prep import SystemPrep

SimilarityMetric = Literal["tanimoto", "dice"]
VectorMode = Literal["binary", "count"]

PLIP_TAGS = {
    "hydrophobic_interaction": "Hydrophobic",
    "hydrogen_bond": "Hbond",
    "water_bridge": "WaterBridge",
    "salt_bridge": "SaltBridge",
    "pi_stack": "PiStacking",
    "pi_cation_interaction": "PiCation",
    "halogen_bond": "HalogenBond",
    "metal_complex": "MetalComplex",
}


@dataclass(frozen=True)
class LigandRecord:
    """One molecule entry originating from an SDF file."""

    source_file: str
    molecule_index: int
    name: str
    mol: Chem.Mol


@dataclass(frozen=True)
class SkippedMoleculeRecord:
    """One skipped molecule entry originating from an SDF file."""

    source_file: str
    molecule_index: int
    name: str
    error: str


@dataclass(frozen=True)
class InteractionScores:
    """Comparison metrics between a reference and a query interaction pattern."""

    source_file: str
    molecule_index: int
    name: str
    backend: str
    num_reference_features: int
    num_query_features: int
    num_shared_features: int
    binary_recovery: float | None
    count_recovery: float | None
    tanimoto: float
    dice: float
    tanimoto_count: float
    dice_count: float
    reference_feature_string: str
    query_feature_string: str
    error: str = ""


def _require_3d_molecule(mol: Chem.Mol, label: str) -> Chem.Mol:
    if mol is None:
        raise ValueError(f"{label} is empty")
    if mol.GetNumConformers() == 0:
        raise ValueError(f"{label} does not contain 3D coordinates")
    return mol


def _ensure_explicit_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(mol)
    if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        mol = Chem.AddHs(mol, addCoords=True)
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    return mol


def _annotate_ligand_residue(
    mol: Chem.Mol,
    residue_name: str = "LIG",
    residue_number: int = 1,
    chain_id: str = "L",
) -> Chem.Mol:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        if atom.GetPDBResidueInfo() is None:
            info = Chem.AtomPDBResidueInfo(
                f" {atom.GetSymbol():<3.3}",
                residueName=residue_name,
                residueNumber=residue_number,
                chainId=chain_id,
            )
            atom.SetMonomerInfo(info)
    return mol


def rdkit_to_prolif_ligand(
    mol: Chem.Mol,
    residue_name: str = "LIG",
    residue_number: int = 1,
    chain_id: str = "L",
) -> plf.Molecule:
    """Convert an RDKit molecule into a ProLIF ligand with explicit hydrogens."""

    mol = _require_3d_molecule(mol, "Ligand")
    mol = _ensure_explicit_hydrogens(mol)
    mol = _annotate_ligand_residue(
        mol,
        residue_name=residue_name,
        residue_number=residue_number,
        chain_id=chain_id,
    )
    return plf.Molecule(mol)


def _fallback_molecule_name(block: str, sdf_path: Path, idx: int) -> str:
    first_line = block.splitlines()[0].strip() if block.splitlines() else ""
    return first_line or f"{sdf_path.stem}_{idx}"


def iter_sdf_molecules(
    sdf_file: str | Path,
    *,
    verbose: bool = False,
) -> Iterator[LigandRecord]:
    """Yield all valid molecules from an SDF file.

    Invalid records are skipped so that one broken generated molecule does not stop
    the whole batch.
    """

    sdf_path = Path(sdf_file)
    try:
        content = sdf_path.read_text(errors="ignore")
    except OSError:
        return

    skipped = 0
    for idx, block in enumerate(content.split("$$$$")):
        if not block.strip():
            continue

        # Keep leading blank header lines intact. They are legal in mol blocks and
        # removing them shifts the counts line away from the required line 4.
        block = block.rstrip("\r\n")
        block_for_parser = f"{block}\n"
        name = _fallback_molecule_name(block, sdf_path, idx)

        mol = None
        last_error = ""
        for sanitize in (True, False):
            try:
                mol = Chem.MolFromMolBlock(
                    block_for_parser,
                    sanitize=sanitize,
                    removeHs=False,
                    strictParsing=False,
                )
            except Exception as exc:
                last_error = str(exc)
                mol = None
            if mol is not None:
                break

        if mol is None:
            skipped += 1
            if verbose:
                warnings.warn(
                    f"Skipping invalid molecule block {idx} from {sdf_path}: "
                    f"{last_error or 'RDKit returned None'}"
                )
            continue

        try:
            parsed_name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        except Exception:
            parsed_name = ""
        if parsed_name:
            name = parsed_name

        yield LigandRecord(
            source_file=str(sdf_path),
            molecule_index=idx,
            name=name,
            mol=mol,
        )

    if verbose and skipped:
        warnings.warn(f"Skipped {skipped} invalid molecule blocks from {sdf_path}")


def iter_generated_molecules(
    sdf_path_or_dir: str | Path,
    *,
    verbose: bool = False,
) -> Iterator[LigandRecord]:
    """Yield molecules from a single SDF file or from all SDF files in a directory."""

    input_path = Path(sdf_path_or_dir)
    if input_path.is_file():
        yield from iter_sdf_molecules(input_path, verbose=verbose)
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    for sdf_file in sorted(input_path.rglob("*.sdf")):
        yield from iter_sdf_molecules(sdf_file, verbose=verbose)


def build_reference_pocket(
    protein_pdb: str | Path,
    reference_sdf: str | Path,
    system_prep: SystemPrep | None = None,
) -> tuple[plf.Molecule, plf.Molecule]:
    """Prepare the reference ligand and the protein pocket used for scoring."""

    prep = system_prep or SystemPrep()
    reference_ligand = prep.ligand_from_file(str(reference_sdf), add_hydrogens=True)

    attempts = [
        prep,
        replace(prep, sanitize=False),
        replace(prep, sanitize=False, non_standard_method=None),
    ]
    last_error: Exception | None = None
    for pocket_prep in attempts:
        try:
            protein_pocket = pocket_prep.pocket_from_pdb(str(protein_pdb), reference_ligand)
            return reference_ligand, protein_pocket
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Failed to prepare the reference protein pocket even after relaxed fallback modes."
    ) from last_error


def compute_prolif_fingerprint(
    ligand: plf.Molecule,
    protein: plf.Molecule,
    interactions: Sequence[str] | None = None,
    interaction_parameters: dict[str, dict] | None = None,
    count: bool = True,
) -> plf.Fingerprint:
    """Compute one ProLIF fingerprint for a ligand against a prepared protein object."""

    fp = plf.Fingerprint(
        interactions=list(interactions or settings.interactions),
        parameters=dict(interaction_parameters or settings.interaction_parameters),
        count=count,
    )
    fp.run_from_iterable([ligand], protein, progress=False)
    return fp


def prolif_fingerprint_to_counts(fp: plf.Fingerprint) -> dict[str, int]:
    """Convert a ProLIF fingerprint into a flat feature-count dictionary."""

    if not fp.ifp or not fp.ifp.get(0):
        return {}

    frame_df = plf.to_dataframe(fp.ifp, fp.interactions, count=True)
    if frame_df.empty:
        return {}

    counts: dict[str, int] = {}
    row = frame_df.iloc[0]
    for column, value in row.items():
        if not value:
            continue
        if isinstance(column, tuple) and len(column) >= 3:
            _, protein_residue, interaction_type = column[:3]
            feature = f"{protein_residue}|{interaction_type}"
        else:
            feature = str(column)
        counts[feature] = int(value)
    return counts


def feature_counts_to_string(feature_counts: dict[str, int]) -> str:
    """Serialize a feature dictionary for logging or CSV export."""

    return "/".join(
        f"{feature}={count}"
        for feature, count in sorted(feature_counts.items(), key=lambda item: item[0])
    )


def binary_recovery_rate(
    reference_counts: dict[str, int],
    query_counts: dict[str, int],
) -> float | None:
    """Fraction of reference interaction features recovered by the query."""

    reference_features = {feature for feature, count in reference_counts.items() if count > 0}
    if not reference_features:
        return None
    query_features = {feature for feature, count in query_counts.items() if count > 0}
    return len(reference_features & query_features) / len(reference_features)


def count_recovery_rate(
    reference_counts: dict[str, int],
    query_counts: dict[str, int],
) -> float | None:
    """Count-aware recovery used by the repository's original PLIF analysis."""

    if not reference_counts:
        return None
    recovered = sum(
        min(reference_count, query_counts.get(feature, 0))
        for feature, reference_count in reference_counts.items()
    )
    total_reference = sum(reference_counts.values())
    if total_reference == 0:
        return None
    return recovered / total_reference


def _binary_vectors(
    reference_counts: dict[str, int],
    query_counts: dict[str, int],
) -> tuple[list[int], list[int]]:
    keys = sorted(set(reference_counts) | set(query_counts))
    return (
        [1 if reference_counts.get(key, 0) > 0 else 0 for key in keys],
        [1 if query_counts.get(key, 0) > 0 else 0 for key in keys],
    )


def _count_vectors(
    reference_counts: dict[str, int],
    query_counts: dict[str, int],
) -> tuple[list[int], list[int]]:
    keys = sorted(set(reference_counts) | set(query_counts))
    return (
        [int(reference_counts.get(key, 0)) for key in keys],
        [int(query_counts.get(key, 0)) for key in keys],
    )


def vector_similarity(
    reference_counts: dict[str, int],
    query_counts: dict[str, int],
    *,
    metric: SimilarityMetric = "tanimoto",
    mode: VectorMode = "binary",
) -> float:
    """Compute a binary or count-based similarity between two interaction vectors."""

    if mode == "binary":
        ref_vec, qry_vec = _binary_vectors(reference_counts, query_counts)
    else:
        ref_vec, qry_vec = _count_vectors(reference_counts, query_counts)

    if not ref_vec:
        return 1.0

    dot_product = sum(left * right for left, right in zip(ref_vec, qry_vec))
    norm_ref = sum(value * value for value in ref_vec)
    norm_qry = sum(value * value for value in qry_vec)

    if metric == "tanimoto":
        denominator = norm_ref + norm_qry - dot_product
    elif metric == "dice":
        denominator = norm_ref + norm_qry
        dot_product *= 2
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if denominator == 0:
        return 1.0
    return dot_product / denominator


def score_feature_counts(
    reference_counts: dict[str, int],
    query_counts: dict[str, int],
    *,
    source_file: str,
    molecule_index: int,
    name: str,
    backend: str,
    error: str = "",
) -> InteractionScores:
    """Build one result record from a query feature dictionary."""

    reference_features = {feature for feature, count in reference_counts.items() if count > 0}
    query_features = {feature for feature, count in query_counts.items() if count > 0}
    shared_features = reference_features & query_features
    return InteractionScores(
        source_file=source_file,
        molecule_index=molecule_index,
        name=name,
        backend=backend,
        num_reference_features=len(reference_features),
        num_query_features=len(query_features),
        num_shared_features=len(shared_features),
        binary_recovery=binary_recovery_rate(reference_counts, query_counts),
        count_recovery=count_recovery_rate(reference_counts, query_counts),
        tanimoto=vector_similarity(reference_counts, query_counts, metric="tanimoto", mode="binary"),
        dice=vector_similarity(reference_counts, query_counts, metric="dice", mode="binary"),
        tanimoto_count=vector_similarity(
            reference_counts,
            query_counts,
            metric="tanimoto",
            mode="count",
        ),
        dice_count=vector_similarity(reference_counts, query_counts, metric="dice", mode="count"),
        reference_feature_string=feature_counts_to_string(reference_counts),
        query_feature_string=feature_counts_to_string(query_counts),
        error=error,
    )


def evaluate_generated_molecules_prolif(
    protein_pdb: str | Path,
    reference_sdf: str | Path,
    generated_sdf_path: str | Path,
    *,
    system_prep: SystemPrep | None = None,
    interactions: Sequence[str] | None = None,
    interaction_parameters: dict[str, dict] | None = None,
    raise_on_error: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Score a directory of generated ligands with the ProLIF backend."""

    reference_ligand, protein_pocket = build_reference_pocket(
        protein_pdb=protein_pdb,
        reference_sdf=reference_sdf,
        system_prep=system_prep,
    )
    reference_fp = compute_prolif_fingerprint(
        reference_ligand,
        protein_pocket,
        interactions=interactions,
        interaction_parameters=interaction_parameters,
        count=True,
    )
    reference_counts = prolif_fingerprint_to_counts(reference_fp)

    results: list[InteractionScores] = []
    for record in iter_generated_molecules(generated_sdf_path, verbose=verbose):
        try:
            ligand = rdkit_to_prolif_ligand(record.mol)
            query_fp = compute_prolif_fingerprint(
                ligand,
                protein_pocket,
                interactions=interactions,
                interaction_parameters=interaction_parameters,
                count=True,
            )
            query_counts = prolif_fingerprint_to_counts(query_fp)
            results.append(
                score_feature_counts(
                    reference_counts,
                    query_counts,
                    source_file=record.source_file,
                    molecule_index=record.molecule_index,
                    name=record.name,
                    backend="prolif",
                )
            )
        except Exception as exc:
            if raise_on_error:
                raise
            results.append(
                score_feature_counts(
                    reference_counts,
                    {},
                    source_file=record.source_file,
                    molecule_index=record.molecule_index,
                    name=record.name,
                    backend="prolif",
                    error=str(exc),
                )
            )

    if verbose and not results:
        warnings.warn(
            f"No valid generated molecules were loaded from {generated_sdf_path}"
        )
    return pd.DataFrame(asdict(result) for result in results)


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _find_child_text(element: ET.Element, *candidate_tags: str) -> str:
    for child in element:
        tag = _strip_namespace(child.tag)
        if tag in candidate_tags and child.text:
            return child.text.strip()
    return ""


def _normalise_plip_interaction_tag(tag: str) -> str:
    return PLIP_TAGS.get(tag, tag)


def parse_plip_xml_report(xml_file: str | Path) -> dict[str, int]:
    """Parse a PLIP XML report into the same flat feature-count representation."""

    root = ET.parse(xml_file).getroot()
    counts: Counter[str] = Counter()

    for element in root.iter():
        tag = _strip_namespace(element.tag)
        if tag not in PLIP_TAGS:
            continue

        residue_name = _find_child_text(element, "restype", "resname")
        residue_number = _find_child_text(element, "resnr", "resnum")
        chain_id = _find_child_text(element, "reschain", "chain")
        location = _find_child_text(element, "sidechain")

        residue_token = f"{residue_name}{residue_number}"
        if chain_id:
            residue_token = f"{chain_id}:{residue_token}"
        if not residue_token.strip(":"):
            residue_token = "UNK"

        feature = f"{residue_token}|{_normalise_plip_interaction_tag(tag)}"
        if location:
            feature = f"{feature}|{location}"
        counts[feature] += 1

    return dict(counts)


def _protein_pdb_without_trailer(protein_pdb: str | Path) -> str:
    lines = Path(protein_pdb).read_text().splitlines()
    kept = [line for line in lines if not line.startswith(("END", "CONECT", "MASTER"))]
    return "".join(f"{line}\n" for line in kept)


def _ligand_to_pdb_block(mol: Chem.Mol, residue_name: str = "LIG") -> str:
    mol = _require_3d_molecule(mol, "Ligand")
    mol = _ensure_explicit_hydrogens(mol)
    mol = _annotate_ligand_residue(mol, residue_name=residue_name)
    return Chem.MolToPDBBlock(mol)


def write_complex_pdb(
    protein_pdb: str | Path,
    ligand_mol: Chem.Mol,
    output_pdb: str | Path,
    *,
    residue_name: str = "LIG",
) -> Path:
    """Create a temporary protein-ligand complex PDB for PLIP."""

    output_path = Path(output_pdb)
    protein_block = _protein_pdb_without_trailer(protein_pdb)
    ligand_block = _ligand_to_pdb_block(ligand_mol, residue_name=residue_name)
    output_path.write_text(f"{protein_block}TER\n{ligand_block}END\n")
    return output_path


def _find_plip_xml(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    for candidate in output_path.rglob("report.xml"):
        return candidate
    xml_files = sorted(output_path.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No PLIP XML report found under {output_path}")
    return xml_files[0]


def _safe_getattr(obj: object, *names: str) -> str:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return str(value)
    return ""


def _iter_plip_interaction_objects(interaction_set: object) -> Iterator[tuple[str, object]]:
    attr_to_type = (
        ("hydrophobic_contacts", "Hydrophobic"),
        ("hbonds_ldon", "Hbond"),
        ("hbonds_pdon", "Hbond"),
        ("water_bridges", "WaterBridge"),
        ("saltbridge_lneg", "SaltBridge"),
        ("saltbridge_pneg", "SaltBridge"),
        ("pistacking", "PiStacking"),
        ("pication_laro", "PiCation"),
        ("pication_paro", "PiCation"),
        ("halogen_bonds", "HalogenBond"),
        ("metal_complexes", "MetalComplex"),
    )

    seen_ids: set[int] = set()
    for attr_name, interaction_type in attr_to_type:
        interactions = getattr(interaction_set, attr_name, None)
        if not interactions:
            continue
        for interaction in interactions:
            seen_ids.add(id(interaction))
            yield interaction_type, interaction

    for interaction in getattr(interaction_set, "all_itypes", []) or []:
        if id(interaction) in seen_ids:
            continue
        interaction_type = _safe_getattr(
            interaction,
            "itype",
            "type",
            "__class__",
        )
        if interaction_type.startswith("<class"):
            interaction_type = interaction.__class__.__name__
        yield interaction_type, interaction


def _plip_interaction_to_feature(interaction_type: str, interaction: object) -> str:
    residue_name = _safe_getattr(interaction, "restype", "resname")
    residue_number = _safe_getattr(interaction, "resnr", "resnum")
    chain_id = _safe_getattr(interaction, "reschain", "chain")
    sidechain = _safe_getattr(interaction, "sidechain")

    residue_token = f"{residue_name}{residue_number}"
    if chain_id:
        residue_token = f"{chain_id}:{residue_token}"
    if not residue_token.strip(":"):
        residue_token = "UNK"

    feature = f"{residue_token}|{interaction_type}"
    if sidechain:
        feature = f"{feature}|{sidechain}"
    return feature


def _select_plip_interaction_set(analyzer: object, residue_name: str = "LIG") -> object:
    interaction_sets = getattr(analyzer, "interaction_sets", None)
    if not interaction_sets:
        ligand_labels = []
        for ligand in getattr(analyzer, "ligands", []) or []:
            ligand_labels.append(
                ":".join(
                    filter(
                        None,
                        [
                            _safe_getattr(ligand, "hetid"),
                            _safe_getattr(ligand, "chain"),
                            _safe_getattr(ligand, "position"),
                            _safe_getattr(ligand, "longname"),
                        ],
                    )
                )
            )
        raise ValueError(
            "PLIP did not produce any interaction set. "
            f"Detected ligands={ligand_labels or '[]'}"
        )

    for ligand in getattr(analyzer, "ligands", []) or []:
        hetid = _safe_getattr(ligand, "hetid")
        chain = _safe_getattr(ligand, "chain")
        position = _safe_getattr(ligand, "position")
        bsid = ":".join(filter(None, [hetid, chain, position]))
        if hetid == residue_name and bsid in interaction_sets:
            return interaction_sets[bsid]

    for bsid, interaction_set in interaction_sets.items():
        if residue_name in str(bsid):
            return interaction_set

    return next(iter(interaction_sets.values()))


def compute_plip_feature_counts_from_complex(
    complex_pdb: str | Path,
    *,
    ligand_residue_name: str = "LIG",
    verbose: bool = False,
) -> dict[str, int]:
    """Use the installed PLIP Python package to extract interaction feature counts."""

    try:
        from plip.structure.preparation import PDBComplex
    except ImportError as exc:
        raise ImportError(
            "The PLIP Python package is required for the PLIP backend."
        ) from exc

    analyzer = PDBComplex()
    analyzer.output_path = str(Path(complex_pdb).parent)
    analyzer.load_pdb(str(complex_pdb))
    analyzer.analyze()

    if verbose:
        ligand_labels = []
        for ligand in getattr(analyzer, "ligands", []) or []:
            ligand_labels.append(
                ":".join(
                    filter(
                        None,
                        [
                            _safe_getattr(ligand, "hetid"),
                            _safe_getattr(ligand, "chain"),
                            _safe_getattr(ligand, "position"),
                            _safe_getattr(ligand, "longname"),
                        ],
                    )
                )
            )
        warnings.warn(
            "PLIP debug: "
            f"ligands={ligand_labels or []}, "
            f"interaction_sets={list(getattr(analyzer, 'interaction_sets', {}).keys())}"
        )

    interaction_set = _select_plip_interaction_set(
        analyzer,
        residue_name=ligand_residue_name,
    )
    counts: Counter[str] = Counter()
    for interaction_type, interaction in _iter_plip_interaction_objects(interaction_set):
        feature = _plip_interaction_to_feature(interaction_type, interaction)
        counts[feature] += 1
    return dict(counts)


def compute_plip_feature_counts(
    protein_pdb: str | Path,
    ligand_mol: Chem.Mol,
    *,
    ligand_residue_name: str = "LIG",
    verbose: bool = False,
) -> dict[str, int]:
    """Run PLIP on one protein-ligand pair and return flat interaction feature counts."""

    with TemporaryDirectory(prefix="plip_score_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        complex_pdb = write_complex_pdb(
            protein_pdb=protein_pdb,
            ligand_mol=ligand_mol,
            output_pdb=tmpdir_path / "complex.pdb",
            residue_name=ligand_residue_name,
        )
        return compute_plip_feature_counts_from_complex(
            complex_pdb=complex_pdb,
            ligand_residue_name=ligand_residue_name,
            verbose=verbose,
        )


def load_first_molecule_from_sdf(sdf_file: str | Path) -> Chem.Mol:
    """Load the first valid molecule from an SDF file."""

    for record in iter_sdf_molecules(sdf_file):
        return record.mol
    raise ValueError(f"No valid molecule found in {sdf_file}")


def evaluate_generated_molecules_plip(
    protein_pdb: str | Path,
    reference_sdf: str | Path,
    generated_sdf_path: str | Path,
    *,
    ligand_residue_name: str = "LIG",
    raise_on_error: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Score a directory of generated ligands with the PLIP backend."""

    reference_mol = load_first_molecule_from_sdf(reference_sdf)
    reference_counts = compute_plip_feature_counts(
        protein_pdb=protein_pdb,
        ligand_mol=reference_mol,
        ligand_residue_name=ligand_residue_name,
        verbose=verbose,
    )

    results: list[InteractionScores] = []
    for record in iter_generated_molecules(generated_sdf_path, verbose=verbose):
        try:
            query_counts = compute_plip_feature_counts(
                protein_pdb=protein_pdb,
                ligand_mol=record.mol,
                ligand_residue_name=ligand_residue_name,
                verbose=verbose,
            )
            results.append(
                score_feature_counts(
                    reference_counts,
                    query_counts,
                    source_file=record.source_file,
                    molecule_index=record.molecule_index,
                    name=record.name,
                    backend="plip",
                )
            )
        except Exception as exc:
            if raise_on_error:
                raise
            results.append(
                score_feature_counts(
                    reference_counts,
                    {},
                    source_file=record.source_file,
                    molecule_index=record.molecule_index,
                    name=record.name,
                    backend="plip",
                    error=str(exc),
                )
            )

    if verbose and not results:
        warnings.warn(
            f"No valid generated molecules were loaded from {generated_sdf_path}"
        )
    return pd.DataFrame(asdict(result) for result in results)
