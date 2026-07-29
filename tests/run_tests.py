#!/usr/bin/env python3
"""Deterministic validation suite for the packaged diazene workflow."""

from __future__ import annotations

import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from diazene_core import (  # noqa: E402
    DEFAULT_BIT_ORDER,
    N_MODES,
    N_PARTICLES_PER_SPIN,
    analyzer_matrix,
    complete_graph_matchings,
    gamma_from_native_circuit,
    geometry_metrics,
    load_distribution,
    load_geometries,
    load_reference,
    parse_gaussian_scf_energies,
    parse_native_circuit,
    postselect_particle_number,
    reconstruct_gamma,
    rhf_energy_components,
)


def test_geometry_sources() -> None:
    geometries = load_geometries(
        PROJECT_ROOT / "data/geometries_paper.json"
    )
    assert len(geometries) == 18
    assert sum(g["pathway"] == "out_of_plane" for g in geometries) == 9
    assert sum(g["pathway"] == "in_plane" for g in geometries) == 9
    for geometry in geometries:
        metrics = geometry_metrics(geometry)
        for key, value in metrics.items():
            assert math.isfinite(value), (geometry["id"], key, value)
        if geometry["pathway"] == "out_of_plane":
            computed = metrics[
                "dihedral_h1_n1_n2_h2_0_to_360_deg"
            ]
            expected = geometry["reaction_coordinate_deg"] % 360.0
            assert abs(computed - expected) < 2.0e-3, (
                geometry["id"],
                computed,
                expected,
            )
        else:
            # Appendix J's in-plane label is a path coordinate, not the
            # H-N-N-H dihedral.  Its printed structures must be planar.
            assert max(
                abs(atom["xyz_angstrom"][0])
                for atom in geometry["atoms"]
            ) < 2.0e-3


def test_reference_closure() -> None:
    geometries = load_geometries(
        PROJECT_ROOT / "data/geometries_paper.json"
    )
    for geometry in geometries:
        reference = load_reference(
            PROJECT_ROOT / "reference" / f"{geometry['id']}.npz"
        )
        gamma = np.asarray(
            reference["gamma_active_one_spin"], dtype=float
        )
        assert gamma.shape == (N_MODES, N_MODES)
        assert abs(np.trace(gamma) - N_PARTICLES_PER_SPIN) < 1.0e-10
        assert np.linalg.norm(gamma @ gamma - gamma) < 1.0e-9
        energy = rhf_energy_components(
            gamma,
            reference["h1_active"],
            reference["eri_active"],
            float(reference["constant_offset_hartree"]),
        )["total_hartree"]
        assert abs(
            energy - float(reference["active_rhf_energy_hartree"])
        ) < 1.0e-10
        assert abs(
            float(reference["active_rhf_energy_hartree"])
            - float(reference["full_rhf_energy_hartree"])
        ) < 2.0e-6
        electron_count = np.einsum(
            "pq,qp->",
            reference["full_density_ao_spin_summed"],
            reference["overlap_ao"],
        )
        assert abs(electron_count - 16.0) < 1.0e-8


def test_measurement_cover() -> None:
    matchings = complete_graph_matchings(N_MODES)
    assert len(matchings) == N_MODES - 1
    assert all(len(matching) == N_MODES // 2 for matching in matchings)
    edges = [tuple(pair) for matching in matchings for pair in matching]
    assert len(edges) == 45
    assert set(edges) == set(combinations(range(N_MODES), 2))


def test_native_circuits() -> None:
    scan_manifest = json.loads(
        (PROJECT_ROOT / "circuits/scan_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(scan_manifest["geometries"]) == 18
    checked = 0
    worst_error = 0.0
    for geometry in scan_manifest["geometries"]:
        geometry_id = geometry["geometry_id"]
        geometry_dir = PROJECT_ROOT / "circuits" / geometry_id
        manifest = json.loads(
            (geometry_dir / "manifest.json").read_text(encoding="utf-8")
        )
        reference = load_reference(
            PROJECT_ROOT / "reference" / f"{geometry_id}.npz"
        )
        target = np.asarray(
            reference["gamma_active_one_spin"], dtype=float
        )
        assert len(manifest["circuits"]) == 10
        assert (
            manifest["gate_conventions"]["bitstring_order"]
            == DEFAULT_BIT_ORDER
        )
        for circuit in manifest["circuits"]:
            gates = parse_native_circuit(
                (
                    geometry_dir / circuit["gate_body_file"]
                ).read_text(encoding="utf-8")
            )
            x_modes = sorted(
                gate.qubits[0] for gate in gates if gate.name == "X"
            )
            assert x_modes == list(range(N_PARTICLES_PER_SPIN))
            assert all(
                abs(gate.qubits[0] - gate.qubits[1]) == 1
                for gate in gates
                if gate.name == "SQISWAP"
            )
            if circuit["kind"] == "diagonal":
                analyzer = np.eye(N_MODES)
            else:
                analyzer, _ = analyzer_matrix(
                    [tuple(pair) for pair in circuit["logical_pairs"]]
                )
            expected = analyzer @ target @ analyzer.T
            actual = gamma_from_native_circuit(gates)
            error = float(np.max(np.abs(actual - expected)))
            worst_error = max(worst_error, error)
            assert error < 1.0e-8
            checked += 1
    assert checked == 180
    print(f"  circuit worst closure: {worst_error:.3e}")


def test_exact_data_reconstruction() -> None:
    geometries = load_geometries(
        PROJECT_ROOT / "data/geometries_paper.json"
    )
    worst_gamma = 0.0
    worst_energy = 0.0
    for geometry in geometries:
        geometry_id = geometry["id"]
        manifest = json.loads(
            (
                PROJECT_ROOT / "circuits" / geometry_id / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        distributions: dict[str, dict[str, float]] = {}
        for circuit in manifest["circuits"]:
            setting = circuit["setting"]
            loaded = load_distribution(
                PROJECT_ROOT
                / "data/ideal"
                / geometry_id
                / f"{setting}.json"
            )
            selected, pass_probability = postselect_particle_number(
                loaded.probabilities
            )
            assert abs(pass_probability - 1.0) < 1.0e-12
            distributions[setting] = selected
        gamma = reconstruct_gamma(distributions, manifest)
        reference = load_reference(
            PROJECT_ROOT / "reference" / f"{geometry_id}.npz"
        )
        target = np.asarray(
            reference["gamma_active_one_spin"], dtype=float
        )
        gamma_error = float(np.max(np.abs(gamma - target)))
        energy = rhf_energy_components(
            gamma,
            reference["h1_active"],
            reference["eri_active"],
            float(reference["constant_offset_hartree"]),
        )["total_hartree"]
        energy_error = abs(
            energy - float(reference["active_rhf_energy_hartree"])
        )
        worst_gamma = max(worst_gamma, gamma_error)
        worst_energy = max(worst_energy, energy_error)
        assert gamma_error < 1.0e-8
        assert energy_error < 1.0e-9
    print(f"  exact-data worst gamma element: {worst_gamma:.3e}")
    print(f"  exact-data worst energy: {worst_energy:.3e} Ha")


def test_sample_and_error_decomposition() -> None:
    summary = json.loads(
        (
            PROJECT_ROOT
            / "results/sample_1000_shots_scan/scan_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["number_of_geometries"] == 18
    maximum_error = 0.0
    for row in summary["rows"]:
        maximum_error = max(
            maximum_error,
            abs(row["measurement_error_vs_active_millihartree"]),
        )
        reconstructed = (
            row["measurement_error_vs_active_millihartree"]
            + row["frozen_core_bias_millihartree"]
        )
        assert abs(
            reconstructed - row["end_to_end_error_vs_full_millihartree"]
        ) < 1.0e-8
        assert row["minimum_particle_number_pass_probability"] == 1.0
    assert maximum_error < 40.0
    assert abs(
        summary["pathway_statistics"]["full_pyscf_rhf"][
            "in_plane_minus_out_of_plane_ts_gap_millihartree"
        ]
        - (-32.85752987302715)
    ) < 1.0e-8


def test_gaussian_parser_and_provenance() -> None:
    sample = """
 SCF Done:  E(RHF) =  -1.0841813327307733D+02     A.U.
 Normal termination of Gaussian 16
"""
    assert parse_gaussian_scf_energies(sample) == [
        -108.41813327307733
    ]
    with (PROJECT_ROOT / "PARAMETER_PROVENANCE.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    names = {row["parameter"] for row in rows}
    required = {
        "geometry_set",
        "basis",
        "preliminary_scf_cycles",
        "frozen_spatial_orbitals",
        "qubits",
        "measurement_settings",
        "gaussian_route",
    }
    assert required <= names


TESTS = [
    test_geometry_sources,
    test_reference_closure,
    test_measurement_cover,
    test_native_circuits,
    test_exact_data_reconstruction,
    test_sample_and_error_decomposition,
    test_gaussian_parser_and_provenance,
]


def main() -> None:
    for test in TESTS:
        print(f"[RUN] {test.__name__}")
        test()
        print(f"[PASS] {test.__name__}")
    print(f"All {len(TESTS)} tests passed.")


if __name__ == "__main__":
    main()
