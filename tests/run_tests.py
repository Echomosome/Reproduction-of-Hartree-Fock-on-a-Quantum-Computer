#!/usr/bin/env python3
"""Deterministic end-to-end tests for the H4 reproduction package."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from h4_core import (  # noqa: E402
    N_PARTICLES_PER_SPIN,
    analyzer_matrix,
    gamma_from_native_circuit,
    load_distribution,
    load_reference,
    parse_gaussian_scf_energies,
    parse_native_circuit,
    postselect_particle_number,
    rank_projector,
    reconstruct_gamma,
    rhf_energy_components,
    slater_probabilities_from_circuit,
)


class H4ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = load_reference(
            ROOT / "reference/h4_r1.3000_sto3g_reference.npz"
        )
        cls.target = np.asarray(
            cls.reference["gamma_one_spin_orth"], dtype=float
        )
        cls.manifest = json.loads(
            (ROOT / "circuits/manifest.json").read_text(encoding="utf-8")
        )

    def test_reference_energy_closure(self) -> None:
        energy = rhf_energy_components(
            self.target,
            self.reference["h1_orth"],
            self.reference["eri_orth"],
            float(self.reference["nuclear_repulsion"]),
        )["total_hartree"]
        self.assertLess(
            abs(energy - float(self.reference["rhf_energy"])), 1e-10
        )
        self.assertLess(
            np.linalg.norm(self.target @ self.target - self.target), 1e-10
        )
        self.assertAlmostEqual(
            float(np.trace(self.target)),
            N_PARTICLES_PER_SPIN,
            places=10,
        )

    def test_measurement_pairs_cover_complete_graph(self) -> None:
        pairs: set[tuple[int, int]] = set()
        for circuit in self.manifest["circuits"]:
            for mapping in circuit["measurement_map"]:
                pair = tuple(
                    sorted((mapping["logical_i"], mapping["logical_j"]))
                )
                pairs.add(pair)
        self.assertEqual(
            pairs,
            {
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
            },
        )

    def test_all_native_circuits_close(self) -> None:
        for circuit in self.manifest["circuits"]:
            path = ROOT / "circuits" / circuit["gate_body_file"]
            gates = parse_native_circuit(path.read_text(encoding="utf-8"))
            gamma = gamma_from_native_circuit(gates)
            if circuit["kind"] == "diagonal":
                analyzer = np.eye(4)
            else:
                logical_pairs = [
                    (entry["logical_i"], entry["logical_j"])
                    for entry in circuit["measurement_map"]
                ]
                analyzer, _ = analyzer_matrix(logical_pairs)
            expected = analyzer @ self.target @ analyzer.T
            self.assertLess(np.max(np.abs(gamma - expected)), 1e-10)
            probabilities = slater_probabilities_from_circuit(gates)
            self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=12)
            self.assertTrue(
                all(bitstring.count("1") == 2 for bitstring in probabilities)
            )

    def test_exact_json_reconstructs_energy(self) -> None:
        distributions: dict[str, dict[str, float]] = {}
        for circuit in self.manifest["circuits"]:
            loaded = load_distribution(
                ROOT
                / "data/example_ideal_probabilities"
                / f"{circuit['setting']}.json"
            )
            selected, pass_probability = postselect_particle_number(
                loaded.probabilities
            )
            self.assertAlmostEqual(pass_probability, 1.0, places=12)
            distributions[circuit["setting"]] = selected
        reconstructed = reconstruct_gamma(distributions, self.manifest)
        self.assertLess(
            np.max(np.abs(reconstructed - self.target)), 1e-10
        )
        energy = rhf_energy_components(
            reconstructed,
            self.reference["h1_orth"],
            self.reference["eri_orth"],
            float(self.reference["nuclear_repulsion"]),
        )["total_hartree"]
        self.assertLess(
            abs(energy - float(self.reference["rhf_energy"])), 1e-10
        )

    def test_rank_two_projection_and_gaussian_parser(self) -> None:
        perturbed = self.target.copy()
        perturbed[0, 0] += 0.03
        perturbed[0, 2] -= 0.02
        perturbed[2, 0] -= 0.02
        projected = rank_projector(perturbed, 2)
        occupations = np.linalg.eigvalsh(projected)
        self.assertTrue(
            np.allclose(occupations, [0.0, 0.0, 1.0, 1.0], atol=1e-10)
        )
        self.assertLess(
            np.linalg.norm(projected @ projected - projected), 1e-10
        )
        sample = (
            " SCF Done:  E(RHF) =  -1.900000000000     A.U.\n"
            " SCF Done:  E(RHF) =  -1.946741626456     A.U.\n"
        )
        self.assertEqual(
            parse_gaussian_scf_energies(sample),
            [-1.9, -1.946741626456],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
