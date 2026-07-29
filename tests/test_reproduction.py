"""Deterministic checks for physical and file-format closure."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from h8hf.circuits import swap_network_matchings, verify_native_givens
from h8hf.data_processing import bitstring_to_occupations
from h8hf.gaussian import parse_scf_energy
from h8hf.integrals import build_sto3g_integrals, linear_h_chain
from h8hf.rhf import run_rhf


class H8ReproductionTests(unittest.TestCase):
    def test_sto3g_contracted_functions_are_normalized(self) -> None:
        bundle = build_sto3g_integrals(linear_h_chain(2, 1.3))
        np.testing.assert_allclose(np.diag(bundle.overlap), 1.0, atol=2e-8)

    def test_h4_energy_regression(self) -> None:
        bundle = build_sto3g_integrals(linear_h_chain(4, 1.3))
        result = run_rhf(
            bundle.h_core,
            bundle.overlap,
            bundle.eri,
            4,
            bundle.nuclear_repulsion,
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.total_energy, -1.9467416261, places=9)

    def test_native_givens_identity(self) -> None:
        for theta in (-2.4, -0.1, 0.0, 0.7, 2.8):
            self.assertLess(verify_native_givens(theta), 1.0e-12)

    def test_nine_settings_cover_every_offdiagonal_once(self) -> None:
        layers = swap_network_matchings(8)
        self.assertEqual(len(layers), 8)
        pairs = [
            tuple(sorted(pair)) for layer in layers for pair in layer
        ]
        self.assertEqual(len(pairs), 28)
        self.assertEqual(len(set(pairs)), 28)

    def test_bit_order(self) -> None:
        occupations = bitstring_to_occupations(
            "10000101", 8, "q[n-1]...q[0]"
        )
        np.testing.assert_array_equal(
            occupations, np.array([1, 0, 1, 0, 0, 0, 0, 1])
        )

    def test_gaussian_parser_uses_final_scf_done(self) -> None:
        text = (
            " SCF Done:  E(RHF) =  -3.800000000000     A.U.\n"
            " SCF Done:  E(RHF) =  -3.902579787288     A.U.\n"
        )
        self.assertEqual(parse_scf_energy(text), -3.902579787288)

    def test_generated_end_to_end_results(self) -> None:
        key_path = ROOT / "results" / "KEY_RESULTS.json"
        if not key_path.exists():
            self.skipTest("Run code/run_all.py before generated-result checks.")
        key = json.loads(key_path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            key["reference_total_energy_hartree"],
            -3.9025797873,
            places=9,
        )
        self.assertLess(abs(key["exact_energy_error_hartree"]), 1.0e-10)
        manifest = json.loads(
            (ROOT / "circuits" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["settings"]), 9)
        for setting in manifest["settings"]:
            self.assertEqual(setting["givens_count"], 16)
            self.assertLess(setting["fit_max_projector_error"], 1.0e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
