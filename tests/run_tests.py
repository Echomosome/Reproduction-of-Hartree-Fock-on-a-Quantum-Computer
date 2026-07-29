#!/usr/bin/env python3
"""Deterministic tests for the H2 reproduction package."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from h2_core import (  # noqa: E402
    gamma_from_native_circuit,
    load_reference,
    native_givens_lines,
    one_particle_probabilities_from_circuit,
    parse_gaussian_scf_energies,
    parse_native_circuit,
    real_coherence_analyzer_lines,
    reconstruct_real_gamma,
    rhf_energy_components,
)


def gates_from_lines(lines: list[str]):
    return parse_native_circuit("\n".join(lines))


def test_native_givens() -> None:
    theta = 0.371
    gates = gates_from_lines(["X q[0]", *native_givens_lines(0, 1, theta)])
    gamma = gamma_from_native_circuit(gates)
    orbital = np.asarray([math.cos(theta), math.sin(theta)])
    target = np.outer(orbital, orbital)
    np.testing.assert_allclose(gamma, target, atol=2e-12)


def test_two_setting_reconstruction() -> None:
    reference = load_reference(
        ROOT / "reference/h2_r0.7414_sto3g_reference.npz"
    )
    theta = float(reference["givens_theta_radians"])
    prep = ["X q[0]", *native_givens_lines(0, 1, theta)]
    diagonal = one_particle_probabilities_from_circuit(
        gates_from_lines(prep)
    )
    real = one_particle_probabilities_from_circuit(
        gates_from_lines(
            [*prep, *real_coherence_analyzer_lines(0, 1)]
        )
    )
    reconstructed = reconstruct_real_gamma(diagonal, real, "q1q0")
    np.testing.assert_allclose(
        reconstructed,
        reference["gamma_one_spin_orth"],
        atol=2e-12,
    )


def test_energy_closure() -> None:
    reference = load_reference(
        ROOT / "reference/h2_r0.7414_sto3g_reference.npz"
    )
    energy = rhf_energy_components(
        reference["gamma_one_spin_orth"],
        reference["h1_orth"],
        reference["eri_orth"],
        float(reference["nuclear_repulsion"]),
    )["total_hartree"]
    assert abs(energy - float(reference["rhf_energy"])) < 1e-10


def test_gaussian_parser() -> None:
    text = (
        " SCF Done:  E(RHF) =  -1.11668438709     "
        "A.U. after    7 cycles\n"
    )
    assert parse_gaussian_scf_energies(text) == [-1.11668438709]


def main() -> None:
    tests = [
        test_native_givens,
        test_two_setting_reconstruction,
        test_energy_closure,
        test_gaussian_parser,
    ]
    for test in tests:
        test()
        print(f"ok - {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
