# Empirical optimizer for QrackAceBackend's BSEQ (Bell/CHSH) ancilla
# correction gate.
#
# The correction is a single, genuine controlled-U(theta, phi, lambda)
# gate (QrackSimulator.mcu), with each boundary qubit's ancilla as
# control and the boundary qubit itself as target. It is applied
# directly to the live quantum state -- before any probability is read
# -- every time the boundary qubit's state is corrected (prob() / m() /
# force_m() all funnel through this), so it composes correctly even
# while the qubit's Bell-pair partner is still genuinely superposed.
#
# The 3 angles (QrackAceBackend._BSEQ_THETA/_PHI/_LAMBDA) are the only
# trainable parameters. They are fit here by direct, gradient-free
# search against the empirically measured CHSH S statistic (or its 4
# components individually), averaged over repeated Bell-pair trials to
# average down the substantial per-construction measurement variance
# inherent to this architecture (each circuit construction makes its
# own one-time classical collapse choice early in the elision pipeline;
# averaging over MANY independent constructions is what makes the mean
# S a meaningful, optimizable target -- a single fixed gate cannot
# reduce per-shot variance, since that variance is set by an earlier,
# already-classical decision the gate cannot retroactively undo).
#
# Usage:
#   python3 optimize_bseq_correction.py [--iterations N] [--time-budget SECONDS]
#
# Writes the found angles directly into pyqrack's installed
# qrack_ace_backend.py (QrackAceBackend._BSEQ_THETA/_PHI/_LAMBDA), so
# every subsequently-constructed QrackAceBackend in the running
# process (and, if you choose to persist the file, future processes)
# uses the fitted correction automatically -- no separate model file,
# no extra inference dependency. The correction is 3 floats; baking
# them in as class attributes is already the most efficient possible
# packaging, and needs no JIT/torch artifact at inference time.

import argparse
import math
import random
import statistics
import time

from pyqrack import QrackAceBackend, Pauli
import pyqrack.qrack_ace_backend as _ace_mod

PI = math.pi

# Standard CHSH angle set.
THETA_A, THETA_AP = 0.0, PI / 2
THETA_B, THETA_BP = PI / 4, -PI / 4


def _apply_measurement_basis(sim, q, theta):
    sim.r(Pauli.PauliY, theta, q)


def _measure_expectation(sim, shots):
    results = sim.measure_shots([0, 1], shots)
    total = 0
    for res in results:
        a = (res >> 0) & 1
        b = (res >> 1) & 1
        total += 1 if a == b else -1
    return total / shots


def _expectation(theta1, theta2, shots, width=16, long_range_columns=1):
    """One fresh Bell-pair-with-ancilla construction, measured at a given
    basis-angle pair. This is exactly the bseq.py pattern: h(0), cx(0,1),
    then a PauliY rotation on each half before measuring."""
    s = QrackAceBackend(width, long_range_columns=long_range_columns)
    s.h(0)
    s.mcx([0], 1)
    _apply_measurement_basis(s, 0, theta1)
    _apply_measurement_basis(s, 1, theta2)
    return _measure_expectation(s, shots)


def compute_chsh_components(shots, **kwargs):
    """Return the 4 CHSH correlator terms (E(a,b), E(a,b'), E(a',b), E(a',b'))."""
    return (
        _expectation(THETA_A, THETA_B, shots, **kwargs),
        _expectation(THETA_A, THETA_BP, shots, **kwargs),
        _expectation(THETA_AP, THETA_B, shots, **kwargs),
        _expectation(THETA_AP, THETA_BP, shots, **kwargs),
    )


def compute_S(shots, **kwargs):
    e_ab, e_abp, e_apb, e_apbp = compute_chsh_components(shots, **kwargs)
    return e_ab + e_abp + e_apb - e_apbp


def set_correction(theta, phi, lam):
    _ace_mod.QrackAceBackend._BSEQ_THETA = theta
    _ace_mod.QrackAceBackend._BSEQ_PHI = phi
    _ace_mod.QrackAceBackend._BSEQ_LAMBDA = lam


def mean_reward(theta, phi, lam, n_repeats, shots, **kwargs):
    """The actual optimization target: mean S across many independent
    Bell-pair-with-ancilla constructions ("an RCS data set for Bell-pair
    perturbations of the ancillae"), not a single noisy reading."""
    set_correction(theta, phi, lam)
    vals = [compute_S(shots, **kwargs) for _ in range(n_repeats)]
    return statistics.mean(vals)


def optimize(
    n_repeats=4,
    shots=64,
    time_budget=200.0,
    max_iterations=None,
    seed=None,
    initial_step=PI / 4,
    step_decay=0.8,
    decay_every=10,
):
    """Gradient-free random-walk search over (theta, phi, lambda),
    maximizing mean_reward. Measurement sampling is not differentiable,
    so this uses repeated empirical evaluation rather than backprop --
    appropriate given the actual cost here is 3 scalars, not a network.
    """
    if seed is not None:
        random.seed(seed)

    best_params = (0.0, 0.0, 0.0)
    best_score = mean_reward(*best_params, n_repeats, shots)
    print(f"start: mean_S={best_score:.4f}")

    t_start = time.time()
    step = initial_step
    iteration = 0
    while True:
        if max_iterations is not None and iteration >= max_iterations:
            break
        if time_budget is not None and (time.time() - t_start) >= time_budget:
            break

        candidate = tuple(p + random.uniform(-step, step) for p in best_params)
        score = mean_reward(*candidate, n_repeats, shots)
        if score > best_score:
            best_score = score
            best_params = candidate
            print(
                f"iter {iteration}: improved -> "
                f"params={[round(p, 4) for p in candidate]} mean_S={score:.4f}"
            )

        iteration += 1
        if iteration % decay_every == 0:
            step *= step_decay

    print()
    print(f"FINAL params: {[round(p, 6) for p in best_params]}")
    print(f"FINAL mean_S: {best_score:.4f}  (classical bound: 2.0, Tsirelson: {2*math.sqrt(2):.4f})")
    print(f"total iterations: {iteration}")
    return best_params, best_score


def write_params_to_backend_file(theta, phi, lam):
    """Patch the installed pyqrack/qrack_ace_backend.py in place, so the
    fitted correction persists for future processes too, without
    requiring a separate model-loading step at import time."""
    import os
    import re

    path = os.path.join(os.path.dirname(_ace_mod.__file__), "qrack_ace_backend.py")
    with open(path, "r") as f:
        src = f.read()

    src = re.sub(r"_BSEQ_THETA\s*=\s*[-0-9.eE]+", f"_BSEQ_THETA = {theta!r}", src)
    src = re.sub(r"_BSEQ_PHI\s*=\s*[-0-9.eE]+", f"_BSEQ_PHI = {phi!r}", src)
    src = re.sub(r"_BSEQ_LAMBDA\s*=\s*[-0-9.eE]+", f"_BSEQ_LAMBDA = {lam!r}", src)

    with open(path, "w") as f:
        f.write(src)
    print(f"Wrote fitted angles into {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--time-budget", type=float, default=200.0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Write the fitted angles into the installed backend file.",
    )
    args = parser.parse_args()

    params, score = optimize(
        n_repeats=args.repeats,
        shots=args.shots,
        time_budget=args.time_budget,
        max_iterations=args.iterations,
        seed=args.seed,
    )

    if args.persist:
        write_params_to_backend_file(*params)


if __name__ == "__main__":
    main()
