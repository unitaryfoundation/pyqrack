# Empirical optimizer for QrackAceBackend's BSEQ (Bell/CHSH) correction
# gate(s).
#
# The correction has two layers:
#
# 1. A plain, uncontrolled U(theta, phi, lambda) gate (QrackSimulator.u)
#    applied directly to the boundary qubit. Always available; needs
#    no ancilla. U then U-adjoint, with nothing else happening to the
#    qubit in between except a probability read, is mathematically
#    guaranteed to leave the qubit's actual evolution unaffected
#    (U_dag @ U = I), while the intermediate readout genuinely depends
#    on the qubit's value beforehand whenever U doesn't commute with
#    the Z basis.
#
# 2. Optionally (when ancillae=True, the default, both here and at
#    QrackAceBackend construction), a second layer applied AFTER the u
#    gate: the boundary qubit's ancilla is prepped fresh into |+>, then
#    acts as CONTROL on a 4-parameter mcu(theta,phi,lambda,gamma) gate
#    targeting the boundary qubit. Reversed in the opposite order
#    (mcu-adjoint, then u-adjoint) when uncomputing.
#
# Both layers ONLY wrap prob()'s own readout computation: _correct()'s
# classical cascade (which permanently decides the boundary qubit's
# committed bit for the rest of circuit evolution) runs completely
# unaffected, exactly as it did before this correction existed. The
# net effect: the correction can only ever change what a given
# prob()/m()/force_m() call reports, and never anything a subsequent
# gate in the circuit will actually see -- i.e. it is idempotent with
# respect to circuit evolution, applying only to observable output.
#
# PyQrack's mcu follows the same convention as Qiskit's U/CU gates. Per
# Qiskit's own documentation, UGate.inverse() gives U(th,ph,la)^dagger
# = U(-th,-la,-ph) (theta negated, phi/lambda swapped-and-negated) --
# directly verified against the real PyQrack simulator (matching to
# better than 1e-6 on the full state vector) for both the 3-parameter u
# and the 4-parameter mcu (gamma simply negated alongside the same
# swap). Note Qiskit's separately-documented CUGate.inverse() page
# states a different formula (straight negation, no swap) -- tested
# directly against PyQrack's mcu, that formula does NOT match; the
# swap formula is the one verified to actually work here.
#
# With ancillae=True (default), all 7 angles (3 for u, 4 for mcu) are
# fit together. With ancillae=False, only the 3 u angles are fit (and
# QrackAceBackend should be constructed with bseq_ancillae=False too,
# so no ancilla qubits are allocated at all). They default to None
# (each layer inactive, a guaranteed no-op) until set here, fit by
# direct, gradient-free search against the empirically measured CHSH S
# statistic (or its 4 components individually), averaged over repeated
# Bell-pair trials to average down the substantial per-construction
# measurement variance inherent to this architecture (each circuit
# construction makes its own one-time classical collapse choice early
# in the elision pipeline; averaging over MANY independent
# constructions is what makes the mean S a meaningful, optimizable
# target -- a single fixed gate cannot reduce per-shot variance, since
# that variance is set by an earlier, already-classical decision the
# gate cannot retroactively undo).
#
# Usage:
#   python3 optimize_bseq_correction.py [--iterations N] [--time-budget SECONDS] [--no-ancillae]
#
# Writes the found angles directly into pyqrack's installed
# qrack_ace_backend.py, so every subsequently-constructed
# QrackAceBackend in the running process (and, if you choose to
# persist the file, future processes) uses the fitted correction
# automatically -- no separate model file, no extra inference
# dependency, and no torch/JIT artifact needed: the correction is at
# most 7 floats, and baking them in as class attributes is already the
# most efficient possible packaging.

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


def _expectation(theta1, theta2, shots, width=16, long_range_columns=1, ancillae=True):
    """One fresh Bell-pair construction, measured at a given basis-angle
    pair. This is exactly the bseq.py pattern: h(0), cx(0,1), then a
    PauliY rotation on each half before measuring."""
    s = QrackAceBackend(width, long_range_columns=long_range_columns, bseq_ancillae=ancillae)
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


def compute_bell_pair_stats(
    n_constructions, shots, width=3, long_range_columns=1, ancillae=True, control=0
):
    """Simpler, plain Bell-pair statistic (matching bell_state.py): for
    each of n_constructions independent constructions, put qubit
    `control` into superposition and CNOT from it onto the other
    qubit. Samples shots measurements and tracks the 0/1
    (agreement-side) balance and overall correlation. Returns
    (mean_correlation, mean_balance, stdev_balance).
    """
    target = 1 - control
    total_correlated = 0
    balance_vals = []
    for _ in range(n_constructions):
        s = QrackAceBackend(width, long_range_columns=long_range_columns, bseq_ancillae=ancillae)
        s.h(control)
        s.cx(control, target)
        results = s.measure_shots([0, 1], shots)
        zero = sum(1 for r in results if r == 0)
        three = sum(1 for r in results if r == 3)
        correlated = zero + three
        total_correlated += correlated
        if correlated:
            balance_vals.append(zero / correlated)
    mean_corr = total_correlated / (n_constructions * shots)
    mean_bal = statistics.mean(balance_vals) if balance_vals else 0.5
    stdev_bal = statistics.stdev(balance_vals) if len(balance_vals) > 1 else 0.0
    return mean_corr, mean_bal, stdev_bal


def bell_pair_score(params, n_constructions=200, shots=24, ancillae=True, **kwargs):
    """A single scalar reward for the Bell-pair objective: reward high
    correlation, balance centered at 0.5, and low variance in that
    balance across constructions. Weighted simply; adjust to taste.
    params is a 3-tuple (u angles only) when ancillae=False, or a
    7-tuple (u angles + mcu angles) when ancillae=True."""
    set_correction(params, ancillae=ancillae)
    mean_corr, mean_bal, stdev_bal = compute_bell_pair_stats(
        n_constructions, shots, ancillae=ancillae, **kwargs
    )
    return mean_corr - abs(mean_bal - 0.5) - 0.5 * stdev_bal


def bell_correlation_score(params, n_constructions=200, shots=24, ancillae=True, **kwargs):
    """A simpler reward: just the mean Bell-pair correlation itself
    (matching bell_state.py's printed "Correlation:" metric exactly --
    correlated/shots, averaged over many independent constructions),
    with no balance-centering or variance penalty. Note: unlike
    bell_pair_score, this has no guard against the optimizer finding a
    correction that pushes correlation up by forcing deterministic
    agreement between the two qubits rather than by improving genuine
    entanglement fidelity -- it is the more direct, but less
    constrained, objective."""
    set_correction(params, ancillae=ancillae)
    mean_corr, _, _ = compute_bell_pair_stats(n_constructions, shots, ancillae=ancillae, **kwargs)
    return mean_corr


def _prepare_and_measure(gate_fn, n_constructions, shots, width=3, long_range_columns=1, ancillae=True):
    """Build n_constructions fresh QrackAceBackend instances, apply
    gate_fn(sim) to each, sample shots measurements of qubits [0,1],
    and return a Counter-like dict of total outcome counts pooled
    across all constructions and shots (outcome values 0..3, matching
    bell_state.py's encoding: bit 0 = qubit 0, bit 1 = qubit 1)."""
    totals = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(n_constructions):
        s = QrackAceBackend(width, long_range_columns=long_range_columns, bseq_ancillae=ancillae)
        gate_fn(s)
        results = s.measure_shots([0, 1], shots)
        for r in results:
            totals[r] += 1
    return totals


def _outcome_prob(totals, outcomes, n_constructions, shots):
    total_shots = n_constructions * shots
    return sum(totals[o] for o in outcomes) / total_shots


# The 6 components making up the combined BSEQ objective, each paired
# with the gate sequence that prepares it and the outcome(s) that
# count as "ideal" for that preparation:
#
#   Bell-pair subset (50% combined weight, 25% each):
#     1. h(0); cx(0,1)            ideal |00>+|11>  -> reward P(0 or 3)
#     2. h(0); cx(0,1); x(1)      ideal |01>+|10>  -> reward P(1 or 2)
#
#   Deterministic subset (50% combined weight, 12.5% each):
#     3. (no gates at all)        ideal |00>       -> reward P(0)
#     4. x(0); x(1)                ideal |11>       -> reward P(3)
#     5. x(0)                      ideal |10>       -> reward P(2)
#     6. x(1)                      ideal |01>       -> reward P(1)
#
# Components 3-6 exist specifically to guard against the correction
# over-fitting correlation in cases where the two qubits are NOT
# entangled and should NOT show any correlation/anti-correlation
# pattern at all beyond their deterministic, classical preparation --
# without this, an optimizer chasing Bell-pair correlation alone has
# no reason not to learn a correction that makes everything look more
# "agreed" or "disagreed," which would be wrong for these 4 cases.
_BSEQ_COMPONENTS = [
    ("bell_00_11", lambda s: (s.h(0), s.cx(0, 1)), (0, 3), 0.25),
    ("bell_01_10", lambda s: (s.h(0), s.cx(0, 1), s.x(1)), (1, 2), 0.25),
    ("det_00", lambda s: None, (0,), 0.125),
    ("det_11", lambda s: (s.x(0), s.x(1)), (3,), 0.125),
    ("det_10", lambda s: s.x(0), (1,), 0.125),
    ("det_01", lambda s: s.x(1), (2,), 0.125),
]


def compute_bseq_components(n_constructions, shots, **kwargs):
    """Run all 6 components and return a dict of {name: probability of
    ideal outcome(s)}, each averaged over n_constructions independent
    constructions."""
    out = {}
    for name, gate_fn, ideal_outcomes, _weight in _BSEQ_COMPONENTS:
        totals = _prepare_and_measure(gate_fn, n_constructions, shots, **kwargs)
        out[name] = _outcome_prob(totals, ideal_outcomes, n_constructions, shots)
    return out


def bseq_combined_score(params, n_constructions=200, shots=24, ancillae=True, **kwargs):
    """The combined, 6-component, equal-subset-weighted reward: 50%
    weight on the 2 Bell-pair components (rewarding correlation on
    |00>+|11> and anti-correlation on |01>+|10>, 25% each), 50% weight
    on the 4 deterministic components (rewarding the correct,
    classical, non-entangled outcome for each of the 4 single-qubit-X
    preparations, 12.5% each). This is a single linear combination,
    so improving any one component (without harming the others)
    always improves the total -- and an optimizer cannot improve Bell-
    pair correlation by making the deterministic cases look more
    "correlated" without paying an equal-or-greater penalty there."""
    set_correction(params, ancillae=ancillae)
    components = compute_bseq_components(n_constructions, shots, ancillae=ancillae, **kwargs)
    return sum(components[name] * weight for name, _, _, weight in _BSEQ_COMPONENTS)


def set_correction(params, ancillae=True):
    """params is a 3-tuple (theta, phi, lambda) for the u layer alone
    when ancillae=False, or a 7-tuple (theta, phi, lambda, mcu_theta,
    mcu_phi, mcu_lambda, mcu_gamma) when ancillae=True."""
    theta, phi, lam = params[0], params[1], params[2]
    _ace_mod.QrackAceBackend._BSEQ_THETA = theta
    _ace_mod.QrackAceBackend._BSEQ_PHI = phi
    _ace_mod.QrackAceBackend._BSEQ_LAMBDA = lam
    if ancillae:
        mcu_theta, mcu_phi, mcu_lam, mcu_gam = params[3], params[4], params[5], params[6]
        _ace_mod.QrackAceBackend._BSEQ_MCU_THETA = mcu_theta
        _ace_mod.QrackAceBackend._BSEQ_MCU_PHI = mcu_phi
        _ace_mod.QrackAceBackend._BSEQ_MCU_LAMBDA = mcu_lam
        _ace_mod.QrackAceBackend._BSEQ_MCU_GAMMA = mcu_gam
    else:
        _ace_mod.QrackAceBackend._BSEQ_MCU_THETA = None
        _ace_mod.QrackAceBackend._BSEQ_MCU_PHI = None
        _ace_mod.QrackAceBackend._BSEQ_MCU_LAMBDA = None
        _ace_mod.QrackAceBackend._BSEQ_MCU_GAMMA = None


def mean_reward(params, n_repeats, shots, ancillae=True, **kwargs):
    """The actual optimization target: mean S across many independent
    Bell-pair constructions ("an RCS data set for Bell-pair
    perturbations"), not a single noisy reading."""
    set_correction(params, ancillae=ancillae)
    vals = [compute_S(shots, ancillae=ancillae, **kwargs) for _ in range(n_repeats)]
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
    target="bseq_combined",
    ancillae=True,
):
    """Gradient-free random-walk search over the correction angles,
    maximizing the chosen reward. Measurement sampling is not
    differentiable, so this uses repeated empirical evaluation rather
    than backprop -- appropriate given the actual cost here is at most
    7 scalars, not a network.

    ancillae: True (default) fits all 7 parameters (3 for the u layer,
        4 for the ancilla+mcu layer) and constructs QrackAceBackend
        with bseq_ancillae=True. False fits only the 3 u-layer
        parameters and constructs with bseq_ancillae=False (no ancilla
        qubits allocated at all).

    target: "bseq_combined" (default) is a single linear combination
        of 6 components, 50% combined weight on 2 Bell-pair components
        (correlation on h(0);cx(0,1)'s ideal |00>+|11>, and
        anti-correlation on h(0);cx(0,1);x(1)'s ideal |01>+|10>, 25%
        each) and 50% combined weight on 4 deterministic components
        (the correct classical outcome for each of no-op->00,
        x(0);x(1)->11, x(0)->10, x(1)->01, 12.5% each). The
        deterministic components exist specifically so the optimizer
        cannot improve Bell-pair correlation by learning a correction
        that makes everything look more "agreed" or "disagreed,"
        independent of whether the qubits are actually entangled --
        that would show up as a penalty on the deterministic
        components equal to or larger than the Bell-pair gain.
        "correlation" optimizes only the original Bell-pair (control=0)
        case alone, with no deterministic guard. "bell_pair" optimizes
        a composite of correlation, balance-centering, and low
        variance (also control=0 only, no deterministic guard). "S"
        optimizes the CHSH S statistic directly. In direct testing,
        "S" did not show a consistent improvement.
    """
    if seed is not None:
        random.seed(seed)

    if target == "S":
        reward_fn = lambda p: mean_reward(p, n_repeats, shots, ancillae=ancillae)
        label = "mean_S"
    elif target == "bell_pair":
        reward_fn = lambda p: bell_pair_score(
            p, n_constructions=n_repeats * 32, shots=shots, ancillae=ancillae
        )
        label = "bell_pair_score"
    elif target == "correlation":
        reward_fn = lambda p: bell_correlation_score(
            p, n_constructions=n_repeats * 32, shots=shots, ancillae=ancillae
        )
        label = "mean_correlation"
    elif target == "bseq_combined":
        reward_fn = lambda p: bseq_combined_score(
            p, n_constructions=n_repeats * 32, shots=shots, ancillae=ancillae
        )
        label = "bseq_combined_score"
    else:
        raise ValueError(
            f"unknown target: {target!r} (expected 'S', 'bell_pair', "
            "'correlation', or 'bseq_combined')"
        )


    n_params = 7 if ancillae else 3
    best_params = tuple(0.0 for _ in range(n_params))
    best_score = reward_fn(best_params)
    print(f"start: {label}={best_score:.4f}")

    t_start = time.time()
    step = initial_step
    iteration = 0
    while True:
        if max_iterations is not None and iteration >= max_iterations:
            break
        if time_budget is not None and (time.time() - t_start) >= time_budget:
            break

        candidate = tuple(p + random.uniform(-step, step) for p in best_params)
        score = reward_fn(candidate)
        if score > best_score:
            best_score = score
            best_params = candidate
            print(
                f"iter {iteration}: improved -> "
                f"params={[round(p, 4) for p in candidate]} {label}={score:.4f}"
            )

        iteration += 1
        if iteration % decay_every == 0:
            step *= step_decay

    print()
    print(f"FINAL params: {[round(p, 6) for p in best_params]}")
    print(f"FINAL {label}: {best_score:.4f}")
    if target == "S":
        print(f"  (classical bound: 2.0, Tsirelson: {2*math.sqrt(2):.4f})")
    print(f"total iterations: {iteration}")
    return best_params, best_score


def write_params_to_backend_file(params):
    """Patch the installed pyqrack/qrack_ace_backend.py in place, so the
    fitted correction persists for future processes too, without
    requiring a separate model-loading step at import time. params is
    a 3-tuple (u only) or a 7-tuple (u + mcu)."""
    import os
    import re

    path = os.path.join(os.path.dirname(_ace_mod.__file__), "qrack_ace_backend.py")
    with open(path, "r") as f:
        src = f.read()

    theta, phi, lam = params[0], params[1], params[2]
    src = re.sub(r"_BSEQ_THETA\s*=\s*(None|[-0-9.eE]+)", f"_BSEQ_THETA = {theta!r}", src)
    src = re.sub(r"_BSEQ_PHI\s*=\s*(None|[-0-9.eE]+)", f"_BSEQ_PHI = {phi!r}", src)
    src = re.sub(r"_BSEQ_LAMBDA\s*=\s*(None|[-0-9.eE]+)", f"_BSEQ_LAMBDA = {lam!r}", src)

    if len(params) >= 7:
        mcu_theta, mcu_phi, mcu_lam, mcu_gam = params[3], params[4], params[5], params[6]
        src = re.sub(
            r"_BSEQ_MCU_THETA\s*=\s*(None|[-0-9.eE]+)",
            f"_BSEQ_MCU_THETA = {mcu_theta!r}",
            src,
        )
        src = re.sub(
            r"_BSEQ_MCU_PHI\s*=\s*(None|[-0-9.eE]+)", f"_BSEQ_MCU_PHI = {mcu_phi!r}", src
        )
        src = re.sub(
            r"_BSEQ_MCU_LAMBDA\s*=\s*(None|[-0-9.eE]+)",
            f"_BSEQ_MCU_LAMBDA = {mcu_lam!r}",
            src,
        )
        src = re.sub(
            r"_BSEQ_MCU_GAMMA\s*=\s*(None|[-0-9.eE]+)",
            f"_BSEQ_MCU_GAMMA = {mcu_gam!r}",
            src,
        )

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
        "--target",
        choices=["S", "bell_pair", "correlation", "bseq_combined"],
        default="bseq_combined",
        help="Optimization objective: 'bseq_combined' (default) is a "
        "linear combination of 2 Bell-pair components (50% combined "
        "weight: correlation on |00>+|11>, anti-correlation on "
        "|01>+|10>) and 4 deterministic components (50% combined "
        "weight: the correct classical outcome for no-op, x(0);x(1), "
        "x(0) alone, and x(1) alone), guarding against over-fitting "
        "correlation where qubits should NOT correlate; 'correlation' "
        "fits only the original Bell-pair case alone; 'bell_pair' "
        "optimizes a composite of correlation, balance-centering, and "
        "low variance; 'S' optimizes the CHSH S statistic directly.",
    )
    parser.add_argument(
        "--no-ancillae",
        dest="ancillae",
        action="store_false",
        default=True,
        help="Fit only the 3-parameter u layer (no ancilla qubits "
        "allocated at all, QrackAceBackend constructed with "
        "bseq_ancillae=False). Default: fit all 7 parameters (u layer "
        "+ ancilla/mcu layer), with bseq_ancillae=True.",
    )
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
        target=args.target,
        ancillae=args.ancillae,
    )

    if args.persist:
        write_params_to_backend_file(params)


if __name__ == "__main__":
    main()
