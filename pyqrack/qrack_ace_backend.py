# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.
#
# Produced with input from (OpenAI) ChatGPT and (Anthropic) Claude
import math
import os
import random
import sys
import time

from .qrack_simulator import QrackSimulator
from .pauli import Pauli


_IS_QISKIT_AVAILABLE = True
try:
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from qiskit.compiler import transpile
    from qiskit.quantum_info.operators.symplectic.clifford import Clifford
except ImportError:
    _IS_QISKIT_AVAILABLE = False

_IS_QISKIT_AER_AVAILABLE = True
try:
    from qiskit_aer.noise import NoiseModel, depolarizing_error
except ImportError:
    _IS_QISKIT_AER_AVAILABLE = False

_IS_TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
except ImportError:
    _IS_TORCH_AVAILABLE = False

_BSEQ_CORRECTOR_MODEL = None
_BSEQ_CORRECTOR_LOAD_ATTEMPTED = False


def _get_bseq_corrector():
    """Lazily load the trained BSEQ (Bell/CHSH) boundary corrector model.

    Returns None if torch isn't available or the model file isn't found,
    in which case callers fall back to the existing non-learned cascade.
    Memoized at module scope: loaded at most once per process.
    """
    global _BSEQ_CORRECTOR_MODEL, _BSEQ_CORRECTOR_LOAD_ATTEMPTED
    if _BSEQ_CORRECTOR_LOAD_ATTEMPTED:
        return _BSEQ_CORRECTOR_MODEL
    _BSEQ_CORRECTOR_LOAD_ATTEMPTED = True
    if not _IS_TORCH_AVAILABLE:
        return None
    model_path = os.path.join(os.path.dirname(__file__), "bseq_corrector.jit.pt")
    try:
        model = torch.jit.load(model_path)
        model.eval()
        _BSEQ_CORRECTOR_MODEL = model
    except Exception:
        _BSEQ_CORRECTOR_MODEL = None
    return _BSEQ_CORRECTOR_MODEL


# Initial stub and concept produced through conversation with Elara
# (the custom OpenAI GPT)
class LHVQubit:
    def __init__(self, to_clone=None):
        # Initial state in "Bloch vector" terms, defaults to |0⟩
        if to_clone:
            self.bloch = to_clone.bloch.copy()
        else:
            self.reset()

    def reset(self):
        self.bloch = [0.0, 0.0, 1.0]

    def h(self):
        # Hadamard: rotate around Y-axis then X-axis (simplified for LHV)
        x, y, z = self.bloch
        self.bloch = [(x + z) / math.sqrt(2), y, (z - x) / math.sqrt(2)]

    def x(self):
        x, y, z = self.bloch
        self.bloch = [x, y, -z]

    def y(self):
        x, y, z = self.bloch
        self.bloch = [-x, y, z]

    def z(self):
        x, y, z = self.bloch
        self.bloch = [x, -y, z]

    def rx(self, theta):
        # Rotate Bloch vector around X-axis by angle theta
        x, y, z = self.bloch
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        new_y = cos_theta * y - sin_theta * z
        new_z = sin_theta * y + cos_theta * z
        self.bloch = [x, new_y, new_z]

    def ry(self, theta):
        # Rotate Bloch vector around Y-axis by angle theta
        x, y, z = self.bloch
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        new_x = cos_theta * x + sin_theta * z
        new_z = -sin_theta * x + cos_theta * z
        self.bloch = [new_x, y, new_z]

    def rz(self, theta):
        # Rotate Bloch vector around Z-axis by angle theta (in radians)
        x, y, z = self.bloch
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        new_x = cos_theta * x - sin_theta * y
        new_y = sin_theta * x + cos_theta * y
        self.bloch = [new_x, new_y, z]

    def s(self):
        self.rz(math.pi / 2)

    def adjs(self):
        self.rz(-math.pi / 2)

    def t(self):
        self.rz(math.pi / 4)

    def adjt(self):
        self.rz(-math.pi / 4)

    def u(self, theta, phi, lam):
        # Apply general single-qubit unitary gate
        self.rz(lam)
        self.ry(theta)
        self.rz(phi)

    # Provided verbatim by Elara (the custom OpenAI GPT):
    def mtrx(self, matrix):
        """
        Apply a 2x2 unitary matrix to the LHV Bloch vector using only standard math/cmath.
        Matrix format: [a, b, c, d] for [[a, b], [c, d]]
        """
        a, b, c, d = matrix

        # Current Bloch vector
        x, y, z = self.bloch

        # Convert to density matrix ρ = ½ (I + xσx + yσy + zσz)
        rho = [[(1 + z) / 2, (x - 1j * y) / 2], [(x + 1j * y) / 2, (1 - z) / 2]]

        # Compute U * ρ
        u_rho = [
            [a * rho[0][0] + b * rho[1][0], a * rho[0][1] + b * rho[1][1]],
            [c * rho[0][0] + d * rho[1][0], c * rho[0][1] + d * rho[1][1]],
        ]

        # Compute (U * ρ) * U†
        rho_prime = [
            [
                u_rho[0][0] * a.conjugate() + u_rho[0][1] * b.conjugate(),
                u_rho[0][0] * c.conjugate() + u_rho[0][1] * d.conjugate(),
            ],
            [
                u_rho[1][0] * a.conjugate() + u_rho[1][1] * b.conjugate(),
                u_rho[1][0] * c.conjugate() + u_rho[1][1] * d.conjugate(),
            ],
        ]

        # Extract Bloch components: Tr(ρ'σi) = 2 * Re[...]
        new_x = 2 * rho_prime[0][1].real + 2 * rho_prime[1][0].real
        new_y = 2 * (rho_prime[0][1].imag - rho_prime[1][0].imag)
        new_z = 2 * rho_prime[0][0].real - 1  # since Tr(ρ') = 1

        p = math.sqrt(new_x**2 + new_y**2 + new_z**2)

        new_x /= p
        new_y /= p
        new_z /= p

        self.bloch = [new_x, new_y, new_z]

    def prob(self, basis=Pauli.PauliZ):
        """Sample a classical outcome from the current 'quantum' state"""
        if basis == Pauli.PauliZ:
            prob_1 = (1 - self.bloch[2]) / 2
        elif basis == Pauli.PauliX:
            prob_1 = (1 - self.bloch[0]) / 2
        elif basis == Pauli.PauliY:
            prob_1 = (1 - self.bloch[1]) / 2
        else:
            raise ValueError(f"Unsupported basis: {basis}")
        return prob_1

    def m(self):
        result = random.random() < self.prob()
        self.reset()
        if result:
            self.x()
        return result


# Provided by Elara (the custom OpenAI GPT)
def _cpauli_lhv(prob, targ, axis, anti, theta=math.pi):
    """
    Apply a 'soft' controlled-Pauli gate: rotate target qubit
    proportionally to control's Z expectation value.

    theta: full rotation angle if control in |1⟩
    """
    # Control influence is (1 - ctrl.bloch[2]) / 2 = P(|1⟩)
    # BUT we avoid collapse by using the expectation value:
    control_influence = (1 - prob) if anti else prob

    effective_theta = control_influence * theta

    # Apply partial rotation to target qubit:
    if axis == Pauli.PauliX:
        targ.rx(effective_theta)
    elif axis == Pauli.PauliY:
        targ.ry(effective_theta)
    elif axis == Pauli.PauliZ:
        targ.rz(effective_theta)

class QrackAceBackend:
    """A back end for elided quantum error correction

    This back end uses elided repetition code on a nearest-neighbor topology to emulate
    a utility-scale superconducting chip quantum computer in very little memory.4

    The backend was originally designed assuming an (orbifolded) 2D qubit grid like 2019 Sycamore.
    However, it quickly became apparent that users can basically design their own connectivity topologies,
    without breaking the concept. (Not all will work equally well.)

    Consider distributing the different "patches" to different GPUs with self.sim[sim_id].set_device(gpu_id)!
    (If you have 3+ patches, maybe your discrete GPU can do multiple patches in the time it takes an Intel HD
    to do one patch worth of work!)

    Attributes:
        sim(QrackSimulator): Array of simulators corresponding to "patches" between boundary rows.
        long_range_columns(int): How many ideal rows between QEC boundary rows?
        is_transpose(bool): Rows are long if False, columns are long if True
    """

    # Sweepable vote weights for the 4-source boundary correction pool:
    # [slot0, slot1, slot2, lhv]. Must sum to an odd number so the
    # underlying hard-vote tally (independent of the continuous RMS
    # formula) is always tie-free.
    _LHV_VOTE_WEIGHTS = [2, 1, 1, 1]

    def __init__(
        self,
        qubit_count=1,
        long_range_columns=4,
        long_range_rows=4,
        is_transpose=False,
        is_schmidt_decompose_multi=False,
        is_stabilizer_hybrid=False,
        is_binary_decision_tree=False,
        is_gpu=True,
        is_host_pointer=(True if os.environ.get("PYQRACK_HOST_POINTER_DEFAULT_ON") else False),
        is_near_clifford_tableau_writer=False,
        noise=0,
        to_clone=None,
    ):
        if to_clone:
            qubit_count = to_clone.num_qubits()
            long_range_columns = to_clone.long_range_columns
            long_range_rows = to_clone.long_range_rows
            is_transpose = to_clone.is_transpose
        if qubit_count < 0:
            qubit_count = 0
        if long_range_columns < 0:
            long_range_columns = 0

        self._factor_width(qubit_count, is_transpose)
        self.long_range_columns = long_range_columns
        self.long_range_rows = long_range_rows
        self.is_transpose = is_transpose

        fppow = 5
        if "QRACK_FPPOW" in os.environ:
            fppow = int(os.environ.get("QRACK_FPPOW"))
        if fppow < 5:
            self._epsilon = 2**-9
        elif fppow > 5:
            self._epsilon = 2**-51
        else:
            self._epsilon = 2**-22

        self._coupling_map = None

        # If there's only one or zero "False" columns or rows,
        # the entire simulator is connected, anyway.
        len_col_seq = long_range_columns + 1
        col_patch_count = (self._row_length + len_col_seq - 1) // len_col_seq
        if (self._row_length < 3) or ((long_range_columns + 1) >= self._row_length):
            self._is_col_long_range = [True] * self._row_length
        else:
            col_seq = [True] * long_range_columns + [False]
            self._is_col_long_range = (col_seq * col_patch_count)[: self._row_length]
            if long_range_columns < self._row_length:
                self._is_col_long_range[-1] = False
        len_row_seq = long_range_rows + 1
        row_patch_count = (self._col_length + len_row_seq - 1) // len_row_seq
        if (self._col_length < 3) or ((long_range_rows + 1) >= self._col_length):
            self._is_row_long_range = [True] * self._col_length
        else:
            row_seq = [True] * long_range_rows + [False]
            self._is_row_long_range = (row_seq * row_patch_count)[: self._col_length]
            if long_range_rows < self._col_length:
                self._is_row_long_range[-1] = False
        sim_count = col_patch_count * row_patch_count

        # Boundary qubits no longer carry a private classical LHV proxy.
        # Instead, every boundary site (row- and/or column-boundary alike)
        # gets a real qubit in one single, shared "crossbar" QrackSimulator.
        # That simulator's own greedy elision (set_sdrp) is trusted to
        # automatically factor apart whatever boundary sites turn out to be
        # separable (e.g. disjoint rails, rail intersections), exactly the
        # same way it already factors apart unentangled subspaces within
        # any other single QrackSimulator instance. We don't need to special
        # -case "crossbar intersections" by hand; the elision does it for us.
        boundary_sim_id = sim_count
        boundary_count = 0

        self._qubits = []
        self._lhv = {}
        self._ancilla = {}
        self._ancilla_paired = set(to_clone._ancilla_paired) if to_clone else set()
        self._bell_partner = dict(to_clone._bell_partner) if to_clone else {}
        self._last_y_rotation = dict(to_clone._last_y_rotation) if to_clone else {}
        sim_counts = [0] * sim_count
        sim_id = 0
        tot_qubits = 0
        for r in self._is_row_long_range:
            for c in self._is_col_long_range:
                qubit = [(sim_id, sim_counts[sim_id])]
                sim_counts[sim_id] += 1

                if (not c) or (not r):
                    t_sim_id = (sim_id + 1) % sim_count
                    qubit.append((t_sim_id, sim_counts[t_sim_id]))
                    sim_counts[t_sim_id] += 1

                    qubit.append((boundary_sim_id, boundary_count))
                    boundary_count += 1

                    # Reserve a second boundary-sim qubit as the ancilla,
                    # right alongside slot2. It starts fresh (|0>) and is
                    # Bell-paired with slot2 lazily, the first time slot2
                    # is actually touched by a real coupling gate (see
                    # _apply_coupling), so it taps slot2's true evolving
                    # state rather than a disconnected pair prepared at
                    # construction time.
                    self._ancilla[tot_qubits] = (boundary_sim_id, boundary_count)
                    boundary_count += 1

                    self._lhv[tot_qubits] = LHVQubit(
                        to_clone=(to_clone._lhv[tot_qubits] if to_clone else None)
                    )

                if (not c) and (not r):
                    t_sim_id = (sim_id + col_patch_count) % sim_count
                    qubit.append((t_sim_id, sim_counts[t_sim_id]))
                    sim_counts[t_sim_id] += 1

                    t_sim_id = (t_sim_id + 1) % sim_count
                    qubit.append((t_sim_id, sim_counts[t_sim_id]))
                    sim_counts[t_sim_id] += 1

                if not c:
                    sim_id = (sim_id + 1) % sim_count

                self._qubits.append(qubit)
                tot_qubits += 1

        # The crossbar's size is fixed by how many boundary sites exist.
        # When there are none (e.g. a grid small enough, relative to
        # long_range_rows/columns, that the whole thing is "fully
        # connected" with no QEC boundary at all), we must NOT allocate a
        # 0-qubit QrackSimulator for the crossbar: a 0-qubit QrackSimulator
        # can be constructed, but calling .clone() on one crashes the
        # native Qrack core (segfault), and clone() is called routinely by
        # measure_shots()/clone(). So the boundary sim is only created when
        # boundary_count > 0, exactly mirroring how the original LHV-based
        # code never instantiated anything for the boundary case when
        # there were no boundary sites.
        has_boundary = boundary_count > 0
        if has_boundary:
            sim_counts.append(boundary_count)

        self.sim = []
        for i in range(sim_count + (1 if has_boundary else 0)):
            self.sim.append(
                to_clone.sim[i].clone()
                if to_clone
                else QrackSimulator(
                    sim_counts[i],
                    is_schmidt_decompose_multi=is_schmidt_decompose_multi,
                    is_stabilizer_hybrid=is_stabilizer_hybrid,
                    is_binary_decision_tree=is_binary_decision_tree,
                    is_gpu=is_gpu,
                    is_host_pointer=is_host_pointer,
                    is_near_clifford_tableau_writer=is_near_clifford_tableau_writer,
                    noise=noise,
                )
            )

            # You can still "set_sdrp" later, after the constructor.
            # if "QRACK_QUNIT_SEPARABILITY_THRESHOLD" not in os.environ:
            #     # (1 - 1 / sqrt(2)) / 4 (but empirically tuned)
            #     self.sim[i].set_sdrp(0.073223304703363119)

        self._boundary_sim_id = boundary_sim_id if has_boundary else None

    def clone(self):
        return QrackAceBackend(to_clone=self)

    def set_sdrp(self, sdrp):
        for sim in self.sim:
            sim.set_sdrp(sdrp)

    def measure_shots_consensus(self, q, s, n_instances=3, threshold=0.1):
        # Consensus measurement across n_instances independent clones.
        #
        # For each shot, run n_instances clones of the current state and
        # compare their marginal probabilities on every logical qubit via
        # prob(). For each qubit, take the majority vote over all instances:
        # if the average marginal >= 0.5, force_m to |1>; else to |0>.
        #
        # This resolves the parity ambiguity that causes XEB flip-flopping:
        # a bit-flip on a boundary qubit appears as disagreement between
        # instances (one sees p~0.9, another sees p~0.1). The majority vote
        # across n_instances corrects isolated odd-parity errors without
        # requiring a second full circuit run — just cheap prob() queries.
        #
        # threshold: minimum spread in marginals to trigger consensus
        # correction (below threshold, instances agree and no correction needed).
        n_qubits = self.num_qubits()
        samples = []
        for _ in range(s):
            # Run n_instances clones, collect marginals for every qubit
            clones = [self.clone() for _ in range(n_instances)]
            marginals = [
                [c.prob(lq) for lq in range(n_qubits)]
                for c in clones
            ]
            # Majority vote: average marginal across instances per qubit
            avg_marginals = [
                sum(marginals[i][lq] for i in range(n_instances)) / n_instances
                for lq in range(n_qubits)
            ]
            # Force all qubits in first clone to majority-vote outcome
            primary = clones[0]
            for lq in range(n_qubits):
                result = avg_marginals[lq] >= 0.5
                primary.force_m(lq, result)
            # Read out requested qubits
            _sample = primary.m_all()
            sample = 0
            for i in range(len(q)):
                if (_sample >> q[i]) & 1:
                    sample |= 1 << i
            samples.append(sample)
        return samples

    def num_qubits(self):
        return self._row_length * self._col_length

    def get_row_length(self):
        return self._row_length

    def get_column_length(self):
        return self._col_length

    def _factor_width(self, width, is_transpose=False):
        col_len = math.floor(math.sqrt(width))
        while ((width // col_len) * col_len) != width:
            col_len -= 1
        row_len = width // col_len

        self._col_length, self._row_length = (
            (row_len, col_len) if is_transpose else (col_len, row_len)
        )

    def _ct_pair_prob(self, q1, q2):
        p1 = self.sim[q1[0]].prob(q1[1]) if isinstance(q1, tuple) else q1.prob()
        p2 = self.sim[q2[0]].prob(q2[1]) if isinstance(q2, tuple) else q2.prob()

        # When p1 and p2 are within floating-point noise of each other
        # (self._epsilon), they carry no real information about which
        # qubit is "more likely 1" -- e.g. a fresh target right after
        # _cx_shadow's H() always reads ~0.5, indistinguishably from a
        # genuinely-mixed control. This shadow mechanism can only return
        # ONE definite classical pick per call, so resolving every such
        # near-tie to a FIXED qubit (as a plain "<" or "<=" comparison
        # would, once p1/p2 are within epsilon of each other) reintroduces
        # a systematic bias -- it just moves the bias from "favors q1" to
        # "always favors q2" instead of removing it. The aggregate
        # statistics this shadow is meant to approximate (e.g. a real CX
        # from a maximally-mixed control onto a fresh target makes BOTH
        # qubits individually read prob=0.5, perfectly correlated) are
        # only reproduced, across many circuit instances, if a genuine
        # tie is broken at random rather than by a fixed rule.
        if abs(p1 - p2) <= self._epsilon:
            return (p1, q2) if random.random() < 0.5 else (p2, q1)

        if p1 < p2:
            return p2, q1

        return p1, q2

    def _cz_shadow(self, q1, q2):
        prob_max, t = self._ct_pair_prob(q1, q2)
        # NOTE: this must be ">=", not ">". H() applied to any qubit in a
        # definite computational-basis state (e.g. a fresh boundary/ancilla
        # qubit, as in _cx_shadow's H-sandwich) lands at EXACTLY prob=0.5,
        # deterministically -- this is the ordinary case, not a rare
        # floating-point tie. A strict "> 0.5" therefore silently no-ops
        # the shadow gate every time it targets a fresh qubit, which
        # breaks entanglement transfer for the extremely common case of a
        # CX/CY/CZ from a maximally-mixed-looking control onto a fresh
        # target (e.g. H(0); cx(0,1) for a Bell pair).
        if prob_max >= (0.5 - self._epsilon):
            if isinstance(t, tuple):
                self.sim[t[0]].z(t[1])
            else:
                t.z()

    def _qec_x(self, c):
        if isinstance(c, tuple):
            self.sim[c[0]].x(c[1])
        else:
            c.x()

    def _qec_h(self, t):
        if isinstance(t, tuple):
            self.sim[t[0]].h(t[1])
        else:
            t.h()

    def _qec_s(self, t):
        if isinstance(t, tuple):
            self.sim[t[0]].s(t[1])
        else:
            t.s()

    def _qec_adjs(self, t):
        if isinstance(t, tuple):
            self.sim[t[0]].adjs(t[1])
        else:
            t.adjs()

    def _anti_cz_shadow(self, c, t):
        self._qec_x(c)
        self._cz_shadow(c, t)
        self._qec_x(c)

    def _cx_shadow(self, c, t):
        self._qec_h(t)
        self._cz_shadow(c, t)
        self._qec_h(t)

    def _anti_cx_shadow(self, c, t):
        self._qec_x(c)
        self._cx_shadow(c, t)
        self._qec_x(c)

    def _cy_shadow(self, c, t):
        self._qec_adjs(t)
        self._cx_shadow(c, t)
        self._qec_s(t)

    def _anti_cy_shadow(self, c, t):
        self._qec_x(c)
        self._cy_shadow(c, t)
        self._qec_x(c)

    def _unpack(self, lq):
        return self._qubits[lq]

    def _get_qb_lhv_indices(hq):
        # Historically, index 2 (when present) pointed at a private
        # classical LHVQubit proxy and had to be special-cased everywhere.
        # It is now an ordinary (sim_id, idx) tuple into the shared
        # boundary "crossbar" QrackSimulator, so it is just one more
        # coupling target like every other index. We keep this helper's
        # name and signature for minimal call-site churn; "lhv" is now
        # always -1 (no index needs special-casing any more).
        if len(hq) < 2:
            qb = [0]
        elif len(hq) < 4:
            qb = [0, 1, 2]
        else:
            qb = [0, 1, 2, 3, 4]
        lhv = -1

        return qb, lhv

    def _get_bloch_angles(self, hq):
        sim = self.sim[hq[0]].clone()
        q = hq[1]
        sim.separate([q])

        # Z axis
        z = 1 - 2 * sim.prob(q)

        # X axis
        sim.h(q)
        x = 1 - 2 * sim.prob(q)
        sim.h(q)

        # Y axis
        sim.adjs(q)
        sim.h(q)
        y = 1 - 2 * sim.prob(q)
        sim.h(q)
        sim.s(q)

        inclination = math.atan2(math.sqrt(x**2 + y**2), z)
        azimuth = math.atan2(y, x)

        return azimuth, inclination

    def _rotate_to_bloch(self, hq, delta_azimuth, delta_inclination):
        sim = self.sim[hq[0]]
        q = hq[1]

        # Apply rotation as "Azimuth, Inclination" (AI)
        cosA = math.cos(delta_azimuth)
        sinA = math.sin(delta_azimuth)
        cosI = math.cos(delta_inclination / 2)
        sinI = math.sin(delta_inclination / 2)

        m00 = complex(cosI, 0)
        m01 = complex(-cosA, sinA) * sinI
        m10 = complex(cosA, sinA) * sinI
        m11 = complex(cosI, 0)

        sim.mtrx([m00, m01, m10, m11], q)

    @staticmethod
    def _get_lhv_bloch_angles(sim):
        z = sim.bloch[2]
        x = sim.bloch[0]
        y = sim.bloch[1]
        inclination = math.atan2(math.sqrt(x**2 + y**2), z)
        azimuth = math.atan2(y, x)
        return azimuth, inclination

    @staticmethod
    def _rotate_lhv_to_bloch(sim, delta_azimuth, delta_inclination):
        cosA = math.cos(delta_azimuth)
        sinA = math.sin(delta_azimuth)
        cosI = math.cos(delta_inclination / 2)
        sinI = math.sin(delta_inclination / 2)

        m00 = complex(cosI, 0)
        m01 = complex(-cosA, sinA) * sinI
        m10 = complex(cosA, sinA) * sinI
        m11 = complex(cosI, 0)

        sim.mtrx([m00, m01, m10, m11])

    # Trainable BSEQ correction: a genuine controlled-U(theta,phi,lambda)
    # gate (QrackSimulator.mcu), with the ancilla as control and the
    # boundary qubit (slot2) as target -- both live in the same boundary
    # simulator, so this is a real, coherent two-qubit unitary, not a
    # classical shadow approximation. Applied directly to the quantum
    # state with no probability read or collapsed first, so it composes
    # correctly even while the Bell-pair partner is still genuinely
    # superposed. These 3 angles are the only trainable parameters;
    # fit them empirically against the measured BSEQ S statistic (see
    # the companion training/optimization script), not hand-derived.
    _BSEQ_THETA = 0.5524356880065896
    _BSEQ_PHI = 0.5611771084538236
    _BSEQ_LAMBDA = -0.3706652126771467

    def _apply_bseq_correction(self, lq):
        """Apply the trained controlled-U BSEQ correction, if applicable.

        Fires every time this boundary qubit's state is about to be read
        out (prob()/_correct()/m()/force_m() all funnel through here, so
        this covers every simulator-API readout for the qubit). A no-op
        whenever there's no ancilla for this site, or it hasn't been
        Bell-paired with slot2 yet (nothing to condition on).
        """
        anc = self._ancilla.get(lq)
        if anc is None or lq not in self._ancilla_paired:
            return
        hq = self._unpack(lq)
        if len(hq) < 3:
            return
        anc_sim, anc_idx = anc
        slot2_sim, slot2_idx = hq[2]
        if anc_sim != slot2_sim:
            return
        self.sim[anc_sim].mcu(
            [anc_idx],
            slot2_idx,
            QrackAceBackend._BSEQ_THETA,
            QrackAceBackend._BSEQ_PHI,
            QrackAceBackend._BSEQ_LAMBDA,
        )

    def _correct(self, lq, phase=False, skip_rotation=False):
        hq = self._unpack(lq)

        if len(hq) == 1:
            return

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        if phase:
            for q in qb:
                b = hq[q]
                self.sim[b[0]].h(b[1])
            if lq in self._lhv:
                self._lhv[lq].h()

        if len(hq) == 5:
            # RMS
            p = [
                self.sim[hq[0][0]].prob(hq[0][1]),
                self.sim[hq[1][0]].prob(hq[1][1]),
                self.sim[hq[2][0]].prob(hq[2][1]),
                self.sim[hq[3][0]].prob(hq[3][1]),
                self.sim[hq[4][0]].prob(hq[4][1]),
            ]
            # Balancing suggestion from Elara (the custom OpenAI GPT)
            prms = math.sqrt((p[0] ** 2 + p[1] ** 2 + 3 * (p[2] ** 2) + p[3] ** 2 + p[4] ** 2) / 7)
            qrms = math.sqrt(
                (
                    (1 - p[0]) ** 2
                    + (1 - p[1]) ** 2
                    + 3 * ((1 - p[2]) ** 2)
                    + (1 - p[3]) ** 2
                    + (1 - p[4]) ** 2
                )
                / 7
            )
            eff_prob = (prms + (1 - qrms)) / 2
            result = (
                (random.random() < 0.5)
                if abs(eff_prob - 0.5) <= self._epsilon
                else (eff_prob >= 0.5)
            )
            syndrome = (
                [1 - p[0], 1 - p[1], 1 - p[2], 1 - p[3], 1 - p[4]]
                if result
                else [p[0], p[1], p[2], p[3], p[4]]
            )
            for q in range(5):
                if syndrome[q] > (0.5 + self._epsilon):
                    self.sim[hq[q][0]].x(hq[q][1])

            if not skip_rotation:
                a, i = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
                a[0], i[0] = self._get_bloch_angles(hq[0])
                a[1], i[1] = self._get_bloch_angles(hq[1])
                a[2], i[2] = self._get_bloch_angles(hq[2])
                a[3], i[3] = self._get_bloch_angles(hq[3])
                a[4], i[4] = self._get_bloch_angles(hq[4])

                a_target = sum(a) / 5
                i_target = sum(i) / 5
                for x in range(5):
                    self._rotate_to_bloch(hq[x], a_target - a[x], i_target - i[x])

        else:
            lhv = self._lhv.get(lq)
            if lhv is None:
                # RMS
                p = [
                    self.sim[hq[0][0]].prob(hq[0][1]),
                    self.sim[hq[1][0]].prob(hq[1][1]),
                    self.sim[hq[2][0]].prob(hq[2][1]),
                ]
                # Balancing suggestion from Elara (the custom OpenAI GPT)
                prms = math.sqrt((p[0] ** 2 + p[1] ** 2 + p[2] ** 2) / 3)
                qrms = math.sqrt(((1 - p[0]) ** 2 + (1 - p[1]) ** 2 + (1 - p[2]) ** 2) / 3)
                eff_prob = (prms + (1 - qrms)) / 2
                result = (
                    (random.random() < 0.5)
                    if abs(eff_prob - 0.5) <= self._epsilon
                    else (eff_prob >= 0.5)
                )
                syndrome = [1 - p[0], 1 - p[1], 1 - p[2]] if result else [p[0], p[1], p[2]]
                for q in range(3):
                    if syndrome[q] > (0.5 + self._epsilon):
                        self.sim[hq[q][0]].x(hq[q][1])
            else:
                # Conditional tie-breaking cascade, NOT a fixed-weight pool.
                # A fixed weight on slot0/lhv pulls every decision toward
                # 0.5 even when slot1 and slot2 already agree confidently,
                # because slot0 and lhv are legitimately, permanently near
                # 0.5 for genuinely-entangled topologies (e.g. when slot0
                # shares a simulator with the control) -- a "vote" stuck at
                # 0.5 is not neutral in an RMS pool, it actively drags the
                # result toward the center. The actual intent ("LHV serves
                # only to act as a tie-breaker") is a conditional, not a
                # weight: trust slot1/slot2 alone whenever they agree, and
                # only consult slot0, then lhv, when they genuinely don't.
                #
                # The BSEQ correction (a real, coherent controlled-U gate
                # on slot2, conditioned on its ancilla) is applied FIRST,
                # directly to the quantum state -- before any probability
                # is read. This composes correctly even while this qubit's
                # Bell-pair partner is still genuinely superposed, unlike
                # every classical-decision approach that was tried and
                # rejected here (all of them were provably forced into an
                # uninformative coin flip in exactly that case).
                self._apply_bseq_correction(lq)

                p0 = self.sim[hq[0][0]].prob(hq[0][1])
                p1 = self.sim[hq[1][0]].prob(hq[1][1])
                p2 = self.sim[hq[2][0]].prob(hq[2][1])
                p_lhv = lhv.prob()

                if (p1 >= 0.5) == (p2 >= 0.5):
                    prms = math.sqrt((p1**2 + p2**2) / 2)
                    qrms = math.sqrt(((1 - p1) ** 2 + (1 - p2) ** 2) / 2)
                    eff_prob = (prms + (1 - qrms)) / 2
                    result = (
                        (random.random() < 0.5)
                        if abs(eff_prob - 0.5) <= self._epsilon
                        else (eff_prob >= 0.5)
                    )
                elif abs(p0 - 0.5) > self._epsilon:
                    # Genuine deadlock between slot1/slot2; slot0 (a real,
                    # exactly-entangled qubit in this common topology) has
                    # a real opinion, so it breaks the tie.
                    result = p0 >= 0.5
                else:
                    # slot0 is itself ambiguous; only now does the LHV's
                    # continuous, non-collapsing proxy actually decide.
                    result = (
                        (random.random() < 0.5)
                        if abs(p_lhv - 0.5) <= self._epsilon
                        else (p_lhv >= 0.5)
                    )

                p = [p0, p1, p2]
                syndrome = [1 - x for x in p] if result else list(p)
                for q in range(3):
                    if syndrome[q] > (0.5 + self._epsilon):
                        self.sim[hq[q][0]].x(hq[q][1])
                # The LHV proxy is never hard-collapsed via x(); it is only
                # ever updated by its own transversal gate evolution and
                # _cpauli_lhv, preserving the property that makes it usable
                # as a non-collapsing tie-breaker of last resort.

            if not skip_rotation:
                a, i = [0, 0, 0], [0, 0, 0]
                a[0], i[0] = self._get_bloch_angles(hq[0])
                a[1], i[1] = self._get_bloch_angles(hq[1])
                a[2], i[2] = self._get_bloch_angles(hq[2])

                a_target = sum(a) / 3
                i_target = sum(i) / 3
                for x in range(3):
                    self._rotate_to_bloch(hq[x], a_target - a[x], i_target - i[x])

        if phase:
            for q in qb:
                b = hq[q]
                self.sim[b[0]].h(b[1])
            if lq in self._lhv:
                self._lhv[lq].h()

    def apply_magnetic_bias(self, q, b):
        if b == 0:
            return
        b = math.exp(b)
        for x in q:
            hq = self._unpack(x)
            for h in hq:
                a, i = self._get_bloch_angles(h)
                self._rotate_to_bloch(
                    h,
                    math.atan(math.tan(a) * b) - a,
                    math.atan(math.tan(i) * b) - i,
                )

    def u(self, lq, th, ph, lm):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].u(b[1], th, ph, lm)
            return

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].u(b[1], th, ph, lm)

        lhv = self._lhv.get(lq)
        if lhv is not None:
            lhv.u(th, ph, lm)

        # Correction deferred to next 2-qubit gate (_cpauli calls _correct)

    def r(self, p, th, lq):
        if p == Pauli.PauliY:
            self._last_y_rotation[lq] = self._last_y_rotation.get(lq, 0.0) + th

        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].r(p, th, b[1])
            return

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].r(p, th, b[1])

        lhv = self._lhv.get(lq)
        if lhv is not None:
            if p == Pauli.PauliX:
                lhv.rx(th)
            elif p == Pauli.PauliY:
                lhv.ry(th)
            elif p == Pauli.PauliZ:
                lhv.rz(th)

        # Correction deferred to next 2-qubit gate (_cpauli calls _correct)

    def h(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].h(b[1])
            return

        self._correct(lq)

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].h(b[1])

        lhv = self._lhv.get(lq)
        if lhv is not None:
            lhv.h()

        # Correction deferred to next 2-qubit gate (_cpauli calls _correct)

    def s(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].s(b[1])
            return

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].s(b[1])

        lhv = self._lhv.get(lq)
        if lhv is not None:
            lhv.s()

    def adjs(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].adjs(b[1])
            return

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].adjs(b[1])

        lhv = self._lhv.get(lq)
        if lhv is not None:
            lhv.adjs()

    def x(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].x(b[1])
            return

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].x(b[1])

        lhv = self._lhv.get(lq)
        if lhv is not None:
            lhv.x()

    def y(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].y(b[1])
            return

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].y(b[1])

        lhv = self._lhv.get(lq)
        if lhv is not None:
            lhv.y()

    def z(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].z(b[1])
            return

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].z(b[1])

        lhv = self._lhv.get(lq)
        if lhv is not None:
            lhv.z()

    def t(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].t(b[1])
            return

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].t(b[1])

        lhv = self._lhv.get(lq)
        if lhv is not None:
            lhv.t()

    def adjt(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].adjt(b[1])
            return

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].adjt(b[1])

        lhv = self._lhv.get(lq)
        if lhv is not None:
            lhv.adjt()

    def _get_gate(self, pauli, anti, sim_id):
        gate = None
        shadow = None
        if pauli == Pauli.PauliX:
            gate = self.sim[sim_id].macx if anti else self.sim[sim_id].mcx
            shadow = self._anti_cx_shadow if anti else self._cx_shadow
        elif pauli == Pauli.PauliY:
            gate = self.sim[sim_id].macy if anti else self.sim[sim_id].mcy
            shadow = self._anti_cy_shadow if anti else self._cy_shadow
        elif pauli == Pauli.PauliZ:
            gate = self.sim[sim_id].macz if anti else self.sim[sim_id].mcz
            shadow = self._anti_cz_shadow if anti else self._cz_shadow
        else:
            raise RuntimeError("QrackAceBackend._get_gate() should never return identity!")

        return gate, shadow

    def _get_connected(self, i, is_row):
        long_range = self._is_row_long_range if is_row else self._is_col_long_range
        length = self._col_length if is_row else self._row_length

        connected = [i]
        c = (i - 1) % length
        while long_range[c] and (len(connected) < length):
            connected.append(c)
            c = (c - 1) % length
        if len(connected) < length:
            connected.append(c)
        boundary = len(connected)
        c = (i + 1) % length
        while long_range[c] and (len(connected) < length):
            connected.append(c)
            c = (c + 1) % length
        if len(connected) < length:
            connected.append(c)

        return connected, boundary

    def _apply_coupling(self, pauli, anti, qb1, hq1, qb2, hq2, lq1_lr):
        for q1 in qb1:
            b1 = hq1[q1]
            gate_fn, shadow_fn = self._get_gate(pauli, anti, b1[0])
            for q2 in qb2:
                b2 = hq2[q2]
                if b1[0] == b2[0]:
                    gate_fn([b1[1]], b2[1])
                elif lq1_lr or (b1[1] == b2[1]) or ((len(qb1) == 2) and (b1[1] == (b2[1] & 1))):
                    shadow_fn(b1, b2)

    def _cpauli(self, lq1, lq2, anti, pauli):
        lq1_row = lq1 // self._row_length
        lq1_col = lq1 % self._row_length
        lq2_row = lq2 // self._row_length
        lq2_col = lq2 % self._row_length

        hq1 = self._unpack(lq1)
        hq2 = self._unpack(lq2)

        lq1_lr = len(hq1) == 1
        lq2_lr = len(hq2) == 1

        self._correct(lq1)

        qb1, _ = QrackAceBackend._get_qb_lhv_indices(hq1)
        qb2, _ = QrackAceBackend._get_qb_lhv_indices(hq2)
        # Apply cross coupling on every qubit, including former-LHV boundary
        # qubits, which now live as real qubits in the shared boundary sim.
        self._apply_coupling(pauli, anti, qb1, hq1, qb2, hq2, lq1_lr)

        if lq2 in self._ancilla and lq2 not in self._ancilla_paired and len(hq2) >= 3:
            anc_sim, anc_idx = self._ancilla[lq2]
            slot2_sim, slot2_idx = hq2[2]
            if anc_sim == slot2_sim:
                # slot2 as CONTROL, ancilla as target: a CX never modifies
                # its control qubit's own marginal state, so this taps
                # slot2's real, evolving state non-invasively -- the
                # ancilla becomes perfectly correlated with slot2 without
                # disturbing it (unlike the reverse direction, which would
                # mix/dephase slot2 itself every time pairing occurred).
                self.sim[anc_sim].mcx([slot2_idx], anc_idx)
            else:
                # Different simulators (shouldn't normally happen, since
                # ancilla and slot2 are allocated in the same boundary sim,
                # but guard defensively): fall back to a direct two-qubit
                # gate is unavailable cross-sim here, so this path is not
                # expected to be exercised in practice.
                pass
            self._ancilla_paired.add(lq2)
            self._bell_partner[lq2] = lq1

        if lq2 in self._lhv:
            ctrl_prob = self.sim[hq1[0][0]].prob(hq1[0][1])
            _cpauli_lhv(ctrl_prob, self._lhv[lq2], pauli, anti)

        self._correct(lq1, True)
        if pauli != Pauli.PauliZ:
            self._correct(lq2, False, pauli != Pauli.PauliX)
        if pauli != Pauli.PauliX:
            self._correct(lq2, True)

    def cx(self, lq1, lq2):
        self._cpauli(lq1, lq2, False, Pauli.PauliX)

    def cy(self, lq1, lq2):
        self._cpauli(lq1, lq2, False, Pauli.PauliY)

    def cz(self, lq1, lq2):
        self._cpauli(lq1, lq2, False, Pauli.PauliZ)

    def acx(self, lq1, lq2):
        self._cpauli(lq1, lq2, True, Pauli.PauliX)

    def acy(self, lq1, lq2):
        self._cpauli(lq1, lq2, True, Pauli.PauliY)

    def acz(self, lq1, lq2):
        self._cpauli(lq1, lq2, True, Pauli.PauliZ)

    def mcx(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackAceBackend.mcx() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self._cpauli(lq1[0], lq2, False, Pauli.PauliX)

    def mcy(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackAceBackend.mcy() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self._cpauli(lq1[0], lq2, False, Pauli.PauliY)

    def mcz(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackAceBackend.mcz() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self._cpauli(lq1[0], lq2, False, Pauli.PauliZ)

    def macx(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackAceBackend.macx() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self._cpauli(lq1[0], lq2, True, Pauli.PauliX)

    def macy(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackAceBackend.macy() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self._cpauli(lq1[0], lq2, True, Pauli.PauliY)

    def macz(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackAceBackend.macz() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self._cpauli(lq1[0], lq2, True, Pauli.PauliZ)

    def swap(self, lq1, lq2):
        self.cx(lq1, lq2)
        self.cx(lq2, lq1)
        self.cx(lq1, lq2)

    def iswap(self, lq1, lq2):
        self.swap(lq1, lq2)
        self.cz(lq1, lq2)
        self.s(lq1)
        self.s(lq2)

    def adjiswap(self, lq1, lq2):
        self.adjs(lq2)
        self.adjs(lq1)
        self.cz(lq1, lq2)
        self.swap(lq1, lq2)

    def prob(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            return self.sim[b[0]].prob(b[1])

        self._correct(lq)
        if len(hq) == 5:
            # RMS
            p = [
                self.sim[hq[0][0]].prob(hq[0][1]),
                self.sim[hq[1][0]].prob(hq[1][1]),
                self.sim[hq[2][0]].prob(hq[2][1]),
                self.sim[hq[3][0]].prob(hq[3][1]),
                self.sim[hq[4][0]].prob(hq[4][1]),
            ]
            # Balancing suggestion from Elara (the custom OpenAI GPT)
            prms = math.sqrt((p[0] ** 2 + p[1] ** 2 + 3 * (p[2] ** 2) + p[3] ** 2 + p[4] ** 2) / 7)
            qrms = math.sqrt(
                (
                    (1 - p[0]) ** 2
                    + (1 - p[1]) ** 2
                    + 3 * ((1 - p[2]) ** 2)
                    + (1 - p[3]) ** 2
                    + (1 - p[4]) ** 2
                )
                / 7
            )
        else:
            lhv = self._lhv.get(lq)
            if lhv is None:
                # RMS
                p = [
                    self.sim[hq[0][0]].prob(hq[0][1]),
                    self.sim[hq[1][0]].prob(hq[1][1]),
                    self.sim[hq[2][0]].prob(hq[2][1]),
                ]
                # Balancing suggestion from Elara (the custom OpenAI GPT)
                prms = math.sqrt((p[0] ** 2 + p[1] ** 2 + p[2] ** 2) / 3)
                qrms = math.sqrt(((1 - p[0]) ** 2 + (1 - p[1]) ** 2 + (1 - p[2]) ** 2) / 3)
            else:
                p = [
                    self.sim[hq[0][0]].prob(hq[0][1]),
                    self.sim[hq[1][0]].prob(hq[1][1]),
                    self.sim[hq[2][0]].prob(hq[2][1]),
                ]
                # The three real replicas are already mutually consistent
                # here (the _correct() call above already ran the
                # conditional tie-breaking cascade and forced agreement via
                # x()). Re-weighting in the LHV here would reintroduce the
                # same center-dragging distortion the cascade was built to
                # avoid -- a plain RMS over the now-settled replicas is the
                # correct, already-decided answer.
                prms = math.sqrt((p[0] ** 2 + p[1] ** 2 + p[2] ** 2) / 3)
                qrms = math.sqrt(((1 - p[0]) ** 2 + (1 - p[1]) ** 2 + (1 - p[2]) ** 2) / 3)

        return (prms + (1 - qrms)) / 2

    def m(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            return self.sim[b[0]].m(b[1])

        p = self.prob(lq)
        result = ((p + self._epsilon) >= 1) or (random.random() < p)

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        for q in qb:
            b = hq[q]
            p = self.sim[b[0]].prob(b[1]) if result else (1 - self.sim[b[0]].prob(b[1]))
            if p < self._epsilon:
                if self.sim[b[0]].m(b[1]) != result:
                    self.sim[b[0]].x(b[1])
            else:
                self.sim[b[0]].force_m(b[1], result)

        lhv = self._lhv.get(lq)
        if lhv is not None:
            lhv.reset()
            if result:
                lhv.x()

        return result

    def force_m(self, lq, result):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            return self.sim[b[0]].force_m(b[1], result)

        self._correct(lq)

        qb, _ = QrackAceBackend._get_qb_lhv_indices(hq)

        for q in qb:
            b = hq[q]
            p = self.sim[b[0]].prob(b[1]) if result else (1 - self.sim[b[0]].prob(b[1]))
            if p < self._epsilon:
                if self.sim[b[0]].m(b[1]) != result:
                    self.sim[b[0]].x(b[1])
            else:
                self.sim[b[0]].force_m(b[1], result)

        lhv = self._lhv.get(lq)
        if lhv is not None:
            lhv.reset()
            if result:
                lhv.x()

        return result

    def m_all(self):
        # Randomize the order of measurement to amortize error.
        result = 0
        rows = list(range(self._col_length))
        random.shuffle(rows)
        for lq_row in rows:
            row_offset = lq_row * self._row_length
            cols = list(range(self._row_length))
            random.shuffle(cols)
            for lq_col in cols:
                lq = row_offset + lq_col
                if self.m(lq):
                    result |= 1 << lq

        return result

    def measure_shots(self, q, s):
        samples = []
        for _ in range(s):
            clone = self.clone()
            _sample = clone.m_all()
            sample = 0
            for i in range(len(q)):
                if (_sample >> q[i]) & 1:
                    sample |= 1 << i
            samples.append(sample)

        return samples

    def _apply_op(self, operation):
        name = operation.name

        if (name == "id") or (name == "barrier"):
            # Skip measurement logic
            return

        conditional = getattr(operation, "conditional", None)
        if isinstance(conditional, int):
            conditional_bit_set = (self._classical_register >> conditional) & 1
            if not conditional_bit_set:
                return
        elif conditional is not None:
            mask = int(conditional.mask, 16)
            if mask > 0:
                value = self._classical_memory & mask
                while (mask & 0x1) == 0:
                    mask >>= 1
                    value >>= 1
                if value != int(conditional.val, 16):
                    return

        if (name == "u1") or (name == "p"):
            self._sim.u(operation.qubits[0]._index, 0, 0, float(operation.params[0]))
        elif name == "u2":
            self._sim.u(
                operation.qubits[0]._index,
                math.pi / 2,
                float(operation.params[0]),
                float(operation.params[1]),
            )
        elif (name == "u3") or (name == "u"):
            self._sim.u(
                operation.qubits[0]._index,
                float(operation.params[0]),
                float(operation.params[1]),
                float(operation.params[2]),
            )
        elif name == "r":
            self._sim.u(
                operation.qubits[0]._index,
                float(operation.params[0]),
                float(operation.params[1]) - math.pi / 2,
                (-1 * float(operation.params[1])) + math.pi / 2,
            )
        elif name == "rx":
            self._sim.r(Pauli.PauliX, float(operation.params[0]), operation.qubits[0]._index)
        elif name == "ry":
            self._sim.r(Pauli.PauliY, float(operation.params[0]), operation.qubits[0]._index)
        elif name == "rz":
            self._sim.r(Pauli.PauliZ, float(operation.params[0]), operation.qubits[0]._index)
        elif name == "h":
            self._sim.h(operation.qubits[0]._index)
        elif name == "x":
            self._sim.x(operation.qubits[0]._index)
        elif name == "y":
            self._sim.y(operation.qubits[0]._index)
        elif name == "z":
            self._sim.z(operation.qubits[0]._index)
        elif name == "s":
            self._sim.s(operation.qubits[0]._index)
        elif name == "sdg":
            self._sim.adjs(operation.qubits[0]._index)
        elif name == "t":
            self._sim.t(operation.qubits[0]._index)
        elif name == "tdg":
            self._sim.adjt(operation.qubits[0]._index)
        elif name == "cx":
            self._sim.cx(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == "cy":
            self._sim.cy(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == "cz":
            self._sim.cz(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == "dcx":
            self._sim.mcx(operation.qubits[0]._index, operation.qubits[1]._index)
            self._sim.mcx(operation.qubits[1]._index, operation.qubits[0]._index)
        elif name == "swap":
            self._sim.swap(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == "iswap":
            self._sim.iswap(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == "iswap_dg":
            self._sim.adjiswap(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == "reset":
            qubits = operation.qubits
            for qubit in qubits:
                if self._sim.m(qubit._index):
                    self._sim.x(qubit._index)
        elif name == "measure":
            qubits = operation.qubits
            clbits = operation.clbits
            cregbits = (
                operation.register
                if hasattr(operation, "register")
                else len(operation.qubits) * [-1]
            )

            self._sample_qubits += qubits
            self._sample_clbits += clbits
            self._sample_cregbits += cregbits

            if not self._sample_measure:
                for index in range(len(qubits)):
                    qubit_outcome = self._sim.m(qubits[index]._index)

                    clbit = clbits[index]
                    clmask = 1 << clbit
                    self._classical_memory = (self._classical_memory & (~clmask)) | (
                        qubit_outcome << clbit
                    )

                    cregbit = cregbits[index]
                    if cregbit < 0:
                        cregbit = clbit

                    regbit = 1 << cregbit
                    self._classical_register = (self._classical_register & (~regbit)) | (
                        qubit_outcome << cregbit
                    )

        elif name == "bfunc":
            mask = int(operation.mask, 16)
            relation = operation.relation
            val = int(operation.val, 16)

            cregbit = operation.register
            cmembit = operation.memory if hasattr(operation, "memory") else None

            compared = (self._classical_register & mask) - val

            if relation == "==":
                outcome = compared == 0
            elif relation == "!=":
                outcome = compared != 0
            elif relation == "<":
                outcome = compared < 0
            elif relation == "<=":
                outcome = compared <= 0
            elif relation == ">":
                outcome = compared > 0
            elif relation == ">=":
                outcome = compared >= 0
            else:
                raise RuntimeError("Invalid boolean function relation.")

            # Store outcome in register and optionally memory slot
            regbit = 1 << cregbit
            self._classical_register = (self._classical_register & (~regbit)) | (
                int(outcome) << cregbit
            )
            if cmembit is not None:
                membit = 1 << cmembit
                self._classical_memory = (self._classical_memory & (~membit)) | (
                    int(outcome) << cmembit
                )
        else:
            err_msg = 'QrackAceBackend encountered unrecognized operation "{0}"'
            raise RuntimeError(err_msg.format(operation))

    def _add_sample_measure(self, sample_qubits, sample_clbits, num_samples):
        """Generate data samples from current statevector.

        Taken almost straight from the terra source code.

        Args:
            measure_params (list): List of (qubit, clbit) values for
                                   measure instructions to sample.
            num_samples (int): The number of data samples to generate.

        Returns:
            list: A list of data values in hex format.
        """
        # Get unique qubits that are actually measured
        measure_qubit = [qubit for qubit in sample_qubits]
        measure_clbit = [clbit for clbit in sample_clbits]

        # Sample and convert to bit-strings
        if num_samples == 1:
            sample = self._sim.m_all()
            result = 0
            for index in range(len(measure_qubit)):
                qubit = measure_qubit[index]._index
                qubit_outcome = (sample >> qubit) & 1
                result |= qubit_outcome << index
            measure_results = [result]
        else:
            measure_results = self._sim.measure_shots(
                [q._index for q in measure_qubit], num_samples
            )

        data = []
        for sample in measure_results:
            for index in range(len(measure_qubit)):
                qubit_outcome = (sample >> index) & 1
                clbit = measure_clbit[index]._index
                clmask = 1 << clbit
                self._classical_memory = (self._classical_memory & (~clmask)) | (
                    qubit_outcome << clbit
                )

            data.append(bin(self._classical_memory)[2:].zfill(self.num_qubits()))

        return data

    def run_qiskit_circuit(self, experiment, shots=1):
        if not _IS_QISKIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to run_qiskit_circuit() with QrackAceBackend, you must install Qiskit!"
            )

        instructions = []
        if isinstance(experiment, QuantumCircuit):
            instructions = experiment.data
        else:
            raise RuntimeError('Unrecognized "run_input" argument specified for run().')

        self._shots = shots
        self._sample_qubits = []
        self._sample_clbits = []
        self._sample_cregbits = []
        self._sample_measure = True
        _data = []
        shotLoopMax = 1

        is_initializing = True
        boundary_start = -1

        for opcount in range(len(instructions)):
            operation = instructions[opcount]

            if operation.name == "id" or operation.name == "barrier":
                continue

            if is_initializing and ((operation.name == "measure") or (operation.name == "reset")):
                continue

            is_initializing = False

            if (operation.name == "measure") or (operation.name == "reset"):
                if boundary_start == -1:
                    boundary_start = opcount

            if (boundary_start != -1) and (operation.name != "measure"):
                shotsPerLoop = 1
                shotLoopMax = self._shots
                self._sample_measure = False
                break

        preamble_memory = 0
        preamble_register = 0
        preamble_sim = None

        if self._sample_measure or boundary_start <= 0:
            boundary_start = 0
            self._sample_measure = True
            shotsPerLoop = self._shots
            shotLoopMax = 1
        else:
            boundary_start -= 1
            if boundary_start > 0:
                self._sim = self
                self._classical_memory = 0
                self._classical_register = 0

                for operation in instructions[:boundary_start]:
                    self._apply_op(operation)

                preamble_memory = self._classical_memory
                preamble_register = self._classical_register
                preamble_sim = self._sim

        for shot in range(shotLoopMax):
            if preamble_sim is None:
                self._sim = self
                self._classical_memory = 0
                self._classical_register = 0
            else:
                self._sim = QrackAceBackend(to_clone=preamble_sim)
                self._classical_memory = preamble_memory
                self._classical_register = preamble_register

            for operation in instructions[boundary_start:]:
                self._apply_op(operation)

            if not self._sample_measure and (len(self._sample_qubits) > 0):
                _data += [bin(self._classical_memory)[2:].zfill(self.num_qubits())]
                self._sample_qubits = []
                self._sample_clbits = []
                self._sample_cregbits = []

        if self._sample_measure and (len(self._sample_qubits) > 0):
            _data = self._add_sample_measure(self._sample_qubits, self._sample_clbits, self._shots)

        del self._sim

        return _data

    def get_qiskit_basis_gates():
        return [
            "id",
            "u",
            "u1",
            "u2",
            "u3",
            "r",
            "rx",
            "ry",
            "rz",
            "h",
            "x",
            "y",
            "z",
            "s",
            "sdg",
            "sx",
            "sxdg",
            "p",
            "t",
            "tdg",
            "cx",
            "cy",
            "cz",
            "swap",
            "iswap",
            "reset",
            "measure",
        ]

    # Mostly written by Dan, but with a little help from Elara (custom OpenAI GPT)
    def get_logical_coupling_map(self):
        if self._coupling_map:
            return self._coupling_map

        coupling_map = set()
        rows, cols = self._row_length, self._col_length

        # Map each column index to its full list of logical qubit indices
        def logical_index(row, col):
            return row * cols + col

        for col in range(cols):
            connected_cols, _ = self._get_connected(col, False)
            for row in range(rows):
                connected_rows, _ = self._get_connected(row, False)
                a = logical_index(row, col)
                for c in connected_cols:
                    for r in connected_rows:
                        b = logical_index(r, c)
                        if a != b:
                            coupling_map.add((a, b))

        self._coupling_map = sorted(coupling_map)

        return self._coupling_map

    # Designed by Dan, and implemented by Elara:
    def create_noise_model(self, x=0.125, y=0.25):
        if not _IS_QISKIT_AER_AVAILABLE:
            raise RuntimeError(
                "Before trying to run_qiskit_circuit() with QrackAceBackend, you must install Qiskit Aer!"
            )
        noise_model = NoiseModel()

        # Single-qubit depolarizing only on boundary qubits
        boundary_qubits = set()
        for a, b in self.get_logical_coupling_map():
            col_a = a % self._row_length
            col_b = b % self._row_length
            is_long_a = self._is_col_long_range[col_a]
            is_long_b = self._is_col_long_range[col_b]
            if not (is_long_a and is_long_b):
                if not is_long_a:
                    boundary_qubits.add(a)
                if not is_long_b:
                    boundary_qubits.add(b)

        for q in boundary_qubits:
            for gate in ["u", "u1", "u2", "u3", "h", "x", "y", "z",
                         "s", "sdg", "t", "tdg", "rx", "ry", "rz"]:
                noise_model.add_quantum_error(
                    depolarizing_error(x, 1), gate, [q])

        # Two-qubit depolarizing on boundary-crossing and boundary-adjacent gates
        for a, b in self.get_logical_coupling_map():
            col_a = a % self._row_length
            col_b = b % self._row_length
            is_long_a = self._is_col_long_range[col_a]
            is_long_b = self._is_col_long_range[col_b]

            if is_long_a and is_long_b:
                continue

            p2 = 1 - (1 - y) ** 2 if (is_long_a or is_long_b) else y
            p3 = 1 - (1 - y) ** 3 if (is_long_a or is_long_b) else 1 - (1 - y) ** 2

            for gate in ["cx", "cy", "cz"]:
                noise_model.add_quantum_error(depolarizing_error(p2, 2), gate, [a, b])
            for gate in ["swap", "iswap"]:
                noise_model.add_quantum_error(depolarizing_error(p3, 2), gate, [a, b])

        return noise_model
