# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.
#
# Produced with input from (Anthropic) Claude
#
# QrackAceNighthawk: a one-off, hard-coded QrackAceBackend layout for the
# 61-qubit, geometrically irregular RCS experiment IBM and BlueQubit ran on
# a nearest-neighbor subset of the "Nighthawk" processor.
#
# Topology (fixed -- there is deliberately NO qubit-count or patch-size
# option): three patches on a 1-dimensional ring of patch boundaries,
# logical qubit indices running left to right:
#
#   index   0 -  1   boundary column A (2 qubits)   patches 2 | 0  (wraps)
#   index   2 - 19   bulk, patch 0     (18 qubits)
#   index  20 - 22   boundary column B (3 qubits)   patches 0 | 1
#   index  23 - 38   bulk, patch 1     (16 qubits)
#   index  39 - 42   boundary column C (4 qubits)   patches 1 | 2
#   index  43 - 60   bulk, patch 2     (18 qubits)
#                    ...wrapping back to boundary column A.
#
#   61 logical qubits; 52 bulk, 9 boundary. Each patch simulator holds its
#   own bulk plus BOTH boundary columns flanking it:
#     patch 0: A(2) + 18 + B(3) = 23
#     patch 1: B(3) + 16 + C(4) = 23
#     patch 2: C(4) + 18 + A(2) = 24
#   (+1 error-detection ancilla each when is_error_detection, and a shared
#   9-qubit crossbar simulator, +1 ancilla, when use_crossbars.)
#
# Every boundary qubit is an EDGE boundary (no corners exist in a 1D ring),
# with replicas ordered exactly as QrackAceBackend's own grid construction
# orders them, which _correct() depends on:
#     [ (left/"home" patch), (right/adjacent patch), (crossbar, if on) ]
# -- the same convention as the generic class, where a boundary column's
# home is the patch to its left and its adjacent replica is the next patch
# to the right (wrapping). Everything downstream of the replica map --
# gates, shadow couplers, _correct(), LHV proxies, error detection, the
# boundary repetition code, measurement -- is inherited unchanged.
#
# NOTE: logical indices above are this class's own convention. Mapping the
# experiment's physical Nighthawk qubit IDs onto them (each boundary
# column's qubits being the ones whose couplers cross between the two
# adjacent patches) is up to the caller.

import os

from .qrack_system import Qrack
from .qrack_simulator import QrackSimulator
from .qrack_ace_backend import QrackAceBackend, LHVQubit


class QrackAceNighthawk(QrackAceBackend):
    # (boundary-column width to the LEFT of patch k, bulk width of patch k),
    # for k = 0, 1, 2. The boundary to the left of patch 0 is the wraparound
    # column shared with the last patch. Fixed for this experiment; it is a
    # class attribute only so that the layout machinery itself can be
    # validated on scaled-down analogs (by subclassing), never a user option.
    LAYOUT = ((2, 18), (3, 16), (4, 18))

    def __init__(
        self,
        is_schmidt_decompose_multi=False,
        is_stabilizer_hybrid=False,
        is_binary_decision_tree=False,
        is_gpu=True,
        is_host_pointer=(True if os.environ.get("PYQRACK_HOST_POINTER_DEFAULT_ON") else False),
        is_near_clifford_tableau_writer=False,
        noise=0,
        is_error_detection=True,
        is_boundary_repetition_code=False,
        use_crossbars=True,
        to_clone=None,
    ):
        if to_clone:
            is_error_detection = to_clone.is_error_detection
            is_boundary_repetition_code = to_clone.is_boundary_repetition_code
            use_crossbars = to_clone.use_crossbars

        layout = self.LAYOUT
        patch_count = len(layout)
        qubit_count = sum(b + k for b, k in layout)

        # Present as a 1 x N chain to every inherited method that reads grid
        # dimensions (num_qubits(), m_all(), _cpauli's unused row/col math).
        self.is_1d_chain = True
        self._col_length, self._row_length = 1, qubit_count
        self.long_range_columns = None
        self.long_range_rows = None
        self.is_transpose = False
        self.is_torus = True
        self.is_error_detection = is_error_detection
        self.use_crossbars = use_crossbars
        self.is_boundary_repetition_code = is_boundary_repetition_code

        if Qrack.fppow < 5:
            self._epsilon = 2**-11
            self._ps_epsilon = 2**-6
        elif Qrack.fppow > 5:
            self._epsilon = 2**-54
            self._ps_epsilon = 2**-27
        else:
            self._epsilon = 2**-24
            self._ps_epsilon = 2**-12
        self._rot_epsilon = (1.0 - 2**-0.5) / 2

        self._coupling_map = None

        # --- Replica map ---------------------------------------------------
        # self._boundary_column[lq] = boundary-column index (0, 1, 2 for
        # A, B, C), or None for bulk. Used only by the coupling map.
        self._boundary_column = []
        self._is_col_long_range = []
        self._is_row_long_range = [True]
        sim_count = patch_count
        boundary_sim_id = sim_count
        boundary_count = 0
        sim_counts = [0] * sim_count
        # First pass: which simulators each logical qubit lives on, in
        # replica order [home (left) patch, adjacent (right) patch].
        homes = []
        for k, (b_width, k_width) in enumerate(layout):
            left = (k - 1) % patch_count  # patch to the left (wraps)
            for _ in range(b_width):
                homes.append((left, k))
                self._boundary_column.append(k)
                self._is_col_long_range.append(False)
            for _ in range(k_width):
                homes.append((k,))
                self._boundary_column.append(None)
                self._is_col_long_range.append(True)

        # Second pass: allocate physical slots. ORDER MATTERS, not just
        # labels: _apply_coupling() shadow-couples on an incidental
        # physical-index match (b1[1] == b2[1]) between replicas on
        # different simulators, so allocation order changes behavior.
        # Allocate in exactly the order QrackAceBackend's own grid loop
        # would -- bulk first, the wraparound boundary column LAST -- i.e.
        # starting just after column A. With uniform 1-wide boundaries
        # this reproduces generic is_1d_chain ACE exactly.
        qubits = [None] * qubit_count
        lhv_lqs = []
        start = layout[0][0]
        for j in range(qubit_count):
            lq = (start + j) % qubit_count
            qubit = []
            for s in homes[lq]:
                qubit.append((s, sim_counts[s]))
                sim_counts[s] += 1
            if len(homes[lq]) > 1:
                if use_crossbars:
                    qubit.append((boundary_sim_id, boundary_count))
                    boundary_count += 1
                lhv_lqs.append(lq)
            qubits[lq] = qubit

        if use_crossbars and (boundary_count > 0):
            self._boundary_sim_id = boundary_sim_id
            sim_counts.append(boundary_count)
            sim_count += 1
        else:
            self._boundary_sim_id = None

        # Same per-simulator ancilla bookkeeping as QrackAceBackend.
        self._detect_ancilla = []
        if self.is_error_detection:
            for i in range(len(sim_counts)):
                self._detect_ancilla.append(sim_counts[i])
                sim_counts[i] += 1
        self._detect_ancilla_lq = []
        if self.is_error_detection:
            for sim_id, phys_idx in enumerate(self._detect_ancilla):
                self._detect_ancilla_lq.append(len(qubits))
                qubits.append([(sim_id, phys_idx)])

        self._rep_code_ancilla = []
        if self.is_boundary_repetition_code:
            for i in range(len(sim_counts)):
                self._rep_code_ancilla.append(sim_counts[i])
                sim_counts[i] += 1
        self._rep_code_ancilla_lq = []
        if self.is_boundary_repetition_code:
            for sim_id, phys_idx in enumerate(self._rep_code_ancilla):
                self._rep_code_ancilla_lq.append(len(qubits))
                qubits.append([(sim_id, phys_idx)])

        self._sim_counts = sim_counts
        self._in_gadget_capture = False

        if to_clone:
            # swap() can exchange replica lists (and LHV proxies) between
            # logical indices, so a clone copies the CURRENT map, not the
            # freshly built one.
            self._qubits = [list(q) for q in to_clone._qubits]
            self._lhv = {lq: LHVQubit(to_clone=v) for lq, v in to_clone._lhv.items()}
            self._sdrp = to_clone._sdrp
        else:
            self._qubits = qubits
            self._lhv = {lq: LHVQubit() for lq in lhv_lqs}
            if "QRACK_QUNIT_SEPARABILITY_THRESHOLD" in os.environ:
                self._sdrp = min(1, float(os.environ["QRACK_QUNIT_SEPARABILITY_THRESHOLD"]))
            else:
                self._sdrp = 0.0

        self.sim = []
        for i in range(sim_count):
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

    def clone(self):
        return type(self)(to_clone=self)

    def num_qubits(self):
        return self._row_length

    def get_patch_sizes(self):
        """Physical qubit count of each simulator: the patches in order,
        then the crossbar (if any). Includes ancillae."""
        return list(self._sim_counts)

    def get_logical_coupling_map(self):
        # Same ground truth as QrackAceBackend: two logical qubits are
        # coupled iff they share a simulator among their replicas. The
        # crossbar, shared by ALL boundary qubits, would make every
        # boundary pair coupled, so -- matching the generic class's
        # "same single side of one patch" filter -- boundary-to-boundary
        # pairs are kept only within the SAME boundary column. Boundary
        # qubits on opposite sides of a patch, which on the real device
        # are separated by that patch's whole bulk, drop out.
        if self._coupling_map:
            return self._coupling_map

        n = self.num_qubits()
        sim_to_qubits = {}
        for lq in range(n):
            for sim_id, _ in self._qubits[lq]:
                sim_to_qubits.setdefault(sim_id, []).append(lq)

        coupling_map = set()
        for qubits_here in sim_to_qubits.values():
            for a in qubits_here:
                for b in qubits_here:
                    if a == b:
                        continue
                    ca, cb = self._boundary_column[a], self._boundary_column[b]
                    if (ca is not None) and (cb is not None) and (ca != cb):
                        continue
                    coupling_map.add((a, b))

        self._coupling_map = sorted(coupling_map)

        return self._coupling_map
