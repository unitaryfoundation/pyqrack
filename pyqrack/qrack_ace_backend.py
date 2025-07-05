# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.
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
        correction_bias(float): Bias magnitude and direction during pseudo-QEC
    """

    def __init__(
        self,
        qubit_count=1,
        long_range_columns=4,
        long_range_rows=4,
        is_transpose=False,
        correction_bias=0,
        isTensorNetwork=False,
        isSchmidtDecomposeMulti=False,
        isSchmidtDecompose=True,
        isStabilizerHybrid=False,
        isBinaryDecisionTree=False,
        isPaged=True,
        isCpuGpuHybrid=True,
        isOpenCL=True,
        isHostPointer=(
            True if os.environ.get("PYQRACK_HOST_POINTER_DEFAULT_ON") else False
        ),
        noise=0,
        toClone=None,
    ):
        if toClone:
            qubit_count = toClone.num_qubits()
            long_range_columns = toClone.long_range_columns
            long_range_rows = toClone.long_range_rows
            is_transpose = toClone.is_transpose
        if qubit_count < 0:
            qubit_count = 0
        if long_range_columns < 0:
            long_range_columns = 0

        self._factor_width(qubit_count, is_transpose)
        self.long_range_columns = long_range_columns
        self.long_range_rows = long_range_rows
        self.is_transpose = is_transpose
        self.correction_bias = correction_bias

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

        self._qubits = []
        sim_counts = [0] * sim_count
        sim_id = 0
        for r in self._is_row_long_range:
            for c in self._is_col_long_range:
                qubit = [(sim_id, sim_counts[sim_id])]
                sim_counts[sim_id] += 1

                if (not c) or (not r):
                    t_sim_id = (sim_id + 1) % sim_count
                    qubit.append((t_sim_id, sim_counts[t_sim_id]))
                    sim_counts[t_sim_id] += 1

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

        self.sim = []
        for i in range(sim_count):
            self.sim.append(
                toClone.sim[i].clone()
                if toClone
                else QrackSimulator(
                    sim_counts[i],
                    isTensorNetwork=isTensorNetwork,
                    isSchmidtDecomposeMulti=isSchmidtDecomposeMulti,
                    isSchmidtDecompose=isSchmidtDecompose,
                    isStabilizerHybrid=isStabilizerHybrid,
                    isBinaryDecisionTree=isBinaryDecisionTree,
                    isPaged=isPaged,
                    isCpuGpuHybrid=isCpuGpuHybrid,
                    isOpenCL=isOpenCL,
                    isHostPointer=isHostPointer,
                    noise=noise,
                )
            )

            # You can still "monkey-patch" this, after the constructor.
            if "QRACK_QUNIT_SEPARABILITY_THRESHOLD" not in os.environ:
                # (1 - 1 / sqrt(2)) / 4 (but empirically tuned)
                self.sim[i].set_sdrp(0.073223304703363119)

    def clone(self):
        return QrackAceBackend(toClone=self)

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

        if p1 < p2:
            return p2, q1

        return p1, q2

    def _cz_shadow(self, q1, q2):
        prob_max, t = self._ct_pair_prob(q1, q2)
        if prob_max > 0.5:
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

    def _get_qb_indices(self, hq):
        qb = []
        if len(hq) < 2:
            qb = [0]
        elif len(hq) < 4:
            qb = [0, 1]
        else:
            qb = [0, 1, 2, 3]

        return qb

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

    def _correct(self, lq, phase=False, skip_rotation=False):
        hq = self._unpack(lq)

        if len(hq) == 1:
            return

        qb = self._get_qb_indices(hq)

        if phase:
            for q in qb:
                b = hq[q]
                self.sim[b[0]].h(b[1])

        if len(hq) == 5:
            # RMS
            p = [
                self.sim[hq[0][0]].prob(hq[0][1]),
                self.sim[hq[1][0]].prob(hq[1][1]),
                self.sim[hq[2][0]].prob(hq[2][1]),
                self.sim[hq[3][0]].prob(hq[3][1]),
            ]
            # Balancing suggestion from Elara (the custom OpenAI GPT)
            prms = math.sqrt((p[0] ** 2 + p[1] ** 2 + p[2] ** 2 + p[3] ** 2) / 4)
            qrms = math.sqrt(
                ((1 - p[0]) ** 2 + (1 - p[1]) ** 2 + (1 - p[2]) ** 2 + (1 - p[3]) ** 2)
                / 4
            )
            result = ((prms + (1 - qrms)) / 2) >= 0.5
            syndrome = (
                [1 - p[0], 1 - p[1], 1 - p[2], 1 - p[3]]
                if result
                else [p[0], p[1], p[2], p[3]]
            )
            for q in range(4):
                if syndrome[q] > (0.5 + self._epsilon):
                    self.sim[hq[q][0]].x(hq[q][1])

            if not skip_rotation:
                a, i = [0, 0, 0, 0], [0, 0, 0, 0]
                a[0], i[0] = self._get_bloch_angles(hq[0])
                a[1], i[1] = self._get_bloch_angles(hq[1])
                a[2], i[2] = self._get_bloch_angles(hq[3])
                a[3], i[3] = self._get_bloch_angles(hq[4])

                a_target = 0
                i_target = 0
                for x in range(4):
                    a_target += a[x]
                    i_target += i[x]

                a_target /= 4
                i_target /= 4
                for x in range(4):
                    self._rotate_to_bloch(hq[x], a_target - a[x], i_target - i[x])

                self.apply_magnetic_bias([lq], self.correction_bias)

        else:
            # RMS
            p = [
                self.sim[hq[0][0]].prob(hq[0][1]),
                self.sim[hq[1][0]].prob(hq[1][1]),
            ]
            # Balancing suggestion from Elara (the custom OpenAI GPT)
            prms = math.sqrt((p[0] ** 2 + p[1] ** 2) / 3)
            qrms = math.sqrt(((1 - p[0]) ** 2 + (1 - p[1]) ** 2) / 3)
            result = ((prms + (1 - qrms)) / 2) >= 0.5
            syndrome = [1 - p[0], 1 - p[1]] if result else [p[0], p[1]]
            for q in range(2):
                if syndrome[q] > (0.5 + self._epsilon):
                    self.sim[hq[q][0]].x(hq[q][1])

            if not skip_rotation:
                a, i = [0, 0], [0, 0]
                a[0], i[0] = self._get_bloch_angles(hq[0])
                a[1], i[1] = self._get_bloch_angles(hq[1])

                a_target = 0
                i_target = 0
                for x in range(2):
                    a_target += a[x]
                    i_target += i[x]

                a_target /= 2
                i_target /= 2
                for x in range(2):
                    self._rotate_to_bloch(hq[x], a_target - a[x], i_target - i[x])

                self.apply_magnetic_bias([lq], self.correction_bias)

        if phase:
            for q in qb:
                b = hq[q]
                self.sim[b[0]].h(b[1])

    def apply_magnetic_bias(self, q, b):
        if b == 0:
            return
        b = math.exp(b)
        for x in q:
            hq = self._unpack(x)
            for h in hq:
                a, i = self._get_bloch_angles(h)
                self._rotate_to_bloch(
                    h, math.atan(math.tan(a) * b) - a, math.atan(math.tan(i) * b) - i
                )

    def u(self, lq, th, ph, lm):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].u(b[1], th, ph, lm)
            return

        qb = self._get_qb_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].u(b[1], th, ph, lm)

        self._correct(lq, False, True)
        self._correct(lq, True, False)

    def r(self, p, th, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].r(p, th, b[1])
            return

        qb = self._get_qb_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].r(p, th, b[1])

        if p != Pauli.PauliZ:
            self._correct(lq, False, p != Pauli.PauliX)
        if p != Pauli.PauliX:
            self._correct(lq, True)

    def h(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].h(b[1])
            return

        self._correct(lq)

        qb = self._get_qb_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].h(b[1])

        self._correct(lq)

    def s(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].s(b[1])
            return

        qb = self._get_qb_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].s(b[1])

    def adjs(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].adjs(b[1])
            return

        qb = self._get_qb_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].adjs(b[1])

    def x(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].x(b[1])
            return

        qb = self._get_qb_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].x(b[1])

    def y(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].y(b[1])
            return

        qb = self._get_qb_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].y(b[1])

    def z(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].z(b[1])
            return

        qb = self._get_qb_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].z(b[1])

    def t(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].t(b[1])
            return

        qb = self._get_qb_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].t(b[1])

    def adjt(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].adjt(b[1])
            return

        qb = self._get_qb_indices(hq)

        for q in qb:
            b = hq[q]
            self.sim[b[0]].adjt(b[1])

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
            raise RuntimeError(
                "QrackAceBackend._get_gate() should never return identity!"
            )

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
                elif (
                    lq1_lr
                    or (b1[1] == b2[1])
                    or ((len(qb1) == 2) and (b1[1] == (b2[1] & 1)))
                ):
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

        qb1 = self._get_qb_indices(hq1)
        qb2 = self._get_qb_indices(hq2)
        # Apply cross coupling on hardware qubits first
        self._apply_coupling(pauli, anti, qb1, hq1, qb2, hq2, lq1_lr)

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
                hq[2].prob(),
                self.sim[hq[3][0]].prob(hq[3][1]),
                self.sim[hq[4][0]].prob(hq[4][1]),
            ]
            # Balancing suggestion from Elara (the custom OpenAI GPT)
            prms = math.sqrt((p[0] ** 2 + p[1] ** 2 + p[2] ** 2 + p[3] ** 2) / 4)
            qrms = math.sqrt(
                ((1 - p[0]) ** 2 + (1 - p[1]) ** 2 + (1 - p[2]) ** 2 + (1 - p[3]) ** 2)
                / 4
            )
        else:
            # RMS
            p = [
                self.sim[hq[0][0]].prob(hq[0][1]),
                self.sim[hq[1][0]].prob(hq[1][1]),
            ]
            # Balancing suggestion from Elara (the custom OpenAI GPT)
            prms = math.sqrt((p[0] ** 2 + p[1] ** 2) / 3)
            qrms = math.sqrt(((1 - p[0]) ** 2 + (1 - p[1]) ** 2) / 3)

        return (prms + (1 - qrms)) / 2

    def m(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            return self.sim[b[0]].m(b[1])

        p = self.prob(lq)
        result = ((p + self._epsilon) >= 1) or (random.random() < p)

        qb = self._get_qb_indices(hq)

        for q in qb:
            b = hq[q]
            p = self.sim[b[0]].prob(b[1]) if result else (1 - self.sim[b[0]].prob(b[1]))
            if p < self._epsilon:
                if self.sim[b[0]].m(b[1]) != result:
                    self.sim[b[0]].x(b[1])
            else:
                self.sim[b[0]].force_m(b[1], result)

        return result

    def force_m(self, lq, result):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            return self.sim[b[0]].force_m(b[1], result)

        self._correct(lq)

        qb = self._get_qb_indices(hq)

        for q in qb:
            b = hq[q]
            p = self.sim[b[0]].prob(b[1]) if result else (1 - self.sim[b[0]].prob(b[1]))
            if p < self._epsilon:
                if self.sim[b[0]].m(b[1]) != result:
                    self.sim[b[0]].x(b[1])
            else:
                self.sim[b[0]].force_m(b[1], result)

        return c

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
            self._sim.r(
                Pauli.PauliX, float(operation.params[0]), operation.qubits[0]._index
            )
        elif name == "ry":
            self._sim.r(
                Pauli.PauliY, float(operation.params[0]), operation.qubits[0]._index
            )
        elif name == "rz":
            self._sim.r(
                Pauli.PauliZ, float(operation.params[0]), operation.qubits[0]._index
            )
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
                    self._classical_register = (
                        self._classical_register & (~regbit)
                    ) | (qubit_outcome << cregbit)

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
                raise QrackError("Invalid boolean function relation.")

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

            if is_initializing and (
                (operation.name == "measure") or (operation.name == "reset")
            ):
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
                self._sim = QrackAceBackend(toClone=preamble_sim)
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
            _data = self._add_sample_measure(
                self._sample_qubits, self._sample_clbits, self._shots
            )

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
    def create_noise_model(self, x=0.25, y=0.25):
        if not _IS_QISKIT_AER_AVAILABLE:
            raise RuntimeError(
                "Before trying to run_qiskit_circuit() with QrackAceBackend, you must install Qiskit Aer!"
            )
        noise_model = NoiseModel()

        for a, b in self.get_logical_coupling_map():
            col_a, col_b = a % self._row_length, b % self._row_length
            row_a, row_b = a // self._row_length, b // self._row_length
            is_long_a = self._is_col_long_range[col_a]
            is_long_b = self._is_col_long_range[col_b]

            if is_long_a and is_long_b:
                continue  # No noise on long-to-long

            if (col_a == col_b) or (row_a == row_b):
                continue  # No noise for same column

            if is_long_a or is_long_b:
                y_cy = 1 - (1 - y) ** 2
                y_swap = 1 - (1 - y) ** 3
                noise_model.add_quantum_error(depolarizing_error(y, 2), "cx", [a, b])
                noise_model.add_quantum_error(depolarizing_error(y_cy, 2), "cy", [a, b])
                noise_model.add_quantum_error(depolarizing_error(y_cy, 2), "cz", [a, b])
                noise_model.add_quantum_error(
                    depolarizing_error(y_swap, 2), "swap", [a, b]
                )
            else:
                y_cy = 1 - (1 - y) ** 2
                y_swap = 1 - (1 - y) ** 3
                noise_model.add_quantum_error(depolarizing_error(y_cy, 2), "cx", [a, b])
                noise_model.add_quantum_error(depolarizing_error(y_cy, 2), "cy", [a, b])
                noise_model.add_quantum_error(depolarizing_error(y_cy, 2), "cz", [a, b])
                noise_model.add_quantum_error(
                    depolarizing_error(y_swap, 2), "swap", [a, b]
                )

        return noise_model
