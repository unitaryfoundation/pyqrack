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
        row_length(int): Qubits per row.
        col_length(int): Qubits per column.
        long_range_columns(int): How many ideal rows between QEC boundary rows?
    """

    def __init__(
        self,
        qubit_count=1,
        long_range_columns=-1,
        reverse_row_and_col=False,
        isTensorNetwork=False,
        isStabilizerHybrid=False,
        isBinaryDecisionTree=False,
        toClone=None,
    ):
        if qubit_count < 0:
            qubit_count = 0
        if toClone:
            qubit_count = toClone.num_qubits()
            long_range_columns = toClone.long_range_columns

        self._factor_width(qubit_count, reverse_row_and_col)
        if long_range_columns < 0:
            long_range_columns = 3 if (self.row_length % 3) == 1 else 2
        self.long_range_columns = long_range_columns

        self._coupling_map = None

        # If there's only one or zero "False" columns,
        # the entire simulator is connected, anyway.
        len_col_seq = long_range_columns + 1
        sim_count = (self.row_length + len_col_seq - 1) // len_col_seq
        if (long_range_columns + 1) >= self.row_length:
            self._is_col_long_range = [True] * self.row_length
        else:
            col_seq = [True] * long_range_columns + [False]
            self._is_col_long_range = (col_seq * sim_count)[: self.row_length]
            if long_range_columns < self.row_length:
                self._is_col_long_range[-1] = False

        self._qubit_dict = {}
        self._hardware_offset = []
        self._ancilla = [0] * sim_count
        sim_counts = [0] * sim_count
        sim_id = 0
        tot_qubits = 0
        for r in range(self.col_length):
            for c in self._is_col_long_range:
                self._hardware_offset.append(tot_qubits)
                if c:
                    self._qubit_dict[tot_qubits] = (sim_id, sim_counts[sim_id])
                    tot_qubits += 1
                    sim_counts[sim_id] += 1
                elif (r & 1):
                    for _ in range(2):
                        self._qubit_dict[tot_qubits] = (sim_id, sim_counts[sim_id])
                        tot_qubits += 1
                        sim_counts[sim_id] += 1
                    sim_id = (sim_id + 1) % sim_count
                    self._qubit_dict[tot_qubits] = (sim_id, sim_counts[sim_id])
                    tot_qubits += 1
                    sim_counts[sim_id] += 1
                else:
                    self._qubit_dict[tot_qubits] = (sim_id, sim_counts[sim_id])
                    tot_qubits += 1
                    sim_counts[sim_id] += 1
                    sim_id = (sim_id + 1) % sim_count
                    for _ in range(2):
                        self._qubit_dict[tot_qubits] = (sim_id, sim_counts[sim_id])
                        tot_qubits += 1
                        sim_counts[sim_id] += 1

        self.sim = []
        for i in range(sim_count):
            self._ancilla[i] = sim_counts[i]
            sim_counts[i] += 1
            self.sim.append(
                toClone.sim[i].clone()
                if toClone
                else QrackSimulator(
                    sim_counts[i],
                    isTensorNetwork=isTensorNetwork,
                    isStabilizerHybrid=isStabilizerHybrid,
                    isBinaryDecisionTree=isBinaryDecisionTree,
                )
            )

            # You can still "monkey-patch" this, after the constructor.
            if "QRACK_QUNIT_SEPARABILITY_THRESHOLD" not in os.environ:
                self.sim[i].set_sdrp(0.045)

    def clone(self):
        return QrackAceBackend(toClone=self)

    def num_qubits(self):
        return self.row_length * self.col_length

    def _factor_width(self, width, reverse=False):
        col_len = math.floor(math.sqrt(width))
        while ((width // col_len) * col_len) != width:
            col_len -= 1
        row_len = width // col_len

        self.col_length = row_len if reverse else col_len
        self.row_length = col_len if reverse else row_len

    def _ct_pair_prob(self, q1, q2):
        p1 = self.sim[q1[0]].prob(q1[1])
        p2 = self.sim[q2[0]].prob(q2[1])

        if p1 < p2:
            return p2, q1

        return p1, q2

    def _cz_shadow(self, q1, q2):
        prob_max, t = self._ct_pair_prob(q1, q2)
        if prob_max > 0.5:
            self.sim[t[0]].z(t[1])

    def _anti_cz_shadow(self, c, t):
        self.sim[c[0]].x(c[1])
        self._cz_shadow(c, t)
        self.sim[c[0]].x(c[1])

    def _cx_shadow(self, c, t):
        self.sim[t[0]].h(t[1])
        self._cz_shadow(c, t)
        self.sim[t[0]].h(t[1])

    def _anti_cx_shadow(self, c, t):
        self.sim[c[0]].x(c[1])
        self._cx_shadow(c, t)
        self.sim[c[0]].x(c[1])

    def _cy_shadow(self, c, t):
        self.sim[t[0]].adjs(t[1])
        self._cx_shadow(c, t)
        self.sim[t[0]].s(t[1])

    def _anti_cy_shadow(self, c, t):
        self.sim[c[0]].x(c[1])
        self._cy_shadow(c, t)
        self.sim[c[0]].x(c[1])

    def _ccz_shadow(self, c1, q2, q3):
        self.sim[q2[0]].mcx([q2[1]], q3[1])
        self.sim[q3[0]].adjt(q3[1])
        self._cx_shadow(c1, q3)
        self.sim[q3[0]].t(q3[1])
        self.sim[q2[0]].mcx([q2[1]], q3[1])
        self.sim[q3[0]].adjt(q3[1])
        self._cx_shadow(c1, q3)
        self.sim[q3[0]].t(q3[1])
        self.sim[q2[0]].t(q2[1])
        self._cx_shadow(c1, q2)
        self.sim[q2[0]].adjt(q2[1])
        self.sim[c1[0]].t(c1[1])
        self._cx_shadow(c1, q2)

    def _ccx_shadow(self, c1, q2, t):
        self.sim[t[0]].h(t[1])
        self._ccz_shadow(c1, q2, t)
        self.sim[t[0]].h(t[1])

    def _unpack(self, lq):
        offset = self._hardware_offset[lq]

        if self._is_col_long_range[lq % self.row_length]:
            return [self._qubit_dict[offset]]

        return [
            self._qubit_dict[offset],
            self._qubit_dict[offset + 1],
            self._qubit_dict[offset + 2],
        ]

    def _encode_decode(self, lq, hq):
        if len(hq) < 2:
            return
        if hq[0][0] == hq[1][0]:
            b0 = hq[0]
            self.sim[b0[0]].mcx([b0[1]], hq[1][1])
        else:
            b2 = hq[2]
            self.sim[b2[0]].mcx([b2[1]], hq[1][1])

    def _encode_decode_half(self, lq, hq, toward_0):
        if len(hq) < 2:
            return
        if toward_0 and (hq[0][0] == hq[1][0]):
            b0 = hq[0]
            self.sim[b0[0]].mcx([b0[1]], hq[1][1])
        elif not toward_0 and (hq[2][0] == hq[1][0]):
            b2 = hq[2]
            self.sim[b2[0]].mcx([b2[1]], hq[1][1])

    def _correct(self, lq, phase=False):
        if self._is_col_long_range[lq % self.row_length]:
            return
        # We can't use true syndrome-based error correction,
        # because one of the qubits in the code is separated.
        # However, we can get pretty close!
        shots = 512

        single_bit = 0
        other_bits = []
        hq = self._unpack(lq)

        if phase:
            for b in hq:
                self.sim[b[0]].h(b[1])

        if hq[0][0] == hq[1][0]:
            single_bit = 2
            other_bits = [0, 1]
        elif hq[1][0] == hq[2][0]:
            single_bit = 0
            other_bits = [1, 2]
        else:
            raise RuntimeError("Invalid boundary qubit!")

        ancilla_sim = hq[other_bits[0]][0]
        ancilla = self._ancilla[ancilla_sim]

        single_bit_value = self.sim[hq[single_bit][0]].prob(hq[single_bit][1])
        single_bit_polarization = max(single_bit_value, 1 - single_bit_value)

        # Suggestion from Elara (the custom OpenAI GPT):
        # Create phase parity tie before measurement.
        self._ccx_shadow(hq[single_bit], hq[other_bits[0]], [ancilla_sim, ancilla])
        self.sim[ancilla_sim].mcx([hq[other_bits[1]][1]], ancilla)
        self.sim[ancilla_sim].force_m(ancilla, False)

        samples = self.sim[ancilla_sim].measure_shots(
            [hq[other_bits[0]][1], hq[other_bits[1]][1]], shots
        )

        syndrome_indices = (
            [other_bits[1], other_bits[0]]
            if (single_bit_value >= 0.5)
            else [other_bits[0], other_bits[1]]
        )
        syndrome = [0, 0, 0]
        values = []
        for sample in samples:
            match sample:
                case 0:
                    value = single_bit_value
                    syndrome[single_bit] += value
                case 1:
                    value = single_bit_polarization
                    syndrome[syndrome_indices[0]] += value
                case 2:
                    value = single_bit_polarization
                    syndrome[syndrome_indices[1]] += value
                case 3:
                    value = 1 - single_bit_value
                    syndrome[single_bit] += value
            values.append(value)

        # Suggestion from Elara (custom OpenAI GPT):
        # Compute the standard deviation and only correct if we're outside a confidence interval.
        # (This helps avoid limit-point over-correction.)
        syndrome_sum = sum(syndrome)
        z_score_numer = syndrome_sum - shots / 2
        z_score = 0
        if z_score_numer > 0:
            syndrome_component_mean = syndrome_sum / shots
            syndrome_total_variance = sum(
                (value - syndrome_component_mean) ** 2 for value in values
            )
            z_score_denom = math.sqrt(syndrome_total_variance)
            z_score = (
                math.inf
                if math.isclose(z_score_denom, 0)
                else (z_score_numer / z_score_denom)
            )

        force_syndrome = True
        # (From Elara, this is the value that minimizes the sum of Type I and Type II error.)
        if z_score >= (497 / 999):
            # There is an error.
            error_bit = syndrome.index(max(syndrome))
            if error_bit == single_bit:
                # The stand-alone bit carries the error.
                self.sim[hq[error_bit][0]].x(hq[error_bit][1])
            else:
                # The coherent bits carry the error.
                force_syndrome = False
                # Form their syndrome.
                self.sim[ancilla_sim].mcx([hq[other_bits[0]][1]], ancilla)
                self.sim[ancilla_sim].mcx([hq[other_bits[1]][1]], ancilla)
                # Force the syndrome pathological
                self.sim[ancilla_sim].force_m(ancilla, True)
                # Reset the ancilla.
                self.sim[ancilla_sim].x(ancilla)
                # Correct the bit flip.
                self.sim[ancilla_sim].x(hq[error_bit][1])

        # There is no error.
        if force_syndrome:
            # Form the syndrome of the coherent bits.
            self.sim[ancilla_sim].mcx([hq[other_bits[0]][1]], ancilla)
            self.sim[ancilla_sim].mcx([hq[other_bits[1]][1]], ancilla)
            # Force the syndrome non-pathological.
            self.sim[ancilla_sim].force_m(ancilla, False)

        if phase:
            for b in hq:
                self.sim[b[0]].h(b[1])

    def u(self, lq, th, ph, lm):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].u(b[1], th, ph, lm)
            return

        while ph > math.pi:
            ph -= 2 * math.pi
        while ph <= -math.pi:
            ph += 2 * math.pi
        while lm > math.pi:
            lm -= 2 * math.pi
        while lm <= -math.pi:
            lm += 2 * math.pi

        if not math.isclose(ph, -lm) and not math.isclose(abs(ph), math.pi / 2):
            # Produces/destroys superposition
            self._encode_decode(lq, hq)
            b = hq[0]
            self.sim[b[0]].u(b[1], th, ph, lm)
            b = hq[2]
            self.sim[b[0]].u(b[1], th, ph, lm)
            self._encode_decode(lq, hq)
            self._correct(lq)
        else:
            # Shouldn't produce/destroy superposition
            for b in hq:
                self.sim[b[0]].u(b[1], th, ph, lm)

    def r(self, p, th, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].r(p, th, b[1])
            return

        while th > math.pi:
            th -= 2 * math.pi
        while th <= -math.pi:
            th += 2 * math.pi
        if (p == Pauli.PauliZ) or math.isclose(abs(th), math.pi):
            # Doesn't produce/destroy superposition
            for b in hq:
                self.sim[b[0]].r(p, th, b[1])
        else:
            # Produces/destroys superposition
            self._encode_decode(lq, hq)
            b = hq[0]
            self.sim[b[0]].r(p, th, b[1])
            b = hq[2]
            self.sim[b[0]].r(p, th, b[1])
            self._encode_decode(lq, hq)
            self._correct(lq)

    def h(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].h(b[1])
            return

        self._encode_decode(lq, hq)
        b = hq[0]
        self.sim[b[0]].h(b[1])
        b = hq[2]
        self.sim[b[0]].h(b[1])
        self._encode_decode(lq, hq)
        self._correct(lq)

    def s(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].s(b[1])
            return

        for b in hq:
            self.sim[b[0]].s(b[1])

    def adjs(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].adjs(b[1])
            return

        for b in hq:
            self.sim[b[0]].adjs(b[1])

    def x(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].x(b[1])
            return

        for b in hq:
            self.sim[b[0]].x(b[1])

    def y(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].y(b[1])
            return

        for b in hq:
            self.sim[b[0]].y(b[1])

    def z(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].z(b[1])
            return

        for b in hq:
            self.sim[b[0]].z(b[1])

    def t(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].t(b[1])
            return

        for b in hq:
            self.sim[b[0]].t(b[1])

    def adjt(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            self.sim[b[0]].adjt(b[1])
            return

        for b in hq:
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

    def _cpauli(self, lq1, lq2, anti, pauli):
        lq1_lr = self._is_col_long_range[lq1 % self.row_length]
        lq2_lr = self._is_col_long_range[lq2 % self.row_length]

        lq1_row = lq1 // self.row_length
        lq1_col = lq1 % self.row_length
        lq2_row = lq2 // self.row_length
        lq2_col = lq2 % self.row_length

        connected_cols = []
        c = (lq1_col - 1) % self.row_length
        while self._is_col_long_range[c] and (
            len(connected_cols) < (self.row_length - 1)
        ):
            connected_cols.append(c)
            c = (c - 1) % self.row_length
        if len(connected_cols) < (self.row_length - 1):
            connected_cols.append(c)
        boundary = len(connected_cols)
        c = (lq1_col + 1) % self.row_length
        while self._is_col_long_range[c] and (
            len(connected_cols) < (self.row_length - 1)
        ):
            connected_cols.append(c)
            c = (c + 1) % self.row_length
        if len(connected_cols) < (self.row_length - 1):
            connected_cols.append(c)

        hq1 = self._unpack(lq1)
        hq2 = self._unpack(lq2)

        if lq1_lr and lq2_lr:
            b1 = hq1[0]
            b2 = hq2[0]
            gate, shadow = self._get_gate(pauli, anti, b1[0])
            if lq2_col in connected_cols:
                gate([b1[1]], b2[1])
            else:
                shadow(b1, b2)
            return

        if (lq2_col in connected_cols) and (connected_cols.index(lq2_col) < boundary):
            # lq2_col < lq1_col
            self._encode_decode_half(lq1, hq1, True)
            self._encode_decode_half(lq2, hq2, False)
            b = hq1[0]
            if lq1_lr:
                self._get_gate(pauli, anti, hq1[0][0])[0]([b[1]], hq2[2][1])
            elif lq2_lr:
                self._get_gate(pauli, anti, hq2[0][0])[0]([b[1]], hq2[0][1])
            else:
                self._get_gate(pauli, anti, b[0])[0]([b[1]], hq2[2][1])
            self._encode_decode_half(lq2, hq2, False)
            self._encode_decode_half(lq1, hq1, True)
        elif lq2_col in connected_cols:
            # lq1_col < lq2_col
            self._encode_decode_half(lq1, hq1, False)
            self._encode_decode_half(lq2, hq2, True)
            b = hq2[0]
            if lq1_lr:
                self._get_gate(pauli, anti, hq1[0][0])[0]([hq1[0][1]], b[1])
            elif lq2_lr:
                self._get_gate(pauli, anti, hq2[0][0])[0]([hq1[2][1]], b[1])
            else:
                self._get_gate(pauli, anti, b[0])[0]([hq1[2][1]], b[1])
            self._encode_decode_half(lq2, hq2, True)
            self._encode_decode_half(lq1, hq1, False)
        elif lq1_col == lq2_col:
            # Both are in the same boundary column.
            b = hq1[0]
            gate, shadow = self._get_gate(pauli, anti, b[0])
            gate([b[1]], hq2[0][1])
            if (lq1_row & 1) == (lq2_row & 1):
                b = hq1[1]
                gate, shadow = self._get_gate(pauli, anti, b[0])
                gate([b[1]], hq2[1][1])
            else:
                shadow(hq1[1], hq2[1])
            b = hq1[2]
            gate, shadow = self._get_gate(pauli, anti, b[0])
            gate([b[1]], hq2[2][1])
        else:
            # The qubits have no quantum connection.
            gate, shadow = self._get_gate(pauli, anti, hq1[0][0])
            if lq1_lr:
                connected01 = (hq2[0][0] == hq2[1][0])
                self._encode_decode(lq2, hq2)
                shadow(hq1[0], hq2[0] if connected01 else hq2[2])
                self._encode_decode(lq2, hq2)
            elif lq2_lr:
                connected01 = (hq1[0][0] == hq1[1][0])
                self._encode_decode(lq1, hq1)
                shadow(hq1[0] if connected01 else hq1[2], hq2[0])
                self._encode_decode(lq1, hq1)
            else:
                self._encode_decode(lq1, hq1)
                self._encode_decode(lq2, hq2)
                shadow(hq1[0], hq2[0])
                shadow(hq1[2], hq2[2])
                self._encode_decode(lq2, hq2)
                self._encode_decode(lq1, hq1)

        self._correct(lq1, True)
        if pauli != Pauli.PauliZ:
            self._correct(lq2, False)
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

    def m(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            return self.sim[b[0]].m(b[1])

        if hq[0][0] == hq[0][1]:
            single_bit = 2
            other_bits = [0, 1]
        else:
            single_bit = 0
            other_bits = [1, 2]
        # The syndrome of "other_bits" is guaranteed to be fixed, after this.
        self._correct(lq)
        b = hq[other_bits[0]]
        syndrome = self.sim[b[0]].m(b[1])
        b = hq[other_bits[1]]
        syndrome += self.sim[b[0]].force_m(b[1], bool(syndrome))
        # The two separable parts of the code are correlated,
        # but not non-locally, via entanglement.
        # Collapse the other separable part toward agreement.
        b = hq[single_bit]
        syndrome += self.sim[b[0]].force_m(b[1], bool(syndrome))

        return True if (syndrome > 1) else False

    def force_m(self, lq, c):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            return self.sim[b[0]].force_m(b[1])

        self._correct(lq)
        for q in hq:
            self.sim[q[0]].force_m(q[1], c)

        return c

    def m_all(self):
        result = 0
        # Randomize the order of measurement to amortize error.
        # However, locality of collapse matters:
        # measure in row pairs, and always across rows,
        # and by row directionality.
        row_pairs = list(range((self.col_length + 1) // 2))
        random.shuffle(row_pairs)
        for row_pair in row_pairs:
            col_offset = random.randint(0, self.row_length - 1)
            lq_row = row_pair << 1
            for c in range(self.row_length):
                lq_col = (c + col_offset) % self.row_length
                lq = lq_row * self.row_length + lq_col
                if self.m(lq):
                    result |= 1 << lq
            lq_row += 1
            if lq_row == self.col_length:
                continue
            for c in range(self.row_length):
                lq_col = ((self.row_length - (c + 1)) + col_offset) % self.row_length
                lq = lq_row * self.row_length + lq_col
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

    def prob(self, lq):
        hq = self._unpack(lq)
        if len(hq) < 2:
            b = hq[0]
            return self.sim[b[0]].prob(b[1])

        self._correct(lq)
        if hq[0][0] == hq[1][0]:
            other_bits = [0, 1]
        else:
            other_bits = [1, 2]
        b0 = hq[other_bits[0]]
        b1 = hq[other_bits[1]]
        self.sim[b0[0]].mcx([b0[1]], b1[1])
        result = self.sim[b0[0]].prob(b0[1])
        self.sim[b0[0]].mcx([b0[1]], b1[1])

        return result

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
        rows, cols = self.row_length, self.col_length

        # Map each column index to its full list of logical qubit indices
        def logical_index(row, col):
            return row * cols + col

        for col in range(cols):
            connected_cols = [col]
            c = (col - 1) % cols
            while self._is_col_long_range[c] and (
                len(connected_cols) < self.row_length
            ):
                connected_cols.append(c)
                c = (c - 1) % cols
            if len(connected_cols) < self.row_length:
                connected_cols.append(c)
            c = (col + 1) % cols
            while self._is_col_long_range[c] and (
                len(connected_cols) < self.row_length
            ):
                connected_cols.append(c)
                c = (c + 1) % cols
            if len(connected_cols) < self.row_length:
                connected_cols.append(c)

            for row in range(rows):
                a = logical_index(row, col)
                for c in connected_cols:
                    for r in range(0, rows):
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
            col_a, col_b = a % self.row_length, b % self.row_length
            row_a, row_b = a // self.row_length, b // self.row_length
            is_long_a = self._is_col_long_range[col_a]
            is_long_b = self._is_col_long_range[col_b]

            if is_long_a and is_long_b:
                continue  # No noise on long-to-long

            same_col = col_a == col_b

            if same_col:
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
