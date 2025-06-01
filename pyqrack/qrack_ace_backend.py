# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.
import math
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


class QrackAceBackend:
    """A back end for elided quantum error correction

    This back end uses elided repetition code on a nearest-neighbor topology to emulate
    a utility-scale superconducting chip quantum computer in very little memory.4

    The backend was originally designed assuming an (orbifolded) 2D qubit grid like 2019 Sycamore.
    However, it quickly became apparent that users can basically design their own connectivity topologies,
    without breaking the concept. (Not all will work equally well.) For maximum flexibility, set
    "alternating_codes=False". (For best performance on Sycamore-like topologies,leave it "True.")

    Attributes:
        sim(QrackSimulator): Corresponding simulator.
        alternating_codes(bool): Alternate repetition code elision by index?
        row_length(int): Qubits per row.
        col_length(int): Qubits per column.
        long_range_columns(int): How many ideal rows between QEC boundary rows?
    """

    def __init__(
        self,
        qubit_count=1,
        long_range_columns=2,
        alternating_codes=True,
        reverse_row_and_col=False,
        isTensorNetwork=False,
        isStabilizerHybrid=False,
        isBinaryDecisionTree=False,
        toClone=None,
    ):
        if long_range_columns < 0:
            long_range_columns = 0
        if qubit_count < 0:
            qubit_count = 0
        if toClone:
            qubit_count = toClone.num_qubits()
            long_range_columns = toClone.long_range_columns

        self._factor_width(qubit_count, reverse_row_and_col)
        self.alternating_codes = alternating_codes
        self.long_range_columns = long_range_columns
        self._is_init = [False] * qubit_count

        col_seq = [True] * long_range_columns + [False]
        len_col_seq = len(col_seq)
        self._is_col_long_range = (
            col_seq * ((self.row_length + len_col_seq - 1) // len_col_seq)
        )[: self.row_length]
        if long_range_columns < self.row_length:
            self._is_col_long_range[-1] = False
        self._hardware_offset = []
        tot_qubits = 0
        for _ in range(self.col_length):
            for c in self._is_col_long_range:
                self._hardware_offset.append(tot_qubits)
                tot_qubits += 1 if c else 3
        self._ancilla = tot_qubits
        tot_qubits += 1

        self.sim = (
            toClone.sim.clone()
            if toClone
            else QrackSimulator(
                tot_qubits,
                isTensorNetwork=isTensorNetwork,
                isStabilizerHybrid=isStabilizerHybrid,
                isBinaryDecisionTree=isBinaryDecisionTree,
            )
        )

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
        p1 = self.sim.prob(q1)
        p2 = self.sim.prob(q2)

        if p1 < p2:
            return p2, q1

        return p1, q2

    def _cz_shadow(self, q1, q2):
        prob_max, t = self._ct_pair_prob(q1, q2)
        if prob_max > 0.5:
            self.sim.z(t)

    def _anti_cz_shadow(self, q1, q2):
        self.sim.x(q1)
        self._cz_shadow(q1, q2)
        self.sim.x(q1)

    def _cx_shadow(self, c, t):
        self.sim.h(t)
        self._cz_shadow(c, t)
        self.sim.h(t)

    def _anti_cx_shadow(self, c, t):
        self.sim.x(t)
        self._cx_shadow(c, t)
        self.sim.x(t)

    def _cy_shadow(self, c, t):
        self.sim.adjs(t)
        self._cx_shadow(c, t)
        self.sim.s(t)

    def _anti_cy_shadow(self, c, t):
        self.sim.x(t)
        self._cy_shadow(c, t)
        self.sim.x(t)

    def _ccz_shadow(self, c1, q2, q3):
        self.sim.mcx([q2], q3)
        self.sim.adjt(q3)
        self._cx_shadow(c1, q3)
        self.sim.t(q3)
        self.sim.mcx([q2], q3)
        self.sim.adjt(q3)
        self._cx_shadow(c1, q3)
        self.sim.t(q3)
        self.sim.t(q2)
        self._cx_shadow(c1, q2)
        self.sim.adjt(q2)
        self.sim.t(c1)
        self._cx_shadow(c1, q2)

    def _ccx_shadow(self, c1, q2, q3):
        self.sim.h(q3)
        self._ccz_shadow(c1, q2, q3)
        self.sim.h(q3)

    def _unpack(self, lq, reverse=False):
        offset = self._hardware_offset[lq]

        if self._is_col_long_range[lq % self.row_length]:
            return [offset]

        return (
            [offset + 2, offset + 1, offset]
            if reverse
            else [offset, offset + 1, offset + 2]
        )

    def _encode(self, lq, hq, reverse=False):
        even_row = not ((lq // self.row_length) & 1)
        # Encode shadow-first
        if self._is_init[lq]:
            self._cx_shadow(hq[0], hq[2])
        if ((not self.alternating_codes) and reverse) or (even_row == reverse):
            self.sim.mcx([hq[2]], hq[1])
        else:
            self.sim.mcx([hq[0]], hq[1])
        self._is_init[lq] = True

    def _decode(self, lq, hq, reverse=False):
        if not self._is_init[lq]:
            return
        even_row = not ((lq // self.row_length) & 1)
        if ((not self.alternating_codes) and reverse) or (even_row == reverse):
            # Decode entangled-first
            self.sim.mcx([hq[2]], hq[1])
        else:
            # Decode entangled-first
            self.sim.mcx([hq[0]], hq[1])
        self._cx_shadow(hq[0], hq[2])

    def _correct(self, lq):
        if not self._is_init[lq]:
            return
        # We can't use true syndrome-based error correction,
        # because one of the qubits in the code is separated.
        # However, we can get pretty close!
        shots = 512

        single_bit = 0
        other_bits = []
        if not self.alternating_codes or not ((lq // self.row_length) & 1):
            single_bit = 2
            other_bits = [0, 1]
        else:
            single_bit = 0
            other_bits = [1, 2]

        hq = self._unpack(lq)

        single_bit_value = self.sim.prob(hq[single_bit])
        single_bit_polarization = max(single_bit_value, 1 - single_bit_value)

        # Suggestion from Elara (the custom OpenAI GPT):
        # Create phase parity tie before measurement.
        self._ccx_shadow(hq[single_bit], hq[other_bits[0]], self._ancilla)
        self.sim.mcx([hq[other_bits[1]]], self._ancilla)
        self.sim.force_m(self._ancilla, False)

        samples = self.sim.measure_shots([hq[other_bits[0]], hq[other_bits[1]]], shots)

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
                self.sim.x(hq[error_bit])
            else:
                # The coherent bits carry the error.
                force_syndrome = False
                # Form their syndrome.
                self.sim.mcx([hq[other_bits[0]]], self._ancilla)
                self.sim.mcx([hq[other_bits[1]]], self._ancilla)
                # Force the syndrome pathological
                self.sim.force_m(self._ancilla, True)
                # Reset the ancilla.
                self.sim.x(self._ancilla)
                # Correct the bit flip.
                self.sim.x(hq[error_bit])

        # There is no error.
        if force_syndrome:
            # Form the syndrome of the coherent bits.
            self.sim.mcx([hq[other_bits[0]]], self._ancilla)
            self.sim.mcx([hq[other_bits[1]]], self._ancilla)
            # Force the syndrome non-pathological.
            self.sim.force_m(self._ancilla, False)

    def _correct_if_like_h(self, th, lq):
        if not self._is_init[lq]:
            return
        while th > math.pi:
            th -= 2 * math.pi
        while th <= -math.pi:
            th += 2 * math.pi
        th = abs(th)
        if not math.isclose(th, 0):
            self._correct(lq)

    def u(self, lq, th, ph, lm):
        hq = self._unpack(lq)
        if self._is_col_long_range[lq % self.row_length]:
            self.sim.u(hq[0], th, ph, lm)
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
            self._correct_if_like_h(th, lq)
            self._decode(lq, hq)
            self.sim.u(hq[0], th, ph, lm)
            if not self._is_init[lq]:
                self.sim.u(hq[2], th, ph, lm)
            self._encode(lq, hq)
        else:
            # Shouldn't produce/destroy superposition
            for b in hq:
                self.sim.u(b, th, ph, lm)

    def r(self, p, th, lq):
        hq = self._unpack(lq)
        if self._is_col_long_range[lq % self.row_length]:
            self.sim.r(p, th, hq[0])
            return

        while th > math.pi:
            th -= 2 * math.pi
        while th <= -math.pi:
            th += 2 * math.pi
        if p == Pauli.PauliY:
            self._correct_if_like_h(th, lq)

        if (p == Pauli.PauliZ) or math.isclose(abs(th), math.pi):
            # Doesn't produce/destroy superposition
            for b in hq:
                self.sim.r(p, th, b)
        else:
            # Produces/destroys superposition
            self._decode(lq, hq)
            self.sim.r(p, th, hq[0])
            if not self._is_init[lq]:
                self.sim.r(p, th, hq[2])
            self._encode(lq, hq)

    def h(self, lq):
        hq = self._unpack(lq)
        if self._is_col_long_range[lq % self.row_length]:
            self.sim.h(hq[0])
            return

        self._decode(lq, hq)
        self.sim.h(hq[0])
        if not self._is_init[lq]:
            self.sim.h(hq[2])
        self._encode(lq, hq)

    def s(self, lq):
        hq = self._unpack(lq)
        if self._is_col_long_range[lq % self.row_length]:
            self.sim.s(hq[0])
            return

        for b in hq:
            self.sim.s(b)

    def adjs(self, lq):
        hq = self._unpack(lq)
        if self._is_col_long_range[lq % self.row_length]:
            self.sim.adjs(hq[0])
            return

        for b in hq:
            self.sim.adjs(b)

    def x(self, lq):
        hq = self._unpack(lq)
        if self._is_col_long_range[lq % self.row_length]:
            self.sim.x(hq[0])
            return

        for b in hq:
            self.sim.x(b)

    def y(self, lq):
        hq = self._unpack(lq)
        if self._is_col_long_range[lq % self.row_length]:
            self.sim.y(hq[0])
            return

        for b in hq:
            self.sim.y(b)

    def z(self, lq):
        hq = self._unpack(lq)
        if self._is_col_long_range[lq % self.row_length]:
            self.sim.z(hq[0])
            return

        for b in hq:
            self.sim.z(b)

    def t(self, lq):
        hq = self._unpack(lq)
        if self._is_col_long_range[lq % self.row_length]:
            self.sim.t(hq[0])
            return

        for b in hq:
            self.sim.t(b)

    def adjt(self, lq):
        hq = self._unpack(lq)
        if self._is_col_long_range[lq % self.row_length]:
            self.sim.adjt(hq[0])
            return

        for b in hq:
            self.sim.adjt(b)

    def _cpauli(self, lq1, lq2, anti, pauli):
        gate = None
        shadow = None
        if pauli == Pauli.PauliX:
            gate = self.sim.macx if anti else self.sim.mcx
            shadow = self._anti_cx_shadow if anti else self._cx_shadow
        elif pauli == Pauli.PauliY:
            gate = self.sim.macy if anti else self.sim.mcy
            shadow = self._anti_cy_shadow if anti else self._cy_shadow
        elif pauli == Pauli.PauliZ:
            gate = self.sim.macz if anti else self.sim.mcz
            shadow = self._anti_cz_shadow if anti else self._cz_shadow
        else:
            return

        lq1_lr = self._is_col_long_range[lq1 % self.row_length]
        lq2_lr = self._is_col_long_range[lq2 % self.row_length]
        if lq1_lr and lq2_lr:
            gate(self._unpack(lq1), self._unpack(lq2)[0])
            return

        self._correct(lq1)

        if not self._is_init[lq1]:
            hq1 = self._unpack(lq1)
            hq2 = self._unpack(lq2)
            if lq1_lr:
                self._decode(lq2, hq2)
                gate(hq1, hq2[0])
                self._encode(lq2, hq2)
            elif lq2_lr:
                self._decode(lq1, hq1)
                gate([hq1[0]], hq2[0])
                self._encode(lq1, hq1)
            else:
                gate([hq1[0]], hq2[0])
                gate([hq1[1]], hq2[1])
                gate([hq1[2]], hq2[2])

            return

        lq1_row = lq1 // self.row_length
        lq1_col = lq1 % self.row_length
        lq2_row = lq2 // self.row_length
        lq2_col = lq2 % self.row_length

        hq1 = None
        hq2 = None
        if (lq2_row == lq1_row) and (((lq1_col + 1) % self.row_length) == lq2_col):
            if lq1_lr:
                self._correct(lq2)
                hq1 = self._unpack(lq1)
                hq2 = self._unpack(lq2, False)
                self._decode(lq2, hq2, False)
                gate(hq1, hq2[0])
                self._encode(lq2, hq2, False)
            elif lq2_lr:
                hq1 = self._unpack(lq1, True)
                hq2 = self._unpack(lq2)
                self._decode(lq1, hq1, True)
                gate([hq1[0]], hq2[0])
                self._encode(lq1, hq1, True)
            else:
                self._correct(lq2)
                hq1 = self._unpack(lq1, True)
                hq2 = self._unpack(lq2, False)
                self._decode(lq1, hq1, True)
                self._decode(lq2, hq2, False)
                gate([hq1[0]], hq2[0])
                self._encode(lq2, hq2, False)
                self._encode(lq1, hq1, True)
        elif (lq1_row == lq2_row) and (((lq2_col + 1) % self.row_length) == lq1_col):
            if lq1_lr:
                self._correct(lq2)
                hq2 = self._unpack(lq2, True)
                hq1 = self._unpack(lq1)
                self._decode(lq2, hq2, True)
                gate(hq1, hq2[0])
                self._encode(lq2, hq2, True)
            elif lq2_lr:
                hq2 = self._unpack(lq2)
                hq1 = self._unpack(lq1, False)
                self._decode(lq1, hq1, False)
                gate([hq1[0]], hq2[0])
                self._encode(lq1, hq1, False)
            else:
                self._correct(lq2)
                hq2 = self._unpack(lq2, True)
                hq1 = self._unpack(lq1, False)
                self._decode(lq2, hq2, True)
                self._decode(lq1, hq1, False)
                gate([hq1[0]], hq2[0])
                self._encode(lq1, hq1, False)
                self._encode(lq2, hq2, True)
        else:
            hq1 = self._unpack(lq1)
            hq2 = self._unpack(lq2)
            if lq1_lr:
                self._correct(lq2)
                self._decode(lq2, hq2)
                gate(hq1, hq2[0])
                self._encode(lq2, hq2)
            elif lq2_lr:
                self._decode(lq1, hq1)
                gate([hq1[0]], hq2[0])
                self._encode(lq1, hq1)
            else:
                gate([hq1[0]], hq2[0])
                if self.alternating_codes and ((lq2_row & 1) != (lq1_row & 1)):
                    shadow(hq1[1], hq2[1])
                else:
                    gate([hq1[1]], hq2[1])
                gate([hq1[2]], hq2[2])

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
        self._is_init[lq] = False
        hq = self._unpack(lq)
        if self._is_col_long_range[lq % self.row_length]:
            return self.sim.m(hq[0])

        if not self.alternating_codes or not ((lq // self.row_length) & 1):
            single_bit = 2
            other_bits = [0, 1]
        else:
            single_bit = 0
            other_bits = [1, 2]
        # The syndrome of "other_bits" is guaranteed to be fixed, after this.
        self._correct(lq)
        syndrome = self.sim.m(hq[other_bits[0]])
        syndrome += self.sim.force_m(hq[other_bits[1]], bool(syndrome))
        # The two separable parts of the code are correlated,
        # but not non-locally, via entanglement.
        # Collapse the other separable part toward agreement.
        syndrome += self.sim.force_m(hq[single_bit], bool(syndrome))

        return True if (syndrome > 1) else False

    def force_m(self, lq, c):
        hq = self._unpack(lq)
        self._is_init[lq] = False
        if self._is_col_long_range[lq % self.row_length]:
            return self.sim.force_m(hq[0])

        self._correct(lq)
        for q in hq:
            self.sim.force_m(q, c)

        return c

    def m_all(self):
        result = 0
        # Randomize the order of measurement to amortize error.
        # However, locality of collapse matters:
        # always measure across rows, and by row directionality.
        rows = list(range(self.col_length))
        random.shuffle(rows)
        for lq_row in rows:
            col_offset = random.randint(0, self.row_length - 1)
            col_reverse = self.alternating_codes and (lq_row & 1)
            for c in range(self.row_length):
                lq_col = (
                    ((self.row_length - (c + 1)) if col_reverse else c) + col_offset
                ) % self.row_length
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
        if self._is_col_long_range[lq % self.row_length]:
            return self.sim.prob(hq[0])

        self._correct(lq)
        if not self.alternating_codes or not ((lq // self.row_length) & 1):
            other_bits = [0, 1]
        else:
            other_bits = [1, 2]
        self.sim.mcx([hq[other_bits[0]]], hq[other_bits[1]])
        result = self.sim.prob(hq[other_bits[0]])
        self.sim.mcx([hq[other_bits[0]]], hq[other_bits[1]])

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

    # Provided by Elara (custom OpenAI GPT)
    def generate_logical_coupling_map(self):
        coupling_map = []
        for y in range(self._col_length):
            for x in range(self._row_length):
                q = y * self._row_length + x
                # Define neighbors with orbifolding
                neighbors = []
                neighbors.append((x + 1) % self._row_length + y * self._row_length)
                neighbors.append(x + ((y + 1) % self._col_length) * self._row_length)
                for nq in neighbors:
                    coupling_map.append([q, nq])

        return coupling_map
