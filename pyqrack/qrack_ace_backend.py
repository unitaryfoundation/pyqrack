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

    The backend was originally designed assuming a 2D qubit grid like 2019 Sycamore.
    However, it quickly became apparent that users can basically design their own
    connectivity topologies, without breaking the concept. (Not all will work equally well.)
    For maximum flexibility, set "alternating_codes=False". (For best performance on
    Sycamore-like topologies, leave it "True.")

    Attributes:
        sim(QrackSimulator): Corresponding simulator.
        alternating_codes(bool): Alternate repetition code elision by index?
        row_length(int): Qubits per row.
        col_length(int): Qubits per column.
    """

    def __init__(
        self,
        qubit_count=-1,
        alternating_codes=True,
        toClone=None
    ):
        self.sim = toClone.sim.clone() if toClone else QrackSimulator(3 * qubit_count)
        self._factor_width(qubit_count)
        self.alternating_codes = alternating_codes


    def _factor_width(self, width):
        col_len = math.floor(math.sqrt(width))
        while (((width // col_len) * col_len) != width):
            col_len -= 1
        row_len = width // col_len

        self.col_length = col_len
        self.row_length = row_len

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


    def _unpack(self, lq, reverse = False):
        return [3 * lq + 2, 3 * lq + 1, 3 * lq] if reverse else [3 * lq, 3 * lq + 1, 3 * lq + 2]


    def _encode(self, hq, reverse = False):
        row = (hq[0] // 3) // self.row_length
        even_row = not (row & 1)
        if ((not self.alternating_codes) and reverse) or (even_row == reverse):
            self._cx_shadow(hq[0], hq[1])
            self.sim.mcx([hq[1]], hq[2])
        else:
            self.sim.mcx([hq[0]], hq[1])
            self._cx_shadow(hq[1], hq[2])


    def _decode(self, hq, reverse = False):
        row = (hq[0] // 3) // self.row_length
        even_row = not (row & 1)
        if ((not self.alternating_codes) and reverse) or (even_row == reverse):
            self.sim.mcx([hq[1]], hq[2])
            self._cx_shadow(hq[0], hq[1])
        else:
            self._cx_shadow(hq[1], hq[2])
            self.sim.mcx([hq[0]], hq[1])


    def u(self, th, ph, lm, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.u(hq[0], th, ph, lm)
        self._encode(hq)


    def r(self, p, th, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.r(p, th, hq[0])
        self._encode(hq)


    def s(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.s(hq[0])
        self._encode(hq)


    def adjs(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.adjs(hq[0])
        self._encode(hq)


    def x(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.x(hq[0])
        self._encode(hq)


    def y(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.y(hq[0])
        self._encode(hq)


    def z(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.z(hq[0])
        self._encode(hq)


    def h(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.h(hq[0])
        self._encode(hq)


    def t(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.t(hq[0])
        self._encode(hq)


    def adjt(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.adjt(hq[0])
        self._encode(hq)


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

        if lq2 == (lq1 + 1):
            hq1 = self._unpack(lq1, True)
            hq2 = self._unpack(lq2, False)
            self._decode(hq1, True)
            self._decode(hq2, False)
            gate([hq1[0]], hq2[0])
            self._encode(hq2, False)
            self._encode(hq1, True)
        elif lq1 == (lq2 + 1):
            hq2 = self._unpack(lq2, True)
            hq1 = self._unpack(lq1, False)
            self._decode(hq2, True)
            self._decode(hq1, False)
            gate([hq1[0]], hq2[0])
            self._encode(hq1, False)
            self._encode(hq2, True)
        else:
            hq1 = self._unpack(lq1)
            hq2 = self._unpack(lq2)
            gate([hq1[0]], hq2[0])
            if self.alternating_codes:
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
        syndrome = 0
        bits = []
        for q in hq:
            bits.append(self.sim.m(q))
            if bits[-1]:
                syndrome += 1
        result = True if (syndrome > 1) else False
        for i in range(len(hq)): 
            if bits[i] != result:
                self.sim.x(hq[i])

        return result


    def m_all(self):
        result = 0
        for lq in range(self.sim.num_qubits() // 3):
            result <<= 1
            if self.m(lq):
                result |= 1

        return result


    def measure_shots(self, q, s):
        _q = []
        for i in q:
            _q.append(3 * i)
            _q.append(3 * i + 1)
            _q.append(3 * i + 2)

        samples = self.sim.measure_shots(_q, s)

        results = []
        for sample in samples:
            logical_sample = 0
            for i in range(len(q)):
                logical_sample <<= 1
                bit_count = 0
                for _ in range(3):
                    if sample & 1:
                        bit_count += 1
                    sample >>= 1
                if bit_count > 1:
                    logical_sample |= 1
            results.append(logical_sample)

        return results


    def _apply_op(self, operation):
        name = operation.name

        if (name == 'id') or (name == 'barrier'):
            # Skip measurement logic
            return

        conditional = getattr(operation, 'conditional', None)
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

        if (name == 'u1') or (name == 'p'):
            self._sim.u(0, 0, float(operation.params[0]), operation.qubits[0]._index)
        elif name == 'u2':
            self._sim.u(
                math.pi / 2,
                float(operation.params[0]),
                float(operation.params[1]),
                operation.qubits[0]._index
            )
        elif (name == 'u3') or (name == 'u'):
            self._sim.u(
                float(operation.params[0]),
                float(operation.params[1]),
                float(operation.params[2]),
                operation.qubits[0]._index
            )
        elif name == 'r':
            self._sim.u(
                float(operation.params[0]),
                float(operation.params[1]) - math.pi / 2,
                (-1 * float(operation.params[1])) + math.pi / 2,
                operation.qubits[0]._index
            )
        elif name == 'rx':
            self._sim.r(Pauli.PauliX, float(operation.params[0]), operation.qubits[0]._index)
        elif name == 'ry':
            self._sim.r(Pauli.PauliY, float(operation.params[0]), operation.qubits[0]._index)
        elif name == 'rz':
            self._sim.r(Pauli.PauliZ, float(operation.params[0]), operation.qubits[0]._index)
        elif name == 'h':
            self._sim.h(operation.qubits[0]._index)
        elif name == 'x':
            self._sim.x(operation.qubits[0]._index)
        elif name == 'y':
            self._sim.y(operation.qubits[0]._index)
        elif name == 'z':
            self._sim.z(operation.qubits[0]._index)
        elif name == 's':
            self._sim.s(operation.qubits[0]._index)
        elif name == 'sdg':
            self._sim.adjs(operation.qubits[0]._index)
        elif name == 't':
            self._sim.t(operation.qubits[0]._index)
        elif name == 'tdg':
            self._sim.adjt(operation.qubits[0]._index)
        elif name == 'cx':
            self._sim.cx(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == 'cy':
            self._sim.cy(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == 'cz':
            self._sim.cz(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == 'dcx':
            self._sim.mcx(operation.qubits[0]._index, operation.qubits[1]._index)
            self._sim.mcx(operation.qubits[1]._index, operation.qubits[0]._index)
        elif name == 'swap':
            self._sim.swap(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == 'iswap':
            self._sim.iswap(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == 'iswap_dg':
            self._sim.adjiswap(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == 'reset':
            qubits = operation.qubits
            for qubit in qubits:
                if self._sim.m(qubit._index):
                    self._sim.x(qubit._index)
        elif name == 'measure':
            qubits = operation.qubits
            clbits = operation.clbits
            cregbits = (
                operation.register
                if hasattr(operation, 'register')
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

        elif name == 'bfunc':
            mask = int(operation.mask, 16)
            relation = operation.relation
            val = int(operation.val, 16)

            cregbit = operation.register
            cmembit = operation.memory if hasattr(operation, 'memory') else None

            compared = (self._classical_register & mask) - val

            if relation == '==':
                outcome = compared == 0
            elif relation == '!=':
                outcome = compared != 0
            elif relation == '<':
                outcome = compared < 0
            elif relation == '<=':
                outcome = compared <= 0
            elif relation == '>':
                outcome = compared > 0
            elif relation == '>=':
                outcome = compared >= 0
            else:
                raise QrackError('Invalid boolean function relation.')

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
            measure_results = self._sim.measure_shots([q._index for q in measure_qubit], num_samples)

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

            if operation.name == 'id' or operation.name == 'barrier':
                continue

            if is_initializing and (
                (operation.name == 'measure') or (operation.name == 'reset')
            ):
                continue

            is_initializing = False

            if (operation.name == 'measure') or (operation.name == 'reset'):
                if boundary_start == -1:
                    boundary_start = opcount

            if (boundary_start != -1) and (operation.name != 'measure'):
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
