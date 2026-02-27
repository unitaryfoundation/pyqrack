# (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.
import math
import random
import sys

from .qrack_stabilizer import QrackStabilizer
from .pauli import Pauli


_IS_QISKIT_AVAILABLE = True
try:
    from qiskit.circuit.quantumcircuit import QuantumCircuit
except ImportError:
    _IS_QISKIT_AVAILABLE = False


class QrackNearCliffordQecBackend:
    """A back end for near-Clifford quantum error correction

    This back end uses repetition code on a near-Clifford simulator to emulate
    a utility-scale superconducting chip quantum computer in very little memory.

    Attributes:
        sim(QrackSimulator): Array of simulators corresponding to "patches" between boundary rows.
    """

    def __init__(
        self,
        qubit_count=1,
        code_len=3,
        layers_per_qec_round = 2,
        is_eager = False,
        toClone=None,
    ):
        if (code_len < 3) or ((code_len & 1) == 0):
            raise ValueError("QrackNearCliffordQecBackend code_len must be odd and >= 3!")

        if toClone:
            qubit_count = toClone.num_qubits()
            code_len = toClone.code_len
            layers_per_qec_round = toClone.layers
            is_eager = toClone.is_eager
        if qubit_count < 0:
            qubit_count = 0

        self.n_qubits = qubit_count
        self.code_len = code_len
        self.layers = layers_per_qec_round
        self.is_eager = is_eager

        # Allocate (code_len - 1) ancillas
        self.a = [
            self.n_qubits * self.code_len + i
            for i in range(self.code_len - 1)
        ]

        # Apply QEC every n layers
        self.b = [0] * self.n_qubits

        # Only apply QEC or count layers if near-Clifford
        self.c = [False] * self.n_qubits

        total_qubits = self.code_len * self.n_qubits + (self.code_len - 1)

        self.sim = (
            toClone.sim.clone()
            if toClone
            else QrackStabilizer(total_qubits)
        )

    def _correct_bit(self, lq):
        hq = self.code_len * lq

        # --- Compute adjacent parity checks ---
        for i in range(self.code_len - 1):
            self.sim.mcx([hq + i], self.a[i])
            self.sim.mcx([hq + i + 1], self.a[i])

        # --- Measure syndrome ---
        syndrome = [int(self.sim.m(aq)) for aq in self.a]

        # --- Decode syndrome ---
        if any(syndrome):

            # Error on first qubit
            if syndrome[0]:
                error_index = 0

            # Error on last qubit
            elif syndrome[-1]:
                error_index = self.code_len - 1

            else:
                # Find transition from 1 → 0
                error_index = None
                for i in range(len(syndrome) - 1):
                    if syndrome[i] and not syndrome[i + 1]:
                        error_index = i + 1
                        break

                # Fallback (shouldn't happen for single error)
                if error_index is None:
                    error_index = 0

            # Apply correction
            self.sim.x(hq + error_index)

        # --- Reset ancillas ---
        for i, bit in enumerate(syndrome):
            if bit:
                self.sim.x(self.a[i])

    def _correct_phase(self, lq):
        hq = self.code_len * lq

        # Rotate to X basis
        for i in range(self.code_len):
            self.sim.h(hq + i)

        # Run normal bit-flip correction
        self._correct_bit(lq)

        # Rotate back
        for i in range(self.code_len):
            self.sim.h(hq + i)

    def _correct(self, lq, b, p):
        if not self.c[lq]:
            return

        self.b[lq] += 1
        if self.b[lq] % self.layers:
            return
        self.b[lq] = 0

        if p:
            self._correct_phase(lq)
        if b:
            self._correct_bit(lq)

    def _prop_nc(self, lq1, lq2):
        if self.c[lq1]:
            self.c[lq2] = True
        elif self.c[lq2]:
            self.c[lq1] = True

    def clone(self):
        return QrackNearCliffordQecBackend(toClone=self)

    def num_qubits(self):
        return self.n_qubits

    def rz(self, th, lq):
        if math.fmod(abs(th), math.pi / 2) > sys.float_info.epsilon:
            self.c[lq] = True
        hq = self.code_len * lq
        self.sim.set_stochastic(False)
        p = [True, False] * (self.code_len >> 1) + [True]
        random.shuffle(p)
        for q in range(self.code_len):
            self.sim.set_major_quadrant(p[q])
            self.sim.r(Pauli.PauliZ, th, hq + q)

    def h(self, lq):
        hq = self.code_len * lq
        for q in range(self.code_len):
            self.sim.h(hq + q)

    def s(self, lq):
        hq = self.code_len * lq
        for q in range(self.code_len):
            self.sim.s(hq + q)

    def adjs(self, lq):
        hq = self.code_len * lq
        for q in range(self.code_len):
            self.sim.adjs(hq + q)

    def x(self, lq):
        hq = self.code_len * lq
        for q in range(self.code_len):
            self.sim.x(hq + q)

    def y(self, lq):
        hq = self.code_len * lq
        for q in range(self.code_len):
            self.sim.y(hq + q)

    def z(self, lq):
        hq = self.code_len * lq
        for q in range(self.code_len):
            self.sim.z(hq + q)

    def t(self, lq):
        self.c[lq] = True
        hq = self.code_len * lq
        self.sim.set_stochastic(False)
        p = [True, False] * (self.code_len >> 1) + [True]
        random.shuffle(p)
        for q in range(self.code_len):
            self.sim.set_major_quadrant(p[q])
            self.sim.t(hq + q)

    def adjt(self, lq):
        self.c[lq] = True
        hq = self.code_len * lq
        self.sim.set_stochastic(False)
        p = [True, False] * (self.code_len >> 1) + [True]
        random.shuffle(p)
        for q in range(self.code_len):
            self.sim.set_major_quadrant(p[q])
            self.sim.adjt(hq + q)

    def cx(self, lq1, lq2):
        if not self.is_eager:
            self._correct(lq1, True, False)
            self._correct(lq2, False, True)
        self._prop_nc(lq1, lq2)
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.mcx([hq1 + q], hq2 + q)
        if self.is_eager:
            self._correct(lq1, False, True)
            self._correct(lq2, True, False)

    def cy(self, lq1, lq2):
        if not self.is_eager:
            self._correct(lq1, True, False)
            self._correct(lq2, True, True)
        self._prop_nc(lq1, lq2)
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.mcy([hq1 + q], hq2 + q)
        if self.is_eager:
            self._correct(lq1, False, True)
            self._correct(lq2, True, True)

    def cz(self, lq1, lq2):
        if not self.is_eager:
            self._correct(lq1, True, False)
            self._correct(lq2, True, False)
        self._prop_nc(lq1, lq2)
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.mcz([hq1 + q], hq2 + q)
        if self.is_eager:
            self._correct(lq1, False, True)
            self._correct(lq2, False, True)


    def acx(self, lq1, lq2):
        if not self.is_eager:
            self._correct(lq1, True, False)
            self._correct(lq2, False, True)
        self._prop_nc(lq1, lq2)
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.macx([hq1 + q], hq2 + q)
        if self.is_eager:
            self._correct(lq1, False, True)
            self._correct(lq2, True, False)

    def acy(self, lq1, lq2):
        if not self.is_eager:
            self._correct(lq1, True, False)
            self._correct(lq2, True, True)
        self._prop_nc(lq1, lq2)
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.macy([hq1 + q], hq2 + q)
        if self.is_eager:
            self._correct(lq1, False, True)
            self._correct(lq2, True, True)

    def acz(self, lq1, lq2):
        if not self.is_eager:
            self._correct(lq1, True, False)
            self._correct(lq2, True, False)
        self._prop_nc(lq1, lq2)
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.macz([hq1 + q], hq2 + q)
        if self.is_eager:
            self._correct(lq1, False, True)
            self._correct(lq2, False, True)

    def mcx(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackNearCliffordQecBackend.mcx() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self.cx(lq1[0], lq2)

    def mcy(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackNearCliffordQecBackend.mcy() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self.cy(lq1[0], lq2)

    def mcz(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackNearCliffordQecBackend.mcz() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self.cz(lq1[0], lq2)

    def macx(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackNearCliffordQecBackend.macx() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self.acx(lq1[0], lq2)

    def macy(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackNearCliffordQecBackend.macy() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self.acy(lq1[0], lq2)

    def macz(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackNearCliffordQecBackend.macz() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self.acz(lq1[0], lq2)

    def swap(self, lq1, lq2):
        self.b[lq1], self.b[lq2] = self.b[lq2], self.b[lq1]
        self.c[lq1], self.c[lq2] = self.c[lq2], self.c[lq1]
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.swap(hq1 + q, hq2 + q)

    def iswap(self, lq1, lq2):
        if not self.is_eager:
            self._correct(lq1, True, False)
            self._correct(lq2, True, False)
        self._prop_nc(lq1, lq2)
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.iswap(hq1 + q, hq2 + q)
        if self.is_eager:
            self._correct(lq1, False, True)
            self._correct(lq2, False, True)

    def adjiswap(self, lq1, lq2):
        if not self.is_eager:
            self._correct(lq1, True, False)
            self._correct(lq2, True, False)
        self._prop_nc(lq1, lq2)
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.adjiswap(hq1 + q, hq2 + q)
        if self.is_eager:
            self._correct(lq1, False, True)
            self._correct(lq2, False, True)

    def m(self, lq):
        hq = self.code_len * lq
        bits = []
        f = 0
        t = 0
        for q in range(self.code_len):
            # Try to force the most agreement with majority possible
            if ((t << 1) > self.code_len) and (self.sim.prob(hq + q) > 0):
                b = self.sim.force_m(hq + q, True)
            elif ((f << 1) > self.code_len) and (self.sim.prob(hq + q) < 1):
                b = self.sim.force_m(hq + q, False)
            else:
                b = self.sim.m(hq + q)

            if b:
                t += 1
            else:
                f += 1

            bits.append(int(b))

        count = sum(bits)
        result = (count << 1) > self.code_len

        if result:
            for q in range(self.code_len):
                if bits[q] == 0:
                    self.sim.x(hq + q)
        else:
            for q in range(self.code_len):
                if bits[q] == 1:
                    self.sim.x(hq + q)

        self.b[lq] = 0
        self.c[lq] = False

        return result

    def force_m(self, lq, result):
        hq = self.code_len * lq
        bits = []
        for q in range(self.code_len):
            bits.append(int(self.sim.m(hq + q)))

        if result:
            for q in range(self.code_len):
                if bits[q] == 0:
                    self.sim.x(hq + q)
        else:
            for q in range(self.code_len):
                if bits[q] == 1:
                    self.sim.x(hq + q)

        self.b[lq] = 0
        self.c[lq] = False

        return result

    def m_all(self):
        sample = 0
        for i in range(self.n_qubits):
            if self.m(i):
               sample |= 1 << i

        return sample

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

        if (name == "u1") or (name == "p") or (name == "rz"):
            self._sim.rz(float(operation.params[0]), operation.qubits[0]._index)
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
            err_msg = 'QrackNearCliffordQecBackend encountered unrecognized operation "{0}"'
            raise RuntimeError(err_msg.format(operation))

    def _add_sample_measure(self, sample_qubits, sample_clbits):
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
        sample = self._sim.m_all()
        measure_result = 0
        for index in range(len(measure_qubit)):
            qubit = measure_qubit[index]._index
            qubit_outcome = (sample >> qubit) & 1
            measure_result |= qubit_outcome << index

        for index in range(len(measure_qubit)):
            qubit_outcome = (measure_result >> index) & 1
            clbit = measure_clbit[index]._index
            clmask = 1 << clbit
            self._classical_memory = (self._classical_memory & (~clmask)) | (qubit_outcome << clbit)

        data = bin(self._classical_memory)[2:].zfill(self.num_qubits())

        return data

    def run_qiskit_circuit(self, experiment, shots=1):
        if not _IS_QISKIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to run_qiskit_circuit() with QrackNearCliffordQecBackend, you must install Qiskit!"
            )

        instructions = []
        if isinstance(experiment, QuantumCircuit):
            instructions = experiment.data
        else:
            raise RuntimeError('Unrecognized "run_input" argument specified for run().')

        self._sample_qubits = []
        self._sample_clbits = []
        self._sample_cregbits = []
        _data = []

        if (shots < 2):
            self._sim = self
            self._classical_memory = 0
            self._classical_register = 0

            for operation in instructions:
                self._apply_op(operation)

            if (shots > 0) and (len(self._sample_qubits) > 0):
                _data.append(self._add_sample_measure(self._sample_qubits, self._sample_clbits))
        else:
            for shot in range(shots):
                self._sim = QrackNearCliffordQecBackend(toClone=self)
                self._classical_memory = 0
                self._classical_register = 0

                for operation in instructions:
                    self._apply_op(operation)

                if len(self._sample_qubits) > 0:
                    _data.append(self._add_sample_measure(self._sample_qubits, self._sample_clbits))

                del self._sim

        return _data

    @staticmethod
    def get_qiskit_basis_gates():
        return [
            "id",
            "u1",
            "r",
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
