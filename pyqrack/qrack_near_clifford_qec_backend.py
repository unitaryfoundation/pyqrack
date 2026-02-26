# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.
import math

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
        toClone=None,
    ):
        if toClone:
            qubit_count = toClone.num_qubits()
        if qubit_count < 0:
            qubit_count = 0
        self.n_qubits = qubit_count
        self.code_len = 3
        self.a0 = self.n_qubits * self.code_len
        self.a1 = self.n_qubits * self.code_len + 1

        self.sim = toClone.sim.clone() if toClone else QrackStabilizer(self.code_len * self.n_qubits + 2)

    def clone(self):
        return QrackAceBackend(toClone=self)

    def num_qubits(self):
        return self.n_qubits

    def _correct(self, lq):
        hq = self.code_len * lq
        self.sim.mcx([hq], self.a0)
        self.sim.mcx([hq + 1], self.a0)
        self.sim.mcx([hq + 1], self.a1)
        self.sim.mcx([hq + 2], self.a1)
        b0 = self.sim.m(self.a0)
        b1 = self.sim.m(self.a1)
        if b0 and b1:
            self.sim.x(hq + 1)
        elif b0:
            self.sim.x(hq)
        elif b1:
            self.sim.x(hq + 2)
        if b0:
            self.sim.x(self.a0)
        if b1:
            self.sim.x(self.a1)

    def rz(self, th, lq):
        hq = self.code_len * lq
        for q in range(self.code_len):
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
        hq = self.code_len * lq
        for q in range(self.code_len):
            self.sim.t(hq + q)

    def adjt(self, lq):
        hq = self.code_len * lq
        for q in range(self.code_len):
            self.sim.adjt(hq + q)

    def cx(self, lq1, lq2):
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.mcx([hq1 + q], hq2 + q)
        self._correct(lq1)
        self._correct(lq2)

    def cy(self, lq1, lq2):
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.mcy([hq1 + q], hq2 + q)
        self._correct(lq1)
        self._correct(lq2)

    def cz(self, lq1, lq2):
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.mcz([hq1 + q], hq2 + q)
        self._correct(lq1)
        self._correct(lq2)

    def acx(self, lq1, lq2):
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.macx([hq1 + q], hq2 + q)
        self._correct(lq1)
        self._correct(lq2)

    def acy(self, lq1, lq2):
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.macy([hq1 + q], hq2 + q)
        self._correct(lq1)
        self._correct(lq2)

    def acz(self, lq1, lq2):
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.macz([hq1 + q], hq2 + q)
        self._correct(lq1)
        self._correct(lq2)

    def mcx(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackAceBackend.mcx() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self.cx(lq1, lq2)

    def mcy(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackAceBackend.mcy() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self.cy(lq1, lq2)

    def mcz(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackAceBackend.mcz() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self.cz(lq1, lq2)

    def macx(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackAceBackend.macx() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self.acx(lq1, lq2)

    def macy(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackAceBackend.macy() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self.acy(lq1, lq2)

    def macz(self, lq1, lq2):
        if len(lq1) > 1:
            raise RuntimeError(
                "QrackAceBackend.macz() is provided for syntax convenience and only supports 1 control qubit!"
            )
        self.acz(lq1, lq2)

    def swap(self, lq1, lq2):
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.swap([hq1 + q], hq2 + q)

    def iswap(self, lq1, lq2):
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.iswap([hq1 + q], hq2 + q)
        self._correct(lq1)
        self._correct(lq2)

    def adjiswap(self, lq1, lq2):
        hq1 = self.code_len * lq1
        hq2 = self.code_len * lq2
        for q in range(self.code_len):
            self.sim.adjiswap([hq1 + q], hq2 + q)
        self._correct(lq1)
        self._correct(lq2)

    def m(self, lq):
        hq = self.code_len * lq
        bits = []
        for q in range(self.code_len):
            bits.append(int(sim.m(hq + q)))
        count = sum(bits)
        result = count > 1
        for q in range(self.code_len):
            if result:
                if bits[q] == 0:
                    self.sim.x(hq + q)
            else:
                if bits[q] == 1:
                    self.sim.x(hq + q)

        return result

    def force_m(self, lq, result):
        hq = self.code_len * lq
        bits = []
        for q in range(self.code_len):
            bits.append(int(sim.m(hq + q)))
        count = sum(bits)
        for q in range(self.code_len):
            if result:
                if bits[q] == 0:
                    self.sim.x(hq + q)
            else:
                if bits[q] == 1:
                    self.sim.x(hq + q)

        return result

    def m_all(self):
        raw_sample = self.sim.m_all();
        sample = 0
        for i in range(self.n_qubits):
            hq = i * self.code_len
            b = (sample >> hq) & 1
            b += (sample >> (hq + 1)) & 1
            b += (sample >> (hq + 2)) & 1
            if b > 1:
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

        if (name == "u1") or (name == "p"):
            self._sim.u(operation.qubits[0]._index, 0, 0, float(operation.params[0]))
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
        measure_results = []
        for s in range(num_samples):
            sample = self._sim.m_all()
            result = 0
            for index in range(len(measure_qubit)):
                qubit = measure_qubit[index]._index
                qubit_outcome = (sample >> qubit) & 1
                result |= qubit_outcome << index
            measure_results.append(result)

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
            _data = self._add_sample_measure(self._sample_qubits, self._sample_clbits, self._shots)

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
