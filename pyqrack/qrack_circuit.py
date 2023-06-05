# (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import ctypes

from .qrack_system import Qrack

_IS_QISKIT_AVAILABLE = True
try:
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from qiskit.compiler.transpiler import transpile
    import numpy as np
    import math
except ImportError:
    _IS_QISKIT_AVAILABLE = False

_IS_QUIMB_AVAILABLE = True
try:
    import quimb as qu
    import quimb.tensor as qtn
except ImportError:
    _IS_QUIMB_AVAILABLE = False

class QrackCircuit:
    """Class that exposes the QCircuit class of Qrack

    QrackCircuit allows the user to specify a unitary circuit, before running it.
    Upon running the state, the result is a QrackSimulator state. Currently,
    measurement is not supported, but measurement can be run on the resultant
    QrackSimulator.

    Attributes:
        cid(int): Qrack ID of this circuit
    """

    def __init__(self, clone_cid = -1):
        if clone_cid < 0:
            self.cid = Qrack.qrack_lib.init_qcircuit()
        else:
            self.cid = Qrack.qrack_lib.init_qcircuit_clone(clone_cid)

    def __del__(self):
        if self.cid is not None:
            Qrack.qrack_lib.destroy_qcircuit(self.cid)
            self.cid = None

    def _ulonglong_byref(self, a):
        return (ctypes.c_ulonglong * len(a))(*a)

    def _double_byref(self, a):
        return (ctypes.c_double * len(a))(*a)

    def _complex_byref(self, a):
        t = [(c.real, c.imag) for c in a]
        return self._double_byref([float(item) for sublist in t for item in sublist])

    def get_qubit_count(self):
        """Get count of qubits in circuit

        Raises:
            RuntimeError: QracQrackCircuitNeuron C++ library raised an exception.
        """
        return Qrack.qrack_lib.get_qcircuit_qubit_count(self.cid)

    def swap(self, q1, q2):
        """Add a 'Swap' gate to the circuit

        Args:
            q1: qubit index #1
            q2: qubit index #2

        Raises:
            RuntimeError: QrackCircuit C++ library raised an exception.
        """
        Qrack.qrack_lib.qcircuit_swap(self.cid, q1, q2)

    def mtrx(self, m, q):
        """Operation from matrix.

        Applies arbitrary operation defined by the given matrix.

        Args:
            m: row-major complex list representing the operator.
            q: the qubit number on which the gate is applied to.

        Raises:
            ValueError: 2x2 matrix 'm' in QrackCircuit.mtrx() must contain at least 4 elements.
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(m) < 4:
            raise ValueError("2x2 matrix 'm' in QrackCircuit.mtrx() must contain at least 4 elements.")
        Qrack.qrack_lib.qcircuit_append_1qb(self.cid, self._complex_byref(m), q)

    def ucmtrx(self, c, m, q, p):
        """Multi-controlled single-target-qubit gate

        Specify a controlled gate by its control qubits, its single-qubit
        matrix "payload," the target qubit, and the permutation of qubits
        that activates the gate.

        Args:
            c: list of controlled qubits
            m: row-major complex list representing the operator.
            q: target qubit
            p: permutation of target qubits

        Raises:
            ValueError: 2x2 matrix 'm' in QrackCircuit.ucmtrx() must contain at least 4 elements.
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(m) < 4:
            raise ValueError("2x2 matrix 'm' in QrackCircuit.ucmtrx() must contain at least 4 elements.")
        Qrack.qrack_lib.qcircuit_append_mc(
            self.cid, self._complex_byref(m), len(c), self._ulonglong_byref(c), q, p
        )

    def run(self, qsim):
        """Run circuit on simulator

        Run the encoded circuit on a specific simulator. The
        result will remain in this simulator.

        Args:
            qsim: QrackSimulator on which to run circuit

        Raises:
            RuntimeError: QrackCircuit raised an exception.
        """
        qb_count = self.get_qubit_count()
        sim_qb_count = qsim.num_qubits()
        if sim_qb_count < qb_count:
            for i in range(sim_qb_count, qb_count):
                qsim.allocate_qubit(i)
        Qrack.qrack_lib.qcircuit_run(self.cid, qsim.sid)
        qsim._throw_if_error()

    def out_to_file(self, filename):
        """Output optimized circuit to file

        Outputs the (optimized) circuit to a file named
        according to the "filename" parameter.

        Args:
            filename: Name of file
        """
        Qrack.qrack_lib.qcircuit_out_to_file(self.cid, filename.encode('utf-8'))

    def in_from_file(self, filename):
        """Read in optimized circuit from file

        Reads in an (optimized) circuit from a file named
        according to the "filename" parameter.

        Args:
            filename: Name of file
        """
        Qrack.qrack_lib.qcircuit_in_from_file(self.cid, filename.encode('utf-8'))

    def file_to_qiskit_circuit(filename):
        """Convert an output file to a Qiskit circuit

        Reads in an (optimized) circuit from a file named
        according to the "filename" parameter and outputs
        a Qiskit circuit.

        Args:
            filename: Name of file

        Raises:
            RuntimeErorr: Before trying to file_to_qiskit_circuit() with
                QrackCircuit, you must install Qiskit, numpy, and math!
        """
        if not _IS_QISKIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to file_to_qiskit_circuit() with QrackCircuit, you must install Qiskit, numpy, and math!"
            )

        tokens = []
        with open(filename, 'r') as file:
            tokens = file.read().split()

        i = 0
        num_qubits = int(tokens[i])
        i = i + 1
        circ = QuantumCircuit(num_qubits, num_qubits)

        num_gates = int(tokens[i])
        i = i + 1

        for g in range(num_gates):
            target = int(tokens[i])
            i = i + 1

            control_count = int(tokens[i])
            i = i + 1
            controls = []
            for j in range(control_count):
                controls.append(int(tokens[i]))
                i = i + 1

            payload_count = int(tokens[i])
            i = i + 1
            payloads = {}
            for j in range(payload_count):
                key = int(tokens[i])
                i = i + 1
                op = np.zeros((2,2), dtype=complex)
                row = []
                for _ in range(2):
                    amp = tokens[i].replace("(","").replace(")","").split(',')
                    row.append(float(amp[0]) + float(amp[1])*1j)
                    i = i + 1
                l = math.sqrt(np.real(row[0] * np.conj(row[0]) + row[1] * np.conj(row[1])))
                op[0][0] = row[0] / l
                op[0][1] = row[1] / l

                if np.abs(op[0][0] - row[0]) > 1e-5:
                    print("Warning: gate ", str(g), ", payload ", str(j), " might not be unitary!")
                if np.abs(op[0][1] - row[1]) > 1e-5:
                    print("Warning: gate ", str(g), ", payload ", str(j), " might not be unitary!")

                row = []
                for _ in range(2):
                    amp = tokens[i].replace("(","").replace(")","").split(',')
                    row.append(float(amp[0]) + float(amp[1])*1j)
                    i = i + 1
                l = math.sqrt(np.real(row[0] * np.conj(row[0]) + row[1] * np.conj(row[1])))
                op[1][0] = row[0] / l
                op[1][1] = row[1] / l

                ph = np.real(np.log(np.linalg.det(op)) / 1j)

                op[1][0] = -np.exp(1j * ph) * np.conj(op[0][1])
                op[1][1] = np.exp(1j * ph) * np.conj(op[0][0])

                if np.abs(op[1][0] - row[0]) > 1e-5:
                    print("Warning: gate ", str(g), ", payload ", str(j), " might not be unitary!")
                if np.abs(op[1][1] - row[1]) > 1e-5:
                    print("Warning: gate ", str(g), ", payload ", str(j), " might not be unitary!")

                # Qiskit has a lower tolerance for deviation from numerically unitary.
                payloads[key] = np.array(op)

            gate_list = []
            for j in range(1 << control_count):
                if j in payloads:
                    gate_list.append(payloads[j])
                else:
                    gate_list.append(np.array([[1, 0],[0, 1]]))

            circ.uc(gate_list, controls, target)

        return circ

    def in_from_qiskit_circuit(self, circ):
        """Read a Qiskit circuit into a QrackCircuit

        Reads in a circuit from a Qiskit `QuantumCircuit`

        Args:
            circ: Qiskit circuit

        Raises:
            RuntimeErorr: Before trying to file_to_qiskit_circuit() with
                QrackCircuit, you must install Qiskit, numpy, and math!
        """
        if not _IS_QISKIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to file_to_qiskit_circuit() with QrackCircuit, you must install Qiskit, numpy, and math!"
            )

        basis_gates = ["u", "cx"]
        circ = transpile(circ, basis_gates=basis_gates, optimization_level=3)
        for gate in circ.data:
            o = gate.operation
            if o.name == "u":
                th = float(o.params[0])
                ph = float(o.params[1])
                lm = float(o.params[2])

                c = math.cos(th / 2)
                s = math.sin(th / 2)

                op = [
                    c + 0j,
                    -np.exp(1j * lm) * s,
                    np.exp(1j * ph) * s,
                    np.exp(1j * (ph + lm)) * c
                ]
                self.mtrx(op, circ.find_bit(gate.qubits[0])[0])
            else:
                ctrls = []
                for c in gate.qubits[0:1]:
                    ctrls.append(circ.find_bit(c)[0])
                self.ucmtrx(ctrls, [0, 1, 1, 0], circ.find_bit(gate.qubits[1])[0], 1)

    def file_to_quimb_circuit(filename):
        """Convert an output file to a Quimb circuit

        Reads in an (optimized) circuit from a file named
        according to the "filename" parameter and outputs
        a Quimb circuit.

        Args:
            filename: Name of file

        Raises:
            RuntimeErorr: Before trying to file_to_quimb_circuit() with
                QrackCircuit, you must install quimb, Qiskit, numpy, and math!
        """
        if not _IS_QUIMB_AVAILABLE:
            raise RuntimeError(
                "Before trying to file_to_quimb_circuit() with QrackCircuit, you must install quimb, Qiskit, numpy, and math!"
            )

        qcirc = QrackCircuit.file_to_qiskit_circuit(filename)
        basis_gates = ["u", "cx"]
        qcirc = transpile(qcirc, basis_gates=basis_gates, optimization_level=3)

        tcirc = qtn.Circuit(qcirc.num_qubits)
        for gate in qcirc.data:
            o = gate.operation
            if o.name == "u":
                th = float(o.params[0])
                ph = float(o.params[1])
                lm = float(o.params[2])

                tcirc.apply_gate('U3', th, ph, lm, qcirc.find_bit(gate.qubits[0])[0])
            else:
                tcirc.apply_gate('CNOT', qcirc.find_bit(gate.qubits[0])[0], qcirc.find_bit(gate.qubits[1])[0])

        return tcirc
