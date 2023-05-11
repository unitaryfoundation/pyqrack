# (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import ctypes

from .qrack_system import Qrack

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
        qubit_count = Qrack.qrack_lib.get_qcircuit_qubit_count(self.cid)

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
            RuntimeError: QrackCircuit C++ library raised an exception.
        """
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
            RuntimeError: QrackSimulator raised an exception.
        """

        p_list = [((p >> i) & 1) for i in range(len(c))]
        p_list = [x for _, x in sorted(zip(c, p_list))]
        p = 0
        for i in range(len(p_list)):
            p |= p_list[i] << i

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
        Qrack.qrack_lib.qcircuit_run(self.cid, qsim.sid)
        qsim._throw_if_error()
