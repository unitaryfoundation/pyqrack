# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.
from .qrack_simulator import QrackSimulator
from .qrack_system import Qrack


class QrackStabilizer(QrackSimulator):
    """Interface for pure-stabilizer Qrack functionality.

    Like QrackSimulator with isTensorNetwork=True, QrackStabilizer does not implement a general ALU or phase parity operations.
    Unlike isTensorNetwork=True, QrackStabilizer does implement compose(), decompose(), and dispose()
    Even if your operation is non-Clifford in full generality, QrackStabilizer will attempt to reduce it to a Clifford case.
    Hence, QrackStabilizer inherits the full interface of QrackSimulator (via Qrack::QInterface).

    Attributes:
        sid(int): Corresponding simulator id.
    """

    def __init__(
        self,
        qubit_count=-1,
        clone_sid=-1,
        pyzx_circuit=None,
        qiskit_circuit=None,
    ):
        self.sid = None

        if pyzx_circuit is not None:
            qubit_count = pyzx_circuit.qubits
        elif qiskit_circuit is not None and qubitCount < 0:
            raise RuntimeError(
                "Must specify qubitCount with qiskitCircuit parameter in QrackSimulator constructor!"
            )

        if qubit_count > -1 and clone_sid > -1:
            raise RuntimeError(
                "Cannot clone a QrackStabilizer and specify its qubit length at the same time, in QrackStabilizer constructor!"
            )

        self.is_tensor_network = False
        self.is_pure_stabilizer = True

        if clone_sid > -1:
            self.sid = Qrack.qrack_lib.init_clone(clone_sid)
        else:
            if qubit_count < 0:
                qubit_count = 0

            self.sid = Qrack.qrack_lib.init_count_stabilizer(qubit_count)

        self._throw_if_error()

        if pyzx_circuit is not None:
            self.run_pyzx_gates(pyzx_circuit.gates)
        elif qiskit_circuit is not None:
            self.run_qiskit_circuit(qiskit_circuit)

    def set_stochastic(self, s):
        Qrack.qrack_lib.SetStochastic(self.sid, s)
        self._throw_if_error()

    def set_major_quadrant(self, q):
        Qrack.qrack_lib.SetMajorQuadrant(self.sid, q)
        self._throw_if_error()

    def flip_quadrant(self, q):
        Qrack.qrack_lib.FlipQuadrant(self.sid, q)
        self._throw_if_error()

    def set_quadrant(self, q, b):
        Qrack.qrack_lib.SetQuadrant(self.sid, q, b)
        self._throw_if_error()
