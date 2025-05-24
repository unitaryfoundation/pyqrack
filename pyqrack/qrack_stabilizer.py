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
        qubitCount=-1,
        cloneSid=-1,
        pyzxCircuit=None,
        qiskitCircuit=None,
    ):
        self.sid = None

        if pyzxCircuit is not None:
            qubitCount = pyzxCircuit.qubits
        elif qiskitCircuit is not None and qubitCount < 0:
            raise RuntimeError(
                "Must specify qubitCount with qiskitCircuit parameter in QrackSimulator constructor!"
            )

        if qubitCount > -1 and cloneSid > -1:
            raise RuntimeError(
                "Cannot clone a QrackStabilizer and specify its qubit length at the same time, in QrackStabilizer constructor!"
            )

        self.is_tensor_network = False
        self.is_pure_stabilizer = True

        if cloneSid > -1:
            self.sid = Qrack.qrack_lib.init_clone(cloneSid)
        else:
            if qubitCount < 0:
                qubitCount = 0

            self.sid = Qrack.qrack_lib.init_count_stabilizer(qubitCount)

        self._throw_if_error()

        if pyzxCircuit is not None:
            self.run_pyzx_gates(pyzxCircuit.gates)
        elif qiskitCircuit is not None:
            self.run_qiskit_circuit(qiskitCircuit)
