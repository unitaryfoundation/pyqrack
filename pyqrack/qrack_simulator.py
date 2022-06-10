# (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import math
import ctypes
from .qrack_system import Qrack
from .pauli import Pauli

_IS_QISKIT_AVAILABLE = True
try:
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from qiskit.qobj.qasm_qobj import QasmQobjExperiment
    from .util import convert_qiskit_circuit_to_qasm_experiment
except ImportError:
    _IS_QISKIT_AVAILABLE = False


class QrackSimulator:
    """Interface for all the QRack functionality.

    Attributes:
        qubitCount(int): Number of qubits that are to be simulated.
        sid(int): Corresponding simulator id.
    """

    def __init__(
        self,
        qubitCount=-1,
        cloneSid=-1,
        isSchmidtDecomposeMulti=True,
        isSchmidtDecompose=True,
        isStabilizerHybrid=True,
        isBinaryDecisionTree=False,
        is1QbFusion=False,
        isPaged=True,
        isCpuGpuHybrid=True,
        isOpenCL=True,
        isHostPointer=False,
        pyzxCircuit=None,
        qiskitCircuit=None,
    ):
        self.sid = None

        if pyzxCircuit is not None:
            qubitCount = pyzxCircuit.qubits

        if qubitCount > -1 and cloneSid > -1:
            raise RuntimeError(
                "Cannot clone a QrackSimulator and specify its qubit length at the same time, in QrackSimulator constructor!"
            )
        if cloneSid > -1:
            self.sid = Qrack.qrack_lib.init_clone(cloneSid)
        else:
            if qubitCount < 0:
                qubitCount = 0

            if (
                isSchmidtDecompose
                and isStabilizerHybrid
                and not isBinaryDecisionTree
                and not is1QbFusion
                and isPaged
                and isCpuGpuHybrid
                and isOpenCL
            ):
                if isSchmidtDecomposeMulti:
                    self.sid = Qrack.qrack_lib.init_count(qubitCount, isHostPointer)
                else:
                    self.sid = Qrack.qrack_lib.init_count_pager(
                        qubitCount, isHostPointer
                    )
            else:
                self.sid = Qrack.qrack_lib.init_count_type(
                    qubitCount,
                    isSchmidtDecomposeMulti,
                    isSchmidtDecompose,
                    isStabilizerHybrid,
                    isBinaryDecisionTree,
                    isPaged,
                    is1QbFusion,
                    isCpuGpuHybrid,
                    isOpenCL,
                    isHostPointer,
                )

        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

        self._qubitCount = qubitCount

        if pyzxCircuit is not None:
            self.run_pyzx_gates(pyzxCircuit.gates)

    def __del__(self):
        if self.sid is not None:
            Qrack.qrack_lib.destroy(self.sid)
            self.sid = None

    def _ulonglong_byref(self, a):
        return (ctypes.c_ulonglong * len(a))(*a)

    def _double_byref(self, a):
        return (ctypes.c_double * len(a))(*a)

    def _complex_byref(self, a):
        t = [(c.real, c.imag) for c in a]
        return self._double_byref([float(item) for sublist in t for item in sublist])

    def _real1_byref(self, a):
        # This needs to be c_double, if PyQrack is built with fp64.
        if Qrack.fppow < 6:
            return (ctypes.c_float * len(a))(*a)
        return (ctypes.c_double * len(a))(*a)

    def _qrack_complex_byref(self, a):
        t = [(c.real, c.imag) for c in a]
        return self._real1_byref([float(item) for sublist in t for item in sublist])

    def _to_ubyte(self, nv, v):
        c = math.floor((nv - 1) / 8) + 1
        b = (ctypes.c_ubyte * (c * (1 << nv)))()
        n = 0
        for u in v:
            for _ in range(c):
                b[n] = u & 0xFF
                u >>= 8
                n += 1

        return ctypes.byref(b)

    # See https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list#answer-30426000
    def _pairwise(self, it):
        it = iter(it)
        while True:
            try:
                yield next(it), next(it)
            except StopIteration:
                # no more elements in the iterator
                return

    # non-quantum
    def _get_error(self):
        return Qrack.qrack_lib.get_error(self.sid)

    def seed(self, s):
        Qrack.qrack_lib.seed(self.sid, s)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def set_concurrency(self, p):
        Qrack.qrack_lib.set_concurrency(self.sid, p)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    # standard gates

    ## single-qubits gates
    def x(self, q):
        """Applies X gate.

        Applies the Pauli “X” operator to the qubit at position “q.”
        The Pauli “X” operator is equivalent to a logical “NOT.”

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.X(self.sid, q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def y(self, q):
        """Applies Y gate.

        Applies the Pauli “Y” operator to the qubit at “q.”
        The Pauli “Y” operator is equivalent to a logical “NOT" with
        permutation phase.

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.Y(self.sid, q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def z(self, q):
        """Applies Z gate.

        Applies the Pauli “Z” operator to the qubit at “q.”
        The Pauli “Z” operator flips the phase of `|1>`

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.Z(self.sid, q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def h(self, q):
        """Applies H gate.

        Applies the Hadarmard operator to the qubit at “q.”

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.H(self.sid, q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def s(self, q):
        """Applies S gate.

        Applies the 1/4 phase rotation to the qubit at “q.”

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.S(self.sid, q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def t(self, q):
        """Applies T gate.

        Applies the 1/8 phase rotation to the qubit at “q.”

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.T(self.sid, q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def adjs(self, q):
        """Adjoint of S gate

        Applies the gate equivalent to the inverse of S gate.

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.AdjS(self.sid, q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def adjt(self, q):
        """Adjoint of T gate

        Applies the gate equivalent to the inverse of T gate.

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.AdjT(self.sid, q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def u(self, q, th, ph, la):
        """General unitary gate.

        Applies a gate guaranteed to be unitary.
        Spans all possible single bit unitary gates.

        `U(theta, phi, lambda) = RZ(phi + pi/2)RX(theta)RZ(lambda - pi/2)`

        Args:
            q: the qubit number on which the gate is applied to.
            th: theta
            ph: phi
            la: lambda

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.U(
            self.sid, q, ctypes.c_double(th), ctypes.c_double(ph), ctypes.c_double(la)
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mtrx(self, m, q):
        """Operation from matrix.

        Applies arbitrary operation defined by the given matrix.

        Args:
            m: row-major complex list representing the operator.
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.Mtrx(self.sid, self._complex_byref(m), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def r(self, b, ph, q):
        """Rotation gate.

        Rotate the qubit along the given pauli basis by the given angle.


        Args:
            b: Pauli basis
            ph: rotation angle
            q: the qubit number on which the gate is applied to

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.R(self.sid, ctypes.c_ulonglong(b), ctypes.c_double(ph), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def exp(self, b, ph, q):
        """Arbitrary exponentiation

        `exp(b, theta) = e^{i*theta*[b_0 . b_1 ...]}`
        where `.` is the tensor product.


        Args:
            b: Pauli basis
            ph: coefficient of exponentiation
            q: the qubit number on which the gate is applied to

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.Exp(
            self.sid,
            len(b),
            self._ulonglong_byref(b),
            ctypes.c_double(ph),
            self._ulonglong_byref(q),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    ## multi-qubit gates
    def mcx(self, c, q):
        """Multi-controlled X gate

        If all controlled qubits are `|1>` then the target qubit is flipped.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MCX(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcy(self, c, q):
        """Multi-controlled Y gate

        If all controlled qubits are `|1>` then the Pauli "Y" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MCY(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcz(self, c, q):
        """Multi-controlled Z gate

        If all controlled qubits are `|1>` then the Pauli "Z" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MCZ(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mch(self, c, q):
        """Multi-controlled H gate

        If all controlled qubits are `|1>` then the Hadarmard gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MCH(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcs(self, c, q):
        """Multi-controlled S gate

        If all controlled qubits are `|1>` then the "S" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MCS(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mct(self, c, q):
        """Multi-controlled T gate

        If all controlled qubits are `|1>` then the "T" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MCT(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcadjs(self, c, q):
        """Multi-controlled adjs gate

        If all controlled qubits are `|1>` then the adjs gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MCAdjS(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcadjt(self, c, q):
        """Multi-controlled adjt gate

        If all controlled qubits are `|1>` then the adjt gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MCAdjT(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcu(self, c, q, th, ph, la):
        """Multi-controlled arbitraty unitary

        If all controlled qubits are `|1>` then the unitary gate described by
        parameters is applied to the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.
            th: theta
            ph: phi
            la: lambda

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MCU(
            self.sid,
            len(c),
            self._ulonglong_byref(c),
            q,
            ctypes.c_double(th),
            ctypes.c_double(ph),
            ctypes.c_double(la),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcmtrx(self, c, m, q):
        """Multi-controlled arbitraty operator

        If all controlled qubits are `|1>` then the arbitrary operation by
        parameters is applied to the target qubit.

        Args:
            c: list of controlled qubits
            m: row-major complex list representing the operator.
            q: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MCMtrx(
            self.sid, len(c), self._ulonglong_byref(c), self._complex_byref(m), q
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def macx(self, c, q):
        """Anti multi-controlled X gate

        If all controlled qubits are `|0>` then the target qubit is flipped.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MACX(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def macy(self, c, q):
        """Anti multi-controlled Y gate

        If all controlled qubits are `|0>` then the Pauli "Y" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MACY(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def macz(self, c, q):
        """Anti multi-controlled Z gate

        If all controlled qubits are `|0>` then the Pauli "Z" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MACZ(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mach(self, c, q):
        """Anti multi-controlled H gate

        If all controlled qubits are `|0>` then the Hadarmard gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MACH(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def macs(self, c, q):
        """Anti multi-controlled S gate

        If all controlled qubits are `|0>` then the "S" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MACS(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mact(self, c, q):
        """Anti multi-controlled T gate

        If all controlled qubits are `|0>` then the "T" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MACT(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def macadjs(self, c, q):
        """Anti multi-controlled adjs gate

        If all controlled qubits are `|0>` then the adjs gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MACAdjS(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def macadjt(self, c, q):
        """Anti multi-controlled adjt gate

        If all controlled qubits are `|0>` then the adjt gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MACAdjT(self.sid, len(c), self._ulonglong_byref(c), q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def macu(self, c, q, th, ph, la):
        """Anti multi-controlled arbitraty unitary

        If all controlled qubits are `|0>` then the unitary gate described by
        parameters is applied to the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.
            th: theta
            ph: phi
            la: lambda

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MACU(
            self.sid,
            len(c),
            self._ulonglong_byref(c),
            q,
            ctypes.c_double(th),
            ctypes.c_double(ph),
            ctypes.c_double(la),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def macmtrx(self, c, m, q):
        """Anti multi-controlled arbitraty operator

        If all controlled qubits are `|0>` then the arbitrary operation by
        parameters is applied to the target qubit.

        Args:
            c: list of controlled qubits.
            m: row-major complex matrix which defines the operator.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MACMtrx(
            self.sid, len(c), self._ulonglong_byref(c), self._complex_byref(m), q
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def multiplex1_mtrx(self, c, q, m):
        """Multiplex gate

        A multiplex gate with a single target and an arbitrary number of
        controls.

        Args:
            c: list of controlled qubits.
            m: row-major complex matrix which defines the operator.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.Multiplex1Mtrx(
            self.sid, len(c), self._ulonglong_byref(c), q, self._complex_byref(m)
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mx(self, q):
        """Multi X-gate

        Applies the Pauli “X” operator on all qubits.

        Args:
            q: list of qubits to apply X on.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MX(self.sid, len(q), self._ulonglong_byref(q))
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def my(self, q):
        """Multi Y-gate

        Applies the Pauli “Y” operator on all qubits.

        Args:
            q: list of qubits to apply Y on.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MY(self.sid, len(q), self._ulonglong_byref(q))
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mz(self, q):
        """Multi Z-gate

        Applies the Pauli “Z” operator on all qubits.

        Args:
            q: list of qubits to apply Z on.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MZ(self.sid, len(q), self._ulonglong_byref(q))
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcr(self, b, ph, c, q):
        """Multi-controlled arbitrary rotation.

        If all controlled qubits are `|1>` then the arbitrary rotation by
        parameters is applied to the target qubit.

        Args:
            b: Pauli basis
            ph: coefficient of exponentiation.
            c: list of controlled qubits.
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MCR(
            self.sid,
            ctypes.c_ulonglong(b),
            ctypes.c_double(ph),
            len(c),
            self._ulonglong_byref(c),
            q,
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcexp(self, b, ph, cs, q):
        """Multi-controlled arbitrary exponentiation

        If all controlled qubits are `|1>` then the target qubit is
        exponentiated an pauli basis basis with coefficient.

        Args:
            b: Pauli basis
            ph: coefficient of exponentiation.
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MCExp(
            self.sid,
            len(b),
            self._ulonglong_byref(b),
            ctypes.c_double(ph),
            len(cs),
            self._ulonglong_byref(cs),
            self._ulonglong_byref(q),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def swap(self, qi1, qi2):
        """Swap Gate

        Swaps the qubits at two given positions.

        Args:
            qi1: First position of qubit.
            qi2: Second position of qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.SWAP(self.sid, qi1, qi2)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def iswap(self, qi1, qi2):
        """Swap Gate with phase.

        Swaps the qubits at two given positions.
        If the bits are different then there is additional phase of `i`.

        Args:
            qi1: First position of qubit.
            qi2: Second position of qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.ISWAP(self.sid, qi1, qi2)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def adjiswap(self, qi1, qi2):
        """Swap Gate with phase.

        Swaps the qubits at two given positions.
        If the bits are different then there is additional phase of `-i`.

        Args:
            qi1: First position of qubit.
            qi2: Second position of qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.AdjISWAP(self.sid, qi1, qi2)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def fsim(self, th, ph, qi1, qi2):
        """Fsim gate.

        The 2-qubit “fSim” gate
        Useful in the simulation of particles with fermionic statistics

        Args:
            qi1: First position of qubit.
            qi2: Second position of qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.FSim(
            self.sid, ctypes.c_double(th), ctypes.c_double(ph), qi1, qi2
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def cswap(self, c, qi1, qi2):
        """Controlled-swap Gate

        Swaps the qubits at two given positions if the control qubits are `|1>`

        Args:
            c: list of controlled qubits.
            qi1: First position of qubit.
            qi2: Second position of qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.CSWAP(self.sid, len(c), self._ulonglong_byref(c), qi1, qi2)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def acswap(self, c, qi1, qi2):
        """Anti controlled-swap Gate

        Swaps the qubits at two given positions if the control qubits are `|0>`

        Args:
            c: list of controlled qubits.
            qi1: First position of qubit.
            qi2: Second position of qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.ACSWAP(self.sid, len(c), self._ulonglong_byref(c), qi1, qi2)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    # standard operations
    def m(self, q):
        """Measurement gate

        Measures the qubit at "q" and returns Boolean value.
        This operator is not unitary & is probabilistic in nature.

        Args:
            q: qubit to measure

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Measurement result.
        """
        result = Qrack.qrack_lib.M(self.sid, q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return result

    def force_m(self, q, r):
        """Force-Measurement gate

        Acts as if the measurement is applied and the result obtained is `r`

        Args:
            q: qubit to measure
            r: the required result

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Measurement result.
        """
        result = Qrack.qrack_lib.ForceM(self.sid, q, r)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return result

    def m_all(self):
        """Measure-all gate

        Measures measures all qubits.
        This operator is not unitary & is probabilistic in nature.

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Measurement result of all qubits.
        """
        result = Qrack.qrack_lib.MAll(self.sid)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return result

    def measure_pauli(self, b, q):
        """Pauli Measurement gate

        Measures the qubit at "q" with the given pauli basis.
        This operator is not unitary & is probabilistic in nature.

        Args:
            b: Pauli basis
            q: qubit to measure

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Measurement result.
        """
        result = Qrack.qrack_lib.Measure(
            self.sid, len(b), self._ulonglong_byref(b), self._ulonglong_byref(q)
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return result

    def measure_shots(self, q, s):
        """Multi-shot measurement operator

        Measures the qubit at "q" with the given pauli basis.
        This operator is not unitary & is probabilistic in nature.

        Args:
            q: list of qubits to measure
            s: number of shots

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            list of measurement result.
        """
        m = self._ulonglong_byref([0] * s)
        Qrack.qrack_lib.MeasureShots(self.sid, len(q), self._ulonglong_byref(q), s, m)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return [m[i] for i in range(s)]

    def reset_all(self):
        """Reset gate

        Resets all qubits to `|0>`

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.ResetAll(self.sid)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    # arithmetic-logic-unit (ALU)
    def _split_longs(self, a):
        """Split operation

        Splits the given integer into 64 bit numbers.


        Args:
            a: number to split

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            list of split numbers.
        """
        aParts = []
        if a == 0:
            aParts.append(0)
        while a > 0:
            aParts.append(a & 0xFFFFFFFFFFFFFFFF)
            a = a >> 64
        return aParts

    def _split_longs_2(self, a, m):
        """Split simultanoues operation

        Splits 2 integers into same number of 64 bit numbers.

        Args:
            a: first number to split
            m: second number to split

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            pair of lists of split numbers.
        """
        aParts = []
        mParts = []
        if a == 0 and m == 0:
            aParts.append(0)
            mParts.append(0)
        while a > 0 or m > 0:
            aParts.append(a & 0xFFFFFFFFFFFFFFFF)
            a = a >> 64
            mParts.append(m & 0xFFFFFFFFFFFFFFFF)
            m = m >> 64
        return aParts, mParts

    def add(self, a, q):
        """Add integer to qubit

        Adds the given integer to the given set of qubits.

        Args:
            a: first number to split
            q: list of qubits to add the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts = self._split_longs(a)
        Qrack.qrack_lib.ADD(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(q),
            self._ulonglong_byref(q),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def sub(self, a, q):
        """Subtract integer to qubit

        Subtracts the given integer to the given set of qubits.

        Args:
            a: first number to split
            q: list of qubits to subtract the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts = self._split_longs(a)
        Qrack.qrack_lib.SUB(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(q),
            self._ulonglong_byref(q),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def adds(self, a, s, q):
        """Signed Addition integer to qubit

        Signed Addition of the given integer to the given set of qubits,
        if there is an overflow the resultant will become negative.

        Args:
            a: number to add
            s: qubit to store overflow
            q: list of qubits to add the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts = self._split_longs(a)
        Qrack.qrack_lib.ADDS(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            s,
            len(q),
            self._ulonglong_byref(q),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def subs(self, a, s, q):
        """Subtract integer to qubit

        Subtracts the given integer to the given set of qubits,
        if there is an overflow the resultant will become negative.

        Args:
            a: number to subtract
            s: qubit to store overflow
            q: list of qubits to subtract the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts = self._split_longs(a)
        Qrack.qrack_lib.SUBS(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            s,
            len(q),
            self._ulonglong_byref(q),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mul(self, a, q, o):
        """Multiplies integer to qubit

        Multiplies the given integer to the given set of qubits.
        Carry register is required for maintaining the unitary nature of
        operation, and must be as long as the input qubit register. 

        Args:
            a: number to multiply
            q: list of qubits to multiply the number
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts = self._split_longs(a)
        Qrack.qrack_lib.MUL(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(q),
            self._ulonglong_byref(q),
            self._ulonglong_byref(o),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def div(self, a, q, o):
        """Divides qubit by integer

        'Divides' the given qubits by the integer.
        Carry register is required for maintaining the unitary nature of
        operation. 

        Args:
            a: integer to divide by
            q: qubits to divide
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts = self._split_longs(a)
        Qrack.qrack_lib.DIV(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(q),
            self._ulonglong_byref(q),
            self._ulonglong_byref(o),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def muln(self, a, m, q, o):
        """Modulo Multiplication

        Modulo Multiplication of the given integer to the given set of qubits
        Carry register is required for maintaining the unitary nature of
        operation. 

        Args:
            a: number to multiply
            m: modulo number
            q: list of qubits to multiply the number
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts, mParts = self._split_longs_2(a, m)
        Qrack.qrack_lib.MULN(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            self._ulonglong_byref(mParts),
            len(q),
            self._ulonglong_byref(q),
            self._ulonglong_byref(o),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def divn(self, a, m, q, o):
        """Modulo Division

        'Modulo Division' of the given set of qubits by the given integer
        Carry register is required for maintaining the unitary nature of
        operation, and must be as long as the input qubit registe. 

        Args:
            a: integer by which qubit will be divided
            m: modulo integer
            q: qubits to divide
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts, mParts = self._split_longs_2(a, m)
        Qrack.qrack_lib.DIVN(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            self._ulonglong_byref(mParts),
            len(q),
            self._ulonglong_byref(q),
            self._ulonglong_byref(o),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def pown(self, a, m, q, o):
        """Modulo Power

        Raises the qubit to the power `a` to which `mod m` is applied to.
        Out-of-place register is required to store the resultant.

        Args:
            a: number in power
            m: modulo number
            q: list of qubits to exponentiate
            o: out-of-place register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts, mParts = self._split_longs_2(a, m)
        Qrack.qrack_lib.POWN(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            self._ulonglong_byref(mParts),
            len(q),
            self._ulonglong_byref(q),
            self._ulonglong_byref(o),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcadd(self, a, c, q):
        """Controlled-add

        Adds the given integer to the given set of qubits if all controlled
        qubits are `|1>`.

        Args:
            a: number to add.
            c: list of controlled qubits.
            q: list of qubits to add the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts = self._split_longs(a)
        Qrack.qrack_lib.MCADD(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(c),
            self._ulonglong_byref(c),
            len(q),
            self._ulonglong_byref(q),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcsub(self, a, c, q):
        """Controlled-subtract

        Subtracts the given integer to the given set of qubits if all controlled
        qubits are `|1>`.

        Args:
            a: number to subtract.
            c: list of controlled qubits.
            q: list of qubits to add the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts = self._split_longs(a)
        Qrack.qrack_lib.MCSUB(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(c),
            self._ulonglong_byref(c),
            len(q),
            self._ulonglong_byref(q),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcmul(self, a, c, q, o):
        """Controlled-multiply

        Multiplies the given integer to the given set of qubits if all controlled
        qubits are `|1>`.
        Out-of-place register is required to store the resultant.

        Args:
            a: number to multiply
            c: list of controlled qubits.
            q: list of qubits to add the number
            o: carry register
            o: out-of-place register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts = self._split_longs(a)
        Qrack.qrack_lib.MCMUL(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(c),
            self._ulonglong_byref(c),
            len(q),
            self._ulonglong_byref(q),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcdiv(self, a, c, q, o):
        """Controlled-divide.

        'Divides' the given qubits by the integer if all controlled
        qubits are `|1>`.
        Carry register is required for maintaining the unitary nature of
        operation. 

        Args:
            a: number to divide by
            c: list of controlled qubits.
            q: qubits to divide
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts = self._split_longs(a)
        Qrack.qrack_lib.MCDIV(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(c),
            self._ulonglong_byref(c),
            len(q),
            self._ulonglong_byref(q),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcmuln(self, a, c, m, q, o):
        """Controlled-modulo multiplication

        Modulo multiplication of the given integer to the given set of qubits
        if all controlled qubits are `|1>`.
        Carry register is required for maintaining the unitary nature of
        operation. 

        Args:
            a: number to multiply
            c: list of controlled qubits.
            m: modulo number
            q: list of qubits to add the number
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts, mParts = self._split_longs_2(a, m)
        Qrack.qrack_lib.MCMULN(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(c),
            self._ulonglong_byref(c),
            self._ulonglong_byref(mParts),
            len(q),
            self._ulonglong_byref(q),
            self._ulonglong_byref(o),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcdivn(self, a, c, m, q, o):
        """Controlled-divide.

        Modulo division of the given qubits by the given number if all
        controlled qubits are `|1>`.
        Carry register is required for maintaining the unitary nature of
        operation. 

        Args:
            a: number to divide by
            c: list of controlled qubits.
            m: modulo number
            q: qubits to divide
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts, mParts = self._split_longs_2(a, m)
        Qrack.qrack_lib.MCDIVN(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(c),
            self._ulonglong_byref(c),
            self._ulonglong_byref(mParts),
            len(q),
            self._ulonglong_byref(q),
            self._ulonglong_byref(o),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def mcpown(self, a, c, m, q, o):
        """Controlled-modulo Power

        Raises the qubit to the power `a` to which `mod m` is applied to if
        all the controlled qubits are set to `|1>`.
        Out-of-place register is required to store the resultant.

        Args:
            a: number in power
            c: control qubits
            m: modulo number
            q: list of qubits to exponentiate
            o: out-of-place register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        aParts, mParts = self._split_longs_2(a, m)
        Qrack.qrack_lib.MCPOWN(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(c),
            self._ulonglong_byref(c),
            self._ulonglong_byref(mParts),
            len(q),
            self._ulonglong_byref(q),
            self._ulonglong_byref(o),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def lda(self, qi, qv, t):
        """Load Accumalator

        Quantum counterpart for LDA from MOS-6502 assembly. `t` must be of
        the length `2 ** len(qi)`. It loads each list entry index of t into
        the qi register and each list entry value into the qv register.

        Args:
            qi: qubit register for index
            qv: qubit register for value
            t: list of values

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.LDA(
            self.sid,
            len(qi),
            self._ulonglong_byref(qi),
            len(qv),
            self._ulonglong_byref(qv),
            self._to_ubyte(len(qv), t),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def adc(self, s, qi, qv, t):
        """Add with Carry

        Quantum counterpart for ADC from MOS-6502 assembly. `t` must be of
        the length `2 ** len(qi)`.

        Args:
            qi: qubit register for index
            qv: qubit register for value
            t: list of values

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.ADC(
            self.sid,
            s,
            len(qi),
            self._ulonglong_byref(qi),
            len(qv),
            self._ulonglong_byref(qv),
            self._to_ubyte(len(qv), t),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def sbc(self, s, qi, qv, t):
        """Subtract with Carry

        Quantum counterpart for SBC from MOS-6502 assembly. `t` must be of
        the length `2 ** len(qi)`

        Args:
            qi: qubit register for index
            qv: qubit register for value
            t: list of values

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.SBC(
            self.sid,
            s,
            len(qi),
            self._ulonglong_byref(qi),
            len(qv),
            self._ulonglong_byref(qv),
            self._to_ubyte(len(qv), t),
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def hash(self, q, t):
        """Hash function

        Replicates the behaviour of LDA without the index register.
        For the operation to be unitary, the entries present in `t` must be
        unique, and the length of `t` must be `2 ** len(qi)`.


        Args:
            q: qubit register for value
            t: list of values

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.Hash(
            self.sid, len(q), self._ulonglong_byref(q), self._to_ubyte(len(q), t)
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    # boolean logic gates
    def qand(self, qi1, qi2, qo):
        """Logical AND

        Logical AND of 2 qubits whose result is stored in the target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.AND(self.sid, qi1, qi2, qo)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def qor(self, qi1, qi2, qo):
        """Logical OR

        Logical OR of 2 qubits whose result is stored in the target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.OR(self.sid, qi1, qi2, qo)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def qxor(self, qi1, qi2, qo):
        """Logical XOR

        Logical exlusive-OR of 2 qubits whose result is stored in the target
        qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.XOR(self.sid, qi1, qi2, qo)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def qnand(self, qi1, qi2, qo):
        """Logical NAND

        Logical NAND of 2 qubits whose result is stored in the target
        qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.NAND(self.sid, qi1, qi2, qo)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def qnor(self, qi1, qi2, qo):
        """Logical NOR

        Logical NOR of 2 qubits whose result is stored in the target
        qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.NOR(self.sid, qi1, qi2, qo)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def qxnor(self, qi1, qi2, qo):
        """Logical XOR

        Logical exlusive-NOR of 2 qubits whose result is stored in the target
        qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.XNOR(self.sid, qi1, qi2, qo)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def cland(self, ci, qi, qo):
        """Classical AND

        Logical AND with one qubit and one classical bit whose result is
        stored in target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.CLAND(self.sid, ci, qi, qo)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def clor(self, ci, qi, qo):
        """Classical OR

        Logical OR with one qubit and one classical bit whose result is
        stored in target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.CLOR(self.sid, ci, qi, qo)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def clxor(self, ci, qi, qo):
        """Classical XOR

        Logical exlusive-OR with one qubit and one classical bit whose result is
        stored in target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.CLXOR(self.sid, ci, qi, qo)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def clnand(self, ci, qi, qo):
        """Classical NAND

        Logical NAND with one qubit and one classical bit whose result is
        stored in target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.CLNAND(self.sid, ci, qi, qo)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def clnor(self, ci, qi, qo):
        """Classical NOR

        Logical NOR with one qubit and one classical bit whose result is
        stored in target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.CLNOR(self.sid, ci, qi, qo)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def clxnor(self, ci, qi, qo):
        """Classical XNOR

        Logical exlusive-NOR with one qubit and one classical bit whose result is
        stored in target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.CLXNOR(self.sid, ci, qi, qo)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    # Particular Quantum Circuits

    ## fourier transform
    def qft(self, qs):
        """Quantum Fourier Transform

        Applies Quantum Fourier Transform on the list of qubits provided.

        Args:
            qs: list of qubits

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.QFT(self.sid, len(qs), self._ulonglong_byref(qs))
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def iqft(self, qs):
        """Inverse-quantum Fourier Transform

        Applies Inverse-quantum Fourier Transform on the list of qubits
        provided.

        Args:
            qs: list of qubits

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.IQFT(self.sid, len(qs), self._ulonglong_byref(qs))
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    # pseudo-quantum

    ## allocate and release
    def allocate_qubit(self, qid):
        """Allocate Qubit

        Allocate 1 new qubit with the given qubit ID.

        Args:
            qid: qubit id

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.allocateQubit(self.sid, qid)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def release(self, q):
        """Release Qubit

        Release qubit given by the given qubit ID.

        Args:
            q: qubit id

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            If the qubit was in `|0>` state with small tolerance.
        """
        result = Qrack.qrack_lib.release(self.sid, q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return result

    def num_qubits(self):
        """Get Qubit count

        Returns the qubit count of the simulator.

        Args:
            q: qubit id

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Qubit count of the simulator
        """
        result = Qrack.qrack_lib.num_qubits(self.sid)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return result

    ## schmidt decomposition
    def compose(self, other, q):
        """Compose qubits

        Compose quantum description of given qubit with the current system.

        Args:
            q: qubit id

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.Compose(self.sid, other.sid, self._ulonglong_byref(q))
        self._qubitCount = self._qubitCount + other._qubitCount
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def decompose(self, q):
        """Decompose system

        Decompose the given qubit out of the system.
        Warning: The qubit subsystem state must be separable, or the behavior 
        of this method is undefined.

        Args:
            q: qubit id

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            State of the systems.
        """
        other = QrackSimulator()
        Qrack.qrack_lib.destroy(other.sid)
        l = len(q)
        other.sid = Qrack.qrack_lib.Decompose(self.sid, l, self._ulonglong_byref(q))
        self._qubitCount = self._qubitCount - l
        other._qubitCount = l
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return other

    def dispose(self, q):
        """Dispose qubits

        Minimally decompose a set of contiguous bits from the separably
        composed unit, and discard the separable bits.
        Warning: The qubit subsystem state must be separable, or the behavior 
        of this method is undefined.

        Args:
            q: qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            State of the systems.
        """
        l = len(q)
        Qrack.qrack_lib.Dispose(self.sid, l, self._ulonglong_byref(q))
        self._qubitCount = self._qubitCount - l
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    ## miscellaneous
    def dump_ids(self):
        """Dump all IDs

        Dump all IDs from the selected simulator ID into the callback.

        Returns:
            List of ids
        """
        global ids_list
        global ids_list_index
        ids_list = [0] * self._qubitCount
        ids_list_index = 0
        Qrack.qrack_lib.DumpIds(self.sid, self.dump_ids_callback)
        return ids_list

    @ctypes.CFUNCTYPE(None, ctypes.c_ulonglong)
    def dump_ids_callback(i):
        """C callback function"""
        global ids_list
        global ids_list_index
        ids_list[ids_list_index] = i
        ids_list_index = ids_list_index + 1

    def dump(self):
        """Dump state vector

        Dump state vector from the selected simulator ID into the callback.

        Returns:
            State vector list
        """
        global state_vec_list
        global state_vec_list_index
        global state_vec_probability
        state_vec_list = [complex(0, 0)] * (1 << self._qubitCount)
        state_vec_list_index = 0
        state_vec_probability = 0
        Qrack.qrack_lib.Dump(self.sid, self.dump_callback)
        return state_vec_list

    @ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_double, ctypes.c_double)
    def dump_callback(r, i):
        """C callback function"""
        global state_vec_list
        global state_vec_list_index
        global state_vec_probability
        state_vec_list[state_vec_list_index] = complex(r, i)
        state_vec_list_index = state_vec_list_index + 1
        state_vec_probability = state_vec_probability + (r * r) + (i * i)
        if (1.0 - state_vec_probability) <= (7.0 / 3 - 4.0 / 3 - 1):
            return False
        return True

    def in_ket(self, ket):
        """Set state vector

        Set state vector for the selected simulator ID. 
        Warning: State vector is not always the internal representation leading 
        to sub-optimal performance of the method.

        Args:
            ket: the state vector to which simulator will be set

        Raises:
            RuntimeError: Not implemented for the given builds.
        """
        if Qrack.fppow == 5 or Qrack.fppow == 6:
            Qrack.qrack_lib.InKet(self.sid, self._qrack_complex_byref(ket))
            if self._get_error() != 0:
                raise RuntimeError("QrackSimulator C++ library raised exception.")
        else:
            raise NotImplementedError(
                "QrackSimulator.in_ket() not implemented for builds beside float/fp32 and double/fp64, but it can be overloaded."
            )

    def out_ket(self):
        """Set state vector

        Returns the raw state vector of the simulator.
        Warning: State vector is not always the internal representation leading 
        to sub-optimal performance of the method.

        Raises:
            RuntimeError: Not implemented for the given builds.

        Returns:
            list representing the state vector.
        """
        if Qrack.fppow == 5 or Qrack.fppow == 6:
            amp_count = 1 << self._qubitCount
            ket = self._qrack_complex_byref([complex(0, 0)] * amp_count)
            Qrack.qrack_lib.OutKet(self.sid, ket)
            if self._get_error() != 0:
                raise RuntimeError("QrackSimulator C++ library raised exception.")
            return [complex(r, i) for r, i in self._pairwise(ket)]
        raise NotImplementedError(
            "QrackSimulator.out_ket() not implemented for builds beside float/fp32 and double/fp64, but it can be overloaded."
        )

    def prob(self, q):
        """Probability of `|1>`

        Get the probability that a qubit is in the `|1>` state.

        Args:
            q: qubit id

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            probability of qubit being in `|1>`
        """
        result = Qrack.qrack_lib.Prob(self.sid, q)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return result

    def permutation_expectation(self, c):
        """Permutation expectation value

        Get the permutation expectation value, based upon the order of
        input qubits.

        Args:
            c: permutation

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        result = Qrack.qrack_lib.PermutationExpectation(
            self.sid, len(c), self._ulonglong_byref(c)
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return result

    def joint_ensemble_probability(self, b, q):
        """Ensemble probability

        Find the joint probability for all specified qubits under the
        respective Pauli basis transformations.

        Args:
            b: pauli basis
            q: specified qubits

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        result = Qrack.qrack_lib.JointEnsembleProbability(
            self.sid, len(b), self._ulonglong_byref(b), q
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return result

    def phase_parity(self, la, q):
        """Phase to odd parity

        Applies `e^(i*la)` phase factor to all combinations of bits with
        odd parity, based upon permutations of qubits.

        Args:
            la: phase
            q: specified qubits

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.PhaseParity(
            self.sid, ctypes.c_double(la), len(q), self._ulonglong_byref(q)
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def try_separate_1qb(self, qi1):
        """Manual seperation

        Exposes manual control for schmidt decomposition which attempts to
        decompose the qubit with possible performance improvement

        Args:
            qi1: qubit to be decomposed

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            State of the qubit.
        """
        result = Qrack.qrack_lib.TrySeparate1Qb(self.sid, qi1)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return result

    def try_separate_2qb(self, qi1, qi2):
        """Manual two-qubits seperation

        two-qubits counterpart of `try_separate_1qb`.

        Args:
            qi1: first qubit to be decomposed
            qi2: second qubit to be decomposed

        Raises:
            Runtimeerror: QrackSimulator raised an exception.

        Returns:
            State of both the qubits.
        """
        result = Qrack.qrack_lib.TrySeparate2Qb(self.sid, qi1, qi2)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return result

    def try_separate_tolerance(self, qs, t):
        """Manual multi-qubits seperation

        Multi-qubits counterpart of `try_separate_1qb`.

        Args:
            qs: list of qubits to be decomposed
            t: allowed tolerance

        Raises:
            Runtimeerror: QrackSimulator raised an exception.

        Returns:
            State of all the qubits.
        """
        result = Qrack.qrack_lib.TrySeparateTol(
            self.sid, len(qs), self._ulonglong_byref(qs), t
        )
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return result

    def set_reactive_separate(self, irs):
        """Set reactive separation option

        If reactive separation is available, then this method turns it off.
        Note that reactive separation is on by default.

        Args:
            irs: is aggresively separable

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.SetReactiveSeparate(self.sid, irs)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def _apply_pyzx_op(self, gate):
        if gate.name == "XPhase":
            self.r(Pauli.PauliX, math.pi * gate.phase, gate.target)
        elif gate.name == "ZPhase":
            self.r(Pauli.PauliZ, math.pi * gate.phase, gate.target)
        elif gate.name == "Z":
            self.z(gate.target)
        elif gate.name == "S":
            self.s(gate.target)
        elif gate.name == "T":
            self.t(gate.target)
        elif gate.name == "NOT":
            self.x(gate.target)
        elif gate.name == "HAD":
            self.h(gate.target)
        elif gate.name == "CNOT":
            self.mcx([gate.control], gate.target)
        elif gate.name == "CZ":
            self.mcz([gate.control], gate.target)
        elif gate.name == "CX":
            self.h(gate.control)
            if self._get_error() != 0:
                raise RuntimeError("QrackSimulator C++ library raised exception.")
            self.mcx([gate.control], gate.target)
            if self._get_error() != 0:
                raise RuntimeError("QrackSimulator C++ library raised exception.")
            self.h(gate.control)
        elif gate.name == "SWAP":
            self.swap(gate.control, gate.target)
        elif gate.name == "CRZ":
            self.mcr(Pauli.PauliZ, math.pi * gate.phase, [gate.control], gate.target)
        elif gate.name == "CHAD":
            self.mch([gate.control], gate.target)
        elif gate.name == "ParityPhase":
            self.phase_parity(math.pi * gate.phase, gate.targets)
        elif gate.name == "FSim":
            self.fsim(gate.theta, gate.phi, gate.control, gate.target)
        elif gate.name == "CCZ":
            self.mcz([gate.ctrl1, gate.ctrl2], gate.target)
        elif gate.name == "Tof":
            self.mcx([gate.ctrl1, gate.ctrl2], gate.target)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def run_pyzx_gates(self, gates):
        """PYZX Gates

        Converts PYZX gates to `QRackSimulator` and immediately executes them.

        Args:
            gates: list of PYZX gates

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        for gate in gates:
            _apply_pyzx_op(gate)

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
            self._sim.u(operation.qubits[0], 0, 0, operation.params[0])
        elif name == 'u2':
            self._sim.u(
                operation.qubits[0],
                math.pi / 2,
                operation.params[0],
                operation.params[1],
            )
        elif (name == 'u3') or (name == 'u'):
            self._sim.u(
                operation.qubits[0],
                operation.params[0],
                operation.params[1],
                operation.params[2],
            )
        elif name == 'r':
            self._sim.u(
                operation.qubits[0],
                operation.params[0],
                operation.params[1] - math.pi / 2,
                -operation.params[1] + mathh.pi / 2,
            )
        elif name == 'rx':
            self._sim.r(Pauli.PauliX, operation.params[0], operation.qubits[0])
        elif name == 'ry':
            self._sim.r(Pauli.PauliY, operation.params[0], operation.qubits[0])
        elif name == 'rz':
            self._sim.r(Pauli.PauliZ, operation.params[0], operation.qubits[0])
        elif name == 'h':
            self._sim.h(operation.qubits[0])
        elif name == 'x':
            self._sim.x(operation.qubits[0])
        elif name == 'y':
            self._sim.y(operation.qubits[0])
        elif name == 'z':
            self._sim.z(operation.qubits[0])
        elif name == 's':
            self._sim.s(operation.qubits[0])
        elif name == 'sdg':
            self._sim.adjs(operation.qubits[0])
        elif name == 'sx':
            self._sim.mtrx(
                [(1 + 1j) / 2, (1 - 1j) / 2, (1 - 1j) / 2, (1 + 1j) / 2],
                operation.qubits[0],
            )
        elif name == 'sxdg':
            self._sim.mtrx(
                [(1 - 1j) / 2, (1 + 1j) / 2, (1 + 1j) / 2, (1 - 1j) / 2],
                operation.qubits[0],
            )
        elif name == 't':
            self._sim.t(operation.qubits[0])
        elif name == 'tdg':
            self._sim.adjt(operation.qubits[0])
        elif name == 'cu1':
            self._sim.mcu(
                operation.qubits[0:1], operation.qubits[1], 0, 0, operation.params[0]
            )
        elif name == 'cu2':
            self._sim.mcu(
                operation.qubits[0:1],
                operation.qubits[1],
                math.pi / 2,
                operation.params[0],
                operation.params[1],
            )
        elif (name == 'cu3') or (name == 'cu'):
            self._sim.mcu(
                operation.qubits[0:1],
                operation.qubits[1],
                operation.params[0],
                operation.params[1],
                operation.params[2],
            )
        elif name == 'cx':
            self._sim.mcx(operation.qubits[0:1], operation.qubits[1])
        elif name == 'cy':
            self._sim.mcy(operation.qubits[0:1], operation.qubits[1])
        elif name == 'cz':
            self._sim.mcz(operation.qubits[0:1], operation.qubits[1])
        elif name == 'ch':
            self._sim.mch(operation.qubits[0:1], operation.qubits[1])
        elif name == 'cp':
            self._sim.mcmtrx(
                operation.qubits[0:1],
                [
                    1,
                    0,
                    0,
                    math.cos(operation.params[0]) + 1j * math.sin(operation.params[0]),
                ],
                operation.qubits[0],
            )
        elif name == 'csx':
            self._sim.mcmtrx(
                operation.qubits[0:1],
                [(1 + 1j) / 2, (1 - 1j) / 2, (1 - 1j) / 2, (1 + 1j) / 2],
                operation.qubits[1],
            )
        elif name == 'csxdg':
            self._sim.mcmtrx(
                operation.qubits[0:1],
                [(1 - 1j) / 2, (1 + 1j) / 2, (1 + 1j) / 2, (1 - 1j) / 2],
                operation.qubits[1],
            )
        elif name == 'dcx':
            self._sim.mcx(operation.qubits[0:1], operation.qubits[1])
            self._sim.mcx(operation.qubits[1:2], operation.qubits[0])
        elif name == 'ccx':
            self._sim.mcx(operation.qubits[0:2], operation.qubits[2])
        elif name == 'ccy':
            self._sim.mcy(operation.qubits[0:2], operation.qubits[2])
        elif name == 'ccz':
            self._sim.mcz(operation.qubits[0:2], operation.qubits[2])
        elif name == 'mcx':
            self._sim.mcx(operation.qubits[0:-1], operation.qubits[-1])
        elif name == 'mcy':
            self._sim.mcy(operation.qubits[0:-1], operation.qubits[-1])
        elif name == 'mcz':
            self._sim.mcz(operation.qubits[0:-1], operation.qubits[-1])
        elif name == 'swap':
            self._sim.swap(operation.qubits[0], operation.qubits[1])
        elif name == 'iswap':
            self._sim.iswap(operation.qubits[0], operation.qubits[1])
        elif name == 'cswap':
            self._sim.cswap(
                operation.qubits[0:1], operation.qubits[1], operation.qubits[2]
            )
        elif name == 'mcswap':
            self._sim.cswap(
                operation.qubits[:-2], operation.qubits[-2], operation.qubits[-1]
            )
        elif name == 'reset':
            qubits = operation.qubits
            for qubit in qubits:
                if self._sim.m(qubit):
                    self._sim.x(qubit)
        elif name == 'measure':
            qubits = operation.qubits
            clbits = operation.memory
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
                    qubit_outcome = self._sim.m(qubits[index])

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
            err_msg = 'QrackSimulator encountered unrecognized operation "{0}"'
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
        data = []
        if num_samples == 1:
            sample = self._sim.m_all()
            result = 0
            for index in range(len(measure_qubit)):
                qubit = measure_qubit[index]
                qubit_outcome = (sample >> qubit) & 1
                result |= qubit_outcome << index
            measure_results = [result]
        else:
            measure_results = self._sim.measure_shots(measure_qubit, num_samples)

        for sample in measure_results:
            for index in range(len(measure_qubit)):
                qubit_outcome = (sample >> index) & 1
                clbit = measure_clbit[index]
                clmask = 1 << clbit
                self._classical_memory = (self._classical_memory & (~clmask)) | (
                    qubit_outcome << clbit
                )

            data.append(hex(int(bin(self._classical_memory)[2:], 2)))

        return data

    def run_qiskit_circuit(self, experiment, shots=1):
        if not _IS_QISKIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to run_qiskit_circuit() with QrackSimulator, you must install Qiskit!"
            )

        if isinstance(experiment, QuantumCircuit):
            experiment = convert_qiskit_circuit_to_qasm_experiment(experiment)

        instructions = []
        if isinstance(experiment, QasmQobjExperiment):
            instructions = experiment.instructions
        else:
            raise RuntimeError('Unrecognized "run_input" argument specified for run().')

        self._shots = shots
        self._sample_qubits = []
        self._sample_clbits = []
        self._sample_cregbits = []
        self._sample_measure = True
        _data = []
        shotsPerLoop = self._shots
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
                self._sim = QrackSimulator(cloneSid=preamble_sim.sid)
                self._classical_memory = preamble_memory
                self._classical_register = preamble_register

            for operation in instructions[boundary_start:]:
                self._apply_op(operation)

            if not self._sample_measure and (len(self._sample_qubits) > 0):
                _data += [hex(int(bin(self._classical_memory)[2:], 2))]
                self._sample_qubits = []
                self._sample_clbits = []
                self._sample_cregbits = []

        if self._sample_measure and (len(self._sample_qubits) > 0):
            _data = self._add_sample_measure(
                self._sample_qubits, self._sample_clbits, self._shots
            )

        del self._sim
        del self._shots
        del self._sample_qubits
        del self._sample_clbits
        del self._sample_cregbits
        del self._sample_measure

        return _data
