# (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import copy
import ctypes
import math
import re
from .qrack_system import Qrack
from .pauli import Pauli

_IS_QISKIT_AVAILABLE = True
try:
    from qiskit.circuit import QuantumRegister, Qubit
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from qiskit.compiler import transpile
    from qiskit.qobj.qasm_qobj import QasmQobjExperiment
    from qiskit.quantum_info.operators.symplectic.clifford import Clifford
    from .util import convert_qiskit_circuit_to_qasm_experiment
except ImportError:
    _IS_QISKIT_AVAILABLE = False

_IS_NUMPY_AVAILABLE = True
try:
    import numpy as np
except:
    _IS_NUMPY_AVAILABLE = False


class QrackSimulator:
    """Interface for all the QRack functionality.

    Attributes:
        qubitCount(int): Number of qubits that are to be simulated.
        sid(int): Corresponding simulator id.
    """

    def _get_error(self):
        return Qrack.qrack_lib.get_error(self.sid)

    def _throw_if_error(self):
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")

    def __init__(
        self,
        qubitCount=-1,
        cloneSid=-1,
        isTensorNetwork=True,
        isSchmidtDecomposeMulti=True,
        isSchmidtDecompose=True,
        isStabilizerHybrid=False,
        isBinaryDecisionTree=False,
        isPaged=True,
        isCpuGpuHybrid=True,
        isOpenCL=True,
        isHostPointer=False,
        noise=0,
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
                "Cannot clone a QrackSimulator and specify its qubit length at the same time, in QrackSimulator constructor!"
            )

        self.is_tensor_network = isTensorNetwork

        if cloneSid > -1:
            self.sid = Qrack.qrack_lib.init_clone(cloneSid)
        else:
            if qubitCount < 0:
                qubitCount = 0

            self.sid = Qrack.qrack_lib.init_count_type(
                qubitCount,
                isTensorNetwork,
                isSchmidtDecomposeMulti,
                isSchmidtDecompose,
                isStabilizerHybrid,
                isBinaryDecisionTree,
                isPaged,
                (noise > 0),
                isCpuGpuHybrid,
                isOpenCL,
                isHostPointer
            )

        self._throw_if_error()

        if noise > 0:
            self.set_noise_parameter(noise)

        if pyzxCircuit is not None:
            self.run_pyzx_gates(pyzxCircuit.gates)
        elif qiskitCircuit is not None:
            self.run_qiskit_circuit(qiskitCircuit)

    def __del__(self):
        if self.sid is not None:
            Qrack.qrack_lib.destroy(self.sid)
            self.sid = None

    def _int_byref(self, a):
        return (ctypes.c_int * len(a))(*a)

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

    def _bool_byref(self, a):
        return (ctypes.c_bool * len(a))(*a)

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

        return b

    def _to_ulonglong(self, m, v):
        b = (ctypes.c_ulonglong * (m * len(v)))()
        n = 0
        for u in v:
            for _ in range(m):
                b[n] = u & 0xFFFFFFFFFFFFFFFF
                u >>= 64
                n += 1

        return b

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
    def seed(self, s):
        Qrack.qrack_lib.seed(self.sid, s)
        self._throw_if_error()

    def set_concurrency(self, p):
        Qrack.qrack_lib.set_concurrency(self.sid, p)
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

    def h(self, q):
        """Applies H gate.

        Applies the Hadarmard operator to the qubit at “q.”

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.H(self.sid, q)
        self._throw_if_error()

    def s(self, q):
        """Applies S gate.

        Applies the 1/4 phase rotation to the qubit at “q.”

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.S(self.sid, q)
        self._throw_if_error()

    def t(self, q):
        """Applies T gate.

        Applies the 1/8 phase rotation to the qubit at “q.”

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.T(self.sid, q)
        self._throw_if_error()

    def adjs(self, q):
        """Adjoint of S gate

        Applies the gate equivalent to the inverse of S gate.

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.AdjS(self.sid, q)
        self._throw_if_error()

    def adjt(self, q):
        """Adjoint of T gate

        Applies the gate equivalent to the inverse of T gate.

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.AdjT(self.sid, q)
        self._throw_if_error()

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
        self._throw_if_error()

    def mtrx(self, m, q):
        """Operation from matrix.

        Applies arbitrary operation defined by the given matrix.

        Args:
            m: row-major complex list representing the operator.
            q: the qubit number on which the gate is applied to.

        Raises:
            ValueError: 2x2 matrix 'm' in QrackSimulator.mtrx() must contain at least 4 elements.
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(m) < 4:
            raise ValueError("2x2 matrix 'm' in QrackSimulator.mtrx() must contain at least 4 elements.")
        Qrack.qrack_lib.Mtrx(self.sid, self._complex_byref(m), q)
        self._throw_if_error()

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
        self._throw_if_error()

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
        if len(b) != len(q):
            raise RuntimeError("Lengths of list parameters are mismatched.")
        Qrack.qrack_lib.Exp(
            self.sid,
            len(b),
            self._ulonglong_byref(b),
            ctypes.c_double(ph),
            self._ulonglong_byref(q),
        )
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

    def mcmtrx(self, c, m, q):
        """Multi-controlled arbitrary operator

        If all controlled qubits are `|1>` then the arbitrary operation by
        parameters is applied to the target qubit.

        Args:
            c: list of controlled qubits
            m: row-major complex list representing the operator.
            q: target qubit

        Raises:
            ValueError: 2x2 matrix 'm' in QrackSimulator.mcmtrx() must contain at least 4 elements.
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(m) < 4:
            raise ValueError("2x2 matrix 'm' in QrackSimulator.mcmtrx() must contain at least 4 elements.")
        Qrack.qrack_lib.MCMtrx(
            self.sid, len(c), self._ulonglong_byref(c), self._complex_byref(m), q
        )
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

    def macmtrx(self, c, m, q):
        """Anti multi-controlled arbitraty operator

        If all controlled qubits are `|0>` then the arbitrary operation by
        parameters is applied to the target qubit.

        Args:
            c: list of controlled qubits.
            m: row-major complex matrix which defines the operator.
            q: target qubit.

        Raises:
            ValueError: 2x2 matrix 'm' in QrackSimulator.macmtrx() must contain at least 4 elements.
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(m) < 4:
            raise ValueError("2x2 matrix 'm' in QrackSimulator.macmtrx() must contain at least 4 elements.")
        Qrack.qrack_lib.MACMtrx(
            self.sid, len(c), self._ulonglong_byref(c), self._complex_byref(m), q
        )
        self._throw_if_error()

    def ucmtrx(self, c, m, q, p):
        """Multi-controlled arbitrary operator with arbitrary controls

        If all control qubits match 'p' permutation by bit order, then the arbitrary
        operation by parameters is applied to the target qubit.

        Args:
            c: list of control qubits
            m: row-major complex list representing the operator.
            q: target qubit
            p: permutation of list of control qubits

        Raises:
            ValueError: 2x2 matrix 'm' in QrackSimulator.ucmtrx() must contain at least 4 elements.
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(m) < 4:
            raise ValueError("2x2 matrix 'm' in QrackSimulator.ucmtrx() must contain at least 4 elements.")
        Qrack.qrack_lib.UCMtrx(
            self.sid, len(c), self._ulonglong_byref(c), self._complex_byref(m), q, p
        )
        self._throw_if_error()

    def multiplex1_mtrx(self, c, q, m):
        """Multiplex gate

        A multiplex gate with a single target and an arbitrary number of
        controls.

        Args:
            c: list of controlled qubits.
            m: row-major complex matrix which defines the operator.
            q: target qubit.

        Raises:
            ValueError: Multiplex matrix 'm' in QrackSimulator.multiplex1_mtrx() must contain at least 4 elements.
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(m) < ((1 << len(c)) * 4):
            raise ValueError("Multiplex matrix 'm' in QrackSimulator.multiplex1_mtrx() must contain at least (4 * 2 ** len(c)) elements.")
        Qrack.qrack_lib.Multiplex1Mtrx(
            self.sid, len(c), self._ulonglong_byref(c), q, self._complex_byref(m)
        )
        self._throw_if_error()

    def mx(self, q):
        """Multi X-gate

        Applies the Pauli “X” operator on all qubits.

        Args:
            q: list of qubits to apply X on.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MX(self.sid, len(q), self._ulonglong_byref(q))
        self._throw_if_error()

    def my(self, q):
        """Multi Y-gate

        Applies the Pauli “Y” operator on all qubits.

        Args:
            q: list of qubits to apply Y on.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MY(self.sid, len(q), self._ulonglong_byref(q))
        self._throw_if_error()

    def mz(self, q):
        """Multi Z-gate

        Applies the Pauli “Z” operator on all qubits.

        Args:
            q: list of qubits to apply Z on.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.MZ(self.sid, len(q), self._ulonglong_byref(q))
        self._throw_if_error()

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
        self._throw_if_error()

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
        if len(b) != len(q):
            raise RuntimeError("Lengths of list parameters are mismatched.")
        Qrack.qrack_lib.MCExp(
            self.sid,
            len(b),
            self._ulonglong_byref(b),
            ctypes.c_double(ph),
            len(cs),
            self._ulonglong_byref(cs),
            self._ulonglong_byref(q),
        )
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()
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
        self._throw_if_error()
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
        self._throw_if_error()
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
        if len(b) != len(q):
            raise RuntimeError("Lengths of list parameters are mismatched.")
        result = Qrack.qrack_lib.Measure(
            self.sid, len(b), self._int_byref(b), self._ulonglong_byref(q)
        )
        self._throw_if_error()
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
        self._throw_if_error()
        return [m[i] for i in range(s)]

    def reset_all(self):
        """Reset gate

        Resets all qubits to `|0>`

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.ResetAll(self.sid)
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

    def mul(self, a, q, o):
        """Multiplies integer to qubit

        Multiplies the given integer to the given set of qubits.
        Carry register is required for maintaining the unitary nature of
        operation and must be as long as the input qubit register.

        Args:
            a: number to multiply
            q: list of qubits to multiply the number
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot mul()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot mul()! (Turn off just this option, in the constructor.)")

        if len(q) != len(o):
            raise RuntimeError("Lengths of list parameters are mismatched.")
        aParts = self._split_longs(a)
        Qrack.qrack_lib.MUL(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(q),
            self._ulonglong_byref(q),
            self._ulonglong_byref(o),
        )
        self._throw_if_error()

    def div(self, a, q, o):
        """Divides qubit by integer

        'Divides' the given qubits by the integer.
        (This is rather the adjoint of mul().)
        Carry register is required for maintaining the unitary nature of
        operation.

        Args:
            a: integer to divide by
            q: qubits to divide
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot div()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot div()! (Turn off just this option, in the constructor.)")

        if len(q) != len(o):
            raise RuntimeError("Lengths of list parameters are mismatched.")
        aParts = self._split_longs(a)
        Qrack.qrack_lib.DIV(
            self.sid,
            len(aParts),
            self._ulonglong_byref(aParts),
            len(q),
            self._ulonglong_byref(q),
            self._ulonglong_byref(o),
        )
        self._throw_if_error()

    def muln(self, a, m, q, o):
        """Modulo Multiplication

        Modulo Multiplication of the given integer to the given set of qubits
        Out-of-place register is required to store the resultant.

        Args:
            a: number to multiply
            m: modulo number
            q: list of qubits to multiply the number
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(q) != len(o):
            raise RuntimeError("Lengths of list parameters are mismatched.")
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
        self._throw_if_error()

    def divn(self, a, m, q, o):
        """Modulo Division

        'Modulo Division' of the given set of qubits by the given integer
        (This is rather the adjoint of muln().)
        Out-of-place register is required to retrieve the resultant.

        Args:
            a: integer by which qubit will be divided
            m: modulo integer
            q: qubits to divide
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(q) != len(o):
            raise RuntimeError("Lengths of list parameters are mismatched.")
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
        self._throw_if_error()

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
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot pown()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot pown()! (Turn off just this option, in the constructor.)")

        if len(q) != len(o):
            raise RuntimeError("Lengths of list parameters are mismatched.")
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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

    def mcmul(self, a, c, q, o):
        """Controlled-multiply

        Multiplies the given integer to the given set of qubits if all controlled
        qubits are `|1>`.
        Carry register is required for maintaining the unitary nature of
        operation.

        Args:
            a: number to multiply
            c: list of controlled qubits.
            q: list of qubits to add the number
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot mcmul()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot mcmul()! (Turn off just this option, in the constructor.)")

        if len(q) != len(o):
            raise RuntimeError("Lengths of list parameters are mismatched.")
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
        self._throw_if_error()

    def mcdiv(self, a, c, q, o):
        """Controlled-divide.

        'Divides' the given qubits by the integer if all controlled
        qubits are `|1>`.
        (This is rather the adjoint of mcmul().)
        Carry register is required for maintaining the unitary nature of
        operation.

        Args:
            a: number to divide by
            c: list of controlled qubits.
            q: qubits to divide
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot mcdiv()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot mcdiv()! (Turn off just this option, in the constructor.)")

        if len(q) != len(o):
            raise RuntimeError("Lengths of list parameters are mismatched.")
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
        self._throw_if_error()

    def mcmuln(self, a, c, m, q, o):
        """Controlled-modulo multiplication

        Modulo multiplication of the given integer to the given set of qubits
        if all controlled qubits are `|1>`.
        Out-of-place register is required to store the resultant.

        Args:
            a: number to multiply
            c: list of controlled qubits.
            m: modulo number
            q: list of qubits to add the number
            o: out-of-place output register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(q) != len(o):
            raise RuntimeError("Lengths of list parameters are mismatched.")
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
        self._throw_if_error()

    def mcdivn(self, a, c, m, q, o):
        """Controlled-divide.

        Modulo division of the given qubits by the given number if all
        controlled qubits are `|1>`.
        (This is rather the adjoint of mcmuln().)
        Out-of-place register is required to retrieve the resultant.

        Args:
            a: number to divide by
            c: list of controlled qubits.
            m: modulo number
            q: qubits to divide
            o: carry register

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(q) != len(o):
            raise RuntimeError("Lengths of list parameters are mismatched.")
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
        self._throw_if_error()

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
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot mcpown()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot mcpown()! (Turn off just this option, in the constructor.)")

        if len(q) != len(o):
            raise RuntimeError("Lengths of list parameters are mismatched.")
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
        self._throw_if_error()

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
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot lda()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot lda()! (Turn off just this option, in the constructor.)")

        Qrack.qrack_lib.LDA(
            self.sid,
            len(qi),
            self._ulonglong_byref(qi),
            len(qv),
            self._ulonglong_byref(qv),
            self._to_ubyte(len(qv), t),
        )
        self._throw_if_error()

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
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot adc()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot adc()! (Turn off just this option, in the constructor.)")

        Qrack.qrack_lib.ADC(
            self.sid,
            s,
            len(qi),
            self._ulonglong_byref(qi),
            len(qv),
            self._ulonglong_byref(qv),
            self._to_ubyte(len(qv), t),
        )
        self._throw_if_error()

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
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot sbc()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot sbc()! (Turn off just this option, in the constructor.)")

        Qrack.qrack_lib.SBC(
            self.sid,
            s,
            len(qi),
            self._ulonglong_byref(qi),
            len(qv),
            self._ulonglong_byref(qv),
            self._to_ubyte(len(qv), t),
        )
        self._throw_if_error()

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
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot hash()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot hash()! (Turn off just this option, in the constructor.)")

        Qrack.qrack_lib.Hash(
            self.sid, len(q), self._ulonglong_byref(q), self._to_ubyte(len(q), t)
        )
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()

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
        self._throw_if_error()
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
        self._throw_if_error()
        return result

    ## schmidt decomposition
    def compose(self, other, q):
        """Compose qubits

        Compose quantum description of given qubit with the current system.

        Args:
            q: qubit id

        Raises:
            RuntimeError: QrackSimulator raised an exception.
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot compose()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot compose()! (Turn off just this option, in the constructor.)")

        Qrack.qrack_lib.Compose(self.sid, other.sid, self._ulonglong_byref(q))
        self._throw_if_error()

    def decompose(self, q):
        """Decompose system

        Decompose the given qubit out of the system.
        Warning: The qubit subsystem state must be separable, or the behavior 
        of this method is undefined.

        Args:
            q: qubit id

        Raises:
            RuntimeError: QrackSimulator raised an exception.
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot decompose()! (Turn off just this option, in the constructor.)

        Returns:
            State of the systems.
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot decompose()! (Turn off just this option, in the constructor.)")

        other = QrackSimulator()
        Qrack.qrack_lib.destroy(other.sid)
        l = len(q)
        other.sid = Qrack.qrack_lib.Decompose(self.sid, l, self._ulonglong_byref(q))
        self._throw_if_error()
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
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot dispose()! (Turn off just this option, in the constructor.)

        Returns:
            State of the systems.
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot dispose()! (Turn off just this option, in the constructor.)")

        l = len(q)
        Qrack.qrack_lib.Dispose(self.sid, l, self._ulonglong_byref(q))
        self._throw_if_error()

    ## miscellaneous
    def dump_ids(self):
        """Dump all IDs

        Dump all IDs from the selected simulator ID into the callback.

        Returns:
            List of ids
        """
        global ids_list
        global ids_list_index
        ids_list = [0] * self.num_qubits()
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
        state_vec_list = [complex(0, 0)] * (1 << self.num_qubits())
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
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.InKet(self.sid, self._qrack_complex_byref(ket))
        self._throw_if_error()

    def out_ket(self):
        """Get state vector

        Returns the raw state vector of the simulator.
        Warning: State vector is not always the internal representation leading 
        to sub-optimal performance of the method.

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            list representing the state vector.
        """
        amp_count = 1 << self.num_qubits()
        ket = self._qrack_complex_byref([complex(0, 0)] * amp_count)
        Qrack.qrack_lib.OutKet(self.sid, ket)
        self._throw_if_error()
        return [complex(r, i) for r, i in self._pairwise(ket)]

    def out_probs(self):
        """Get basis dimension probabilities

        Returns the probabilities of each basis dimension in the state vector
        of the simulator.

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            list representing the basis dimension probabilities.
        """
        prob_count = 1 << self.num_qubits()
        probs = self._real1_byref([0.0] * prob_count)
        Qrack.qrack_lib.OutProbs(self.sid, probs)
        self._throw_if_error()
        return list(probs)

    def prob_all(self, q):
        """Probabilities of all subset permutations

        Get the probabilities of all permutations of the subset.

        Args:
            q: list of qubit ids

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            list representing the state vector.
        """
        probs = self._real1_byref([0.0] * (1 << len(q)))
        Qrack.qrack_lib.ProbAll(self.sid, len(q), self._ulonglong_byref(q), probs)
        self._throw_if_error()
        return list(probs)

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
        self._throw_if_error()
        return result

    def prob_rdm(self, q):
        """Probability of `|1>`, (tracing out the reduced
        density matrix without stabilizer ancillary qubits)

        Get the probability that a qubit is in the `|1>` state.

        Args:
            q: qubit id

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            probability of qubit being in `|1>`
        """
        result = Qrack.qrack_lib.ProbRdm(self.sid, q)
        self._throw_if_error()
        return result

    def prob_perm(self, q, c):
        """Probability of permutation

        Get the probability that the qubit IDs in "q" have the truth values
        in "c", directly corresponding by list index.

        Args:
            q: list of qubit ids
            c: list of qubit truth values bools

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            probability that each qubit in "q[i]" has corresponding truth
            value in "c[i]", at once
        """

        if len(q) != len(c):
            raise RuntimeError("prob_perm argument lengths do not match.")
        result = Qrack.qrack_lib.PermutationProb(self.sid, len(q), self._ulonglong_byref(q), self._bool_byref(c));
        self._throw_if_error()
        return result

    def prob_perm_rdm(self, q, c, r = True):
        """Probability of permutation, (tracing out the reduced
        density matrix without stabilizer ancillary qubits)

        Get the probability that the qubit IDs in "q" have the truth
        values in "c", directly corresponding by list index.

        Args:
            q: list of qubit ids
            c: list of qubit truth values bools
            r: round Rz gates down from T^(1/2)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            probability that each qubit in "q[i]" has corresponding truth
            value in "c[i]", at once
        """

        if len(q) != len(c):
            raise RuntimeError("prob_perm argument lengths do not match.")
        result = Qrack.qrack_lib.PermutationProbRdm(self.sid, len(q), self._ulonglong_byref(q), self._bool_byref(c), r);
        self._throw_if_error()
        return result

    def permutation_expectation(self, q):
        """Permutation expectation value

        Get the permutation expectation value, based upon the order of
        input qubits.

        Args:
            q: qubits, from low to high

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        result = Qrack.qrack_lib.PermutationExpectation(
            self.sid, len(q), self._ulonglong_byref(q)
        )
        self._throw_if_error()
        return result

    def permutation_expectation_rdm(self, q, r = True):
        """Permutation expectation value, (tracing out the reduced
        density matrix without stabilizer ancillary qubits)

        Get the permutation expectation value, based upon the order of
        input qubits.

        Args:
            q: qubits, from low to high
            r: round Rz gates down from T^(1/2)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        result = Qrack.qrack_lib.PermutationExpectationRdm(
            self.sid, len(q), self._ulonglong_byref(q), r
        )
        self._throw_if_error()
        return result

    def factorized_expectation(self, q, c):
        """Factorized expectation value

        Get the factorized expectation value, where each entry
        in "c" is an expectation value for corresponding "q"
        being false, then true, repeated for each in "q".

        Args:
            q: qubits, from low to high
            c: qubit falsey/truthy values, from low to high

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        if (len(q) << 1) != len(c):
            raise RuntimeError("factorized_expectation argument lengths do not match.")
        m = max([(x.bit_length() + 63) // 64 for x in c])
        result = Qrack.qrack_lib.FactorizedExpectation(
            self.sid, len(q), self._ulonglong_byref(q), m, self._to_ulonglong(m, c)
        )
        self._throw_if_error()
        return result

    def factorized_expectation_rdm(self, q, c, r = True):
        """Factorized expectation value, (tracing out the reduced
        density matrix without stabilizer ancillary qubits)

        Get the factorized expectation value, where each entry
        in "c" is an expectation value for corresponding "q"
        being false, then true, repeated for each in "q".

        Args:
            q: qubits, from low to high
            c: qubit falsey/truthy values, from low to high
            r: round Rz gates down from T^(1/2)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        if (len(q) << 1) != len(c):
            raise RuntimeError("factorized_expectation_rdm argument lengths do not match.")
        m = max([(x.bit_length() + 63) // 64 for x in c])
        result = Qrack.qrack_lib.FactorizedExpectationRdm(
            self.sid, len(q), self._ulonglong_byref(q), m, self._to_ulonglong(m, c), r
        )
        self._throw_if_error()
        return result

    def factorized_expectation_fp(self, q, c):
        """Factorized expectation value (floating-point)

        Get the factorized expectation value, where each entry
        in "c" is an expectation value for corresponding "q"
        being false, then true, repeated for each in "q".

        Args:
            q: qubits, from low to high
            c: qubit falsey/truthy values, from low to high

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        if (len(q) << 1) != len(c):
            raise RuntimeError("factorized_expectation_rdm argument lengths do not match.")
        result = Qrack.qrack_lib.FactorizedExpectationFp(
            self.sid, len(q), self._ulonglong_byref(q), self._real1_byref(c)
        )
        self._throw_if_error()
        return result

    def factorized_expectation_fp_rdm(self, q, c, r = True):
        """Factorized expectation value, (tracing out the reduced
        density matrix without stabilizer ancillary qubits)

        Get the factorized expectation value, where each entry
        in "c" is an expectation value for corresponding "q"
        being false, then true, repeated for each in "q".

        Args:
            q: qubits, from low to high
            c: qubit falsey/truthy values, from low to high
            r: round Rz gates down from T^(1/2)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        if (len(q) << 1) != len(c):
            raise RuntimeError("factorized_expectation_fp_rdm argument lengths do not match.")
        result = Qrack.qrack_lib.FactorizedExpectationFpRdm(
            self.sid, len(q), self._ulonglong_byref(q), self._real1_byref(c), r
        )
        self._throw_if_error()
        return result

    def unitary_expectation(self, q, b):
        """3-parameter unitary tensor product expectation value

        Get the single-qubit (3-parameter) operator
        expectation value for the array of qubits and bases.

        Args:
            q: qubits, from low to high
            b: 3-parameter, single-qubit, unitary bases (flat over wires)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        if (3 * len(q)) != len(b):
            raise RuntimeError("unitary_expectation argument lengths do not match.")
        result = Qrack.qrack_lib.UnitaryExpectation(
            self.sid, len(q), self._ulonglong_byref(q), self._real1_byref(b)
        )
        self._throw_if_error()
        return result

    def matrix_expectation(self, q, b):
        """Single-qubit operator tensor product expectation value

        Get the single-qubit (3-parameter) operator
        expectation value for the array of qubits and bases.

        Args:
            q: qubits, from low to high
            b: single-qubit (2x2) operator unitary bases (flat over wires)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        if (len(q) << 2) != len(b):
            raise RuntimeError("matrix_expectation argument lengths do not match.")
        result = Qrack.qrack_lib.MatrixExpectation(
            self.sid, len(q), self._ulonglong_byref(q), self._complex_byref(b)
        )
        self._throw_if_error()
        return result

    def unitary_expectation_eigenval(self, q, b, e):
        """3-parameter unitary tensor product expectation value

        Get the single-qubit (3-parameter) operator
        expectation value for the array of qubits and bases.

        Args:
            q: qubits, from low to high
            b: 3-parameter, single-qubit, unitary bases (flat over wires)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        if (3 * len(q)) != len(b):
            raise RuntimeError("unitary_expectation_eigenval qubit and basis argument lengths do not match.")
        if (len(q) << 1) != len(e):
            raise RuntimeError("unitary_expectation_eigenval qubit and eigenvalue argument lengths do not match.")
        result = Qrack.qrack_lib.UnitaryExpectationEigenVal(
            self.sid, len(q), self._ulonglong_byref(q), self._real1_byref(b), self._real1_byref(e)
        )
        self._throw_if_error()
        return result

    def matrix_expectation_eigenval(self, q, b, e):
        """Single-qubit operator tensor product expectation value

        Get the single-qubit (3-parameter) operator
        expectation value for the array of qubits and bases.

        Args:
            q: qubits, from low to high
            b: single-qubit (2x2) operator unitary bases (flat over wires)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        if (len(q) << 2) != len(b):
            raise RuntimeError("matrix_expectation_eigenval qubit and basis argument lengths do not match.")
        if (len(q) << 1) != len(e):
            raise RuntimeError("matrix_expectation_eigenval qubit and eigenvalue argument lengths do not match.")
        result = Qrack.qrack_lib.MatrixExpectationEigenVal(
            self.sid, len(q), self._ulonglong_byref(q), self._complex_byref(b), self._real1_byref(e)
        )
        self._throw_if_error()
        return result

    def pauli_expectation(self, q, b):
        """Pauli tensor product expectation value

        Get the Pauli tensor product expectation value,
        where each entry in "b" is a Pauli observable for
        corresponding "q", as the product for each in "q".

        Args:
            q: qubits, from low to high
            b: qubit Pauli bases

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Expectation value
        """
        if len(q) != len(b):
            raise RuntimeError("pauli_expectation argument lengths do not match.")
        result = Qrack.qrack_lib.PauliExpectation(
            self.sid, len(q), self._ulonglong_byref(q), self._ulonglong_byref(b)
        )
        self._throw_if_error()
        return result

    def variance(self, q):
        """Variance of probabilities of all subset permutations

        Get the overall variance of probabilities of all
        permutations of the subset.

        Args:
            q: list of qubit ids

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            float variance
        """
        result = Qrack.qrack_lib.Variance(self.sid, len(q), self._ulonglong_byref(q))
        self._throw_if_error()
        return result

    def variance_rdm(self, q, r = True):
        """Permutation variance, (tracing out the reduced
        density matrix without stabilizer ancillary qubits)

        Get the permutation variance, based upon the order of
        input qubits.

        Args:
            q: qubits, from low to high
            r: round Rz gates down from T^(1/2)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            variance
        """
        result = Qrack.qrack_lib.VarianceRdm(
            self.sid, len(q), self._ulonglong_byref(q), r
        )
        self._throw_if_error()
        return result

    def factorized_variance(self, q, c):
        """Factorized variance

        Get the factorized variance, where each entry
        in "c" is an variance for corresponding "q"
        being false, then true, repeated for each in "q".

        Args:
            q: qubits, from low to high
            c: qubit falsey/truthy values, from low to high

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            variance
        """
        if (len(q) << 1) != len(c):
            raise RuntimeError("factorized_variance argument lengths do not match.")
        m = max([(x.bit_length() + 63) // 64 for x in c])
        result = Qrack.qrack_lib.FactorizedVariance(
            self.sid, len(q), self._ulonglong_byref(q), m, self._to_ulonglong(m, c)
        )
        self._throw_if_error()
        return result

    def factorized_variance_rdm(self, q, c, r = True):
        """Factorized variance, (tracing out the reduced
        density matrix without stabilizer ancillary qubits)

        Get the factorized variance, where each entry
        in "c" is an variance for corresponding "q"
        being false, then true, repeated for each in "q".

        Args:
            q: qubits, from low to high
            c: qubit falsey/truthy values, from low to high
            r: round Rz gates down from T^(1/2)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            variance
        """
        if (len(q) << 1) != len(c):
            raise RuntimeError("factorized_variance_rdm argument lengths do not match.")
        m = max([(x.bit_length() + 63) // 64 for x in c])
        result = Qrack.qrack_lib.FactorizedVarianceRdm(
            self.sid, len(q), self._ulonglong_byref(q), m, self._to_ulonglong(m, c), r
        )
        self._throw_if_error()
        return result

    def factorized_variance_fp(self, q, c):
        """Factorized variance (floating-point)

        Get the factorized variance, where each entry
        in "c" is an variance for corresponding "q"
        being false, then true, repeated for each in "q".

        Args:
            q: qubits, from low to high
            c: qubit falsey/truthy values, from low to high

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            variance
        """
        if (len(q) << 1) != len(c):
            raise RuntimeError("factorized_variance_rdm argument lengths do not match.")
        result = Qrack.qrack_lib.FactorizedVarianceFp(
            self.sid, len(q), self._ulonglong_byref(q), self._real1_byref(c)
        )
        self._throw_if_error()
        return result

    def factorized_variance_fp_rdm(self, q, c, r = True):
        """Factorized variance, (tracing out the reduced
        density matrix without stabilizer ancillary qubits)

        Get the factorized variance, where each entry
        in "c" is an variance for corresponding "q"
        being false, then true, repeated for each in "q".

        Args:
            q: qubits, from low to high
            c: qubit falsey/truthy values, from low to high
            r: round Rz gates down from T^(1/2)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            variance
        """
        if (len(q) << 1) != len(c):
            raise RuntimeError("factorized_variance_fp_rdm argument lengths do not match.")
        result = Qrack.qrack_lib.FactorizedVarianceFpRdm(
            self.sid, len(q), self._ulonglong_byref(q), self._real1_byref(c), r
        )
        self._throw_if_error()
        return result

    def unitary_variance(self, q, b):
        """3-parameter unitary tensor product variance

        Get the single-qubit (3-parameter) operator
        variance for the array of qubits and bases.

        Args:
            q: qubits, from low to high
            b: 3-parameter, single-qubit, unitary bases (flat over wires)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            variance
        """
        if (3 * len(q)) != len(b):
            raise RuntimeError("unitary_variance argument lengths do not match.")
        result = Qrack.qrack_lib.UnitaryVariance(
            self.sid, len(q), self._ulonglong_byref(q), self._real1_byref(b)
        )
        self._throw_if_error()
        return result

    def matrix_variance(self, q, b):
        """Single-qubit operator tensor product variance

        Get the single-qubit (3-parameter) operator
        variance for the array of qubits and bases.

        Args:
            q: qubits, from low to high
            b: single-qubit (2x2) operator unitary bases (flat over wires)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            variance
        """
        if (len(q) << 2) != len(b):
            raise RuntimeError("matrix_variance argument lengths do not match.")
        result = Qrack.qrack_lib.MatrixVariance(
            self.sid, len(q), self._ulonglong_byref(q), self._complex_byref(b)
        )
        self._throw_if_error()
        return result

    def unitary_variance_eigenval(self, q, b, e):
        """3-parameter unitary tensor product variance

        Get the single-qubit (3-parameter) operator
        variance for the array of qubits and bases.

        Args:
            q: qubits, from low to high
            b: 3-parameter, single-qubit, unitary bases (flat over wires)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            variance
        """
        if (3 * len(q)) != len(b):
            raise RuntimeError("unitary_variance_eigenval qubit and basis argument lengths do not match.")
        if (len(q) << 1) != len(e):
            raise RuntimeError("unitary_variance_eigenval qubit and eigenvalue argument lengths do not match.")
        result = Qrack.qrack_lib.UnitaryVarianceEigenVal(
            self.sid, len(q), self._ulonglong_byref(q), self._real1_byref(b), self._real1_byref(e)
        )
        self._throw_if_error()
        return result

    def matrix_variance_eigenval(self, q, b, e):
        """Single-qubit operator tensor product variance

        Get the single-qubit (3-parameter) operator
        variance for the array of qubits and bases.

        Args:
            q: qubits, from low to high
            b: single-qubit (2x2) operator unitary bases (flat over wires)

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            variance
        """
        if (len(q) << 2) != len(b):
            raise RuntimeError("matrix_variance_eigenval qubit and basis argument lengths do not match.")
        if (len(q) << 1) != len(e):
            raise RuntimeError("matrix_variance_eigenval qubit and eigenvalue argument lengths do not match.")
        result = Qrack.qrack_lib.MatrixVarianceEigenVal(
            self.sid, len(q), self._ulonglong_byref(q), self._complex_byref(b), self._real1_byref(e)
        )
        self._throw_if_error()
        return result

    def pauli_variance(self, q, b):
        """Pauli tensor product variance

        Get the Pauli tensor product variance,
        where each entry in "b" is a Pauli observable for
        corresponding "q", as the product for each in "q".

        Args:
            q: qubits, from low to high
            b: qubit Pauli bases

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            variance
        """
        if len(q) != len(b):
            raise RuntimeError("pauli_variance argument lengths do not match.")
        result = Qrack.qrack_lib.PauliVariance(
            self.sid, len(q), self._ulonglong_byref(q), self._ulonglong_byref(b)
        )
        self._throw_if_error()
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
            Variance
        """
        if len(b) != len(q):
            raise RuntimeError("Lengths of list parameters are mismatched.")
        result = Qrack.qrack_lib.JointEnsembleProbability(
            self.sid, len(b), self._ulonglong_byref(b), q
        )
        self._throw_if_error()
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
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot phase_parity()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot phase_parity()! (Turn off just this option, in the constructor.)")

        Qrack.qrack_lib.PhaseParity(
            self.sid, ctypes.c_double(la), len(q), self._ulonglong_byref(q)
        )
        self._throw_if_error()

    def phase_root_n(self, n, q):
        """Phase to root n

        Applies `-2 * math.pi / (2**N)` phase rotation to each qubit.

        Args:
            n: Phase root
            q: specified qubits

        Raises:
            RuntimeError: QrackSimulator raised an exception.
            RuntimeError: QrackSimulator with isTensorNetwork=True option cannot phase_root_n()! (Turn off just this option, in the constructor.)
        """
        if self.is_tensor_network:
            raise RuntimeError("QrackSimulator with isTensorNetwork=True option cannot phase_root_n()! (Turn off just this option, in the constructor.)")

        Qrack.qrack_lib.PhaseRootN(
            self.sid, n, len(q), self._ulonglong_byref(q)
        )
        self._throw_if_error()

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
        self._throw_if_error()
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
        self._throw_if_error()
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
        self._throw_if_error()
        return result

    def separate(self, qs):
        """Force manual multi-qubits seperation

        Force separation as per `try_separate_tolerance`.

        Args:
            qs: list of qubits to be decomposed

        Raises:
            Runtimeerror: QrackSimulator raised an exception.
        """
        result = Qrack.qrack_lib.Separate(
            self.sid, len(qs), self._ulonglong_byref(qs)
        )
        self._throw_if_error()


    def get_unitary_fidelity(self):
        """Get fidelity estimate

        When using "Schmidt decomposition rounding parameter" ("SDRP")
        approximate simulation, QrackSimulator() can make an excellent
        estimate of its overall fidelity at any time, tested against a
        nearest-neighbor variant of quantum volume circuits.

        Resetting the fidelity calculation to 1.0 happens automatically
        when calling `mall` are can be done manually with
        `reset_unitary_fidelity()`.

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Fidelity estimate
        """
        result = Qrack.qrack_lib.GetUnitaryFidelity(self.sid)
        self._throw_if_error()
        return result

    def reset_unitary_fidelity(self):
        """Reset fidelity estimate

        When using "Schmidt decomposition rounding parameter" ("SDRP")
        approximate simulation, QrackSimulator() can make an excellent
        estimate of its overall fidelity at any time, tested against a
        nearest-neighbor variant of quantum volume circuits.

        Resetting the fidelity calculation to 1.0 happens automatically
        when calling `m_all` or can be done manually with
        `reset_unitary_fidelity()`.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.ResetUnitaryFidelity(self.sid)
        self._throw_if_error()

    def set_sdrp(self, sdrp):
        """Set "Schmidt decomposition rounding parameter"

        When using "Schmidt decomposition rounding parameter" ("SDRP")
        approximate simulation, QrackSimulator() can make an excellent
        estimate of its overall fidelity at any time, tested against a
        nearest-neighbor variant of quantum volume circuits.

        Resetting the fidelity calculation to 1.0 happens automatically
        when calling `m_all` or can be done manually with
        `reset_unitary_fidelity()`.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.SetSdrp(self.sid, sdrp)
        self._throw_if_error()

    def set_ncrp(self, ncrp):
        """Set "Near-Clifford rounding parameter"

        When using "near-Clifford rounding parameter" ("NCRP")
        approximate simulation, QrackSimulator() can make an excellent
        estimate of its overall fidelity after measurement, tested against
        a nearest-neighbor variant of quantum volume circuits.

        Resetting the fidelity calculation to 1.0 happens automatically
        when calling `m_all` or can be done manually with
        `reset_unitary_fidelity()`.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.SetNcrp(self.sid, ncrp)
        self._throw_if_error()

    def set_reactive_separate(self, irs):
        """Set reactive separation option

        If reactive separation is available, then this method turns it off/on.
        Note that reactive separation is on by default.

        Args:
            irs: is aggresively separable

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.SetReactiveSeparate(self.sid, irs)
        self._throw_if_error()

    def set_t_injection(self, iti):
        """Set t-injection option

        If t-injection is available, then this method turns it off/on.
        Note that t-injection is on by default.

        Args:
            iti: use "reverse t-injection gadget"

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.SetTInjection(self.sid, iti)
        self._throw_if_error()

    def set_noise_parameter(self, np):
        """Set noise parameter option

        If noisy simulation is on, then this set the depolarization
        parameter per qubit per gate. (Default is 0.01.)

        Args:
            np: depolarizing noise parameter

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.SetNoiseParameter(self.sid, np)
        self._throw_if_error()

    def normalize(self):
        """Normalize the state

        This should never be necessary to use unless
        decompose() is "abused." ("Abusing" decompose()
        might lead to efficient entanglement-breaking
        channels, though.)

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.Normalize(self.sid)
        self._throw_if_error()

    def out_to_file(self, filename):
        """Output state to file (stabilizer only!)

        Outputs the hybrid stabilizer state to file.

        Args:
            filename: Name of file
        """
        Qrack.qrack_lib.qstabilizer_out_to_file(self.sid, filename.encode('utf-8'))
        self._throw_if_error()

    def in_from_file(filename, is_binary_decision_tree = False, is_paged = True, is_cpu_gpu_hybrid = False, is_opencl = True, is_host_pointer = False, is_noisy = False):
        """Input state from file (stabilizer only!)

        Reads in a hybrid stabilizer state from file.

        Args:
            filename: Name of file
        """
        qb_count = 1
        with open(filename) as f:
            qb_count = int(f.readline())
        out = QrackSimulator(
            qubitCount=qb_count,
            isTensorNetwork=False,
            isSchmidtDecomposeMulti=False,
            isSchmidtDecompose=False,
            isStabilizerHybrid=True,
            isBinaryDecisionTree=is_binary_decision_tree,
            isPaged=is_paged,
            isCpuGpuHybrid=is_cpu_gpu_hybrid,
            isOpenCL=is_opencl,
            isHostPointer=is_host_pointer,
            isNoisy=is_noisy
        )
        Qrack.qrack_lib.qstabilizer_in_from_file(out.sid, filename.encode('utf-8'))
        out._throw_if_error()

        return out

    def file_to_qiskit_circuit(filename, is_hardware_encoded=False):
        """Convert an output state file to a Qiskit circuit

        Reads in an (optimized) circuit from a file named
        according to the "filename" parameter and outputs
        a Qiskit circuit.

        Args:
            filename: Name of file

        Raises:
            RuntimeErorr: Before trying to file_to_qiskit_circuit() with
                QrackCircuit, you must install Qiskit, numpy, and math!
        """
        if not (_IS_QISKIT_AVAILABLE and _IS_NUMPY_AVAILABLE):
            raise RuntimeError(
                "Before trying to file_to_qiskit_circuit() with QrackCircuit, you must install Qiskit, numpy, and math!"
            )

        lines = []
        with open(filename, 'r') as file:
            lines = file.readlines()

        logical_qubits = int(lines[0])
        stabilizer_qubits = int(lines[1])

        stabilizer_count = int(lines[2])

        reg = QuantumRegister(stabilizer_qubits, name="q")
        circ_qubits = [Qubit(reg, i) for i in range(stabilizer_qubits)]
        clifford_circ = QuantumCircuit(reg)
        line_number = 3
        for i in range(stabilizer_count):
            shard_map_size = int(lines[line_number])
            line_number += 1

            shard_map = {}
            for j in range(shard_map_size):
                line = lines[line_number].split()
                line_number += 1
                shard_map[int(line[0])] = int(line[1])

            sub_reg = []
            for index, _ in sorted(shard_map.items(), key=lambda x: x[1]):
                sub_reg.append(circ_qubits[index])

            line_number += 1
            tableau = []
            row_count = shard_map_size << 1
            for line in lines[line_number:(line_number + row_count)]:
                bits = line.split()
                if len(bits) != (row_count + 1):
                    raise QrackException("Invalid Qrack hybrid stabilizer file!")
                row = []
                for b in range(row_count):
                    row.append(bool(int(bits[b])))
                row.append(bool((int(bits[-1]) >> 1) & 1))
                tableau.append(row)
            line_number += (shard_map_size << 1)
            tableau = np.array(tableau, bool)

            clifford = Clifford(tableau, validate=False, copy=False)
            circ = clifford.to_circuit()

            for instr in circ.data:
                qubits = instr.qubits
                n_qubits = []
                for qubit in qubits:
                    n_qubits.append(sub_reg[circ.find_bit(qubit)[0]])
                instr.qubits = tuple(n_qubits)
                clifford_circ.data.append(instr)
            del circ

        non_clifford_gates = []
        g = 0
        for line in lines[line_number:]:
            i = 0
            tokens = line.split()
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
                print("Warning: gate ", str(g), " might not be unitary!")
            if np.abs(op[0][1] - row[1]) > 1e-5:
                print("Warning: gate ", str(g), " might not be unitary!")

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
                print("Warning: gate ", str(g), " might not be unitary!")
            if np.abs(op[1][1] - row[1]) > 1e-5:
                print("Warning: gate ", str(g), " might not be unitary!")

            non_clifford_gates.append(op)
            g = g + 1

        basis_gates = ["rz", "h", "x", "y", "z", "sx", "sxdg", "sy", "sydg", "s", "sdg", "t", "tdg", "cx", "cy", "cz", "swap"]
        try:
            circ = transpile(clifford_circ, basis_gates=basis_gates, optimization_level=2)
        except:
            circ = clifford_circ

        for i in range(len(non_clifford_gates)):
            circ.unitary(non_clifford_gates[i], [i])

        if is_hardware_encoded:
            for i in range(logical_qubits, stabilizer_qubits, 2):
                circ.h(i + 1)
                circ.cz(i, i + 1)
                circ.h(i + 1)

        return circ

    def file_to_optimized_qiskit_circuit(filename):
        """Convert an output state file to a Qiskit circuit

        Reads in a circuit from a file named according to the "filename"
        parameter and outputs a 'hyper-optimized' Qiskit circuit that
        favors maximum reduction in gate count and depth at the potential
        expense of additional non-Clifford gates. (Ancilla qubits are
        left included in the output, though they probably have no gates.)

        Args:
            filename: Name of file

        Raises:
            RuntimeErorr: Before trying to file_to_qiskit_circuit() with
                QrackCircuit, you must install Qiskit, numpy, and math!
        """
        circ = QrackSimulator.file_to_qiskit_circuit(filename)
        
        width = 0
        with open(filename, "r", encoding="utf-8") as file:
            width = int(file.readline())

        sqrt_pi = np.sqrt(1j)
        sqrt_ni = np.sqrt(-1j)
        sqrt1_2 = 1 / math.sqrt(2)
        ident = np.eye(2, dtype=np.complex128)
        # passable_gates = ["unitary", "rz", "h", "x", "y", "z", "sx", "sxdg", "sy", "sydg", "s", "sdg", "t", "tdg"]

        passed_swaps = []
        for i in range(0, circ.width()):
            # We might trace out swap, but we want to maintain the iteration order of qubit channels.
            non_clifford = np.copy(ident)
            j = 0
            while j < len(circ.data):
                op = circ.data[j].operation
                qubits = circ.data[j].qubits
                if len(qubits) > 2:
                    raise RuntimeError("Something went wrong while optimizing circuit! (Found a gate with 3 or more qubits)")
                q1 = circ.find_bit(qubits[0])[0]
                if (len(qubits) < 2) and (q1 == i):
                    if op.name == "unitary":
                        non_clifford = np.matmul(op.params[0], non_clifford)
                    elif op.name == "rz":
                        lm = float(op.params[0])
                        non_clifford = np.matmul([[np.exp(-1j * lm / 2), 0], [0, np.exp(1j * lm / 2)]], non_clifford)
                    elif op.name == "h":
                        non_clifford = np.matmul(np.array([[sqrt1_2, sqrt1_2], [sqrt1_2, -sqrt1_2]], np.complex128), non_clifford)
                    elif op.name == "x":
                        non_clifford = np.matmul(np.array([[0, 1], [1, 0]], np.complex128), non_clifford)
                    elif op.name == "y":
                        non_clifford = np.matmul(np.array([[0, -1j], [1j, 0]], np.complex128), non_clifford)
                    elif op.name == "z":
                        non_clifford = np.matmul(np.array([[1, 0], [0, -1]], np.complex128), non_clifford)
                    elif op.name == "sx":
                        non_clifford = np.matmul(np.array([[(1+1j)/2, (1-1j)/2], [(1-1j)/2, (1+1j)/2]], np.complex128), non_clifford)
                    elif op.name == "sxdg":
                        non_clifford = np.matmul(np.array([[(1-1j)/2, (1+1j)/2], [(1+1j)/2, (1-1j)/2]], np.complex128), non_clifford)
                    elif op.name == "sy":
                        non_clifford = np.matmul(np.array([[(1+1j)/2, -(1+1j)/2], [(1+1j)/2, (1+1j)/2]], np.complex128), non_clifford)
                    elif op.name == "sydg":
                        non_clifford = np.matmul(np.array([[(1-1j)/2, (1-1j)/2], [(-1+1j)/2, (1-1j)/2]], np.complex128), non_clifford)
                    elif op.name == "s":
                        non_clifford = np.matmul(np.array([[1, 0], [0, 1j]], np.complex128), non_clifford)
                    elif op.name == "sdg":
                        non_clifford = np.matmul(np.array([[1, 0], [0, -1j]], np.complex128), non_clifford)
                    elif op.name == "t":
                        non_clifford = np.matmul(np.array([[1, 0], [0, sqrt_pi]], np.complex128), non_clifford)
                    elif op.name == "tdg":
                        non_clifford = np.matmul(np.array([[1, 0], [0, sqrt_ni]], np.complex128), non_clifford)
                    else:
                        raise RuntimeError("Something went wrong while optimizing circuit! (Dropped a single-qubit gate.)")

                    del circ.data[j]
                    continue

                if len(qubits) < 2:
                    j += 1
                    continue

                q2 = circ.find_bit(qubits[1])[0]

                if (i != q1) and (i != q2):
                    j += 1
                    continue

                if op.name == "swap":
                    i = (q2 if i == q1 else q1)

                    if circ.data[j] in passed_swaps:
                        passed_swaps.remove(circ.data[j])
                        del circ.data[j]
                        continue

                    passed_swaps.append(circ.data[j])

                    j += 1
                    continue 

                if (q1 == i) and ((op.name == "cx") or (op.name == "cy") or (op.name == "cz")):
                    if (np.isclose(np.abs(non_clifford[0][1]), 0) and np.isclose(np.abs(non_clifford[1][0]), 0)):
                        # If we're not buffering anything but phase, the blocking gate has no effect, and we're safe to continue.
                        del circ.data[j]
                        continue

                    if (np.isclose(np.abs(non_clifford[0][0]), 0) and np.isclose(np.abs(non_clifford[1][1]), 0)):
                        # If we're buffering full negation (plus phase), the control qubit can be dropped.
                        c = QuantumCircuit(1)
                        if op.name == "cx":
                            c.x(0)
                        elif op.name == "cy":
                            c.y(0)
                        else:
                            c.z(0)
                        instr = c.data[0]
                        instr.qubits = (qubits[1],)
                        circ.data[j] = copy.deepcopy(instr)

                        j += 1
                        continue

                if np.allclose(non_clifford, ident):
                    # No buffer content to write to circuit definition
                    non_clifford = np.copy(ident)
                    break

                # We're blocked, so we insert our buffer at this place in the circuit definition.
                c = QuantumCircuit(1)
                c.unitary(non_clifford, 0)
                instr = c.data[0]
                instr.qubits = (qubits[0],)
                circ.data.insert(j, copy.deepcopy(instr))

                non_clifford = np.copy(ident)
                break

            if (j == len(circ.data)) and not np.allclose(non_clifford, ident):
                # We're at the end of the wire, so add the buffer gate.
                circ.unitary(non_clifford, i)

        passed_swaps.clear()
        for i in range(width, circ.width()):
            # We might trace out swap, but we want to maintain the iteration order of qubit channels.
            non_clifford = np.copy(ident)
            j = len(circ.data) - 1
            while j >= 0:
                op = circ.data[j].operation
                qubits = circ.data[j].qubits
                if len(qubits) > 2:
                    raise RuntimeError("Something went wrong while optimizing circuit! (Found a gate with 3 or more qubits.)")
                q1 = circ.find_bit(qubits[0])[0]
                if (len(qubits) < 2) and (q1 == i):
                    if op.name == "unitary":
                        non_clifford = np.matmul(non_clifford, op.params[0])
                    elif op.name == "rz":
                        lm = float(op.params[0])
                        non_clifford = np.matmul(non_clifford, [[np.exp(-1j * lm / 2), 0], [0, np.exp(1j * lm / 2)]])
                    elif op.name == "h":
                        non_clifford = np.matmul(non_clifford, np.array([[sqrt1_2, sqrt1_2], [sqrt1_2, -sqrt1_2]], np.complex128))
                    elif op.name == "x":
                        non_clifford = np.matmul(non_clifford, np.array([[0, 1], [1, 0]], np.complex128))
                    elif op.name == "y":
                        non_clifford = np.matmul(non_clifford, np.array([[0, -1j], [1j, 0]], np.complex128))
                    elif op.name == "z":
                        non_clifford = np.matmul(non_clifford, np.array([[1, 0], [0, -1]], np.complex128))
                    elif op.name == "sx":
                        non_clifford = np.matmul(non_clifford, np.array([[(1+1j)/2, (1-1j)/2], [(1-1j)/2, (1+1j)/2]], np.complex128))
                    elif op.name == "sxdg":
                        non_clifford = np.matmul(non_clifford, np.array([[(1-1j)/2, (1+1j)/2], [(1+1j)/2, (1-1j)/2]], np.complex128))
                    elif op.name == "sy":
                        non_clifford = np.matmul(non_clifford, np.array([[(1+1j)/2, -(1+1j)/2], [(1+1j)/2, (1+1j)/2]], np.complex128))
                    elif op.name == "sydg":
                        non_clifford = np.matmul(non_clifford, np.array([[(1-1j)/2, (1-1j)/2], [(-1+1j)/2, (1-1j)/2]], np.complex128))
                    elif op.name == "s":
                        non_clifford = np.matmul(non_clifford, np.array([[1, 0], [0, 1j]], np.complex128))
                    elif op.name == "sdg":
                        non_clifford = np.matmul(non_clifford, np.array([[1, 0], [0, -1j]], np.complex128))
                    elif op.name == "t":
                        non_clifford = np.matmul(non_clifford, np.array([[1, 0], [0, sqrt_pi]], np.complex128))
                    elif op.name == "tdg":
                        non_clifford = np.matmul(non_clifford, np.array([[1, 0], [0, sqrt_ni]], np.complex128))
                    else:
                        raise RuntimeError("Something went wrong while optimizing circuit! (Dropped a single-qubit gate.)")

                    del circ.data[j]
                    j -= 1
                    continue

                if len(qubits) < 2:
                    j -= 1
                    continue

                q2 = circ.find_bit(qubits[1])[0]

                if (i != q1) and (i != q2):
                    j -= 1
                    continue

                if (op.name == "swap") and (q1 >= width) and (q2 >= width):
                    i = (q2 if i == q1 else q1)
                    if circ.data[j] in passed_swaps:
                        passed_swaps.remove(circ.data[j])
                        del circ.data[j]
                    else:
                        passed_swaps.append(circ.data[j])

                    j -= 1
                    continue

                if (q1 == i) and ((op.name == "cx") or (op.name == "cy") or (op.name == "cz")) and (np.isclose(np.abs(non_clifford[0][1]), 0) and np.isclose(np.abs(non_clifford[1][0]), 0)):
                    # If we're not buffering anything but phase, this commutes with control, and we're safe to continue.
                    j -= 1
                    continue

                if (q1 == i) and (op.name == "cx"):
                    orig_instr = circ.data[j]
                    del circ.data[j]

                    h = QuantumCircuit(1)
                    h.h(0)
                    instr = h.data[0]

                    # We're replacing CNOT with CNOT in the opposite direction plus four H gates
                    instr.qubits = (qubits[0],)
                    circ.data.insert(j, copy.deepcopy(instr))
                    instr.qubits = (qubits[1],)
                    circ.data.insert(j, copy.deepcopy(instr))
                    orig_instr.qubits = (qubits[1], qubits[0])
                    circ.data.insert(j, copy.deepcopy(orig_instr))
                    instr.qubits = (qubits[0],)
                    circ.data.insert(j, copy.deepcopy(instr))
                    instr.qubits = (qubits[1],)
                    circ.data.insert(j, copy.deepcopy(instr))

                    j += 4
                    continue

                if (q1 == i) or (op.name != "cx"):
                    if np.allclose(non_clifford, ident):
                        # No buffer content to write to circuit definition
                        break

                    # We're blocked, so we insert our buffer at this place in the circuit definition.
                    c = QuantumCircuit(1)
                    c.unitary(non_clifford, 0)
                    instr = c.data[0]
                    instr.qubits = (qubits[0],)
                    circ.data.insert(j + 1, copy.deepcopy(instr))

                    break

                # Re-injection branch (apply gadget to target)
                to_inject = np.matmul(non_clifford, np.array([[sqrt1_2, sqrt1_2], [sqrt1_2, -sqrt1_2]]))
                if np.allclose(to_inject, ident):
                    # No buffer content to write to circuit definition
                    del circ.data[j]
                    j -= 1
                    continue

                c = QuantumCircuit(1)
                c.unitary(to_inject, 0)
                instr = c.data[0]
                instr.qubits = (qubits[0],)
                circ.data[j] = copy.deepcopy(instr)
                j -= 1

        basis_gates=["u", "rz", "h", "x", "y", "z", "sx", "sxdg", "sy", "sydg", "s", "sdg", "t", "tdg", "cx", "cy", "cz", "swap"]
        circ = transpile(circ, basis_gates=basis_gates, optimization_level=2)

        #Eliminate unused ancillae
        qasm = circ.qasm()
        qasm = qasm.replace("qreg q[" + str(circ.width()) + "];", "qreg q[" + str(width) + "];")
        highest_index = max([int(x) for x in re.findall(r"\[(.*?)\]", qasm) if x.isdigit()])
        if highest_index != width:
            qasm = qasm.replace("qreg q[" + str(width) + "];", "qreg q[" + str(highest_index) + "];")

        orig_circ = circ
        try:
            circ = QuantumCircuit.from_qasm_str(qasm)
        except:
            circ = orig_circ

        return circ

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
            self.mcx([gate.control], gate.target)
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
        self._throw_if_error()

    def run_pyzx_gates(self, gates):
        """PYZX Gates

        Converts PYZX gates to `QRackSimulator` and immediately executes them.

        Args:
            gates: list of PYZX gates

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        for gate in gates:
            self._apply_pyzx_op(gate)

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
            self._sim.u(operation.qubits[0], 0, 0, float(operation.params[0]))
        elif name == 'u2':
            self._sim.u(
                operation.qubits[0],
                math.pi / 2,
                float(operation.params[0]),
                float(operation.params[1]),
            )
        elif (name == 'u3') or (name == 'u'):
            self._sim.u(
                operation.qubits[0],
                float(operation.params[0]),
                float(operation.params[1]),
                float(operation.params[2]),
            )
        elif (name == 'unitary') and (len(operation.qubits) == 1):
            self._sim.mtrx(operation.params[0].flatten(), operation.qubits[0])
        elif name == 'r':
            self._sim.u(
                operation.qubits[0],
                float(operation.params[0]),
                float(operation.params[1]) - math.pi / 2,
                (-1 * float(operation.params[1])) + math.pi / 2,
            )
        elif name == 'rx':
            self._sim.r(Pauli.PauliX, float(operation.params[0]), operation.qubits[0])
        elif name == 'ry':
            self._sim.r(Pauli.PauliY, float(operation.params[0]), operation.qubits[0])
        elif name == 'rz':
            self._sim.r(Pauli.PauliZ, float(operation.params[0]), operation.qubits[0])
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
                operation.qubits[0:1], operation.qubits[1], 0, 0, float(operation.params[0])
            )
        elif name == 'cu2':
            self._sim.mcu(
                operation.qubits[0:1],
                operation.qubits[1],
                math.pi / 2,
                float(operation.params[0]),
                float(operation.params[1]),
            )
        elif (name == 'cu3') or (name == 'cu'):
            self._sim.mcu(
                operation.qubits[0:1],
                operation.qubits[1],
                float(operation.params[0]),
                float(operation.params[1]),
                float(operation.params[2]),
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
                    math.cos(float(operation.params[0])) + 1j * math.sin(float(operation.params[0])),
                ],
                operation.qubits[1],
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
        elif name == 'iswap_dg':
            self._sim.adjiswap(operation.qubits[0], operation.qubits[1])
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

            data.append(bin(self._classical_memory)[2:].zfill(self.num_qubits()))

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
