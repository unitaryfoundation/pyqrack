# (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import math, cmath
from ctypes import *
from .qrack_system import Qrack
from .pauli import Pauli


class QrackSimulator:

    # non-quantum

    def __init__(self, qubitCount=-1, stabilizerQubitCount=-1, cloneSid=-1, isSchmidtDecomposeMulti=True, isSchmidtDecompose=True, isStabilizerHybrid=True, isBinaryDecisionTree=False, isPaged=True, is1QbFusion=True, isCpuGpuHybrid=True, isOpenCL=True, isHostPointer=False, pyzxCircuit=None):
        self.sid = None

        if pyzxCircuit is not None:
            qubitCount = pyzxCircuit.qubits

        if qubitCount > -1 and cloneSid > -1:
            raise RuntimeError('Cannot clone a QrackSimulator and specify its qubit length at the same time, in QrackSimulator constructor!')
        if cloneSid > -1:
            self.sid = Qrack.qrack_lib.init_clone(cloneSid)
        else:
            if qubitCount < 0:
                qubitCount = 0
            if stabilizerQubitCount < 0:
                stabilizerQubitCount = 0

            if isBinaryDecisionTree and stabilizerQubitCount > 0:
                self.sid = Qrack.qrack_lib.init_qbdt_stabilizer(qubitCount, stabilizerQubitCount)
            if isSchmidtDecompose and isStabilizerHybrid and not isBinaryDecisionTree and isPaged and is1QbFusion and isCpuGpuHybrid and isOpenCL:
                if isSchmidtDecomposeMulti:
                    self.sid = Qrack.qrack_lib.init_count(qubitCount, isHostPointer)
                else:
                    self.sid = Qrack.qrack_lib.init_count_pager(qubitCount, isHostPointer)
            else:
                self.sid = Qrack.qrack_lib.init_count_type(qubitCount, isSchmidtDecomposeMulti, isSchmidtDecompose, isStabilizerHybrid, isBinaryDecisionTree, isPaged, is1QbFusion, isCpuGpuHybrid, isOpenCL, isHostPointer)

        if pyzxCircuit is not None:
            self.run_pyzx_gates(pyzxCircuit.gates)

    def __del__(self):
        if self.sid is not None:
            Qrack.qrack_lib.destroy(self.sid)
            self.sid = None

    def _uint_byref(self, a):
        return (c_uint * len(a))(*a)

    def _double_byref(self, a):
        return (c_double * len(a))(*a)

    def _complex_byref(self, a):
        t = [(c.real, c.imag) for c in a]
        return self._double_byref([float(item) for sublist in t for item in sublist])

    def _to_ubyte(self, nv, v):
        c = math.floor((nv - 1) / 8) + 1
        b = (c_ubyte * (c * (1 << nv)))()
        n = 0
        for u in v:
            for i in range(c):
                b[n] = u & 0xFF
                u >>= 8
                n += 1

        return byref(b)

    def get_error(self):
        return Qrack.qrack_lib.get_error(self.sid)

    def seed(self, s):
        Qrack.qrack_lib.seed(self.sid, s)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def set_concurrency(self, p):
        Qrack.qrack_lib.set_concurrency(self.sid, p)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # pseudo-quantum

    def prob(self, q):
        result = Qrack.qrack_lib.Prob(self.sid, q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")
        return result

    def permutation_expectation(self, c):
        result = Qrack.qrack_lib.PermutationExpectation(self.sid, len(c), self._uint_byref(c))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def joint_ensemble_probability(self, b, q):
        result = Qrack.qrack_lib.JointEnsembleProbability(self.sid, len(b), self._uint_byref(b), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def phase_parity(self, la, q):
        Qrack.qrack_lib.PhaseParity(self.sid, c_double(la), len(q), self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def reset_all(self):
        Qrack.qrack_lib.ResetAll(self.sid)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # allocate and release

    def allocate_qubit(self, qid):
        Qrack.qrack_lib.allocateQubit(self.sid, qid)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def release(self, q):
        result = Qrack.qrack_lib.release(self.sid, q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")
        return result

    def num_qubits(self):
        result = Qrack.qrack_lib.num_qubits(self.sid)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")
        return result

    # single-qubit gates

    def x(self, q):
        Qrack.qrack_lib.X(self.sid, q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def y(self, q):
        Qrack.qrack_lib.Y(self.sid, q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def z(self, q):
        Qrack.qrack_lib.Z(self.sid, q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def h(self, q):
        Qrack.qrack_lib.H(self.sid, q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def s(self, q):
        Qrack.qrack_lib.S(self.sid, q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def t(self, q):
        Qrack.qrack_lib.T(self.sid, q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def adjs(self, q):
        Qrack.qrack_lib.AdjS(self.sid, q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def adjt(self, q):
        Qrack.qrack_lib.AdjT(self.sid, q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def u(self, q, th, ph, la):
        Qrack.qrack_lib.U(self.sid, q, c_double(th), c_double(ph), c_double(la))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mtrx(self, m, q):
        Qrack.qrack_lib.Mtrx(self.sid, self._complex_byref(m), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # multi-controlled single-qubit gates

    def mcx(self, c, q):
        Qrack.qrack_lib.MCX(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcy(self, c, q):
        Qrack.qrack_lib.MCY(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcz(self, c, q):
        Qrack.qrack_lib.MCZ(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mch(self, c, q):
        Qrack.qrack_lib.MCH(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcs(self, c, q):
        Qrack.qrack_lib.MCS(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mct(self, c, q):
        Qrack.qrack_lib.MCT(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcadjs(self, c, q):
        Qrack.qrack_lib.MCAdjS(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcadjt(self, c, q):
        Qrack.qrack_lib.MCAdjT(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcu(self, c, q, th, ph, la):
        Qrack.qrack_lib.MCU(self.sid, len(c), self._uint_byref(c), q, c_double(th), c_double(ph), c_double(la))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcmtrx(self, c, m, q):
        Qrack.qrack_lib.MCMtrx(self.sid, len(c), self._uint_byref(c), self._complex_byref(m), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # multi-anti-controlled single-qubit gates

    def macx(self, c, q):
        Qrack.qrack_lib.MACX(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def macy(self, c, q):
        Qrack.qrack_lib.MACY(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def macz(self, c, q):
        Qrack.qrack_lib.MACZ(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mach(self, c, q):
        Qrack.qrack_lib.MACH(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def macs(self, c, q):
        Qrack.qrack_lib.MACS(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mact(self, c, q):
        Qrack.qrack_lib.MACT(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def macadjs(self, c, q):
        Qrack.qrack_lib.MACAdjS(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def macadjt(self, c, q):
        Qrack.qrack_lib.MACAdjT(self.sid, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def macu(self, c, q, th, ph, la):
        Qrack.qrack_lib.MACU(self.sid, len(c), self._uint_byref(c), q, c_double(th), c_double(ph), c_double(la))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def macmtrx(self, c, m, q):
        Qrack.qrack_lib.MACMtrx(self.sid, len(c), self._uint_byref(c), self._complex_byref(m), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def multiplex1_mtrx(self, c, q, m):
        Qrack.qrack_lib.Multiplex1Mtrx(self.sid, len(c), self._uint_byref(c), q, self._complex_byref(m))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # rotations

    def r(self, b, ph, q):
        Qrack.qrack_lib.R(self.sid, c_uint(b), c_double(ph), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcr(self, b, ph, c, q):
        Qrack.qrack_lib.MCR(self.sid, c_uint(b), ph, len(c), self._uint_byref(c), q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # exponential of Pauli operators

    def exp(self, b, ph, q):
        Qrack.qrack_lib.Exp(self.sid, len(b), self._uint_byref(b), c_double(ph), self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcexp(self, b, ph, cs, q):
        Qrack.qrack_lib.MCExp(self.sid, len(b), self._uint_byref(b), c_double(ph), len(cs), self._uint_byref(cs), self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # measurements

    def m(self, q):
        result = Qrack.qrack_lib.M(self.sid, q)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")
        return result

    def m_all(self):
        result = Qrack.qrack_lib.MAll(self.sid)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")
        return result

    def measure_pauli(self, b, q):
        result = Qrack.qrack_lib.Measure(self.sid, len(b), self._uint_byref(b), self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")
        return result

    def measure_shots(self, q, s):
        m = self._uint_byref([0] * s)
        Qrack.qrack_lib.MeasureShots(self.sid, len(q), self._uint_byref(q), s, m)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")
        return [m[i] for i in range(s)]

    #swap

    def swap(self, qi1, qi2):
        Qrack.qrack_lib.SWAP(self.sid, qi1, qi2)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def iswap(self, qi1, qi2):
        Qrack.qrack_lib.ISWAP(self.sid, qi1, qi2)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def fsim(self, th, ph, qi1, qi2):
        Qrack.qrack_lib.FSim(self.sid, c_double(th), c_double(ph), qi1, qi2)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def cswap(self, c, qi1, qi2):
        Qrack.qrack_lib.CSWAP(self.sid, len(c), self._uint_byref(c), qi1, qi2)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def acswap(self, c, qi1, qi2):
        Qrack.qrack_lib.ACSWAP(self.sid, len(c), self._uint_byref(c), qi1, qi2)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # Schmidt decomposition

    def compose(self, other, q):
        Qrack.qrack_lib.Compose(self.sid, other.sid, self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def decompose(self, q):
        other = QrackSimulator()
        Qrack.qrack_lib.destroy(other.sid)
        other.sid = Qrack.qrack_lib.Decompose(self.sid, len(q), self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")
        return other

    def dispose(self, q):
        Qrack.qrack_lib.Dispose(self.sid, len(q), self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # (quasi-)Boolean gates

    def qand(self, qi1, qi2, qo):
        Qrack.qrack_lib.AND(self.sid, qi1, qi2, qo)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def qor(self, qi1, qi2, qo):
        Qrack.qrack_lib.OR(self.sid, qi1, qi2, qo)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def qxor(self, qi1, qi2, qo):
        Qrack.qrack_lib.XOR(self.sid, qi1, qi2, qo)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def qnand(self, qi1, qi2, qo):
        Qrack.qrack_lib.NAND(self.sid, qi1, qi2, qo)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def qnor(self, qi1, qi2, qo):
        Qrack.qrack_lib.NOR(self.sid, qi1, qi2, qo)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def qxnor(self, qi1, qi2, qo):
        Qrack.qrack_lib.XNOR(self.sid, qi1, qi2, qo)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # half classical (quasi-)Boolean gates

    def cland(self, ci, qi, qo):
        Qrack.qrack_lib.CLAND(self.sid, ci, qi, qo)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def clor(self, ci, qi, qo):
        Qrack.qrack_lib.CLOR(self.sid, ci, qi, qo)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def clxor(self, ci, qi, qo):
        Qrack.qrack_lib.CLXOR(self.sid, ci, qi, qo)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def clnand(self, ci, qi, qo):
        Qrack.qrack_lib.CLNAND(self.sid, ci, qi, qo)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def clnor(self, ci, qi, qo):
        Qrack.qrack_lib.CLNOR(self.sid, ci, qi, qo)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def clxnor(self, ci, qi, qo):
        Qrack.qrack_lib.CLXNOR(self.sid, ci, qi, qo)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # Fourier transform

    def qft(self, qs):
        Qrack.qrack_lib.QFT(self.sid, len(qs), self._uint_byref(qs))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def iqft(self, qs):
        Qrack.qrack_lib.IQFT(self.sid, len(qs), self._uint_byref(qs))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # Arithmetic-Logic-Unit (ALU)

    def add(self, a, q):
        Qrack.qrack_lib.ADD(self.sid, a, len(q), self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def sub(self, a, q):
        Qrack.qrack_lib.SUB(self.sid, a, len(q), self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def adds(self, a, s, q):
        Qrack.qrack_lib.ADDS(self.sid, a, s, len(q), self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def subs(self, a, s, q):
        Qrack.qrack_lib.SUBS(self.sid, a, s, len(q), self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mul(self, a, q, o):
        Qrack.qrack_lib.MUL(self.sid, a, len(q), self._uint_byref(q), self._uint_byref(o))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def div(self, a, q, o):
        Qrack.qrack_lib.DIV(self.sid, a, len(q), self._uint_byref(q), self._uint_byref(o))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def muln(self, a, m, q, o):
        Qrack.qrack_lib.MULN(self.sid, a, m, len(q), self._uint_byref(q), self._uint_byref(o))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def divn(self, a, m, q, o):
        Qrack.qrack_lib.DIVN(self.sid, a, m, len(q), self._uint_byref(q), self._uint_byref(o))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def pown(self, a, m, q, o):
        Qrack.qrack_lib.POWN(self.sid, a, m, len(q), self._uint_byref(q), self._uint_byref(o))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcadd(self, a, c, q):
        Qrack.qrack_lib.MCADD(self.sid, a, len(c), c, len(q), self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcsub(self, a, c, q):
        Qrack.qrack_lib.MCSUB(self.sid, a, len(c), c, len(q), self._uint_byref(q))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcmul(self, a, c, q, o):
        Qrack.qrack_lib.MCMUL(self.sid, a, len(c), c, len(q), self._uint_byref(q), self._uint_byref(o))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcdiv(self, a, c, q, o):
        Qrack.qrack_lib.MCDIV(self.sid, a, len(c), c, len(q), self._uint_byref(q), self._uint_byref(o))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcmuln(self, a, c, m, q, o):
        Qrack.qrack_lib.MCMULN(self.sid, a, len(c), c, m, len(q), self._uint_byref(q), self._uint_byref(o))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcdivn(self, a, c, m, q, o):
        Qrack.qrack_lib.MCDIVN(self.sid, a, len(c), c, m, len(q), self._uint_byref(q), self._uint_byref(o))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def mcpown(self, a, c, m, q, o):
        Qrack.qrack_lib.MCPOWN(self.sid, a, len(c), c, m, len(q), self._uint_byref(q), self._uint_byref(o))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def lda(self, qi, qv, t):
        Qrack.qrack_lib.LDA(self.sid, len(qi), self._uint_byref(qi), len(qv), self._uint_byref(qv), self._to_ubyte(len(qv), t))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def adc(self, s, qi, qv, t):
        Qrack.qrack_lib.ADC(self.sid, s, len(qi), self._uint_byref(qi), len(qv), self._uint_byref(qv), self._to_ubyte(len(qv), t))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def sbc(self, s, qi, qv, t):
        Qrack.qrack_lib.SBC(self.sid, s, len(qi), self._uint_byref(qi), len(qv), self._uint_byref(qv), self._to_ubyte(len(qv), t))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def hash(self, q, t):
        Qrack.qrack_lib.Hash(self.sid, len(q), self._uint_byref(q), self._to_ubyte(len(q), t))
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    # miscellaneous

    def try_separate_1qb(self, qi1):
        result = Qrack.qrack_lib.TrySeparate1Qb(self.sid, qi1)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")
        return result

    def try_separate_2qb(self, qi1, qi2):
        result = Qrack.qrack_lib.TrySeparate2Qb(self.sid, qi1, qi2)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")
        return result

    def try_separate_tolerance(self, qs, t):
        result = Qrack.qrack_lib.TrySeparateTol(self.sid, len(qs), self._uint_byref(qs), tol)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")
        return result

    def set_reactive_separate(self, irs):
        Qrack.qrack_lib.SetReactiveSeparate(self.sid, irs)
        if self.get_error() != 0:
            raise Exception("QrackSimulator C++ library raised exception.")

    def run_pyzx_gates(self, gates):
        for gate in gates:
            if gate.name == 'XPhase':
                self.r(Pauli.PauliX, math.pi * gate.phase, gate.target)
            elif gate.name == 'ZPhase':
                self.r(Pauli.PauliZ, math.pi * gate.phase, gate.target)
            elif gate.name == 'Z':
                self.z(gate.target)
            elif gate.name == 'S':
                self.s(gate.target)
            elif gate.name == 'T':
                self.t(gate.target)
            elif gate.name == 'NOT':
                self.x(gate.target)
            elif gate.name == 'HAD':
                self.h(gate.target)
            elif gate.name == 'CNOT':
                self.mcx([gate.control], gate.target)
            elif gate.name == 'CZ':
                self.mcz([gate.control], gate.target)
            elif gate.name == 'CX':
                self.h(gate.control)
                if self.get_error() != 0:
                    raise Exception("QrackSimulator C++ library raised exception.")
                self.mcx([gate.control], gate.target)
                if self.get_error() != 0:
                    raise Exception("QrackSimulator C++ library raised exception.")
                self.h(gate.control)
            elif gate.name == 'SWAP':
                self.swap(gate.control, gate.target)
            elif gate.name == 'CRZ':
                self.mcr(Pauli.PauliZ, math.pi * gate.phase, [gate.control], gate.target)
            elif gate.name == 'CHAD':
                self.mch([gate.control], gate.target)
            elif gate.name == 'ParityPhase':
                self.phase_parity(math.pi * gate.phase, gate.targets)
            elif game.name == 'FSim':
                self.fsim(gate.theta, gate.phi, gate.control, gate.target)
            elif gate.name == 'CCZ':
                self.mcz([gate.ctrl1, gate.ctrl2], gate.target)
            elif gate.name == 'Tof':
                self.mcx([gate.ctrl1, gate.ctrl2], gate.target)
            if self.get_error() != 0:
                raise Exception("QrackSimulator C++ library raised exception.")
