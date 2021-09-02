# (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

from .qrack_system import Qrack


class QrackSimulator:

    # non-quantum

    def __init__(self, isClone = False, *args):
        if len(args) == 0:
            self.sid = Qrack.qrack_lib.init()
        elif isClone:
            self.sid = Qrack.qrack_lib.init_clone(sid)
        else:
            self.sid = Qrack.qrack_lib.init_count(args[0])

    def __del__(self):
        Qrack.qrack_lib.destroy(self.sid)

    def seed(self, s):
        Qrack.qrack_lib.seed(self.sid, s)

    def set_concurrency(self, p):
        Qrack.qrack_lib.set_concurrency(self.sid, p)

    # pseudo-quantum

    def prob(self, q):
        return Qrack.qrack_lib.Prob(self.sid, q)

    def permutation_expectation(self, n, c):
        return Qrack.qrack_lib.PermutationExpectation(self.sid, n, c)

    def joint_ensemble_probability(self, n, b, q):
        Qrack.qrack_lib.JointEnsembleProbability(self.sid, n, b, q)

    def reset_all(self):
        Qrack.qrack_lib.ResetAll(self.sid)

    # allocate and release

    def allocate_qubit(self, qid):
        Qrack.qrack_lib.allocateQubit(self.sid, qid)

    def release(self, q):
        return Qrack.qrack_lib.release(self.sid, q)

    def num_qubits(self):
        return Qrack.qrack_lib.num_qubits(self.sid)

    # single-qubit gates

    def x(self, q):
        Qrack.qrack_lib.X(self.sid, q)

    def y(self, q):
        Qrack.qrack_lib.Y(self.sid, q)

    def z(self, q):
        Qrack.qrack_lib.Z(self.sid, q)

    def h(self, q):
        Qrack.qrack_lib.H(self.sid, q)

    def s(self, q):
        Qrack.qrack_lib.S(self.sid, q)

    def t(self, q):
        Qrack.qrack_lib.T(self.sid, q)

    def adjs(self, q):
        Qrack.qrack_lib.AdjS(self.sid, q)

    def adjt(self, q):
        Qrack.qrack_lib.AdjT(self.sid, q)

    def u(self, q, th, ph, la):
        Qrack.qrack_lib.U(self.sid, q, th, ph, la)

    def mtrx(self, m, q):
        Qrack.qrack_lib.Mtrx(self.sid, m, q)

    # multi-controlled single-qubit gates

    def mcx(self, c, q):
        Qrack.qrack_lib.MCX(self.sid, len(c), c, q)

    def mcy(self, c, q):
        Qrack.qrack_lib.MCY(self.sid, len(c), c, q)

    def mcz(self, c, q):
        Qrack.qrack_lib.MCZ(self.sid, len(c), c, q)

    def mch(self, c, q):
        Qrack.qrack_lib.MCH(self.sid, len(c), c, q)

    def mcs(self, c, q):
        Qrack.qrack_lib.MCS(self.sid, len(c), c, q)

    def mct(self, c, q):
        Qrack.qrack_lib.MCT(self.sid, len(c), c, q)

    def mcadjs(self, c, q):
        Qrack.qrack_lib.MCAdjS(self.sid, len(c), c, q)

    def mcadjt(self, c, q):
        Qrack.qrack_lib.MCAdjT(self.sid, len(c), c, q)

    def mcu(self, n, c, q, th, ph, la):
        Qrack.qrack_lib.MCU(self.sid, n, c, q, th, ph, la)

    def mcmtrx(self, c, m, q):
        Qrack.qrack_lib.MCMtrx(self.sid, len(c), c, m, q)

    # multi-anti-controlled single-qubit gates

    def macx(self, c, q):
        Qrack.qrack_lib.MACX(self.sid, len(c), c, q)

    def macy(self, c, q):
        Qrack.qrack_lib.MACY(self.sid, len(c), c, q)

    def macz(self, c, q):
        Qrack.qrack_lib.MACZ(self.sid, len(c), c, q)

    def mach(self, c, q):
        Qrack.qrack_lib.MACH(self.sid, len(c), c, q)

    def macs(self, c, q):
        Qrack.qrack_lib.MACS(self.sid, len(c), c, q)

    def mact(self, c, q):
        Qrack.qrack_lib.MACT(self.sid, len(c), c, q)

    def macadjs(self, c, q):
        Qrack.qrack_lib.MACAdjS(self.sid, len(c), c, q)

    def macadjt(self, c, q):
        Qrack.qrack_lib.MACAdjT(self.sid, len(c), c, q)

    def macu(self, c, q, th, ph, la):
        Qrack.qrack_lib.MACU(self.sid, len(c), c, q, th, ph, la)

    def macmtrx(self, c, m, q):
        Qrack.qrack_lib.MACMtrx(self.sid, len(c), c, m, q)

    # rotations

    def r(self, b, ph, q):
        Qrack.qrack_lib.R(self.sid, b, ph, q)

    def mcr(self, b, ph, c, q):
        Qrack.qrack_lib.MCR(self.sid, b, ph, len(c), c, q)

    # exponential of Pauli operators

    def exp(self, b, ph, q):
        Qrack.qrack_lib.Exp(self.sid, len(b), ph, q)

    def mcexp(self, b, ph, cs, q):
        Qrack.qrack_lib.MCExp(self.sid, len(b), b, ph, len(cs), cs, q)

    # measurements

    def m(self, q):
        return Qrack.qrack_lib.M(self.sid, q)

    def measure_pauli(self, b, q):
        return Qrack.qrack_lib.Measure(self.sid, len(b), b, q)

    #swap

    def swap(self, qi1, qi2):
        Qrack.qrack_lib.SWAP(self.sid, qi1, qi2)

    def cswap(self, c, qi1, qi2):
        Qrack.qrack_lib.CSWAP(self.sid, len(c), c, qi1, qi2)

    # (quasi-)Boolean gates

    def qand(self, qi1, qi2, qo):
        Qrack.qrack_lib.AND(self.sid, qi1, qi2, qo)

    def qor(self, qi1, qi2, qo):
        Qrack.qrack_lib.OR(self.sid, qi1, qi2, qo)

    def qxor(self, qi1, qi2, qo):
        Qrack.qrack_lib.XOR(self.sid, qi1, qi2, qo)

    def qnand(self, qi1, qi2, qo):
        Qrack.qrack_lib.NAND(self.sid, qi1, qi2, qo)

    def qnor(self, qi1, qi2, qo):
        Qrack.qrack_lib.NOR(self.sid, qi1, qi2, qo)

    def qxnor(self, qi1, qi2, qo):
        Qrack.qrack_lib.XNOR(self.sid, qi1, qi2, qo)

    # half classical (quasi-)Boolean gates

    def cland(self, ci, qi, qo):
        Qrack.qrack_lib.CLAND(self.sid, ci, qi, qo)

    def clor(self, ci, qi, qo):
        Qrack.qrack_lib.CLOR(self.sid, ci, qi, qo)

    def clxor(self, ci, qi, qo):
        Qrack.qrack_lib.CLXOR(self.sid, ci, qi, qo)

    def clnand(self, ci, qi, qo):
        Qrack.qrack_lib.CLNAND(self.sid, ci, qi, qo)

    def clnor(self, ci, qi, qo):
        Qrack.qrack_lib.CLNOR(self.sid, ci, qi, qo)

    def clxnor(self, ci, qi, qo):
        Qrack.qrack_lib.CLXNOR(self.sid, ci, qi, qo)
