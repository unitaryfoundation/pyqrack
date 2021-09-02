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

    def mcx(self, n, c, q):
        Qrack.qrack_lib.MCX(self.sid, n, c, q)

    def mcy(self, n, c, q):
        Qrack.qrack_lib.MCY(self.sid, n, c, q)

    def mcz(self, n, c, q):
        Qrack.qrack_lib.MCZ(self.sid, n, c, q)

    def mch(self, n, c, q):
        Qrack.qrack_lib.MCH(self.sid, n, c, q)

    def mcs(self, n, c, q):
        Qrack.qrack_lib.MCS(self.sid, n, c, q)

    def mct(self, n, c, q):
        Qrack.qrack_lib.MCT(self.sid, n, c, q)

    def mcadjs(self, n, c, q):
        Qrack.qrack_lib.MCAdjS(self.sid, n, c, q)

    def mcadjt(self, n, c, q):
        Qrack.qrack_lib.MCAdjT(self.sid, n, c, q)

    def mcu(self, n, c, q, th, ph, la):
        Qrack.qrack_lib.MCU(self.sid, n, c, q, th, ph, la)

    def mcmtrx(self, n, c, m, q):
        Qrack.qrack_lib.MCMtrx(self.sid, n, c, m, q)

    # multi-anti-controlled single-qubit gates

    def macx(self, n, c, q):
        Qrack.qrack_lib.MACX(self.sid, n, c, q)

    def macy(self, n, c, q):
        Qrack.qrack_lib.MACY(self.sid, n, c, q)

    def macz(self, n, c, q):
        Qrack.qrack_lib.MACZ(self.sid, n, c, q)

    def mach(self, n, c, q):
        Qrack.qrack_lib.MACH(self.sid, n, c, q)

    def macs(self, n, c, q):
        Qrack.qrack_lib.MACS(self.sid, n, c, q)

    def mact(self, n, c, q):
        Qrack.qrack_lib.MACT(self.sid, n, c, q)

    def macadjs(self, n, c, q):
        Qrack.qrack_lib.MACAdjS(self.sid, n, c, q)

    def macadjt(self, n, c, q):
        Qrack.qrack_lib.MACAdjT(self.sid, n, c, q)

    def macu(self, n, c, q, th, ph, la):
        Qrack.qrack_lib.MACU(self.sid, n, c, q, th, ph, la)

    def macmtrx(self, n, c, m, q):
        Qrack.qrack_lib.MACMtrx(self.sid, n, c, m, q)
