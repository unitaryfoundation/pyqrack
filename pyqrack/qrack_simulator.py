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

    def adj_s(self, q):
        Qrack.qrack_lib.AdjS(self.sid, q)

    def adj_t(self, q):
        Qrack.qrack_lib.AdjT(self.sid, q)

    def u(self, q, th, ph, la):
        Qrack.qrack_lib.U(self.sid, q, th, ph, la)

    def mtrx(self, m, q):
        Qrack.qrack_lib.Mtrx(self.sid, m, q)
