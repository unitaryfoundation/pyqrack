# (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

from .qrack_system import Qrack

class QrackSimulator:

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

    def prob(self, q):
        return Qrack.qrack_lib.Prob(self.sid, q)

    def permutation_expectation(self, n, c):
        return Qrack.qrack_lib.PermutationExpectation(self.sid, n, c)

    def joint_ensemble_probability(self, n, b, q):
        Qrack.qrack_lib.JointEnsembleProbability(self.sid, n, b, q)

    def reset_all(self):
        Qrack.qrack_lib.ResetAll(self.sid)
