# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.
import math
import random
import sys
import time

from .qrack_simulator import QrackSimulator
from .pauli import Pauli


class QrackAceBackend:
    """A back end for elided quantum error correction

    This back end uses elided repetition code on a nearest-neighbor topology to emulate
    a utility-scale superconducting chip quantum computer in very little memory.

    The backend was originally designed assuming a 2D qubit grid like 2019 Sycamore.
    However, it quickly became apparent that users can basically design their own
    connectivity topologies, without breaking the concept. (Not all will work equally well.)

    Attributes:
        sim(QrackSimulator): Corresponding simulator.
    """

    def __init__(
        self,
        qubit_count=-1,
    ):
        self.sim = QrackSimulator(3 * qubit_count)


    def _ct_pair_prob(self, q1, q2):
        p1 = self.sim.prob(q1)
        p2 = self.sim.prob(q2)

        if p1 < p2:
            return p2, q1

        return p1, q2


    def _cz_shadow(self, q1, q2):
        prob_max, t = self._ct_pair_prob(q1, q2)
        if prob_max > 0.5:
            self.sim.z(t)


    def _anti_cz_shadow(self, q1, q2):
        self.sim.x(q1)
        self._cz_shadow(q1, q2)
        self.sim.x(q1)


    def _cx_shadow(self, c, t):
        self.sim.h(t)
        self._cz_shadow(c, t)
        self.sim.h(t)


    def _anti_cx_shadow(self, c, t):
        self.sim.x(t)
        self._cx_shadow(c, t)
        self.sim.x(t)


    def _cy_shadow(self, c, t):
        self.sim.adjs(t)
        self._cx_shadow(c, t)
        self.sim.s(t)


    def _anti_cy_shadow(self, c, t):
        self.sim.x(t)
        self._cy_shadow(c, t)
        self.sim.x(t)


    def _unpack(self, lq, reverse = False):
        return [3 * lq + 2, 3 * lq + 1, 3 * lq] if reverse else [3 * lq, 3 * lq + 1, 3 * lq + 2]


    def _encode(self, hq, reverse = False):
        if reverse:
            self._cx_shadow(hq[0], hq[1])
            self.sim.mcx([hq[1]], hq[2])
        else:
            self.sim.mcx([hq[0]], hq[1])
            self._cx_shadow(hq[1], hq[2])


    def _decode(self, hq, reverse = False):
        if reverse:
            self.sim.mcx([hq[1]], hq[2])
            self._cx_shadow(hq[0], hq[1])
        else:
            self._cx_shadow(hq[1], hq[2])
            self.sim.mcx([hq[0]], hq[1])


    def u(self, th, ph, lm, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.u(hq[0], th, ph, lm)
        self._encode(hq)


    def s(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.s(hq[0])
        self._encode(hq)


    def adjs(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.adjs(hq[0])
        self._encode(hq)


    def x(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.x(hq[0])
        self._encode(hq)


    def y(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.y(hq[0])
        self._encode(hq)


    def z(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.z(hq[0])
        self._encode(hq)


    def h(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.h(hq[0])
        self._encode(hq)


    def t(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.t(hq[0])
        self._encode(hq)


    def adjt(self, lq):
        hq = self._unpack(lq)
        self._decode(hq)
        self.sim.adjt(hq[0])
        self._encode(hq)


    def _cpauli(self, lq1, lq2, anti, pauli):
        gate = None
        if pauli == Pauli.PauliX:
            gate = self.sim.macx if anti else self.sim.mcx
        elif pauli == Pauli.PauliY:
            gate = self.sim.macy if anti else self.sim.mcy
        elif pauli == Pauli.PauliZ:
            gate = self.sim.macz if anti else self.sim.mcz
        else:
            return

        if (lq2 == (lq1 + 1)) or (lq1 == (lq2 + 1)):
            hq1 = self._unpack(lq1, True)
            hq2 = self._unpack(lq2, False)
            self._decode(hq1, True)
            self._decode(hq2, False)
            gate([hq1[0]], hq2[0])
            self._encode(hq2, False)
            self._encode(hq1, True)
        else:
            hq1 = self._unpack(lq1)
            hq2 = self._unpack(lq2)
            gate([hq1[0]], hq2[0])
            gate([hq1[1]], hq2[1])
            gate([hq1[2]], hq2[2])


    def cx(self, lq1, lq2):
        self._cpauli(lq1, lq2, False, Pauli.PauliX)


    def cy(self, lq1, lq2):
        self._cpauli(lq1, lq2, False, Pauli.PauliY)


    def cz(self, lq1, lq2):
        self._cpauli(lq1, lq2, False, Pauli.PauliZ)


    def acx(self, lq1, lq2):
        self._cpauli(lq1, lq2, True, Pauli.PauliX)


    def acy(self, lq1, lq2):
        self._cpauli(lq1, lq2, True, Pauli.PauliY)


    def acz(self, lq1, lq2):
        self._cpauli(lq1, lq2, True, Pauli.PauliZ)


    def swap(self, lq1, lq2):
        self.cx(lq1, lq2)
        self.cx(lq2, lq1)
        self.cx(lq1, lq2)


    def iswap(self, lq1, lq2):
        self.swap(lq1, lq2)
        self.cz(lq1, lq2)
        self.s(lq1)
        self.s(lq2)


    def adjiswap(self, lq1, lq2):
        self.adjs(lq2)
        self.adjs(lq1)
        self.cz(lq1, lq2)
        self.swap(lq1, lq2)


    def m(self, lq):
        hq = self._unpack(lq)
        syndrome = 0
        bits = []
        for q in hq:
            bits.append(self.sim.m(q))
            if bits[-1]:
                syndrome += 1
        result = True if (syndrome > 1) else False
        for i in range(len(hq)): 
            if bits[i] != result:
                self.sim.x(hq[i])

        return result


    def m_all(self):
        result = 0
        for lq in range(self.sim.num_qubits() // 3):
            result <<= 1
            if self.m(lq):
                result |= 1

        return result

