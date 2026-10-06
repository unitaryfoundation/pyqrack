# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.
#
# Produced with input from (Anthropic) Claude
#
# QrackAceBackend over matrix-product-state patches.
#
# QrackAceBackend's elision scheme (replicated boundary qubits, shadow
# couplers, _correct() reconciliation, LHV tie-breakers, error-detection
# gadgets) never inspects HOW a patch stores its state: it only drives each
# patch through a small "just-in-time state machine" interface -- single-
# qubit gates, (anti-)controlled Paulis, swap, prob(), m(), force_m(),
# clone(). QrackMPSPatch implements exactly that interface on an MPS, so
# every line of ACE's own logic runs unchanged on top of it.
#
# What MPS buys: a patch's cost is set by its entanglement (bond dimension
# chi), not by 2^(patch width), so patches far wider than a state vector
# could hold become possible when in-patch entanglement allows -- and, with
# max_bond/cutoff, a second, tunable approximation layer (truncation INSIDE
# a patch) composes with ACE's first one (elision BETWEEN patches).
#
# Implementation notes:
#   * Plain numpy MPS, site tensors A[s] of shape (Dl, 2, Dr), with an
#     explicitly tracked orthogonality center. (A quimb-backed prototype
#     was ~180x slower inside ACE, almost entirely from re-deriving the
#     orthogonality center on every prob() call -- and ACE calls prob()
#     constantly. Tracking it explicitly makes prob() O(chi^2) plus the
#     cost of moving the center, usually zero or one QR step.)
#   * Logical qubits map to MPS sites through a permutation, so swap() is a
#     free, exact relabeling -- the same reason ACE's own native swap is
#     exact -- and non-adjacent 2-qubit gates are routed by adjacent SWAPs
#     that are LEFT in place (relabeled), not undone.
#   * Truncation: max_bond caps chi; cutoff is a per-split budget on
#     discarded probability weight (sum of discarded s^2 / total s^2).
#     set_sdrp() sets cutoff, as the MPS analog of Qrack's SDRP.

import math
import random

_IS_NUMPY_AVAILABLE = True
try:
    import numpy as np
except:
    _IS_NUMPY_AVAILABLE = False

from .pauli import Pauli
from .qrack_ace_backend import QrackAceBackend


_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_S = np.diag([1, 1j]).astype(complex)
_SDG = _S.conj().T
_T = np.diag([1, np.exp(1j * math.pi / 4)]).astype(complex)
_TDG = _T.conj().T
_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
_SXDG = _SX.conj().T
_P0 = np.diag([1, 0]).astype(complex)
_P1 = np.diag([0, 1]).astype(complex)
_SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)
_PAULI_MTRX = {Pauli.PauliX: _X, Pauli.PauliY: _Y, Pauli.PauliZ: _Z}


def _controlled(u, anti=False):
    # 4x4 in the (control, target) basis, control = first (most significant).
    g = np.eye(4, dtype=complex)
    if anti:
        g[0:2, 0:2] = u
    else:
        g[2:4, 2:4] = u
    return g


_CX = _controlled(_X)


class QrackMPSPatch:
    """One ACE patch held as an MPS, exposing the subset of the
    QrackSimulator interface that QrackAceBackend calls on its patches.

    Args:
        qubit_count: patch width.
        max_bond: bond-dimension cap (None = no cap).
        cutoff: per-split discarded-weight budget (0 = exact up to
            numerical zeros).
    """

    _ZERO_TOL = 1e-14

    def __init__(self, qubit_count, max_bond=None, cutoff=0.0, _state=None):
        if not _IS_NUMPY_AVAILABLE:
            raise RuntimeError(
                "Before instantiating QrackAceMPSPatch, you must install numpy!"
            )
        self.n = qubit_count
        self.max_bond = max_bond
        self.cutoff = float(cutoff)
        self.truncation_error = 0.0  # accumulated discarded weight
        if _state is not None:
            self.A, self.site_of, self.q_at, self.center = _state
        else:
            t = np.zeros((1, 2, 1), dtype=complex)
            t[0, 0, 0] = 1.0
            self.A = [t.copy() for _ in range(max(1, qubit_count))]
            self.site_of = list(range(qubit_count))
            self.q_at = list(range(qubit_count))
            self.center = 0

    # --- bookkeeping -------------------------------------------------------

    def clone(self):
        c = QrackMPSPatch(
            self.n, self.max_bond, self.cutoff,
            _state=([a.copy() for a in self.A], list(self.site_of), list(self.q_at), self.center),
        )
        c.truncation_error = self.truncation_error
        return c

    def set_sdrp(self, sdrp):
        self.cutoff = max(0.0, float(sdrp))

    def set_device(self, device_id):
        # MPS patches run on the host; accepted for interface compatibility.
        pass

    def num_qubits(self):
        return self.n

    def max_bond_dim(self):
        return max(a.shape[2] for a in self.A)

    # --- canonical-form machinery -------------------------------------------

    def _move_center(self, s):
        A = self.A
        while self.center < s:
            c = self.center
            dl, d, dr = A[c].shape
            q, r = np.linalg.qr(A[c].reshape(dl * d, dr))
            A[c] = q.reshape(dl, d, q.shape[1])
            A[c + 1] = np.tensordot(r, A[c + 1], axes=(1, 0))
            self.center += 1
        while self.center > s:
            c = self.center
            dl, d, dr = A[c].shape
            q, r = np.linalg.qr(A[c].reshape(dl, d * dr).T)
            A[c] = q.T.reshape(q.shape[1], d, dr)
            A[c - 1] = np.tensordot(A[c - 1], r.T, axes=(2, 0))
            self.center -= 1

    def _apply_site_1q(self, mat, s, unitary=True):
        if not unitary:
            # Non-unitary (projector): only safe at the orthogonality center.
            self._move_center(s)
        self.A[s] = np.einsum("ab,lbr->lar", mat, self.A[s])

    def _apply_adjacent_2q(self, g4, s):
        # g4 acts on (site s, site s+1), site s = most significant.
        self._move_center(s)
        a, b = self.A[s], self.A[s + 1]
        dl, dr = a.shape[0], b.shape[2]
        theta = np.tensordot(a, b, axes=(2, 0))                      # (dl,2,2,dr)
        theta = np.einsum("ijkl,aklb->aijb", g4.reshape(2, 2, 2, 2), theta)
        u, sv, vh = np.linalg.svd(theta.reshape(dl * 2, 2 * dr), full_matrices=False)
        w = sv ** 2
        total = float(np.sum(w))
        keep = len(sv)
        # Drop numerical zeros always; drop more only within the budget.
        tail = np.cumsum(w[::-1])[::-1]  # tail[k] = sum_{j>=k} w[j]
        budget = max(self.cutoff * total, self._ZERO_TOL * total)
        while keep > 1 and tail[keep - 1] <= budget:
            keep -= 1
        if self.max_bond is not None and keep > self.max_bond:
            keep = self.max_bond
        discarded = float(np.sum(w[keep:]))
        if discarded > 0 and total > 0:
            self.truncation_error += discarded / total
        u, sv, vh = u[:, :keep], sv[:keep], vh[:keep, :]
        if discarded > 0:
            # Restore the pre-truncation norm (the center carries it all).
            sv = sv * math.sqrt(total / float(np.sum(sv ** 2)))
        self.A[s] = u.reshape(dl, 2, keep)
        self.A[s + 1] = (sv[:, None] * vh).reshape(keep, 2, dr)
        self.center = s + 1

    def _swap_sites(self, s):
        # Physically exchange sites s, s+1 and relabel.
        self._apply_adjacent_2q(_SWAP, s)
        qa, qb = self.q_at[s], self.q_at[s + 1]
        self.q_at[s], self.q_at[s + 1] = qb, qa
        self.site_of[qa], self.site_of[qb] = s + 1, s

    # --- gate application --------------------------------------------------

    def _g1(self, mat, q, unitary=True):
        self._apply_site_1q(mat, self.site_of[q], unitary)

    def _g2(self, g4, q1, q2):
        # g4 acts on (q1, q2), q1 = most significant.
        if q1 == q2:
            raise ValueError("Two-qubit gate on identical qubits.")
        # Route q2 next to q1 with adjacent swaps, left in place.
        while abs(self.site_of[q1] - self.site_of[q2]) > 1:
            s1, s2 = self.site_of[q1], self.site_of[q2]
            if s2 > s1:
                self._swap_sites(s2 - 1)
            else:
                self._swap_sites(s2)
        s1, s2 = self.site_of[q1], self.site_of[q2]
        if s1 < s2:
            self._apply_adjacent_2q(g4, s1)
        else:
            self._apply_adjacent_2q(_SWAP @ g4 @ _SWAP, s2)

    def mtrx(self, m, q):
        self._g1(np.array([[m[0], m[1]], [m[2], m[3]]], dtype=complex), q)

    def u(self, q, th, ph, la):
        c, s = math.cos(th / 2), math.sin(th / 2)
        self._g1(np.array(
            [[c, -np.exp(1j * la) * s],
             [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]], dtype=complex), q)

    def r(self, b, ph, q):
        p = _PAULI_MTRX.get(b)
        if p is None:
            return  # PauliI: global phase only
        self._g1(math.cos(ph / 2) * _I2 - 1j * math.sin(ph / 2) * p, q)

    def h(self, q): self._g1(_H, q)
    def x(self, q): self._g1(_X, q)
    def y(self, q): self._g1(_Y, q)
    def z(self, q): self._g1(_Z, q)
    def s(self, q): self._g1(_S, q)
    def adjs(self, q): self._g1(_SDG, q)
    def t(self, q): self._g1(_T, q)
    def adjt(self, q): self._g1(_TDG, q)
    def sx(self, q): self._g1(_SX, q)
    def adjsx(self, q): self._g1(_SXDG, q)

    def _toffoli(self, c1, c2, t):
        self.h(t)
        self._g2(_CX, c2, t); self.adjt(t)
        self._g2(_CX, c1, t); self.t(t)
        self._g2(_CX, c2, t); self.adjt(t)
        self._g2(_CX, c1, t); self.t(t)
        self.h(t)
        self.t(c2)
        self._g2(_CX, c1, c2); self.t(c1); self.adjt(c2)
        self._g2(_CX, c1, c2)

    def _mc_pauli(self, c, q, u, anti):
        c = list(c)
        if len(c) == 0:
            self._g1(u, q)
        elif len(c) == 1:
            self._g2(_controlled(u, anti), c[0], q)
        elif len(c) == 2:
            # Exact 1-/2-qubit Toffoli decomposition, conjugated into
            # X, Y, or Z on the target.
            if anti:
                for x in c:
                    self.x(x)
            pre, post = None, None
            if u is _Y:
                pre, post = _SDG, _S
            elif u is _Z:
                pre, post = _H, _H
            if pre is not None:
                self._g1(pre, q)
            self._toffoli(c[0], c[1], q)
            if post is not None:
                self._g1(post, q)
            if anti:
                for x in c:
                    self.x(x)
        else:
            raise NotImplementedError("QrackMPSPatch supports at most 2 controls per gate.")

    def mcx(self, c, q): self._mc_pauli(c, q, _X, False)
    def mcy(self, c, q): self._mc_pauli(c, q, _Y, False)
    def mcz(self, c, q): self._mc_pauli(c, q, _Z, False)
    def macx(self, c, q): self._mc_pauli(c, q, _X, True)
    def macy(self, c, q): self._mc_pauli(c, q, _Y, True)
    def macz(self, c, q): self._mc_pauli(c, q, _Z, True)

    def swap(self, q1, q2):
        # Exact and free: relabel which site each logical qubit lives on.
        if q1 == q2:
            return
        s1, s2 = self.site_of[q1], self.site_of[q2]
        self.site_of[q1], self.site_of[q2] = s2, s1
        self.q_at[s1], self.q_at[s2] = q2, q1

    def cswap(self, c, q1, q2):
        c = list(c)
        if q1 == q2:
            return
        if len(c) != 1:
            raise NotImplementedError("QrackMPSPatch.cswap() supports exactly 1 control.")
        self._g2(_CX, q2, q1)
        self._toffoli(c[0], q1, q2)
        self._g2(_CX, q2, q1)

    # --- readout -----------------------------------------------------------

    def prob(self, q):
        s = self.site_of[q]
        self._move_center(s)
        a = self.A[s]
        total = float(np.sum(np.abs(a) ** 2))
        if total <= 0:
            return 0.0
        one = float(np.sum(np.abs(a[:, 1, :]) ** 2))
        return min(1.0, max(0.0, one / total))

    def force_m(self, q, r):
        s = self.site_of[q]
        self._move_center(s)
        a = self.A[s].copy()
        a[:, 0 if r else 1, :] = 0
        nrm = math.sqrt(float(np.sum(np.abs(a) ** 2)))
        if nrm > 0:
            self.A[s] = a / nrm
        return bool(r)

    def m(self, q):
        return self.force_m(q, random.random() < self.prob(q))

    def m_all(self):
        result = 0
        for q in range(self.n):
            if self.m(q):
                result |= 1 << q
        return result

    # --- dense readout (testing only; 2^n) ----------------------------------

    def out_ket(self):
        v = self.A[0]
        for a in self.A[1:]:
            v = np.tensordot(v, a, axes=(v.ndim - 1, 0))
        v = v.reshape([2] * self.n)  # axis k = site k
        # Reorder axes into Qrack's little-endian logical order: the
        # C-order flatten treats axis 0 as most significant, so axis 0
        # must be logical qubit n-1.
        v = np.transpose(v, [self.site_of[q] for q in range(self.n - 1, -1, -1)])
        return v.reshape(-1)

    def out_probs(self):
        return np.abs(self.out_ket()) ** 2


class QrackAceMPSBackend(QrackAceBackend):
    """QrackAceBackend whose patches are MPS (QrackMPSPatch) instead of
    QrackSimulator. All elision, shadow-coupling, reconciliation and
    error-detection logic is inherited unchanged.

    Extra args:
        max_bond: per-patch MPS bond-dimension cap (None = no cap).
        mps_cutoff: per-split discarded-weight budget (0 = exact).
    QrackSimulator-only construction options (is_gpu, is_stabilizer_hybrid,
    ...) are accepted and ignored.
    """

    def __init__(self, *args, max_bond=None, mps_cutoff=0.0, to_clone=None, **kwargs):
        if not _IS_NUMPY_AVAILABLE:
            raise RuntimeError(
                "Before instantiating QrackAceMPSBackend, you must install numpy!"
            )
        if to_clone is not None:
            max_bond = to_clone.max_bond
            mps_cutoff = to_clone.mps_cutoff
        self.max_bond = max_bond
        self.mps_cutoff = mps_cutoff
        super().__init__(*args, to_clone=to_clone, **kwargs)

    def _new_patch_sim(self, qubit_count, sim_kwargs):
        return QrackMPSPatch(qubit_count, max_bond=self.max_bond, cutoff=self.mps_cutoff)

    def max_bond_dims(self):
        return [s.max_bond_dim() for s in self.sim]

    def truncation_errors(self):
        return [s.truncation_error for s in self.sim]
