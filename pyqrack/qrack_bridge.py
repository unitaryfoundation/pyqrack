# Bridged patching on QrackSimulator patches, recompressed through out_ket()/in_ket().
#
# By (Anthropic) Claude and Dan Strano
#
# Two-patch split: patch A = qubits [0, n // 2), patch B = the rest (first patch rounded down).
#
# Representation. Each patch is ONE QrackSimulator holding its data qubits (low indices) plus exactly
# b bond qubits (high indices), allocated and released as the bridge grows and shrinks. Bond j of
# patch A is paired with bond j of patch B. The represented state is both patch states contracted on
# the live bond pairs with the unnormalized virtual Bell projection sum_k <k|<k|:
#
#     |Psi>  ~  sum_k <k|psi_A> (x) <k|psi_B>          (k over the 2^b bond basis states)
#
# i.e. with each patch's ket read as a (2^n_patch x 2^b) matrix M_A, M_B:  Psi = M_A M_B^T.
# Rank chi = 2^b. Every in-patch gate is applied natively, once, to all chi branches at once.
#
# Cross-cut gates by gate teleportation through a fresh bond pair (exact, gates only):
#     controlled-U from c (one patch) to t (the other):
#       control patch:  CNOT  c -> bond_b                 (copy the control into the bond)
#       target patch:   H bond_b ;  controlled-U  bond_b -> t
#     Contracting on (bond_b, bond_b) enforces equality, which is exactly controlled-U(c -> t).
#     SWAP is 3 CNOTs; iSWAP is (S x S), CZ, SWAP; iSWAP^dagger is SWAP, CZ, (S^dag x S^dag).
#
# Recompression after each bridged two-qubit gate: out_ket() both patches, thin QR of each
# (2^n_patch x chi) matrix, SVD of the chi x chi core, keep the largest singular values (exact when
# only zeros are dropped; above max_bonds, greedy or Born-rule group selection), and in_ket() back.
# The singular values are absorbed into patch A, so patch B's bond columns stay orthonormal (B is
# isometric), which makes patch A's own marginal over its data qubits the exact marginal.
# Nothing larger than the kets themselves and a chi x chi core is ever formed: no density matrix,
# no (2^n x 2^n) object of any kind.
#
# Readout (exact for the stored state): sample patch A's data natively with its own measure_shots
# (exact because B is isometric); for each distinct outcome a, patch B's conditional state is
# M_B @ M_A[a, :], sampled in numpy from the exported B ket.
#
# fidelity_estimate multiplies the kept Schmidt weight (greedy) or sum_g p_g^2 (born) per cut.

import ctypes
import math

import numpy as np
from .qrack_simulator import QrackSimulator
from .qrack_system import Qrack

# Ket I/O straight into/out of numpy buffers (PyQrack's out_ket()/in_ket() round-trip through Python
# lists of complex, which dominated runtime). Same library calls, OutKet/InKet, without the copies.
_REAL = np.float32 if Qrack.fppow < 6 else np.float64
_CREAL = ctypes.c_float if Qrack.fppow < 6 else ctypes.c_double


def _out_ket(sim):
    buf = np.empty(2 << sim.num_qubits(), dtype=_REAL)
    Qrack.qrack_lib.OutKet(sim.sid, buf.ctypes.data_as(ctypes.POINTER(_CREAL)))
    sim._throw_if_error()
    return buf.view(np.complex64 if _REAL is np.float32 else np.complex128).astype(complex)


def _in_ket(sim, v):
    buf = np.ascontiguousarray(np.asarray(v, dtype=complex).view(np.float64).astype(_REAL))
    Qrack.qrack_lib.InKet(sim.sid, buf.ctypes.data_as(ctypes.POINTER(_CREAL)))
    sim._throw_if_error()

_X = [0, 1, 1, 0]
_Y = [0, -1j, 1j, 0]
_Z = [1, 0, 0, -1]
_TOL = 1e-6          # relative singular-value floor treated as exactly zero (Qrack is single precision)


class QrackBridge:
    def __init__(self, qubit_count, max_bonds=6, seed=None, truncation="greedy", _clone=None):
        if _clone is not None:
            src = _clone
            self.n, self.qa, self.qb, self.side = src.n, src.qa, src.qb, src.side
            self.max_bonds, self.truncation = src.max_bonds, src.truncation
            self.sim = [s.clone() for s in src.sim]
            self.b = src.b
            self.fidelity_estimate, self.truncations, self.peak_bonds = src.fidelity_estimate, src.truncations, src.peak_bonds
            self._defer = False
            self.rng = np.random.default_rng(int(src.rng.integers(1 << 62)))
            return
        if truncation not in ("greedy", "born"):
            raise ValueError("truncation must be 'greedy' or 'born'")
        self.n = qubit_count
        self.qa = list(range(qubit_count // 2))
        self.qb = list(range(qubit_count // 2, qubit_count))
        self.side = {q: (0, i) for i, q in enumerate(self.qa)}
        self.side.update({q: (1, i) for i, q in enumerate(self.qb)})
        self.max_bonds = max_bonds
        self.truncation = truncation
        self.sim = [QrackSimulator(len(self.qa)), QrackSimulator(len(self.qb))]
        self.b = 0
        self.fidelity_estimate = 1.0
        self.truncations = 0
        self.peak_bonds = 0
        self._defer = False
        self.rng = np.random.default_rng(seed)

    def clone(self):
        return QrackBridge(None, _clone=self)

    def _nd(self, p):
        return len(self.qa) if p == 0 else len(self.qb)

    def _loc(self, q):
        p, i = self.side[q]
        return self.sim[p], p, i

    # --- local gates ---------------------------------------------------------

    def u(self, q, th, ph, lm):
        s, _, i = self._loc(q)
        s.u(i, th, ph, lm)

    def x(self, q):
        s, _, i = self._loc(q)
        s.x(i)

    def h(self, q):
        s, _, i = self._loc(q)
        s.h(i)

    def s(self, q):
        s, _, i = self._loc(q)
        s.s(i)

    def adjs(self, q):
        s, _, i = self._loc(q)
        s.adjs(i)

    # --- controlled gates, local or bridged ----------------------------------

    def _controlled(self, c, t, mtrx, anti):
        sc, pc, ic = self._loc(c)
        st, pt, it = self._loc(t)
        if pc == pt:
            (sc.macmtrx if anti else sc.mcmtrx)([ic], mtrx, it)
            return
        kc = self._nd(pc) + self.b
        kt = self._nd(pt) + self.b
        sc.allocate_qubit(kc)
        st.allocate_qubit(kt)
        (sc.macx if anti else sc.mcx)([ic], kc)
        st.h(kt)
        st.mcmtrx([kt], mtrx, it)
        self.b += 1
        self.peak_bonds = max(self.peak_bonds, self.b)
        # Inside a SWAP-type gate, recompress exactly between its CNOTs; truncate once at its end.
        self._compress(exact_only=self._defer)

    def mcx(self, c, t): self._controlled(c[0], t, _X, False)
    def mcy(self, c, t): self._controlled(c[0], t, _Y, False)
    def mcz(self, c, t): self._controlled(c[0], t, _Z, False)
    def macx(self, c, t): self._controlled(c[0], t, _X, True)
    def macy(self, c, t): self._controlled(c[0], t, _Y, True)
    def macz(self, c, t): self._controlled(c[0], t, _Z, True)

    def cx(self, a, b): self.mcx([a], b)
    def cy(self, a, b): self.mcy([a], b)
    def cz(self, a, b): self.mcz([a], b)

    def _composite(self, steps):
        outer = self._defer
        self._defer = True
        for f in steps:
            f()
        self._defer = outer
        if not outer:
            self._compress()

    def swap(self, a, b):
        if self.side[a][0] == self.side[b][0]:
            s, _, i = self._loc(a)
            s.swap(i, self.side[b][1])
            return
        self._composite([lambda: self.mcx([a], b), lambda: self.mcx([b], a), lambda: self.mcx([a], b)])

    def iswap(self, a, b):
        if self.side[a][0] == self.side[b][0]:
            s, _, i = self._loc(a)
            s.iswap(i, self.side[b][1])
            return
        self._composite([lambda: self.s(a), lambda: self.s(b), lambda: self.mcz([a], b), lambda: self.swap(a, b)])

    def adjiswap(self, a, b):
        if self.side[a][0] == self.side[b][0]:
            s, _, i = self._loc(a)
            s.adjiswap(i, self.side[b][1])
            return
        self._composite([lambda: self.swap(a, b), lambda: self.mcz([a], b), lambda: self.adjs(a), lambda: self.adjs(b)])

    # --- recompression through the kets --------------------------------------

    def _matrices(self):
        """Each patch ket as a (2^n_patch x 2^b) matrix over the live bonds (dead slots are |0>)."""
        out = []
        for p in (0, 1):
            nd = self._nd(p)
            v = _out_ket(self.sim[p])
            out.append(v.reshape(1 << self.b, 1 << nd).T)  # (data, bond): bond = high qubits
        return out

    def _store(self, p, mat, b_new):
        nd = self._nd(p)
        full = np.zeros((1 << b_new, 1 << nd), dtype=complex)
        full[: mat.shape[1], :] = mat.T
        v = full.reshape(-1)
        v = v / np.linalg.norm(v)
        self.sim[p] = QrackSimulator(nd + b_new)
        _in_ket(self.sim[p], v)

    def _compress(self, exact_only=False):
        if self.b == 0:
            return
        ma, mb = self._matrices()
        qa_, ra = np.linalg.qr(ma)
        qb_, rb = np.linalg.qr(mb)
        u, sv, vh = np.linalg.svd(ra @ rb.T)
        if sv[0] <= 0:
            return
        live = int(np.sum(sv > _TOL * sv[0]))
        b_exact = int(math.ceil(math.log2(max(live, 1))))
        b_new = b_exact if exact_only else min(b_exact, self.max_bonds)
        keep = min(live, 1 << b_new)
        w = sv ** 2
        if keep < live:
            self.truncations += 1
            if self.truncation == "greedy":
                idx = np.arange(keep)
                self.fidelity_estimate *= float(np.sum(w[:keep]) / np.sum(w[:live]))
            else:
                groups = [np.arange(i, min(i + keep, live)) for i in range(0, live, keep)]
                pg = np.array([np.sum(w[g]) for g in groups]) / np.sum(w[:live])
                idx = groups[int(self.rng.choice(len(groups), p=pg))]
                self.fidelity_estimate *= float(np.sum(pg ** 2))
        else:
            idx = np.arange(keep)
        new_a = (qa_ @ u[:, idx]) * sv[idx][None, :]      # singular values absorbed into patch A
        new_b = qb_ @ vh[idx, :].T                         # orthonormal columns: patch B isometric
        self._store(0, new_a, b_new)
        self._store(1, new_b, b_new)
        self.b = b_new

    # --- readout -----------------------------------------------------------------

    def measure_shots(self, qubits, shots):
        self._compress(exact_only=True)            # canonical: patch B isometric
        na, nb = len(self.qa), len(self.qb)
        a_draws = self.sim[0].measure_shots(list(range(na)), shots)   # exact marginal of patch A
        ma, mb = self._matrices()
        out = []
        for a, cnt in zip(*np.unique(np.asarray(a_draws, dtype=np.int64), return_counts=True)):
            amp = mb @ ma[a, :]
            pb = np.abs(amp) ** 2
            pb /= pb.sum()
            for bidx in self.rng.choice(len(pb), size=int(cnt), p=pb):
                full = int(a) | (int(bidx) << na)
                out.append(sum(((full >> q) & 1) << j for j, q in enumerate(qubits)))
        self.rng.shuffle(out)
        return out

    def out_ket(self):
        """Dense contracted state (2^n), little-endian over logical qubits; for testing."""
        ma, mb = self._matrices()
        psi = (ma @ mb.T)                               # [a, b]
        v = psi.T.reshape(-1)                           # index = a + 2^na * b
        return v / np.linalg.norm(v)
