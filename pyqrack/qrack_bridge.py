# Bridged patching, Qrack-native: two exact QrackSimulator patches joined by virtual Bell bonds.
# No out_ket(), no in_ket(), no SVD.
#
# By (Anthropic) Claude and Dan Strano
#
# Representation. Each patch is one QrackSimulator holding its data qubits plus a few "bond" qubits.
# Bond qubits come in pairs (one in each patch). The represented state is the two patch states
# contracted on every bond pair with the unnormalized virtual Bell projection <Phi| = sum_k <k|<k|:
#
#     |Psi>  ~  (prod_pairs <Phi|) ( |psi_A> (x) |psi_B> )
#
# No amplitude ever crosses between the two simulators; only small reduced density matrices of the
# bond qubits (out_rdm, 2^b x 2^b for b live bonds) are read, to evaluate contraction norms.
#
# Cross-cut gates by gate teleportation through a fresh bond pair (exact, gates only):
#     controlled-U from c (one patch) to t (the other):
#       control patch:  CNOT  c -> k_c                 (copy the control into the bond)
#       target patch:   H k_t ;  controlled-U  k_t -> t
#     Contracting <Phi| on (k_c, k_t) enforces k_t = c, which is exactly controlled-U(c -> t).
#     Anti-controlled gates copy with an anti-controlled NOT. SWAP is 3 CNOTs; iSWAP is
#     (S x S), CZ, SWAP; iSWAP^dagger is SWAP, CZ, (S^dag x S^dag).
# Each controlled gate across the cut adds one bond pair (Schmidt rank x2), so bond count is the
# log2 of the bridge rank: max_bonds = b caps the rank at 2^b. max_bonds large enough is exact.
#
# Compression after every bridged gate (see _compress), from the two bond RDMs only:
#   * exact: rotate the bond basis (V on one half, V* on the other, which leaves the contraction
#     invariant) so one patch's bond RDM is diagonal, and drop the bond states it gives zero weight;
#     bond count then tracks log2 of the actual rank, not the number of gates.
#   * above max_bonds: first a heralded gauge transform, G on patch A's bonds and (G^-1)^T on
#     patch B's with G = rho_A^(-1/2) (each applied as a unitary dilation on bonds + one ancilla,
#     then force_m(ancilla, 0)), so patch A's bond RDM becomes the identity; then the unitary step
#     diagonalizes patch B's bond RDM, whose eigenvalues are now the squared Schmidt values, and the
#     cut keeps the largest (greedy) or picks a group by Born rule. That is the optimal rank
#     truncation, the same answer an SVD would give, computed from 2^b x 2^b matrices.
#   * SWAP-type gates (several CNOTs) are compressed exactly between their CNOTs and truncated once.
#   V and the dilations are applied with Qrack's multiplexed single-qubit gates (multiplex1_mtrx),
#   via recursive cosine-sine decomposition, so the patches never expose amplitudes.
#
# Cost note: everything is gates, force_m and out_rdm on bond qubits. In this PyQrack build,
# out_rdm (QInterface::GetReducedDensityMatrix) reads amplitudes one GetAmplitude call at a time,
# dimKept * (dimKept + 1) per environment pattern, and is ~90% of the runtime.
#
# Readout. Exact sampling of the contracted state, qubit by qubit: for each data qubit, the outcome
# probability is the contraction of that patch's RDM over (qubit + its bonds) with the other patch's
# bond RDM; the outcome is then imposed with force_m. Patch A is read first (the B bond RDM is fixed
# meanwhile), then patch B.
#
# Interface: the subset of QrackSimulator / QrackAceBackend that nn_qab.py circuits call.

import math

import numpy as np
from .qrack_simulator import QrackSimulator
from scipy.linalg import cossin

_X = [0, 1, 1, 0]
_Y = [0, -1j, 1j, 0]
_Z = [1, 0, 0, -1]
_EPS = 1e-12
_RANK_TOL = 1e-6    # relative eigenvalue floor for "exactly zero" (Qrack runs in single precision)
_CANON_TOL = 1e-4   # relative floor below which bond directions are not rescaled (conditioning)


def apply_unitary(sim, targets, U, controls=None, Us=None):
    """Apply a 2^t x 2^t unitary U (index bit i = targets[i], LSB first) using only Qrack
    multiplexed single-qubit gates (multiplex1_mtrx), by recursive cosine-sine decomposition.
    With controls/Us: a uniformly controlled unitary, Us[c] applied when controls read c (LSB first)."""
    if Us is None:
        controls, Us = [], [np.asarray(U, dtype=complex)]
    t = len(targets)
    if t == 1:
        sim.multiplex1_mtrx(list(controls), targets[0], [complex(x) for u in Us for x in np.asarray(u).reshape(-1)])
        return
    half = 1 << (t - 1)
    top, rest = targets[-1], targets[:-1]
    L, CS, R = [], [], []
    for u in Us:
        (u1, u2), theta, (v1h, v2h) = cossin(u, p=half, q=half, separate=True)
        L.append((u1, u2)); R.append((v1h, v2h)); CS.append(theta)
    nc = len(Us)
    # time order: R (block-diag on rest, multiplexed by controls + top), then CS on top, then L
    apply_unitary(sim, rest, None, list(controls) + [top], [r[0] for r in R] + [r[1] for r in R])
    mlist = []
    for idx in range(nc * half):
        r_i = idx % half          # rest value (low bits)
        c_i = idx // half         # outer control value (high bits)
        th = CS[c_i][r_i]
        cth, sth = np.cos(th), np.sin(th)
        mlist.append(np.array([[cth, -sth], [sth, cth]], dtype=complex))
    sim.multiplex1_mtrx(list(rest) + list(controls), top, [complex(x) for m in mlist for x in m.reshape(-1)])
    apply_unitary(sim, rest, None, list(controls) + [top], [l[0] for l in L] + [l[1] for l in L])



class QrackBridge:
    def __init__(self, qubit_count, max_bonds=6, seed=None, truncation="greedy", canonical=True, _clone=None):
        if _clone is not None:
            src = _clone
            self.n, self.qa, self.qb, self.side = src.n, src.qa, src.qb, src.side
            self.max_bonds = src.max_bonds
            self.truncation = src.truncation
            self.canonical = src.canonical
            self.anc = list(src.anc)
            self._defer = False
            self.compressions = src.compressions
            self.sim = [s.clone() for s in src.sim]
            self.pairs = list(src.pairs)
            self.free = [list(f) for f in src.free]
            self.fidelity_estimate = src.fidelity_estimate
            self.dephasings = src.dephasings
            self.peak_bonds = src.peak_bonds
            self.rng = np.random.default_rng(int(src.rng.integers(1 << 62)))
            return
        self.n = qubit_count
        self.qa = list(range(qubit_count>>1))
        self.qb = list(range(qubit_count>>1, qubit_count))
        self.side = {q: (0, i) for i, q in enumerate(self.qa)}
        self.side.update({q: (1, i) for i, q in enumerate(self.qb)})
        self.max_bonds = max_bonds
        if truncation not in ("greedy", "born"):
            raise ValueError("truncation must be 'greedy' or 'born'")
        self.truncation = truncation
        self.compressions = 0
        # Bond slots: max_bonds live pairs plus one transient slot for the pair being created.
        self.canonical = canonical
        nb = max_bonds + 4      # room for a whole multi-CNOT gate (SWAP-type) before compressing
        self._defer = False
        # One more qubit per patch: the ancilla for heralded (non-unitary) gauge transforms.
        self.anc = [len(self.qa) + nb, len(self.qb) + nb]
        self.sim = [QrackSimulator(len(self.qa) + nb + 1), QrackSimulator(len(self.qb) + nb + 1)]
        self.free = [list(range(len(self.qa), len(self.qa) + nb)), list(range(len(self.qb), len(self.qb) + nb))]
        self.pairs = []            # list of (slot in patch A, slot in patch B)
        self.fidelity_estimate = 1.0
        self.dephasings = 0
        self.peak_bonds = 0
        self.rng = np.random.default_rng(seed)

    def clone(self):
        return QrackBridge(None, None, _clone=self)

    # --- local gates ---------------------------------------------------------

    def _loc(self, q):
        p, i = self.side[q]
        return self.sim[p], p, i

    def u(self, q, th, ph, lm):
        s, _, i = self._loc(q)
        s.u(i, th, ph, lm)

    def x(self, q):
        s, _, i = self._loc(q)
        s.x(i)

    def s(self, q):
        s, _, i = self._loc(q)
        s.s(i)

    def adjs(self, q):
        s, _, i = self._loc(q)
        s.adjs(i)

    def h(self, q):
        s, _, i = self._loc(q)
        s.h(i)

    # --- controlled gates, local or bridged ----------------------------------

    def _controlled(self, c, t, mtrx, anti):
        sc, pc, ic = self._loc(c)
        st, pt, it = self._loc(t)
        if pc == pt:
            (sc.macmtrx if anti else sc.mcmtrx)([ic], mtrx, it)
            return
        kc = self.free[pc].pop()
        kt = self.free[pt].pop()
        (sc.macx if anti else sc.mcx)([ic], kc)
        st.h(kt)
        st.mcmtrx([kt], mtrx, it)
        self.pairs.append((kc, kt) if pc == 0 else (kt, kc))
        self.peak_bonds = max(self.peak_bonds, len(self.pairs))
        # Inside a multi-CNOT gate (SWAP-type), compress exactly only; truncate once at its end.
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

    def swap(self, a, b):
        if self.side[a][0] == self.side[b][0]:
            s, _, i = self._loc(a)
            s.swap(i, self.side[b][1])
            return
        outer = self._defer
        self._defer = True
        self.mcx([a], b)
        self.mcx([b], a)
        self.mcx([a], b)
        self._defer = outer
        if not outer:
            self._compress()

    def iswap(self, a, b):
        if self.side[a][0] == self.side[b][0]:
            s, _, i = self._loc(a)
            s.iswap(i, self.side[b][1])
            return
        self._defer = True
        self.s(a)
        self.s(b)
        self.mcz([a], b)
        self.swap(a, b)
        self._defer = False
        self._compress()

    def adjiswap(self, a, b):
        if self.side[a][0] == self.side[b][0]:
            s, _, i = self._loc(a)
            s.adjiswap(i, self.side[b][1])
            return
        self._defer = True
        self.swap(a, b)
        self.mcz([a], b)
        self.adjs(a)
        self.adjs(b)
        self._defer = False
        self._compress()

    # --- contraction bookkeeping (bond RDMs only) ------------------------------

    def _bond_rdm(self, p, extra=None):
        slots = [pr[p] for pr in self.pairs]
        qs = ([extra] if extra is not None else []) + slots
        if not qs:
            return np.ones((1, 1), dtype=complex)
        d = 1 << len(qs)
        return np.array(self.sim[p].out_rdm(qs)).reshape(d, d)

    def _apply_contraction(self, p, k):
        """Apply the non-unitary map k (spectral norm <= 1) to patch p's bond register, heralded:
        unitary dilation [[k, sqrt(1 - k k^+)], [sqrt(1 - k^+ k), -k^+]] on bonds + ancilla, then
        force_m(ancilla, 0). Gates and a forced measurement only; the global norm is irrelevant to
        the represented state, which is defined up to normalization."""
        d = k.shape[0]
        def psd_sqrt(m):
            w, v = np.linalg.eigh((m + m.conj().T) / 2)
            return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.conj().T
        eye = np.eye(d)
        u = np.block([[k, psd_sqrt(eye - k @ k.conj().T)], [psd_sqrt(eye - k.conj().T @ k), -k.conj().T]])
        slots = self._bond_slots(p) + [self.anc[p]]
        apply_unitary(self.sim[p], slots, u)
        self.sim[p].force_m(self.anc[p], False)

    def _canonicalize(self, ea, va):
        """Gauge transform G on patch A's bonds and (G^-1)^T on patch B's (contraction invariant),
        with G = rho_A^(-1/2) on its support, so patch A's bond RDM becomes the identity there."""
        top = max(ea[-1], _EPS)
        keep = ea > _CANON_TOL * top
        lam = ea[keep]
        e = va[:, keep]
        g = (e * (1.0 / np.sqrt(lam))) @ e.conj().T                  # rho_A^(-1/2) on support
        g_inv_t = ((e * np.sqrt(lam)) @ e.conj().T).T                 # (rho_A^(1/2))^T
        self._apply_contraction(0, g * math.sqrt(lam.min()))
        self._apply_contraction(1, g_inv_t / math.sqrt(lam.max()))

    def _bond_slots(self, p):
        return [pr[p] for pr in self.pairs]

    def _compress(self, exact_only=False):
        """Keep the bond register as small as the contracted state allows, and within max_bonds.

        All decisions come from the two bond RDMs (2^b x 2^b). The bond basis is rotated by a unitary
        V on one patch's bond qubits and V* on the other's -- which leaves the virtual Bell
        contraction invariant -- so that (1) the pivot patch's bond RDM is diagonal and (2) basis
        states are sorted by their contribution to the contracted norm. Bond states the pivot RDM
        gives zero weight carry zero amplitude, so dropping them is exact; beyond max_bonds the
        least-weighted states are cut (greedy, or by Born-rule group selection). Dropping bond
        qubits is force_m on both halves, then reset. V is applied with multiplexed single-qubit
        gates only (recursive cosine-sine decomposition), so no amplitude leaves either simulator."""
        b = len(self.pairs)
        if b == 0:
            return
        ra, rb = self._bond_rdm(0), self._bond_rdm(1)
        ea, va = np.linalg.eigh(ra)
        eb, vb = np.linalg.eigh(rb)
        ra_rank = int(np.sum(ea > _RANK_TOL * max(ea[-1], _EPS)))
        rb_rank = int(np.sum(eb > _RANK_TOL * max(eb[-1], _EPS)))
        pivot = 0 if ra_rank <= rb_rank else 1
        rank = min(ra_rank, rb_rank)
        c_exact = max(0, int(math.ceil(math.log2(max(rank, 1)))))
        c = c_exact if exact_only else min(c_exact, self.max_bonds)
        if c >= b:
            return
        if c < c_exact and self.canonical:
            # Truncation ahead: first move to canonical gauge, where patch A's bond RDM is the
            # identity on its support. Then the remaining unitary step diagonalizes patch B's bond
            # RDM and its eigenvalues ARE the squared Schmidt values of the contracted state, so
            # the cut below is the optimal (SVD-equivalent) rank truncation.
            self._canonicalize(ea, va)
            ra, rb = self._bond_rdm(0), self._bond_rdm(1)
            eb, vb = np.linalg.eigh(rb)
            ea, va = np.linalg.eigh(ra)
            pivot = 1
        lam, vecs = (ea, va) if pivot == 0 else (eb, vb)
        rp, ro = (ra, rb) if pivot == 0 else (rb, ra)
        v_p = vecs.conj().T                       # pivot RDM -> diagonal
        v_o = v_p.conj()
        rp2 = v_p @ rp @ v_p.conj().T
        ro2 = v_o @ ro @ v_o.conj().T
        score = np.real(np.diag(rp2)) * np.real(np.diag(ro2))
        order = np.argsort(-score, kind="stable")
        perm = np.zeros((1 << b, 1 << b))
        perm[np.arange(1 << b), order] = 1.0
        v_p = perm @ v_p
        v_o = v_p.conj()
        rp2 = v_p @ rp @ v_p.conj().T
        ro2 = v_o @ ro @ v_o.conj().T
        w = rp2 * ro2
        total = float(np.real(np.sum(w)))
        groups = 1 << (b - c)
        size = 1 << c
        ng = np.array([float(np.real(np.sum(w[g * size:(g + 1) * size, g * size:(g + 1) * size]))) for g in range(groups)])
        ng = np.clip(ng, 0.0, None)
        if total <= 0 or ng.sum() <= 0:
            return
        pg = ng / ng.sum()
        if self.truncation == "born" and c < c_exact:
            g = int(self.rng.choice(groups, p=pg))
            self.fidelity_estimate *= float(np.sum(pg ** 2))
        else:
            g = 0
            self.fidelity_estimate *= float(pg[0])
        if c < c_exact:
            self.dephasings += 1
        self.compressions += 1
        # Rotate both bond registers, then fix the dropped bond qubits to group g on both halves.
        slots_p = self._bond_slots(pivot)
        slots_o = self._bond_slots(1 - pivot)
        apply_unitary(self.sim[pivot], slots_p, v_p)
        apply_unitary(self.sim[1 - pivot], slots_o, v_o)
        for j in range(b - 1, c - 1, -1):
            m = bool((g >> (j - c)) & 1)
            for side in (0, 1):
                slot = self.pairs[j][side]
                self.sim[side].force_m(slot, m)
                if m:
                    self.sim[side].x(slot)
                self.free[side].append(slot)
            del self.pairs[j]

    # --- readout -----------------------------------------------------------------

    def _sample_once(self):
        bits = {}
        for p, qs in ((0, self.qa), (1, self.qb)):
            other = self._bond_rdm(1 - p)
            for q in qs:
                i = self.side[q][1]
                r = self._bond_rdm(p, extra=i)          # bit 0 = data qubit, bits 1.. = bonds
                d = r.shape[0]
                idx = np.arange(d)
                w1 = 0.0
                w = 0.0
                for m in (0, 1):
                    sel = (idx & 1) == m
                    sub = r[np.ix_(sel, sel)]
                    val = float(np.real(np.sum(sub * other)))
                    w += val
                    if m:
                        w1 = val
                p1 = min(1.0, max(0.0, w1 / w)) if w > 0 else 0.0
                m = bool(self.rng.random() < p1)
                self.sim[p].force_m(i, m)
                bits[q] = m
        return bits

    def measure_shots(self, qubits, shots):
        out = []
        for _ in range(shots):
            b = self.clone()._sample_once()
            out.append(sum(int(b[q]) << j for j, q in enumerate(qubits)))
        return out

    # --- testing only: dense contraction (2^n), never used by the bridge itself -------

    def _test_dense_ket(self):
        """For validation only: contract the two patches densely (uses out_ket)."""
        kets = []
        for p, qs in ((0, self.qa), (1, self.qb)):
            s = self.sim[p]
            nloc = s.num_qubits()
            v = np.array(s.out_ket()).reshape([2] * nloc)        # axis 0 = highest qubit index
            kets.append((v, nloc, len(qs)))
        (va, na, da), (vb, nb, db) = kets
        ax = lambda nloc, q: nloc - 1 - q
        slots_a = [pr[0] for pr in self.pairs]
        slots_b = [pr[1] for pr in self.pairs]
        free_a = [q for q in range(da, na) if q not in slots_a]
        free_b = [q for q in range(db, nb) if q not in slots_b]
        va = va[tuple(0 if (na - 1 - a) in free_a else slice(None) for a in range(na))]
        vb = vb[tuple(0 if (nb - 1 - a) in free_b else slice(None) for a in range(nb))]
        keep_a = [q for q in range(na - 1, -1, -1) if q not in free_a]     # remaining axes, in order
        keep_b = [q for q in range(nb - 1, -1, -1) if q not in free_b]
        psi = np.tensordot(va, vb, axes=([keep_a.index(s) for s in slots_a], [keep_b.index(s) for s in slots_b]))
        rem = [("A", q) for q in keep_a if q not in slots_a] + [("B", q) for q in keep_b if q not in slots_b]
        logical = [self.qa[q] if s == "A" else self.qb[q] for s, q in rem]
        psi = np.transpose(psi, [logical.index(q) for q in range(self.n - 1, -1, -1)]).reshape(-1)
        nrm = np.linalg.norm(psi)
        return psi / nrm if nrm > 0 else psi
