# Bridged patching on QrackSimulator patches, recompressed through out_ket()/in_ket().
#
# By (Anthropic) Claude and Dan Strano
#
# Patches. `patches` (default 2) contiguous blocks of logical qubits, equal up to the rounding
# remainder: every patch gets n // patches qubits and the LAST n % patches patches get one more (so
# for 2 patches the first is rounded down, as before). Patches sit on a 1D chain; only direct
# nearest-neighbour patches share bonds. With wrap=True and 3+ patches, the chain closes into a ring
# (patch P-1 shares bonds with patch 0). With 2 patches there is one cut and all-to-all coupling.
#
# Representation. Each patch is ONE QrackSimulator holding its data qubits plus its live bond qubits:
# "L" bonds on the cut to its left neighbour and "R" bonds on the cut to its right neighbour. Bond j
# on the right of patch p is paired with bond j on the left of patch p+1. The represented state is
# all patch kets contracted on every live bond pair with the unnormalized virtual Bell projection
# sum_k <k|<k|. Reading patch p's ket as a tensor A_p[l, d, r] (left bonds, data, right bonds), the
# state is the matrix-product contraction A_0 A_1 ... A_{P-1} (traced through the wrap cut on a ring).
# Every in-patch gate is applied natively, once, to all bond branches at once.
#
# Cross-patch gates by gate teleportation along the chain (exact, gates only). Controlled-U from c
# (patch pc) to t (patch pt), along the shorter way round on a ring:
#     control patch:       CNOT c -> fresh bond on the first cut              (copy the control)
#     each relay patch:    H incoming bond ; CNOT incoming -> fresh bond on the next cut
#     target patch:        H incoming bond ; controlled-U  incoming -> t
# Contracting on each bond pair enforces equality, so every relay bond carries a copy of c and the
# result is exactly controlled-U(c -> t). A gate between patches k apart costs one bond on each of
# the k cuts in between; nothing ever couples non-adjacent patches directly.
# SWAP is 3 CNOTs; iSWAP is (S x S), CZ, SWAP; iSWAP^dagger is SWAP, CZ, (S^dag x S^dag).
#
# Recompression after each bridged two-qubit gate (one MPS-style sweep over the touched segment):
#   - Gates on data qubits are unitary on each patch's (left, data) and (data, right) row spaces, so
#     they preserve left/right isometry. Only patches on a gate's path lose it.
#   - An orthogonality centre is tracked. QR-sweep left-to-right over the touched segment (and up to
#     the centre), then SVD-sweep right-to-left, keeping the largest singular values per cut (exact
#     when only zeros are dropped; above max_bonds, greedy or Born-rule group selection).
#   - On an open chain the environment is isometric at every SVD, so these are the true Schmidt values
#     of the state and each truncation is the optimal one for that cut, as with 2 patches.
#   - On a ring, the wrap cut is first recompressed between patches P-1 and 0 locally (QR of both
#     sides, SVD of the core), then the open sweep runs with the wrap bond treated as a physical leg.
#     Dropping only zeros is still exact. Truncating the wrap cut, and truncating inner cuts while wrap
#     bonds are live, is locally optimal but not globally optimal (a periodic MPS has no canonical
#     form), so fidelity_estimate is an estimate there and not the kept Schmidt weight.
# Nothing larger than the patch kets and per-cut cores is ever formed: no density matrix.
#
# Readout. Exact for the stored state: sequential conditional sampling, patch by patch, from the
# exported kets, with shots grouped by prefix. On an open chain the canonical form (centre at patch 0)
# makes each step a local marginal. On a ring, right environments carry the open wrap index, at a cost
# of (chi * chi_wrap)^2 per environment.
#
# fidelity_estimate multiplies the kept Schmidt weight (greedy) or sum_g p_g^2 (born) per truncation.

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


def _bits(dim):
    """Bond qubits needed for bond dimension dim."""
    return (max(int(dim), 1) - 1).bit_length()


_X = [0, 1, 1, 0]
_Y = [0, -1j, 1j, 0]
_Z = [1, 0, 0, -1]
_TOL = 1e-6          # relative singular-value floor treated as exactly zero (Qrack is single precision)


class QrackBridge:
    def __init__(self, qubit_count, max_bonds=6, seed=None, truncation="greedy", patches=2, wrap=False,
                 _clone=None):
        if _clone is not None:
            src = _clone
            for k in ("n", "P", "ring", "max_bonds", "truncation", "fidelity_estimate", "truncations",
                      "peak_bonds", "center", "_wrap_touched"):
                setattr(self, k, getattr(src, k))
            self.part = [list(x) for x in src.part]
            self.start = list(src.start)
            self.side = dict(src.side)
            self.layout = [list(x) for x in src.layout]
            self.bond = list(src.bond)
            self.sim = [s.clone() for s in src.sim]
            self._touched = set(src._touched)
            self._defer = False
            self.rng = np.random.default_rng(int(src.rng.integers(1 << 62)))
            return
        if truncation not in ("greedy", "born"):
            raise ValueError("truncation must be 'greedy' or 'born'")
        if not 1 <= patches <= qubit_count:
            raise ValueError("need 1 <= patches <= qubit_count")
        self.n = qubit_count
        self.P = patches
        self.ring = bool(wrap) and patches >= 3
        base, rem = divmod(qubit_count, patches)
        sizes = [base + (1 if p >= patches - rem else 0) for p in range(patches)]
        self.start = [sum(sizes[:p]) for p in range(patches)]
        self.part = [list(range(self.start[p], self.start[p] + sizes[p])) for p in range(patches)]
        self.side = {q: (p, i) for p, qs in enumerate(self.part) for i, q in enumerate(qs)}
        # internal qubit order of each patch sim: ("d", i) data, ("L", j) / ("R", j) bond slots
        self.layout = [[("d", i) for i in range(len(qs))] for qs in self.part]
        self.sim = [QrackSimulator(len(qs)) for qs in self.part]
        self.bond = [0] * (patches if self.ring else patches - 1)   # cut c joins patch c (R) to c+1 (L)
        self.max_bonds = max_bonds
        self.truncation = truncation
        self.fidelity_estimate = 1.0
        self.truncations = 0
        self.peak_bonds = 0
        self.center = 0
        self._touched = set()
        self._wrap_touched = False
        self._defer = False
        self.rng = np.random.default_rng(seed)

    def clone(self):
        return QrackBridge(None, _clone=self)

    # --- chain geometry --------------------------------------------------------

    def _nd(self, p):
        return len(self.part[p])

    def _lcut(self, p):
        if p > 0:
            return p - 1
        return self.P - 1 if self.ring else None

    def _rcut(self, p):
        return p if (p < self.P - 1 or self.ring) else None

    def _dims(self, p):
        lc, rc = self._lcut(p), self._rcut(p)
        bl = 0 if lc is None else self.bond[lc]
        br = 0 if rc is None else self.bond[rc]
        return bl, br

    def _path(self, pa, pb):
        """Patches visited from pa to pb along the chain (shorter way round on a ring)."""
        step = 1 if pb > pa else -1
        if self.ring:
            right = (pb - pa) % self.P
            step = 1 if right <= self.P - right else -1
        path = [pa]
        while path[-1] != pb:
            path.append((path[-1] + step) % self.P)
        return path

    def _new_bond(self, here, there):
        """Fresh bond pair on the cut between adjacent patches here and there; returns (id here, id there)."""
        rightward = there == (here + 1) % self.P
        c = here if rightward else there
        j = self.bond[c]
        self.bond[c] += 1
        self.peak_bonds = max(self.peak_bonds, self.bond[c])
        if self.ring and c == self.P - 1:
            self._wrap_touched = True
        ids = {}
        for p, tag in ((c, ("R", j)), ((c + 1) % self.P, ("L", j))):
            qid = len(self.layout[p])
            self.sim[p].allocate_qubit(qid)
            self.layout[p].append(tag)
            ids[p] = qid
        return ids[here], ids[there]

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
        path = self._path(pc, pt)
        out, inc = self._new_bond(path[0], path[1])
        (sc.macx if anti else sc.mcx)([ic], out)
        for k in range(1, len(path) - 1):            # relay patches: pass the copy of c along
            s = self.sim[path[k]]
            s.h(inc)
            out, nxt = self._new_bond(path[k], path[k + 1])
            s.mcx([inc], out)
            inc = nxt
        st.h(inc)
        st.mcmtrx([inc], mtrx, it)
        self._touched.update(path)
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

    # --- patch tensors through the kets --------------------------------------

    def _tensor(self, p):
        """Patch p's ket as A[l, d, r] (left bonds, data, right bonds), whatever its internal order."""
        lay = self.layout[p]
        k = len(lay)
        bl, br = self._dims(p)
        nd = self._nd(p)
        v = _out_ket(self.sim[p])
        order = ([("L", j) for j in reversed(range(bl))] + [("d", i) for i in reversed(range(nd))]
                 + [("R", j) for j in reversed(range(br))])
        pos = {tag: x for x, tag in enumerate(lay)}
        t = v.reshape([2] * k).transpose([k - 1 - pos[tag] for tag in order])
        return t.reshape(1 << bl, 1 << nd, 1 << br)

    def _store(self, p, a):
        """Write A[l, d, r] back as a fresh patch sim: data low, then L bonds, then R bonds."""
        bl, br = self._dims(p)
        nd = self._nd(p)
        full = np.zeros((1 << bl, 1 << nd, 1 << br), dtype=complex)
        full[: a.shape[0], :, : a.shape[2]] = a
        v = full.transpose(2, 0, 1).reshape(-1)
        v = v / np.linalg.norm(v)
        self.layout[p] = [("d", i) for i in range(nd)] + [("L", j) for j in range(bl)] + [("R", j) for j in range(br)]
        self.sim[p] = QrackSimulator(nd + bl + br)
        _in_ket(self.sim[p], v)

    # --- recompression -------------------------------------------------------

    def _select(self, sv, exact_only):
        if sv[0] <= 0:
            return np.arange(1)
        live = int(np.sum(sv > _TOL * sv[0]))
        b_exact = _bits(live)
        b_new = b_exact if exact_only else min(b_exact, self.max_bonds)
        keep = min(live, 1 << b_new)
        if keep >= live:
            return np.arange(keep)
        w = sv[:live] ** 2
        self.truncations += 1
        if self.truncation == "greedy":
            self.fidelity_estimate *= float(np.sum(w[:keep]) / np.sum(w))
            return np.arange(keep)
        groups = [np.arange(i, min(i + keep, live)) for i in range(0, live, keep)]
        pg = np.array([np.sum(w[g]) for g in groups]) / np.sum(w)
        self.fidelity_estimate *= float(np.sum(pg ** 2))
        return groups[int(self.rng.choice(len(groups), p=pg))]

    def _sweep(self, lo, hi, exact_only):
        """Canonicalize and truncate inner cuts over [lo, hi] (plus up to the centre); centre ends at lo."""
        start, end = min(lo, self.center), max(hi, self.center)
        if start == end:
            self.center = lo
            return
        T = {k: self._tensor(k) for k in range(start, end + 1)}
        for k in range(start, end):                       # left-isometric up to the centre at `end`
            l, d, r = T[k].shape
            q, rr = np.linalg.qr(T[k].reshape(l * d, r))
            T[k] = q.reshape(l, d, -1)
            T[k + 1] = np.einsum("ab,bdr->adr", rr, T[k + 1])
            self.bond[k] = _bits(q.shape[1])
        for k in range(end, lo, -1):                      # right-isometric back down to `lo`
            l, d, r = T[k].shape
            u, sv, vh = np.linalg.svd(T[k].reshape(l, d * r), full_matrices=False)
            idx = self._select(sv, exact_only)
            T[k] = vh[idx].reshape(len(idx), d, r)
            T[k - 1] = np.einsum("ldr,rs->lds", T[k - 1], u[:, idx] * sv[idx][None, :])
            self.bond[k - 1] = _bits(len(idx))
        for k, a in T.items():
            self._store(k, a)
        self.center = lo

    def _compress_wrap(self, exact_only):
        """Ring only: recompress the wrap cut (P-1 | 0) locally, QR both sides and SVD of the core."""
        P = self.P
        a, b = self._tensor(P - 1), self._tensor(0)
        l, d, w = a.shape
        _, d0, r0 = b.shape
        qa, ra = np.linalg.qr(a.reshape(l * d, w))
        qb, rb = np.linalg.qr(b.reshape(w, d0 * r0).T)
        u, sv, vh = np.linalg.svd(ra @ rb.T)
        idx = self._select(sv, exact_only)
        self.bond[P - 1] = _bits(len(idx))
        self._store(P - 1, ((qa @ u[:, idx]) * sv[idx][None, :]).reshape(l, d, -1))
        self._store(0, (qb @ vh[idx, :].T).T.reshape(-1, d0, r0))

    def _compress(self, exact_only=False):
        if not self._touched:
            return
        if self._wrap_touched:
            self._compress_wrap(exact_only)
            self._touched.update((0, self.P - 1))
        self._sweep(min(self._touched), max(self._touched), exact_only)
        if not exact_only:
            self._touched.clear()
            self._wrap_touched = False

    # --- readout -----------------------------------------------------------------

    def measure_shots(self, qubits, shots):
        self._compress()
        self._sweep(0, 0, exact_only=True)                # centre at patch 0: patches 1.. right-isometric
        T = [self._tensor(p) for p in range(self.P)]
        R = None
        if self.ring:                                     # right environments with the wrap index open
            cw = T[0].shape[0]
            eye = np.eye(cw)
            R = [None] * (self.P + 1)
            R[self.P] = np.einsum("rw,sv->rwsv", eye, eye)
            for k in range(self.P - 1, 0, -1):
                R[k] = np.einsum("ldr,rwsv,mds->lwmv", T[k], R[k + 1], T[k].conj(), optimize=True)
            groups = [(eye, shots, 0)]
        else:
            groups = [(np.ones((1, 1)), shots, 0)]
        for k in range(self.P):
            nxt = []
            for X, cnt, bits in groups:
                M = np.einsum("wl,ldr->wdr", X, T[k])
                if R is None:
                    p = np.sum(np.abs(M) ** 2, axis=(0, 2))
                else:
                    p = np.einsum("wdr,rwsv,vds->d", M, R[k + 1], M.conj(), optimize=True).real
                p = np.clip(p, 0, None)
                p /= p.sum()
                for d, c in enumerate(self.rng.multinomial(cnt, p)):
                    if c:
                        Y = M[:, d, :]
                        nxt.append((Y / np.linalg.norm(Y), int(c), bits | (d << self.start[k])))
            groups = nxt
        out = []
        for _, c, full in groups:
            out += [sum(((full >> q) & 1) << j for j, q in enumerate(qubits))] * c
        self.rng.shuffle(out)
        return out

    def out_ket(self):
        """Dense contracted state (2^n), little-endian over logical qubits; for testing."""
        C = None
        for p in range(self.P):
            A = self._tensor(p)
            if C is None:
                C = np.einsum("wl,ldr->wdr", np.eye(A.shape[0]), A)
            else:
                w, D, _ = C.shape
                C = np.einsum("wDr,rdq->wdDq", C, A).reshape(w, -1, A.shape[2])
        v = np.einsum("wDw->D", C)
        return v / np.linalg.norm(v)
