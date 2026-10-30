# (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
#
# QrackBridge: bridged patching, backed by Qrack's native QBridge.
#
# By (Anthropic) Claude and Dan Strano
#
# The qubits are split into `patches` contiguous blocks (equal up to the remainder; the last
# qubit_count % patches blocks get one more), each simulated exactly by its own Qrack engine. Patches
# sit on a 1D chain (a ring with wrap=True and 3+ patches) and share "bond" qubits only with their
# nearest neighbours: the state is a matrix-product state whose sites are whole state-vector simulators.
# Gates within a patch run natively; gates between patches are teleported through fresh bond pairs and
# each cut is recompressed to at most 2^max_bonds Schmidt components (greedy, or Born-rule groups).
#
# Same API as the pure-Python reference implementation (pyqrack.QrackPyBridge), plus pauli_expectation().
# If the loaded Qrack library predates QBridge, pyqrack.QrackBridge is QrackPyBridge instead.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import contextlib
import ctypes
import math
import os

import numpy as np

from .qrack_system import Qrack

_SQ = 1 / math.sqrt(2)
_X = [0, 1, 1, 0]
_Y = [0, -1j, 1j, 0]
_Z = [1, 0, 0, -1]
_H = [_SQ, _SQ, _SQ, -_SQ]
_S = [1, 0, 0, 1j]
_ADJS = [1, 0, 0, -1j]
_T = [1, 0, 0, complex(_SQ, _SQ)]
_ADJT = [1, 0, 0, complex(_SQ, -_SQ)]


def _mtrx_byref(m):
    flat = []
    for c in m:
        c = complex(c)
        flat += [c.real, c.imag]
    return (ctypes.c_double * 8)(*flat)


def _ulonglong_byref(a):
    return (ctypes.c_ulonglong * len(a))(*a)


class QrackBridge:
    """Bridged patching on native Qrack patches (see module docstring).

    Args:
        qubit_count: number of logical qubits
        max_bonds: bond qubits per cut (bond dimension 2^max_bonds)
        seed: sampler seed (None: random)
        truncation: "greedy" (keep the largest Schmidt values) or "born" (Born-rule group selection)
        patches: number of patches (default 2: one cut, all-to-all coupling)
        wrap: close the patch chain into a ring (3+ patches)
        lazy_bonds: extra bond qubits a cut may buffer before a truncating sweep (0 = eager)
        max_patch_qubits: optional cap on any patch's total width that also forces a sweep
        is_*: engine layers for every patch, as for QrackSimulator
    """

    def __init__(
        self,
        qubit_count,
        max_bonds=6,
        seed=None,
        truncation="greedy",
        patches=2,
        wrap=False,
        lazy_bonds=0,
        max_patch_qubits=None,
        _clone=None,
        is_tensor_network=True,
        is_schmidt_decompose=True,
        is_schmidt_decompose_multi=False,
        is_stabilizer_hybrid=False,
        is_binary_decision_tree=False,
        is_paged=True,
        is_cpu_gpu_hybrid=True,
        is_gpu=True,
    ):
        self.bid = None
        if _clone is not None:
            self.bid = Qrack.qrack_lib.init_qbridge_clone(_clone.bid)
            for k in ("n", "P", "ring", "max_bonds", "lazy_bonds", "max_patch_qubits", "truncation"):
                setattr(self, k, getattr(_clone, k))
            self._throw_if_error()
            self._partition()
            return
        if truncation not in ("greedy", "born"):
            raise ValueError("truncation must be 'greedy' or 'born'")
        if not 1 <= patches <= qubit_count:
            raise ValueError("need 1 <= patches <= qubit_count")
        if lazy_bonds < 0:
            raise ValueError("lazy_bonds must be >= 0")
        self.n = qubit_count
        self.P = patches
        self.ring = bool(wrap) and patches >= 3
        self.max_bonds = max_bonds
        self.lazy_bonds = lazy_bonds
        self.max_patch_qubits = max_patch_qubits
        self.truncation = truncation
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "little")
        self.bid = Qrack.qrack_lib.init_qbridge(
            qubit_count,
            max_bonds,
            patches,
            bool(wrap),
            truncation == "born",
            lazy_bonds,
            max_patch_qubits or 0,
            seed & ((1 << 64) - 1),
            is_tensor_network,
            is_schmidt_decompose_multi,
            is_schmidt_decompose,
            is_stabilizer_hybrid,
            is_binary_decision_tree,
            is_paged,
            is_cpu_gpu_hybrid,
            is_gpu,
        )
        self._throw_if_error()
        self._partition()

    def _partition(self):
        base, rem = divmod(self.n, self.P)
        sizes = [base + (1 if p >= self.P - rem else 0) for p in range(self.P)]
        self.start = [sum(sizes[:p]) for p in range(self.P)]
        self.part = [list(range(self.start[p], self.start[p] + sizes[p])) for p in range(self.P)]
        self.side = {q: (p, i) for p, qs in enumerate(self.part) for i, q in enumerate(qs)}

    def __del__(self):
        if getattr(self, "bid", None) is not None:
            Qrack.qrack_lib.destroy_qbridge(self.bid)
            self.bid = None

    def _throw_if_error(self):
        if Qrack.qrack_lib.get_qbridge_error(self.bid) != 0:
            raise RuntimeError("QrackBridge C++ library raised exception.")

    def clone(self):
        return QrackBridge(None, _clone=self)

    def num_qubits(self):
        return self.n

    # --- statistics ----------------------------------------------------------

    @property
    def fidelity_estimate(self):
        return Qrack.qrack_lib.qbridge_get_fidelity_estimate(self.bid)

    @property
    def truncations(self):
        return Qrack.qrack_lib.qbridge_get_truncations(self.bid)

    @property
    def flushes(self):
        return Qrack.qrack_lib.qbridge_get_flushes(self.bid)

    @property
    def peak_bonds(self):
        return Qrack.qrack_lib.qbridge_get_peak_bonds(self.bid)

    @property
    def bond(self):
        """Live bond qubits per cut (cut c joins patch c to patch c + 1, mod P on a ring)."""
        c = Qrack.qrack_lib.qbridge_get_cut_count(self.bid)
        out = (ctypes.c_ulonglong * max(c, 1))()
        Qrack.qrack_lib.qbridge_get_bonds(self.bid, out)
        return [int(out[i]) for i in range(c)]

    # --- single-qubit gates --------------------------------------------------

    def mtrx(self, m, q):
        Qrack.qrack_lib.qbridge_mtrx(self.bid, _mtrx_byref(m), q)
        self._throw_if_error()

    def u(self, q, th, ph, lm):
        Qrack.qrack_lib.qbridge_u(self.bid, q, th, ph, lm)
        self._throw_if_error()

    def x(self, q):
        self.mtrx(_X, q)

    def y(self, q):
        self.mtrx(_Y, q)

    def z(self, q):
        self.mtrx(_Z, q)

    def h(self, q):
        self.mtrx(_H, q)

    def s(self, q):
        self.mtrx(_S, q)

    def adjs(self, q):
        self.mtrx(_ADJS, q)

    def t(self, q):
        self.mtrx(_T, q)

    def adjt(self, q):
        self.mtrx(_ADJT, q)

    # --- controlled gates, local or bridged ----------------------------------
    # One control across patches; several controls only when all qubits share a patch.

    def mcmtrx(self, c, m, t):
        Qrack.qrack_lib.qbridge_mcmtrx(self.bid, len(c), _ulonglong_byref(c), _mtrx_byref(m), t)
        self._throw_if_error()

    def macmtrx(self, c, m, t):
        Qrack.qrack_lib.qbridge_macmtrx(self.bid, len(c), _ulonglong_byref(c), _mtrx_byref(m), t)
        self._throw_if_error()

    def mcx(self, c, t): self.mcmtrx(c, _X, t)
    def mcy(self, c, t): self.mcmtrx(c, _Y, t)
    def mcz(self, c, t): self.mcmtrx(c, _Z, t)
    def macx(self, c, t): self.macmtrx(c, _X, t)
    def macy(self, c, t): self.macmtrx(c, _Y, t)
    def macz(self, c, t): self.macmtrx(c, _Z, t)

    def cx(self, a, b): self.mcx([a], b)
    def cy(self, a, b): self.mcy([a], b)
    def cz(self, a, b): self.mcz([a], b)

    # --- two-qubit gates (ship-and-return across patches: 2 bonds per cut, one sweep) ----------

    def swap(self, a, b):
        Qrack.qrack_lib.qbridge_swap(self.bid, a, b)
        self._throw_if_error()

    def iswap(self, a, b):
        Qrack.qrack_lib.qbridge_iswap(self.bid, a, b)
        self._throw_if_error()

    def adjiswap(self, a, b):
        Qrack.qrack_lib.qbridge_adjiswap(self.bid, a, b)
        self._throw_if_error()

    def fsim(self, th, ph, a, b):
        Qrack.qrack_lib.qbridge_fsim(self.bid, th, ph, a, b)
        self._throw_if_error()

    # --- truncation control --------------------------------------------------

    @contextlib.contextmanager
    def deferred(self):
        """Truncate once, after the whole block, never inside it (e.g. around a transversal logical
        gate); inside, bridged gates are recompressed only exactly. Nests."""
        Qrack.qrack_lib.qbridge_begin_deferred(self.bid)
        try:
            yield self
        finally:
            Qrack.qrack_lib.qbridge_end_deferred(self.bid)
            self._throw_if_error()

    def flush(self):
        """Truncate any buffered bonds back to max_bonds now (readout does this automatically)."""
        Qrack.qrack_lib.qbridge_flush(self.bid)
        self._throw_if_error()

    # --- readout -----------------------------------------------------------------

    def measure_shots(self, qubits, shots):
        """Exact samples of the stored state; bit j of each result is qubits[j]."""
        out = (ctypes.c_ulonglong * shots)()
        Qrack.qrack_lib.qbridge_measure_shots(self.bid, len(qubits), _ulonglong_byref(qubits), shots, out)
        self._throw_if_error()
        return list(out)

    def pauli_expectation(self, q, b):
        """Exact <P> for the Pauli product with basis b[i] (pyqrack.Pauli) on qubit q[i]."""
        if len(q) != len(b):
            raise RuntimeError("pauli_expectation argument lengths do not match.")
        result = Qrack.qrack_lib.qbridge_pauli_expectation(
            self.bid, len(q), _ulonglong_byref(q), _ulonglong_byref([int(x) for x in b])
        )
        self._throw_if_error()
        return result

    def out_ket(self):
        """Dense contracted state (2^n), little-endian over logical qubits; for testing."""
        real = ctypes.c_float if Qrack.fppow < 6 else ctypes.c_double
        buf = (real * (2 << self.n))()
        Qrack.qrack_lib.qbridge_out_ket(self.bid, buf)
        self._throw_if_error()
        a = np.frombuffer(buf, dtype=np.float32 if Qrack.fppow < 6 else np.float64)
        return a[0::2].astype(float) + 1j * a[1::2].astype(float)
