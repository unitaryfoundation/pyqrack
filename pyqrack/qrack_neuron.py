# (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import ctypes
import sys

from .qrack_system import Qrack

class QrackNeuron:
    """Class that exposes the QNeuron class of Qrack

    Attributes:
        nid(int): Qrack ID of this neuron
        simulator(QrackSimulator): Simulator instance for all synaptic clefts of the neuron
        controls(list(int)): Indices of all "control" qubits, for neuron input
        target(int): Index of "target" qubit, for neuron output
        tolerance(double): Rounding tolerance
    """

    def _get_error(self):
        return Qrack.qrack_lib.get_error(self.simulator.sid)

    def _throw_if_error(self):
        if self._get_error() != 0:
            raise RuntimeError("QrackNeuron C++ library raised exception.")

    def __init__(
        self,
        simulator,
        controls,
        target,
        tolerance = sys.float_info.epsilon,
        _init = True
    ):
        self.simulator = simulator
        self.controls = controls
        self.target = target
        self.tolerance = tolerance

        self.amp_count = 1 << (len(controls) + 1)

        if not _init:
            return

        self.nid = Qrack.qrack_lib.init_qneuron(simulator.sid, len(controls), self._ulonglong_byref(controls), target, tolerance)

        self._throw_if_error()

    def __del__(self):
        if self.nid is not None:
            Qrack.qrack_lib.destroy_qneuron(self.nid)
            self.nid = None

    def clone(self):
        """Clones this neuron.

        Create a new, independent neuron instance with identical angles,
        inputs, output, and tolerance, for the same QrackSimulator.

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        result = QrackNeuron(self.simulator, self.controls, self.target, self.tolerance)
        self.nid = Qrack.qrack_lib.clone_qneuron(self.simulator.sid)
        self._throw_if_error()
        return result

    def _ulonglong_byref(self, a):
        return (ctypes.c_ulonglong * len(a))(*a)

    def _real1_byref(self, a):
        # This needs to be c_double, if PyQrack is built with fp64.
        if Qrack.fppow < 6:
            return (ctypes.c_float * len(a))(*a)
        return (ctypes.c_double * len(a))(*a)

    def set_qneuron_angles(self, a):
        """Directly sets the neuron parameters.

        Set all parameters of the neuron directly, by a list
        enumerated over the integer permutations of input quibts.

        Args:
            a(list(double)): List of input permutation angles

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        Qrack.qrack_lib.set_qneuron_angles(self.nid, self._real1_byref(a))
        self._throw_if_error()

    def get_qneuron_angles(self):
        """Directly gets the neuron parameters.

        Get all parameters of the neuron directly, as a list
        enumerated over the integer permutations of input quibts.

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        ket = self._real1_byref([0.0] * self.amp_count)
        Qrack.qrack_lib.get_qneuron_angles(self.nid, ket)
        if self._get_error() != 0:
            raise RuntimeError("QrackSimulator C++ library raised exception.")
        return list(ket)

    def predict(self, e=True, r=True):
        """Predict based on training

        'Predict' the anticipated output, based on input and training.

        Args:
            e(bool): If False, predict the opposite
            r(bool): If True, start by resetting the output to 50/50

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        result = Qrack.qrack_lib.qneuron_predict(self.nid, e, r)
        self._throw_if_error()
        return result

    def unpredict(self, e=True):
        """Uncompute a prediction

        Uncompute a 'prediction' of the anticipated output, based on
        input and training.

        Args:
            e(bool): If False, unpredict the opposite

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        result = Qrack.qrack_lib.qneuron_unpredict(self.nid, e)
        self._throw_if_error()
        return result

    def learn_cycle(self, e=True):
        """Run a learning cycle

        A learning cycle consists of predicting a result, saving the
        classical outcome, and uncomputing the prediction.

        Args:
            e(bool): If False, predict the opposite

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        Qrack.qrack_lib.qneuron_learn_cycle(self.nid, e)
        self._throw_if_error()

    def learn(self, eta, e=True, r=True):
        """Learn from current qubit state

        Learn to associate current inputs with output

        Args:
            eta(double): Training volatility, 0 to 1
            e(bool): If False, predict the opposite
            r(bool): If True, start by resetting the output to 50/50

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        Qrack.qrack_lib.qneuron_learn(self.nid, eta, e, r)
        self._throw_if_error()

    def learn_permutation(self, eta, e=True, r=True):
        """Learn from current classical state

        Learn to associate current inputs with output, under the
        assumption that the inputs and outputs are "classical."

        Args:
            eta(double): Training volatility, 0 to 1
            e(bool): If False, predict the opposite
            r(bool): If True, start by resetting the output to 50/50

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        Qrack.qrack_lib.qneuron_learn_permutation(self.nid, eta, e, r)
        self._throw_if_error()
