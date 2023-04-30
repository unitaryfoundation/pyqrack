# (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import ctypes
import sys

from .qrack_system import Qrack

class QrackNeuron:
    """Class that exposes the QNeuron class of Qrack

    This model of a "quantum neuron" is based on the concept of a "uniformly controlled"
    rotation of a single output qubit around the Pauli Y axis, and has been developed by
    others. (See https://arxiv.org/abs/quant-ph/0407010 for an introduction to "uniformly
    controlled" gates, which could also be called single-qubit-target multiplexer gates.)

    QrackNeuron is meant to be interchangeable with a single classical neuron, as in
    conventional neural net software. It differs from classical neurons in conventional
    neural nets, in that the "synaptic cleft" is modelled as a single qubit. Hence, this
    neuron can train and predict in superposition.

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

        Set all synaptic parameters of the neuron directly, by a list
        enumerated over the integer permutations of input qubits.

        Args:
            a(list(double)): List of input permutation angles

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        Qrack.qrack_lib.set_qneuron_angles(self.nid, self._real1_byref(a))
        self._throw_if_error()

    def get_qneuron_angles(self):
        """Directly gets the neuron parameters.

        Get all synaptic parameters of the neuron directly, as a list
        enumerated over the integer permutations of input qubits.

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

        "Predict" the anticipated output, based on input and training.
        By default, "predict()" will initialize the output qubit as by
        reseting to |0> and then acting a Hadamard gate. From that
        state, the method amends the output qubit upon the basis of
        the state of its input qubits, applying a rotation around
        Pauli Y axis according to the angle learned for the input.

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

        "Learn" to associate current inputs with output. Based on
        input qubit states and volatility 'eta,' the input state
        angle is updated to prefer the "e" ("expected") output.

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
        Based on input qubit states and volatility 'eta,' the input
        state angle is updated to prefer the "e" ("expected") output.

        Args:
            eta(double): Training volatility, 0 to 1
            e(bool): If False, predict the opposite
            r(bool): If True, start by resetting the output to 50/50

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        Qrack.qrack_lib.qneuron_learn_permutation(self.nid, eta, e, r)
        self._throw_if_error()
