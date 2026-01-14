# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import ctypes
import sys

from .qrack_system import Qrack
from .neuron_activation_fn import NeuronActivationFn


class QrackNeuron:
    """Class that exposes the QNeuron class of Qrack

    This model of a "quantum neuron" is based on the concept of a "uniformly controlled"
    rotation of a single output qubit around the Pauli Y axis, and has been developed by
    others. In our case, the primary relevant gate could also be called a
    single-qubit-target multiplexer.

    (See https://arxiv.org/abs/quant-ph/0407010 for an introduction to "uniformly controlled
    gates.)

    QrackNeuron is meant to be interchangeable with a single classical neuron, as in
    conventional neural net software. It differs from classical neurons in conventional
    neural nets, in that the "synaptic cleft" is modelled as a single qubit. Hence, this
    neuron can train and predict in superposition.

    Attributes:
        nid(int): Qrack ID of this neuron
        simulator(QrackSimulator): Simulator instance for all synaptic clefts of the neuron
        controls(list(int)): Indices of all "control" qubits, for neuron input
        target(int): Index of "target" qubit, for neuron output
        activation_fn(NeuronActivationFn): Activation function choice
        alpha(float): Activation function parameter, if required
        angles(list[ctypes.c_float]): (or c_double) Memory for neuron prediction angles
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
        activation_fn=NeuronActivationFn.Sigmoid,
        alpha=1.0,
        _init=True,
    ):
        self.simulator = simulator
        self.controls = controls
        self.target = target
        self.activation_fn = activation_fn
        self.alpha = alpha
        self.angles = QrackNeuron._real1_byref([0.0] * (1 << len(controls)))

        if not _init:
            return

        self.nid = Qrack.qrack_lib.init_qneuron(
            simulator.sid,
            len(controls),
            QrackNeuron._ulonglong_byref(controls),
            target,
        )

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
        result = QrackNeuron(
            self.simulator,
            self.controls,
            self.target,
        )
        result.nid = Qrack.qrack_lib.clone_qneuron(self.simulator.sid)
        result.angles = self.angles[:]
        self._throw_if_error()
        return result

    @staticmethod
    def _ulonglong_byref(a):
        return (ctypes.c_ulonglong * len(a))(*a)

    @staticmethod
    def _real1_byref(a):
        # This needs to be c_double, if PyQrack is built with fp64.
        if Qrack.fppow < 6:
            return (ctypes.c_float * len(a))(*a)
        return (ctypes.c_double * len(a))(*a)

    def set_simulator(self, s):
        """Set the neuron simulator

        Set the simulator used by this neuron

        Args:
            s(QrackSimulator): The simulator to use

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        Qrack.qrack_lib.set_qneuron_sim(
            self.nid,
            s.sid,
            len(self.controls),
            QrackNeuron._ulonglong_byref(self.controls),
            self.target,
        )
        self._throw_if_error()
        self.simulator = s

    def set_angles(self, a):
        """Directly sets the neuron parameters.

        Set all synaptic parameters of the neuron directly, by a list
        enumerated over the integer permutations of input qubits.

        Args:
            a(list(double)): List of input permutation angles

        Raises:
            ValueError: Angles 'a' in QrackNeuron.set_angles() must contain at least (2 ** len(self.controls)) elements.
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(a) < (1 << len(self.controls)):
            raise ValueError(
                "Angles 'a' in QrackNeuron.set_angles() must contain at least (2 ** len(self.controls)) elements."
            )
        self.angles = QrackNeuron._real1_byref(a)

    def get_angles(self):
        """Directly gets the neuron parameters.

        Get all synaptic parameters of the neuron directly, as a list
        enumerated over the integer permutations of input qubits.

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        return list(self.angles)

    def set_alpha(self, a):
        """Set the neuron 'alpha' parameter.

        To enable nonlinear activation, `QrackNeuron` has an 'alpha'
        parameter that is applied as a power to its angles, before
        learning and prediction. This makes the activation function
        sharper (or less sharp).
        """
        self.alpha = a

    def set_activation_fn(self, f):
        """Sets the activation function of this QrackNeuron

        Nonlinear activation functions can be important to neural net
        applications, like DNN. The available activation functions are
        enumerated in `NeuronActivationFn`.
        """
        self.activation_fn = f

    def predict(self, e=True, r=True):
        """Predict based on training

        "Predict" the anticipated output, based on input and training.
        By default, "predict()" will initialize the output qubit as by
        resetting to |0> and then acting a Hadamard gate. From that
        state, the method amends the output qubit upon the basis of
        the state of its input qubits, applying a rotation around
        Pauli Y axis according to the angle learned for the input.

        Args:
            e(bool): If False, predict the opposite
            r(bool): If True, start by resetting the output to 50/50

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        result = Qrack.qrack_lib.qneuron_predict(self.nid, self.angles, e, r, self.activation_fn, self.alpha)
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
        result = Qrack.qrack_lib.qneuron_unpredict(self.nid, self.angles, e, self.activation_fn, self.alpha)
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
        Qrack.qrack_lib.qneuron_learn_cycle(self.nid, self.angles, e, self.activation_fn, self.alpha)
        self._throw_if_error()

    def learn(self, eta, e=True, r=True):
        """Learn from current qubit state

        "Learn" to associate current inputs with output. Based on
        input qubit states and volatility 'eta,' the input state
        synaptic parameter is updated to prefer the "e" ("expected")
        output.

        Args:
            eta(double): Training volatility, 0 to 1
            e(bool): If False, predict the opposite
            r(bool): If True, start by resetting the output to 50/50

        Raises:
            RuntimeError: QrackNeuron C++ library raised an exception.
        """
        Qrack.qrack_lib.qneuron_learn(self.nid, self.angles, eta, e, r, self.activation_fn, self.alpha)
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
        Qrack.qrack_lib.qneuron_learn_permutation(self.nid, self.angles, eta, e, r, self.activation_fn, self.alpha)
        self._throw_if_error()

    @staticmethod
    def quantile_bounds(vec, bits):
        """Calculate vector quantile bounds

        This is a static helper method to calculate the quantile
        bounds of 2 ** bits worth of quantiles.

        Args:
            vec: numerical vector
            bits: log2() of quantile count

        Returns:
            Quantile (n + 1) bounds for n-quantile division, including
            minimum and maximum values
        """

        bins = 1 << bits
        n = len(vec)
        vec_sorted = sorted(vec)

        return (
            [vec_sorted[0]]
            + [vec_sorted[(k * n) // bins] for k in range(1, bins)]
            + [vec_sorted[-1]]
        )

    @staticmethod
    def discretize(vec, bounds):
        """Discretize vector by quantile bounds

        This is a static helper method to discretize a numerical
        vector according to quantile bounds calculated by the
        quantile_bounds(vec, bits) static method.

        Args:
            vec: numerical vector
            bounds: (n + 1) n-quantile bounds including extrema

        Returns:
            Discretized bit-row vector, least-significant first
        """

        bounds = bounds[1:]
        bounds_len = len(bounds)
        bits = bounds_len.bit_length() - 1
        n = len(vec)
        vec_discrete = [[False] * n for _ in range(bits)]
        for i, v in enumerate(vec):
            p = 0
            while (p < bounds_len) and (v > bounds[p]):
                p += 1
            for b in range(bits):
                vec_discrete[b][i] = bool((p >> b) & 1)

        return vec_discrete

    @staticmethod
    def flatten_and_transpose(arr):
        """Flatten and transpose feature matrix

        This is a static helper method to convert a multi-feature
        bit-row matrix to an observation-row matrix with flat
        feature columns.

        Args:
            arr: bit-row matrix

        Returns:
            Observation-row matrix with flat feature columns
        """
        return list(zip(*[item for sublist in arr for item in sublist]))

    @staticmethod
    def bin_endpoints_average(bounds):
        """Bin endpoints average

        This is a static helper method that accepts the output
        bins from quantile_bounds() and returns the average points
        between the bin endpoints. (This is NOT always necessarily
        the best heuristic for how to convert binned results back
        to numerical results, but it is often a reasonable way.)

        Args:
            bounds: (n + 1) n-quantile bounds including extrema

        Returns:
            List of average points between the bin endpoints
        """
        return [((bounds[i] + bounds[i + 1]) / 2) for i in range(len(bounds) - 1)]
