# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Initial draft by Elara (OpenAI custom GPT)
# Refined and architecturally clarified by Dan Strano
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import itertools
import math
import sys

_IS_TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    from torch.autograd import Function
except ImportError:
    _IS_TORCH_AVAILABLE = False

from .pauli import Pauli
from .qrack_neuron import QrackNeuron
from .qrack_simulator import QrackSimulator
from .neuron_activation_fn import NeuronActivationFn


# Parameter-shift rule
angle_eps = math.pi / 2


if not _IS_TORCH_AVAILABLE:
    class TorchContextMock(object):
        def __init__(self):
            pass

        def save_for_backward(self, *args):
            self.saved_tensors = args


class QrackNeuronTorchFunction(Function if _IS_TORCH_AVAILABLE else object):
    """Static forward/backward/apply functions for QrackNeuronTorch"""

    if not _IS_TORCH_AVAILABLE:
        @staticmethod
        def apply(x, neuron_wrapper):
            return forward(TorchContextMock(), x, neuron_wrapper)

    @staticmethod
    def forward(ctx, x, neuron_wrapper):
        ctx.neuron_wrapper = neuron_wrapper
        ctx.save_for_backward(x)

        neuron = neuron_wrapper.neuron

        angles = (x.detach().cpu().numpy() if x.requires_grad else x.numpy()) if _IS_TORCH_AVAILABLE else x
        neuron.set_angles(angles)
        neuron.predict(True, False)

        post_prob = neuron.simulator.prob(neuron.target)

        # Return shape: (1,)
        return x.new_tensor([post_prob]) if _IS_TORCH_AVAILABLE else post_prob

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        neuron = ctx.neuron_wrapper.neuron

        angles = (x.detach().cpu().numpy() if x.requires_grad else x.numpy()) if _IS_TORCH_AVAILABLE else x

        # Reset simulator state
        neuron.set_angles(angles)
        neuron.unpredict()
        pre_sim = neuron.simulator

        grad_x = torch.zeros_like(x) if _IS_TORCH_AVAILABLE else ([0.0] * len(x))

        for i in range(x.shape[0]):
            angle = angles[i]

            # x + π/2
            angles[i] = angle + angle_eps
            neuron.set_angles(angles)
            neuron.simulator = pre_sim.clone()
            neuron.predict(True, False)
            p_plus = neuron.simulator.prob(neuron.target)

            # x − π/2
            angles[i] = angle - angle_eps
            neuron.set_angles(angles)
            neuron.simulator = pre_sim.clone()
            neuron.predict(True, False)
            p_minus = neuron.simulator.prob(neuron.target)

            # parameter-shift rule
            grad_x[i] = 0.5 * (p_plus - p_minus)

            angles[i] = angle

        neuron.simulator = pre_sim

        # Chain rule: scalar grad_output
        grad_x *= grad_output[0]

        return grad_x, None


class QrackNeuronTorch(nn.Module if _IS_TORCH_AVAILABLE else object):
    """Torch wrapper for QrackNeuron

    Attributes:
        neuron(QrackNeuron): QrackNeuron backing this torch wrapper
    """

    def __init__(self, neuron, x):
        super().__init__()
        self.neuron = neuron
        self.weights = nn.Parameter(x) if _IS_TORCH_AVAILABLE else x

    def forward(self):
        return QrackNeuronTorchFunction.apply(self.weights, self.neuron)


class QrackNeuronTorchLayer(nn.Module if _IS_TORCH_AVAILABLE else object):
    """Torch layer wrapper for QrackNeuron (with maximally expressive set of neurons between inputs and outputs)

    Attributes:
        simulator (QrackSimulator): Prototype simulator that batching copies to use with QrackNeuron instances
        simulators (list[QrackSimulator]): In-flight copies of prototype simulator corresponding to batch count
        input_indices (list[int], read-only): simulator qubit indices used as QrackNeuron inputs
        output_indices (list[int], read-only): simulator qubit indices used as QrackNeuron outputs
        hidden_indices (list[int], read-only): simulator qubit indices used as QrackNeuron hidden inputs (in maximal superposition)
        neurons (ModuleList[QrackNeuronTorch]): QrackNeuronTorch wrappers (for PyQrack QrackNeurons) in this layer, corresponding to weights
        weights (ParameterList): List of tensors corresponding one-to-one with weights of list of neurons
        apply_fn (Callable[Tensor, QrackNeuronTorch]): Corresponds to QrackNeuronTorchFunction.apply(x, neuron_wrapper) (or override with a custom implementation)
        backward_fn (Callable[Tensor, Tensor]): Corresponds to QrackNeuronTorchFunction._backward(x, neuron_wrapper) (or override with a custom implementation)
    """

    def __init__(
        self,
        input_qubits,
        output_qubits,
        hidden_qubits=None,
        lowest_combo_count=0,
        highest_combo_count=2,
        activation=int(NeuronActivationFn.Generalized_Logistic),
        dtype=torch.float,
        parameters=None,
    ):
        """
        Initialize a QrackNeuron layer for PyTorch with a power set of neurons connecting inputs to outputs.
        The inputs and outputs must take the form of discrete, binary features (loaded manually into the backing QrackSimulator)

        Args:
            sim (QrackSimulator): Simulator into which predictor features are loaded
            input_qubits (int): Count of inputs (1 per qubit)
            output_qubits (int): Count of outputs (1 per qubit)
            hidden_qubits (int): Count of "hidden" inputs (1 per qubit, always initialized to |+>, suggested to be same a highest_combo_count)
            lowest_combo_count (int): Lowest combination count of input qubits iterated (0 is bias)
            highest_combo_count (int): Highest combination count of input qubits iterated
            activation (int): Integer corresponding to choice of activation function from NeuronActivationFn
            parameters (list[float]): (Optional) Flat list of initial neuron parameters, corresponding to little-endian basis states of input + hidden qubits, repeated for ascending combo count, repeated for each output index
        """
        super(QrackNeuronTorchLayer, self).__init__()
        if hidden_qubits is None:
            hidden_qubits = highest_combo_count
        self.simulator = QrackSimulator(input_qubits + hidden_qubits + output_qubits)
        self.simulators = []
        self.input_indices = list(range(input_qubits))
        self.hidden_indices = list(range(input_qubits, input_qubits + hidden_qubits))
        self.output_indices = list(range(input_qubits + hidden_qubits, input_qubits + hidden_qubits + output_qubits))
        self.activation = NeuronActivationFn(activation)
        self.dtype = dtype
        self.apply_fn = QrackNeuronTorchFunction.apply

        # Create neurons from all input combinations, projecting to coherent output qubits
        neurons = []
        param_count = 0
        for output_id in self.output_indices:
            for k in range(lowest_combo_count, highest_combo_count + 1):
                for input_subset in itertools.combinations(self.input_indices + self.hidden_indices, k):
                    p_count = 1 << len(input_subset)
                    neurons.append(QrackNeuronTorch(
                        QrackNeuron(self.simulator, input_subset, output_id, activation),
                        (torch.tensor(parameters[param_count : (param_count + p_count)], dtype=dtype) if parameters else torch.zeros(p_count, dtype=dtype)) if _IS_TORCH_AVAILABLE else ([0.0] * p_count)
                    ))
                    param_count += p_count
        self.neurons = nn.ModuleList(neurons) if _IS_TORCH_AVAILABLE else neurons

    def forward(self, x):
        if _IS_TORCH_AVAILABLE:
            B = x.shape[0]
            x = x.view(B, -1)
        else:
            B = len(x)

        self.simulators.clear()
        for b in range(B):
            simulator = self.simulator.clone()
            self.simulators.append(simulator)
            for q, input_id in enumerate(self.input_indices):
                simulator.r(Pauli.PauliY, math.pi * x[b][q], q)

        if _IS_TORCH_AVAILABLE:
            y = x.new_zeros((B, len(self.output_indices)))
        else:
            y = [([0.0] * len(self.output_indices)) for _ in range(B)]
        for b in range(B):
            simulator = self.simulators[b]
            # Prepare a maximally uncertain output state.
            for output_id in self.output_indices:
                simulator.h(output_id)
            # Prepare hidden predictors
            for h in self.hidden_indices:
                simulator.h(h)

            # Set QrackNeurons' internal parameters:
            for idx, neuron_wrapper in enumerate(self.neurons):
                neuron_wrapper.neuron.simulator = simulator
                o = self.output_indices.index(neuron_wrapper.neuron.target)
                y[b][o] = self.apply_fn(neuron_wrapper.weights, neuron_wrapper)

        return y
