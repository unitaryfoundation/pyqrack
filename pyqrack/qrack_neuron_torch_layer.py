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


# Should be safe for 16-bit
angle_eps = math.pi * (2 ** -8)


class QrackNeuronTorchFunction(Function if _IS_TORCH_AVAILABLE else object):
    """Static forward/backward/apply functions for QrackNeuronTorch"""

    @staticmethod
    def forward(ctx, x, neuron_wrapper):
        ctx.neuron_wrapper = neuron_wrapper
        ctx.save_for_backward(x)
        neuron = neuron_wrapper.neuron

        angles = (x.detach().cpu().numpy() if x.requires_grad else x.numpy()) if _IS_TORCH_AVAILABLE else x
        neuron.set_angles(angles)
        neuron.predict(True, False)
        post_prob = neuron.simulator.prob(neuron.target)
        if _IS_TORCH_AVAILABLE:
            post_prob = torch.tensor([post_prob], dtype=torch.float32, device=x.device)

        return post_prob

    @staticmethod
    def _backward(x, neuron_wrapper):
        neuron = neuron_wrapper.neuron
        angles = (x.detach().cpu().numpy() if x.requires_grad else x.numpy()) if _IS_TORCH_AVAILABLE else x

        # Uncompute
        neuron.set_angles(angles)
        neuron.unpredict()
        pre_sim = neuron.simulator
        pre_prob = pre_sim.prob(neuron.target)

        param_count = 1 << len(neuron.controls)
        delta = [0.0] * param_count
        for param in range(param_count):
            angle = angles[param]

            # x + angle_eps
            angles[param] = angle + angle_eps
            neuron.set_angles(angles)
            neuron.simulator = pre_sim.clone()
            neuron.predict(True, False)
            p_plus = neuron.simulator.prob(neuron.target)

            # x - angle_eps
            angles[param] = angle - angle_eps
            neuron.set_angles(angles)
            neuron.simulator = pre_sim.clone()
            neuron.predict(True, False)
            p_minus = neuron.simulator.prob(neuron.target)

            # Central difference
            delta[param] = (p_plus - p_minus) / (2 * angle_eps)

            angles[param] = angle

        neuron.simulator = pre_sim

        if _IS_TORCH_AVAILABLE:
            delta = torch.tensor(delta, dtype=torch.float32, device=x.device)

        return delta

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        neuron_wrapper = ctx.neuron_wrapper
        delta = _backward(x, neuron_wrapper, grad_output)
        if _IS_TORCH_AVAILABLE:
            # grad_output: (O,)
            # delta:       (O, I)
            grad_input = torch.matmul(grad_output, delta)  # result: (I,)
        else:
            grad_input = [
                sum(o * d for o, d in zip(grad_output, col))
                for col in zip(*delta)
            ]

        return grad_input, None

class QrackNeuronTorch(nn.Module if _IS_TORCH_AVAILABLE else object):
    """Torch wrapper for QrackNeuron

    Attributes:
        neuron(QrackNeuron): QrackNeuron backing this torch wrapper
    """

    def __init__(self, neuron: QrackNeuron):
        super().__init__()
        self.neuron = neuron

    def forward(self, x):
        return QrackNeuronTorchFunction.apply(x, self.neuron)


class QrackNeuronTorchLayer(nn.Module if _IS_TORCH_AVAILABLE else object):
    """Torch layer wrapper for QrackNeuron (with maximally expressive set of neurons between inputs and outputs)

    Attributes:
        simulator (QrackSimulator): Prototype simulator that batching copies to use with QrackNeuron instances
        simulators (list[QrackSimulator]): In-flight copies of prototype simulator corresponding to batch count
        input_indices (list[int], read-only): simulator qubit indices used as QrackNeuron inputs
        output_indices (list[int], read-only): simulator qubit indices used as QrackNeuron outputs
        hidden_indices (list[int], read-only): simulator qubit indices used as QrackNeuron hidden inputs (in maximal superposition)
        neurons (list[QrackNeuron]): QrackNeurons in this layer, corresponding to weights
        weights (ParameterList): List of tensors corresponding one-to-one with weights of list of neurons
    """

    def __init__(
        self,
        input_qubits,
        output_qubits,
        hidden_qubits=None,
        lowest_combo_count=0,
        highest_combo_count=2,
        activation=int(NeuronActivationFn.Generalized_Logistic),
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
        self.apply_fn = (
            QrackNeuronTorchFunction.apply
            if _IS_TORCH_AVAILABLE
            else lambda x: QrackNeuronTorchFunction.forward(object(), x)
        )
        self.forward_fn = (
            QrackNeuronTorchFunction.forward
            if _IS_TORCH_AVAILABLE
            else lambda x: QrackNeuronTorchFunction.forward(object(), x)
        )
        self.backward_fn = (
            QrackNeuronTorchFunction.backward
            if _IS_TORCH_AVAILABLE
            else lambda x: QrackNeuronTorchFunction.backward(object(), x)
        )
        self._backward_fn = (
            QrackNeuronTorchFunction._backward
            if _IS_TORCH_AVAILABLE
            else lambda x: QrackNeuronTorchFunction._backward(object(), x)
        )

        # Create neurons from all input combinations, projecting to coherent output qubits
        self.neurons = nn.ModuleList(
            [
                QrackNeuronTorch(
                    QrackNeuron(self.simulator, input_subset, output_id, activation)
                )
                for output_id in self.output_indices
                for k in range(lowest_combo_count, highest_combo_count + 1)
                for input_subset in itertools.combinations(self.input_indices + self.hidden_indices, k)
            ]
        )   

        # Set Qrack's internal parameters:
        if parameters:
            param_count = 0
            self.weights = nn.ParameterList()
            for neuron_wrapper in self.neurons:
                neuron = neuron_wrapper.neuron
                p_count = 1 << len(neuron.controls)
                neuron.set_angles(parameters[param_count : (param_count + p_count)])
                self.weights.append(nn.Parameter(torch.tensor(parameters[param_count : (param_count + p_count)])))
                param_count += p_count
        else:
            self.weights = nn.ParameterList()
            for neuron_wrapper in self.neurons:
                neuron = neuron_wrapper.neuron
                p_count = 1 << len(neuron.controls)
                self.weights.append(nn.Parameter(torch.zeros(p_count)))

    def forward(self, x):
        return QrackNeuronTorchLayerFunction.apply(x, self)


class QrackNeuronTorchLayerFunction(Function if _IS_TORCH_AVAILABLE else object):
    """Static forward/backward/apply functions for QrackNeuronTorch"""

    @staticmethod
    def forward(ctx, x, neuron_layer):
        # Save for backward
        ctx.save_for_backward(x)
        ctx.neuron_layer = neuron_layer

        input_indices = neuron_layer.input_indices
        hidden_indices = neuron_layer.hidden_indices
        output_indices = neuron_layer.output_indices
        simulators = neuron_layer.simulators
        weights = neuron_layer.weights

        if _IS_TORCH_AVAILABLE:
            B = x.shape[0]
            x = x.view(B, -1)
        else:
            B = len(x)

        simulators.clear()
        if _IS_TORCH_AVAILABLE:
            for b in range(B):
                simulator = neuron_layer.simulator.clone()
                simulators.append(simulator)
                for q, input_id in enumerate(input_indices):
                    simulator.r(Pauli.PauliY, math.pi * x[b, q].item(), q)
        else:
            for b in range(B):
                simulator = neuron_layer.simulator.clone()
                simulators.append(simulator)
                for q, input_id in enumerate(input_indices):
                    simulator.r(Pauli.PauliY, math.pi * x[b][q], q)

        y = [([0.0] * len(output_indices)) for _ in range(B)]
        for b in range(B):
            simulator = simulators[b]
            # Prepare a maximally uncertain output state.
            for output_id in output_indices:
                simulator.h(output_id)
            # Prepare hidden predictors
            for h in hidden_indices:
                simulator.h(h)

            # Set Qrack's internal parameters:
            for idx, neuron_wrapper in enumerate(neuron_layer.neurons):
                neuron_wrapper.neuron.simulator = simulator
                neuron_layer.apply_fn(weights[idx], neuron_wrapper)

            for q, output_id in enumerate(output_indices):
                y[b][q] = simulator.prob(output_id)

        if _IS_TORCH_AVAILABLE:
            y = torch.tensor(y, dtype=torch.float32, device=x.device)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        neuron_layer = ctx.neuron_layer

        input_indices = neuron_layer.input_indices
        hidden_indices = neuron_layer.hidden_indices
        output_indices = neuron_layer.output_indices
        simulators = neuron_layer.simulators
        neurons = neuron_layer.neurons
        backward_fn = neuron_layer._backward_fn

        input_count = len(input_indices)
        output_count = len(output_indices)

        if _IS_TORCH_AVAILABLE:
            B = x.shape[0]
            x = x.view(B, -1)
        else:
            B = len(x)

        # Uncompute prediction
        if _IS_TORCH_AVAILABLE:
            delta = torch.zeros((B, output_count, input_count), dtype=torch.float32, device=x.device)
            for b in range(B):
                simulator = simulators[b]
                for neuron_wrapper in neurons:
                    neuron = neuron_wrapper.neuron
                    neuron.simulator = simulator
                    angles = torch.tensor(neuron.get_angles(), dtype=torch.float32, device=x.device, requires_grad=True)
                    o = output_indices.index(neuron.target)
                    neuron_grad = backward_fn(angles, neuron_wrapper)
                    for idx, c in enumerate(neuron.controls):
                        if c not in input_indices:
                            continue
                        i = input_indices.index(c)
                        delta[b, o, i] += neuron_grad[idx]
        else:
            delta = [[[0.0] * input_count for _ in range(output_count)] for _ in range(B)]
            for b in range(B):
                simulator = simulators[b]
                for neuron_wrapper in neurons:
                    neuron = neuron_wrapper.neuron
                    neuron.simulator = simulator
                    angles = neuron.get_angles()
                    o = output_indices.index(neuron.target)
                    neuron_grad = backward_fn(angles, neuron_wrapper)
                    for idx, c in enumerate(neuron.controls):
                        if c not in input_indices:
                            continue
                        i = input_indices.index(c)
                        delta[b][o][i] += neuron_grad[idx]

        # Uncompute output state prep
        for simulator in simulators:
            for output_id in output_indices:
                simulator.h(output_id)
            for h in hidden_indices:
                simulator.h(output_id)

        if _IS_TORCH_AVAILABLE:
            for b in range(B):
                simulator = simulators[b]
                for q, input_id in enumerate(input_indices):
                    simulator.r(Pauli.PauliY, -math.pi * x[b, q].item(), q)
        else:
            for b in range(B):
                simulator = simulators[b]
                for q, input_id in enumerate(input_indices):
                    simulator.r(Pauli.PauliY, -math.pi * x[b][q].item(), q)

        if _IS_TORCH_AVAILABLE:
            grad_input = torch.matmul(grad_output.view(B, 1, -1), delta).view_as(x)
        else:
            grad_input = [[0.0] * output_count for _ in range(B)]
            for b in range(B):
                for o in range(output_indices):
                    for i in range(input_indices):
                        grad_input[b][o] += grad_output[b][o] * delta[b][o][i]

        return grad_input, None
