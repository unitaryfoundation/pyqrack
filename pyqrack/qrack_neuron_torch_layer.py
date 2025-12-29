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
from .neuron_activation_fn import NeuronActivationFn


class QrackNeuronTorchFunction(Function if _IS_TORCH_AVAILABLE else object):
    """Static forward/backward/apply functions for QrackNeuronTorch"""

    @staticmethod
    def forward(ctx, x, neuron_wrapper):
        ctx.neuron_wrapper = neuron_wrapper
        ctx.save_for_backward(x)
        neuron = neuron_wrapper.neuron

        pre_prob = neuron.simulator.prob(neuron.target)
        neuron.set_angles(x.detach().cpu().numpy() if _IS_TORCH_AVAILABLE else x)
        neuron.predict(True, False)
        post_prob = neuron.simulator.prob(neuron.target)

        ctx.delta = pre_prob - post_prob
        if _IS_TORCH_AVAILABLE:
            post_prob = torch.tensor([post_prob], dtype=torch.float32)

        return post_prob

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        neuron_wrapper = ctx.neuron_wrapper
        neuron = neuron_wrapper.neuron

        neuron.set_angles(x.detach().cpu().numpy() if _IS_TORCH_AVAILABLE else x)
        neuron.unpredict()

        if _IS_TORCH_AVAILABLE:
            grad_input = grad_output * ctx.delta
        else:
            grad_input = [o * d for o, d in zip(grad_output, ctx.delta)]

        return grad_input


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
    """Torch layer wrapper for QrackNeuron (with maximally expressive set of neurons between inputs and outputs)"""

    def __init__(
        self,
        simulator,
        input_indices,
        output_indices,
        activation=int(NeuronActivationFn.Generalized_Logistic),
        lowest_combo_count=0,
        highest_combo_count=2,
        parameters=None,
    ):
        """
        Initialize a QrackNeuron layer for PyTorch with a power set of neurons connecting inputs to outputs.
        The inputs and outputs must take the form of discrete, binary features (loaded manually into the backing QrackSimulator)

        Args:
            sim (QrackSimulator): Simulator into which predictor features are loaded
            input_indices (list[int]): List of input bits
            output_indices (list[int]): List of output bits
            activation (int): Integer corresponding to choice of activation function from NeuronActivationFn
            lowest_combo_count (int): Lowest combination count of input qubits iterated (0 is bias)
            highest_combo_count (int): Highest combination count of input qubits iterated
            parameters (list[float]): (Optional) Flat list of initial neuron parameters, corresponding to little-endian basis states of input qubits, repeated for ascending combo count, repeated for each output index
        """
        super(QrackNeuronTorchLayer, self).__init__()
        self.simulator = simulator
        self.simulators = []
        self.input_indices = input_indices
        self.output_indices = output_indices
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
            else lambda x: QrackNeuronTorchFunction.forward(object(), x)
        )

        # Create neurons from all input combinations, projecting to coherent output qubits
        self.neurons = nn.ModuleList(
            [
                QrackNeuronTorch(
                    QrackNeuron(simulator, input_subset, output_id, activation)
                )
                for output_id in output_indices
                for k in range(lowest_combo_count, highest_combo_count + 1)
                for input_subset in itertools.combinations(input_indices, k)
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
        if _IS_TORCH_AVAILABLE:
            B = x.shape[0]
            x = x.view(B, -1)
        else:
            B = len(x)

        self.simulators.clear()
        if _IS_TORCH_AVAILABLE:
            for b in range(B):
                simulator = self.simulator.clone()
                self.simulators.append(simulator)
                for q, input_id in enumerate(self.input_indices):
                    simulator.r(Pauli.PauliY, math.pi * x[b, q].item(), q)
        else:
            for b in range(B):
                simulator = self.simulator.clone()
                self.simulators.append(simulator)
                for q, input_id in enumerate(self.input_indices):
                    simulator.r(Pauli.PauliY, math.pi * x[b][q], q)

        y = [([0.0] * len(self.output_indices)) for _ in range(B)]
        for b in range(B):
            simulator = self.simulators[b]
            # Prepare a maximally uncertain output state.
            for output_id in self.output_indices:
                simulator.h(output_id)

            # Set Qrack's internal parameters:
            for idx, neuron_wrapper in enumerate(self.neurons):
                self.apply_fn(self.weights[idx], neuron_wrapper)

            for q, output_id in enumerate(self.output_indices):
                y[b][q] = simulator.prob(output_id)

        if _IS_TORCH_AVAILABLE:
            y = torch.tensor(y, dtype=torch.float32, device=x.device, requires_grad=True)

        return y


class QrackNeuronTorchLayerFunction(Function if _IS_TORCH_AVAILABLE else object):
    """Static forward/backward/apply functions for QrackNeuronTorch"""

    @staticmethod
    def forward(ctx, x, neuron_layer):
        final_probs = neuron_layer.forward(x)

        # Save for backward
        ctx.save_for_backward(x, final_probs)
        ctx.neuron_layer = neuron_layer

        return final_probs

    @staticmethod
    def backward(ctx, grad_output):
        x, final_probs = ctx.saved_tensors
        neuron_layer = ctx.neuron_layer
        output_indices = neuron_layer.output_indices
        simulators = neuron_layer.simulators
        neurons = neuron_layer.neurons

        if _IS_TORCH_AVAILABLE:
            B = x.shape[0]
            x = x.view(B, -1)
        else:
            B = len(x)

        # Uncompute prediction
        if _IS_TORCH_AVAILABLE:
            delta = torch.zeros((B, len(output_indices)), dtype=torch.float32, device=x.device, requires_grad=True)
            for b in range(B):
                for neuron_wrapper in neurons:
                    neuron = neuron_wrapper.neuron
                    delta[b, output_indices.index(neuron.target)] += neuron_layer.backward_fn(neuron.neuron)
        else:
            delta = [[0.0] * len(output_indices) for _ in range(B)]
            for b in range(B):
                for neuron_wrapper in neurons:
                    neuron = neuron_wrapper.neuron
                    delta[b][output_indices.index(neuron.target)] += neuron_layer.backward_fn(neuron.neuron)

        # Uncompute output state prep
        for simulator in simulators:
            for output_id in output_indices:
                simulator.h(output_id)

        if _IS_TORCH_AVAILABLE:
            grad_input = grad_output * delta
        else:
            grad_input = [o * d for o, d in zip(grad_output, delta)]

        return grad_input
