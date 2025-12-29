# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Initial draft by Elara (OpenAI custom GPT)
# Refined and architecturally clarified by Dan Strano
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

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

from itertools import chain, combinations


# From https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset#answer-1482316
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3,) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class QrackTorchNeuron(nn.Module if _IS_TORCH_AVAILABLE else object):
    """Torch wrapper for QrackNeuron

    Attributes:
        neuron(QrackNeuron): QrackNeuron backing this torch wrapper
    """

    def __init__(self, neuron: QrackNeuron):
        super().__init__()
        self.neuron = neuron

    def forward(self, x):
        neuron = self.neuron
        neuron.predict(True, False)

        return neuron.simulator.prob(neuron.target)


class QrackNeuronFunction(Function if _IS_TORCH_AVAILABLE else object):
    """Static forward/backward/apply functions for QrackTorchNeuron"""

    @staticmethod
    def forward(ctx, neuron):
        # Save for backward
        ctx.neuron = neuron

        init_prob = neuron.simulator.prob(neuron.target)
        neuron.predict(True, False)
        final_prob = neuron.simulator.prob(neuron.target)
        ctx.delta = final_prob - init_prob

        return (
            torch.tensor([ctx.delta], dtype=torch.float32, requires_grad=True)
            if _IS_TORCH_AVAILABLE
            else ctx.delta
        )

    @staticmethod
    def backward(ctx, grad_output):
        neuron = ctx.neuron

        pre_unpredict = neuron.simulator.prob(neuron.output_id)
        neuron.unpredict()
        post_unpredict = neuron.simulator.prob(neuron.output_id)
        reverse_delta = pre_unpredict - post_unpredict

        grad = reverse_delta - ctx.delta

        return (
            torch.tensor([grad], dtype=torch.float32, requires_grad=True)
            if _IS_TORCH_AVAILABLE
            else grad
        )


class QrackNeuronTorchLayer(nn.Module if _IS_TORCH_AVAILABLE else object):
    """Torch layer wrapper for QrackNeuron (with power set of neurons between inputs and outputs)"""

    def __init__(
        self,
        simulator,
        input_indices,
        output_indices,
        activation=int(NeuronActivationFn.Generalized_Logistic),
    ):
        """
        Initialize a QrackNeuron layer for PyTorch with a power set of neurons connecting inputs to outputs.
        The inputs and outputs must take the form of discrete, binary features (loaded manually into the backing QrackSimulator)

        Args:
            sim (QrackSimulator): Simulator into which predictor features are loaded
            input_indices (list[int]): List of input bits
            output_indices (list[int]): List of output bits
            activation (int): Integer corresponding to choice of activation function from NeuronActivationFn
        """
        super(QrackNeuronTorchLayer, self).__init__()
        self.simulator = simulator
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.activation = NeuronActivationFn(activation)
        self.apply_fn = (
            QrackNeuronFunction.apply
            if _IS_TORCH_AVAILABLE
            else lambda x: QrackNeuronFunction.forward(object(), x)
        )
        self.forward_fn = (
            QrackNeuronFunction.forward
            if _IS_TORCH_AVAILABLE
            else lambda x: QrackNeuronFunction.forward(object(), x)
        )
        self.backward_fn = (
            QrackNeuronFunction.backward
            if _IS_TORCH_AVAILABLE
            else lambda x: QrackNeuronFunction.forward(object(), x)
        )

        # Create neurons from all powerset input combinations, projecting to coherent output qubits
        self.neurons = nn.ModuleList(
            [
                QrackTorchNeuron(
                    QrackNeuron(simulator, list(input_subset), output_id, activation)
                )
                for input_subset in powerset(input_indices)
                for output_id in output_indices
            ]
        )   

    def forward(self, x):
        perm_0_prob = self.simulator.prob_perm(self.input_indices,
            [False] * len(self.input_indices)
        )

        if _IS_TORCH_AVAILABLE:
            B = x.shape[0]
            x = x.view(B, -1)
        else:
            B = len(x)

        # If the inputs are not reset, they're effectively the input from the last layer.
        if (1.0 - perm_0_prob) <= sys.float_info.epsilon:
            b = 0
            q = 0
            # The simulator is effectively reset and we need to re-prepare the input
            if _IS_TORCH_AVAILABLE:
                for q, input_id in enumerate(self.input_indices):
                    self.simulator.r(Pauli.PauliY, math.pi * x[b, q].item(), q)
                    b += 1
                    if b == B:
                        q += 1
                        b = 0
            else:
                for q, input_id in enumerate(self.input_indices):
                    self.simulator.r(Pauli.PauliY, math.pi * x[b][q], q)
                    b += 1
                    if b == B:
                        q += 1
                        b = 0

        # Prepare a maximally uncertain output state.
        for output_id in self.output_indices:
            self.simulator.h(output_id)

        for neuron_wrapper in self.neurons:
            self.apply_fn(neuron_wrapper.neuron)

        b = 0
        q = 0
        y = [([0.0] * len(self.output_indices)) for _ in range(B)]
        for output_id in self.output_indices:
            y[b][q] = self.simulator.prob(output_id)
            b += 1
            if b == B:
                q += 1
                b = 0

        return (
            torch.tensor(y, dtype=torch.float32, device=x.device, requires_grad=True)
            if _IS_TORCH_AVAILABLE
            else y
        )

class QrackNeuronTorchLayerFunction(Function if _IS_TORCH_AVAILABLE else object):
    """Static forward/backward/apply functions for QrackTorchNeuron"""

    @staticmethod
    def forward(ctx, neuron_layer):
        # Save for backward
        ctx.neuron_layer = neuron_layer

        init_probs = [neuron_layer.simulator.prob(target) for target in neuron_layer.output_indices]
        neuron_layer.forward()
        final_probs = [neuron_layer.simulator.prob(target) for target in neuron_layer.output_indices]
        ctx.delta = [(f - i) for i, f in zip(init_probs, final_probs)]

        return (
            torch.tensor(ctx.delta, dtype=torch.float32, requires_grad=True)
            if _IS_TORCH_AVAILABLE
            else ctx.delta
        )

    @staticmethod
    def backward(ctx, x):
        neuron_layer = ctx.neuron_layer

        final_probs = [neuron_layer.simulator.prob(target) for target in neuron_layer.output_indices]

        # Uncompute prediction
        for neuron_wrapper in neuron_layer.neurons:
            neuron_layer.backward_fn(neuron_wrapper.neuron)

        # Uncompute output state prep
        for output_id in neuron_layer.output_indices:
            neuron_layer.simulator.h(output_id)

        init_probs = [neuron_layer.simulator.prob(target) for target in neuron_layer.output_indices]

        grad = [0.0] * len(init_probs)
        for idx in range(len(init_probs)):
            grad[idx] = (final_probs[idx] - init_probs[idx]) - ctx.delta[idx]

        return (
            torch.tensor(grad, dtype=torch.float32, requires_grad=True)
            if _IS_TORCH_AVAILABLE
            else grad
        )
