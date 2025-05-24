# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Initial draft by Elara (OpenAI custom GPT)
# Refined and architecturally clarified by Dan Strano
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

_IS_TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    from torch.autograd import Function
except ImportError:
    _IS_TORCH_AVAILABLE = False

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
            torch.tensor([ctx.delta], dtype=torch.float32)
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
            torch.tensor([grad], dtype=torch.float32) if _IS_TORCH_AVAILABLE else grad
        )


class QrackNeuronTorchLayer(nn.Module if _IS_TORCH_AVAILABLE else object):
    """Torch layer wrapper for QrackNeuron (with power set of neurons between inputs and outputs)"""

    def __init__(
        self,
        simulator,
        input_indices,
        output_indices,
        activation=int(NeuronActivationFn.Generalized_Logistic),
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
            parameters (list[float]): (Optional) Flat list of initial neuron parameters, corresponding to little-endian basis states of power set of input indices, repeated for each output index (with empty set being constant bias)
        """
        super(QrackNeuronTorchLayer, self).__init__()
        self.simulator = simulator
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.activation = NeuronActivationFn(activation)
        self.fn = (
            QrackNeuronFunction.apply
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

        # Set Qrack's internal parameters:
        param_count = 0
        for neuron_wrapper in self.neurons:
            neuron = neuron_wrapper.neuron
            p_count = 1 << len(neuron.controls)
            neuron.set_angles(
                parameters[param_count : (param_count + p_count + 1)]
                if parameters
                else ([0.0] * p_count)
            )
            param_count += p_count

        self.weights = nn.ParameterList()
        for pid in range(param_count):
            self.weights.append(
                nn.Parameter(torch.tensor(parameters[pid] if parameters else 0.0))
            )

    def forward(self, _):
        # Assume quantum outputs should overwrite the simulator state
        for output_id in self.output_indices:
            if self.simulator.m(output_id):
                self.simulator.x(output_id)
            self.simulator.h(output_id)

        # Set Qrack's internal parameters:
        param_count = 0
        for neuron_wrapper in self.neurons:
            neuron = neuron_wrapper.neuron
            p_count = 1 << len(neuron.controls)
            angles = [
                w.item() for w in self.weights[param_count : (param_count + p_count)]
            ]
            neuron.set_angles(angles)
            param_count += p_count

        # Assume quantum inputs already loaded into simulator state
        for neuron_wrapper in self.neurons:
            self.fn(neuron_wrapper.neuron)

        # These are classical views over quantum state; simulator still maintains full coherence
        outputs = [self.simulator.prob(output_id) for output_id in self.output_indices]

        return torch.tensor(outputs, dtype=torch.float32)
