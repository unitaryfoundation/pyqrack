# (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
#
# Initial draft by Elara (OpenAI custom GPT)
# Refined and architecturally clarified by Dan Strano
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import itertools
import math
import random
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
param_shift_eps = math.pi / 2
# Neuron angle initialization
init_phi = math.asin(0.5)


class QrackNeuronTorchFunction(Function if _IS_TORCH_AVAILABLE else object):
    """Static forward/backward/apply functions for QrackNeuronTorch"""

    @staticmethod
    def forward(ctx, x, neuron):
        ctx.neuron = neuron
        ctx.simulator = neuron.simulator
        ctx.save_for_backward(x)

        # Baseline probability BEFORE applying this neuron's unitary
        pre_prob = neuron.simulator.prob(neuron.target)

        angles = x.detach().cpu().numpy() if x.requires_grad else x.numpy()
        neuron.set_angles(angles)
        neuron.predict(True, False)

        # Probability AFTER applying this neuron's unitary
        post_prob = neuron.simulator.prob(neuron.target)
        ctx.post_prob = post_prob

        delta = math.asin(post_prob) - math.asin(pre_prob)
        ctx.delta = delta

        # Return shape: (1,)
        return x.new_tensor([delta])

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        neuron = ctx.neuron
        neuron.set_simulator(ctx.simulator)
        post_prob = ctx.post_prob

        angles = x.detach().cpu().numpy() if x.requires_grad else x.numpy()

        # Restore simulator to state BEFORE this neuron's unitary
        neuron.set_angles(angles)
        neuron.unpredict()
        pre_sim = neuron.simulator

        grad_x = torch.zeros_like(x)

        for i in range(x.shape[0]):
            angle = angles[i]

            # θ + π/2
            angles[i] = angle + param_shift_eps
            neuron.set_angles(angles)
            neuron.simulator = pre_sim.clone()
            neuron.predict(True, False)
            p_plus = neuron.simulator.prob(neuron.target)

            # θ − π/2
            angles[i] = angle - param_shift_eps
            neuron.set_angles(angles)
            neuron.simulator = pre_sim.clone()
            neuron.predict(True, False)
            p_minus = neuron.simulator.prob(neuron.target)

            # Parameter-shift gradient
            grad_x[i] = 0.5 * (p_plus - p_minus)

            angles[i] = angle

        # Restore simulator
        neuron.set_simulator(pre_sim)

        # Apply chain rule and upstream gradient
        grad_x *= grad_output[0] / math.sqrt(max(1.0 - post_prob * post_prob, 1e-6))

        return grad_x, None


class QrackNeuronTorch(nn.Module if _IS_TORCH_AVAILABLE else object):
    """Torch wrapper for QrackNeuron

    Attributes:
        neuron(QrackNeuron): QrackNeuron backing this torch wrapper
    """

    def __init__(self, neuron, x):
        super().__init__()
        self.neuron = neuron
        self.weights = nn.Parameter(x)

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
        dtype=torch.float if _IS_TORCH_AVAILABLE else float,
        parameters=None,
        **kwargs
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
        self.simulator = QrackSimulator(input_qubits + hidden_qubits + output_qubits, **kwargs)
        self.simulators = []
        self.input_indices = list(range(input_qubits))
        self.hidden_indices = list(range(input_qubits, input_qubits + hidden_qubits))
        self.output_indices = list(
            range(input_qubits + hidden_qubits, input_qubits + hidden_qubits + output_qubits)
        )
        self.activation = NeuronActivationFn(activation)
        self.dtype = dtype
        self.apply_fn = QrackNeuronTorchFunction.apply

        # Create neurons from all input combinations, projecting to coherent output qubits
        neurons = []
        param_count = 0
        for output_id in self.output_indices:
            for k in range(lowest_combo_count, highest_combo_count + 1):
                for input_subset in itertools.combinations(self.input_indices, k):
                    p_count = 1 << len(input_subset)
                    angles = (
                        (
                            torch.tensor(
                                parameters[param_count : (param_count + p_count)], dtype=dtype
                            )
                            if parameters
                            else torch.zeros(p_count, dtype=dtype)
                        )
                    )
                    neurons.append(
                        QrackNeuronTorch(
                            QrackNeuron(self.simulator, input_subset, output_id, activation), angles
                        )
                    )
                    param_count += p_count
        self.neurons = nn.ModuleList(neurons)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)

        self.simulators.clear()

        self.simulator.reset_all()
        # Prepare hidden predictors
        for hidden_id in self.hidden_indices:
            self.simulator.h(hidden_id)
        # Prepare a maximally uncertain output state.
        for output_id in self.output_indices:
            self.simulator.h(output_id)

        # Group neurons by output target once
        by_out = {out: [] for out in self.output_indices}
        for neuron_wrapper in self.neurons:
            by_out[neuron_wrapper.neuron.target].append(neuron_wrapper)

        batch_rows = []
        for b in range(B):
            simulator = self.simulator.clone()
            self.simulators.append(simulator)

            for q, input_id in enumerate(self.input_indices):
                simulator.r(Pauli.PauliY, math.pi * x[b, q].item(), input_id)

            row = []
            for out in self.output_indices:
                phi = torch.tensor(init_phi, device=x.device, dtype=x.dtype)

                for neuron_wrapper in by_out[out]:
                    neuron_wrapper.neuron.set_simulator(simulator)
                    phi += self.apply_fn(
                        neuron_wrapper.weights,
                        neuron_wrapper.neuron
                    ).squeeze()

                # Convert angle back to probability
                p = torch.clamp(torch.sin(phi), min=0.0)
                row.append(p)

            batch_rows.append(torch.stack(row))

        return torch.stack(batch_rows)
