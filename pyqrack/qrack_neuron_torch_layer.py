# Initial draft by Elara (OpenAI custom GPT)
# Refined and architecturally clarified by Dan Strano

_IS_TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    from torch.autograd import Function
except ImportError:
    _IS_TORCH_AVAILABLE = False

from .qrack_simulator import QrackSimulator
from .qrack_neuron import QrackNeuron
from .neuron_activation_fn import NeuronActivationFn

from itertools import chain, combinations


# From https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset#answer-1482316
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3,) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class QrackTorchNeuron(nn.Module):
    def __init__(self, neuron: QrackNeuron) -> None:
        super().__init__()
        self.neuron = neuron

    def forward(self, x):
        neuron = self.neuron
        neuron.predict(True, False)

        return neuron.simulator.prob(neuron.target)


class QrackNeuronFunction(Function if _IS_TORCH_AVAILABLE else object):
    @staticmethod
    def forward(ctx, neuron: object):
        # Save for backward
        ctx.neuron = neuron

        init_prob = neuron.simulator.prob(neuron.target)
        neuron.predict(True, False)
        final_prob = neuron.simulator.prob(neuron.target)
        ctx.delta = final_prob - init_prob

        return torch.tensor([ctx.delta], dtype=torch.float32) if _IS_TORCH_AVAILABLE else ctx.delta

    @staticmethod
    def backward(ctx, grad_output):
        neuron = ctx.neuron

        pre_unpredict = neuron.simulator.prob(neuron.output_id)
        neuron.unpredict()
        post_unpredict = neuron.simulator.prob(neuron.output_id)
        reverse_delta = pre_unpredict - post_unpredict

        grad = reverse_delta - ctx.delta

        return torch.tensor([grad], dtype=torch.float32) if _IS_TORCH_AVAILABLE else grad


class QrackNeuronTorchLayer(nn.Module if _IS_TORCH_AVAILABLE else object):
    def __init__(self, simulator: object, input_indices: list[int], output_size: int,
                 activation: int = int(NeuronActivationFn.Generalized_Logistic)):
        super(QrackNeuronTorchLayer, self).__init__()
        self.simulator = simulator
        self.input_indices = input_indices
        self.output_size = output_size
        self.activation = NeuronActivationFn(activation)
        self.fn = QrackNeuronFunction.apply if _IS_TORCH_AVAILABLE else lambda x: QrackNeuronFunction.forward(object(), x)

        # Create neurons from all powerset input combinations, projecting to coherent output qubits
        self.neurons = nn.ModuleList([
            QrackTorchNeuron(QrackNeuron(simulator, list(input_subset), len(input_indices) + output_id, activation))
            for input_subset in powerset(input_indices)
            for output_id in range(output_size)
        ])

    def forward(self, _):
        # Assume quantum outputs should overwrite the simulator state
        for output_id in range(self.output_size):
            if self.simulator.m(len(self.input_indices) + output_id):
                self.simulator.x(output_id)
            self.simulator.h(output_id)

        # Assume quantum inputs already loaded into simulator state
        for neuron_wrapper in self.neurons:
            self.fn(neuron_wrapper.neuron)

        # These are classical views over quantum state; simulator still maintains full coherence
        outputs = [
            self.simulator.prob(len(self.input_indices) + output_id)
            for output_id in range(self.output_size)
        ]

        return torch.tensor(outputs, dtype=torch.float32) if _IS_TORCH_AVAILABLE else outputs

