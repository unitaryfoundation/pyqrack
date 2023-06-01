import random
import math
from collections import Counter
from typing import List, Dict

import pytest

from pyqrack import QrackCircuit, QrackSimulator

X = [0, 1, 1, 0]
Y = [0, -1j, 1j, 0]
Z = [1, 0, 0, -1]
SQRT1_2 = 1 / math.sqrt(2)


def mc_gate(circ: QrackCircuit, c: List[int], mat: List[complex], q: int, p: int):
    """add a Multi-controlled mat gate, controlled on qubits c, targeting qubit q"""
    circ.ucmtrx([c], mat, q, p)


def gen_random_1q_gates(n_qubits, gate_count_1qb, depth):
    return [
        [random.randint(0, gate_count_1qb) for _ in range(n_qubits)]
        for _ in range(depth)
    ]


def gen_random_multiq_gates(n_qubits, gate_count_2qb, gate_count_multiqb, depth):
    unused_bits = list(range(n_qubits))
    gates = []
    for _ in range(depth):
        layer = []
        while len(unused_bits) > 1:
            b1 = random.choice(unused_bits)
            unused_bits.remove(b1)
            b2 = random.choice(unused_bits)
            unused_bits.remove(b2)

            max_gates = gate_count_multiqb if len(unused_bits) > 0 else gate_count_2qb

            gate = random.randint(0, max_gates)

            if gate > gate_count_2qb:
                b3 = random.choice(unused_bits)
                unused_bits.remove(b3)
            else:
                b3 = 0
            layer.append({"gate": gate, "b1": b1, "b2": b2, "b3": b3})
        gates.append(layer)
    return gates


def random_bit_string(n_qubits):
    return random.randint(0, (1 << n_qubits) - 1)


def single_qubit_gates(n):
    if n == 0:
        return [SQRT1_2, SQRT1_2, SQRT1_2, -SQRT1_2]
    elif n == 1:
        return X
    elif n == 2:
        return Y
    elif n == 3:
        return Z
    elif n == 4:
        return [1, 0, 0, 1j]
    elif n == 5:
        return [1, 0, 0, -1j]
    elif n == 6:
        return [1, 0, 0, SQRT1_2 * (1 + 1j)]
    else:
        return [1, 0, 0, SQRT1_2 * (1 - 1j)]


def mirrored_single_qubit_gate(sim: QrackSimulator, gate, q):
    if gate == 0:
        sim.h(q)
    elif gate == 1:
        sim.x(q)
    elif gate == 2:
        sim.y(q)
    elif gate == 3:
        sim.z(q)
    elif gate == 4:
        sim.adjs(q)
    elif gate == 5:
        sim.s(q)
    elif gate == 6:
        sim.adjt(q)
    else:
        sim.t(q)


def multi_qubit_gates(circuit, gate, b1, b2, b3):
    control = [b1]
    controls = [b1, b2]
    if gate == 0:
        circuit.swap(b1, b2)
    elif gate == 1:
        circuit.ucmtrx(control, [0, 1, 1, 0], b2, 1)
    elif gate == 2:
        circuit.ucmtrx(control, [0, -1j, 1j, 0], b2, 1)
    elif gate == 3:
        circuit.ucmtrx(control, [1, 0, 0, -1], b2, 1)
    elif gate == 4:
        circuit.ucmtrx(control, [0, 1, 1, 0], b2, 0)
    elif gate == 5:
        circuit.ucmtrx(control, [0, -1j, 1j, 0], b2, 0)
    elif gate == 6:
        circuit.ucmtrx(control, [1, 0, 0, -1], b2, 0)
    elif gate == 7:
        circuit.ucmtrx(controls, [0, 1, 1, 0], b3, 3)
    elif gate == 8:
        circuit.ucmtrx(controls, [0, -1j, 1j, 0], b3, 3)
    elif gate == 9:
        circuit.ucmtrx(controls, [1, 0, 0, -1], b3, 3)
    elif gate == 10:
        circuit.ucmtrx(controls, [0, 1, 1, 0], b3, 0)
    elif gate == 11:
        circuit.ucmtrx(controls, [0, -1j, 1j, 0], b3, 0)
    else:
        circuit.ucmtrx(controls, [1, 0, 0, -1], b3, 0)


def mirrored_multi_qubit_gates(sim: QrackSimulator, gate, b1, b2, b3):
    control = [b1]
    controls = [b1, b2]
    if gate == 0:
        sim.swap(b1, b2)
    elif gate == 1:
        sim.mcx(control, b2)
    elif gate == 2:
        sim.mcy(control, b2)
    elif gate == 3:
        sim.mcz(control, b2)
    elif gate == 4:
        sim.macx(control, b2)
    elif gate == 5:
        sim.macy(control, b2)
    elif gate == 6:
        sim.macz(control, b2)
    elif gate == 7:
        sim.mcx(controls, b3)
    elif gate == 8:
        sim.mcy(controls, b3)
    elif gate == 9:
        sim.mcz(controls, b3)
    elif gate == 10:
        sim.macx(controls, b3)
    elif gate == 11:
        sim.macy(controls, b3)
    else:
        sim.macz(controls, b3)


def mirror_circuit(
    qsim: QrackSimulator,
    n_qubits: int,
    initial_bitstr: int,
    depth: int,
    random_1q_gates: List[List[int]],
    random_multiq_gates: List[List[Dict[str, int]]],
    shots: int,
):
    circuit = QrackCircuit()

    # State preparation
    for i in range(n_qubits):
        if (initial_bitstr >> i) & 1 == 1:
            circuit.mtrx(X, i)

    # Forward pass
    for d in range(depth):
        random_1q_layer = random_1q_gates[d]
        for i in range(len(random_1q_layer)):
            circuit.mtrx(single_qubit_gates(random_1q_layer[i]), i)

        random_mq_layer = random_multiq_gates[d]
        for i in range(len(random_mq_layer)):
            gate, b1, b2, b3 = random_mq_layer[i].values()
            multi_qubit_gates(circuit, gate, b1, b2, b3)

    circuit.run(qsim)

    # Reversing the circuit
    for d in reversed(range(depth)):
        random_mq_layer = random_multiq_gates[d]
        for i in reversed(range(len(random_mq_layer))):
            gate, b1, b2, b3 = random_mq_layer[i].values()
            mirrored_multi_qubit_gates(qsim, gate, b1, b2, b3)

        random_1q_layer = random_1q_gates[d]
        for i in reversed(range(len(random_1q_layer))):
            mirrored_single_qubit_gate(qsim, random_1q_layer[i], i)

    return dict(Counter(qsim.measure_shots(list(range(n_qubits)), shots)))


@pytest.fixture()
def trials():
    return 100


@pytest.fixture()
def n_shots():
    return 100


@pytest.fixture()
def n_qubits():
    return 12


@pytest.fixture()
def depth():
    return 12


@pytest.fixture()
def gate_count_1qb():
    return 7


@pytest.fixture()
def gate_count_multiqb():
    return 12


@pytest.fixture()
def gate_count_2qb():
    return 6


class TestMirrorCircuits:
    def test_mirror_circuits(
        self,
        depth: int,
        trials: int,
        gate_count_1qb: int,
        n_qubits: int,
        gate_count_multiqb: int,
        gate_count_2qb: int,
        n_shots: int,
    ):
        for _ in range(trials):
            simulator = QrackSimulator(n_qubits)
            bit_string = random_bit_string(n_qubits)
            results = mirror_circuit(
                simulator,
                n_qubits,
                bit_string,
                depth,
                gen_random_1q_gates(n_qubits, gate_count_1qb, depth),
                gen_random_multiq_gates(
                    n_qubits, gate_count_2qb, gate_count_multiqb, depth
                ),
                n_shots,
            )
            assert all(k == bit_string for k in results.keys())
