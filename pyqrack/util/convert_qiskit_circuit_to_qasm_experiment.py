_IS_QISKIT_AVAILABLE = True
try:
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from qiskit.qobj.qasm_qobj import QasmQobjExperiment, QasmQobjInstruction
except ImportError:
    _IS_QISKIT_AVAILABLE = False


class QrackQasmQobjInstructionConditional:
    def __init__(self, mask, val):
        self.mask = mask
        self.val = val


def convert_qiskit_circuit_to_qasm_experiment(experiment, config=None, header=None):
    if not _IS_QISKIT_AVAILABLE:
        raise RuntimeError(
            "Before trying to convert_circuit_to_qasm_experiment() with QrackSimulator, you must install Qiskit!"
        )

    instructions = []
    for datum in experiment._data:
        qubits = []
        for qubit in datum[1]:
            qubits.append(experiment.qubits.index(qubit))

        clbits = []
        for clbit in datum[2]:
            clbits.append(experiment.clbits.index(clbit))

        conditional = None
        condition = datum[0].condition
        if condition is not None:
            if isinstance(condition[0], Clbit):
                conditional = experiment.clbits.index(condition[0])
            else:
                creg_index = experiment.cregs.index(condition[0])
                size = experiment.cregs[creg_index].size
                offset = 0
                for i in range(creg_index):
                    offset += len(experiment.cregs[i])
                mask = ((1 << offset) - 1) ^ ((1 << (offset + size)) - 1)
                val = condition[1]
                conditional = (
                    offset
                    if (size == 1)
                    else QrackQasmQobjInstructionConditional(mask, val)
                )

        instructions.append(
            QasmQobjInstruction(
                datum[0].name,
                qubits=qubits,
                memory=clbits,
                condition=condition,
                conditional=conditional,
                params=datum[0].params,
            )
        )

    return QasmQobjExperiment(config=config, header=header, instructions=instructions)
