# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import ctypes

from .qrack_system import Qrack
from .quimb_circuit_type import QuimbCircuitType

_IS_QISKIT_AVAILABLE = True
try:
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from qiskit.compiler.transpiler import transpile
    from qiskit.circuit.library import UCGate
    import numpy as np
    import math
    import sys
except ImportError:
    _IS_QISKIT_AVAILABLE = False

_IS_QUIMB_AVAILABLE = True
try:
    import quimb as qu
    import quimb.tensor as qtn
except ImportError:
    _IS_QUIMB_AVAILABLE = False

_IS_TENSORCIRCUIT_AVAILABLE = True
try:
    import tensorcircuit as tc
except ImportError:
    _IS_TENSORCIRCUIT_AVAILABLE = False


class QrackCircuit:
    """Class that exposes the QCircuit class of Qrack

    QrackCircuit allows the user to specify a unitary circuit, before running it.
    Upon running the state, the result is a QrackSimulator state. Currently,
    measurement is not supported, but measurement can be run on the resultant
    QrackSimulator.

    Attributes:
        cid(int): Qrack ID of this circuit
    """

    def __init__(
        self,
        is_collapse=True,
        is_near_clifford=False,
        clone_cid=-1,
        is_inverse=False,
        past_light_cone=[],
    ):
        if clone_cid < 0:
            self.cid = Qrack.qrack_lib.init_qcircuit(is_collapse, is_near_clifford)
        elif is_inverse:
            self.cid = Qrack.qrack_lib.qcircuit_inverse(clone_cid)
        elif len(past_light_cone) > 0:
            self.cid = Qrack.qrack_lib.qcircuit_past_light_cone(
                clone_cid, len(past_light_cone), self._ulonglong_byref(past_light_cone)
            )
        else:
            self.cid = Qrack.qrack_lib.init_qcircuit_clone(clone_cid)

    def __del__(self):
        if self.cid is not None:
            Qrack.qrack_lib.destroy_qcircuit(self.cid)
            self.cid = None

    def _ulonglong_byref(self, a):
        return (ctypes.c_ulonglong * len(a))(*a)

    def _double_byref(self, a):
        return (ctypes.c_double * len(a))(*a)

    def _complex_byref(self, a):
        t = [(c.real, c.imag) for c in a]
        return self._double_byref([float(item) for sublist in t for item in sublist])

    def _mtrx_to_u4(m):
        nrm = abs(m[0])
        if (nrm * nrm) < sys.float_info.epsilon:
            phase = 1.0 + 0.0j
            th = math.pi
        else:
            phase = m[0] / nrm
            if nrm > 1.0:
                nrm = 1.0
            th = 2 * math.acos(nrm)

        nrm1 = abs(m[1])
        nrm2 = abs(m[2])
        nrm1 *= nrm1
        nrm2 *= nrm2
        if (nrm1 < sys.float_info.epsilon) or (nrm2 < sys.float_info.epsilon):
            ph = np.angle(m[3] / phase)
            lm = 0.0
        else:
            ph = np.angle(m[2] / phase)
            lm = np.angle(-m[1] / phase)

        return th, ph, lm, np.angle(phase)

    def _u3_to_mtrx(params):
        th = float(params[0])
        ph = float(params[1])
        lm = float(params[2])

        c = math.cos(th / 2)
        s = math.sin(th / 2)
        el = np.exp(1j * lm)
        ep = np.exp(1j * ph)

        return [c + 0j, -el * s, ep * s, ep * el * c]

    def _u4_to_mtrx(params):
        m = QrackCircuit._u3_to_mtrx(params)
        g = np.exp(1j * float(params[3]))
        for i in range(4):
            m[i] *= g

        return m

    def _make_mtrx_unitary(m):
        return QrackCircuit._u4_to_mtrx(QrackCircuit._mtrx_to_u4(m))

    def clone(self):
        """Make a new circuit that is an exact clone of this circuit

        Raises:
            RuntimeError: QrackCircuit C++ library raised an exception.
        """
        return QrackCircuit(clone_cid=self.cid, is_inverse=False)

    def inverse(self):
        """Make a new circuit that is the exact inverse of this circuit

        Raises:
            RuntimeError: QrackCircuit C++ library raised an exception.
        """
        return QrackCircuit(clone_cid=self.cid, is_inverse=True)

    def past_light_cone(self, q):
        """Make a new circuit with just this circuits' past light cone for certain qubits.

        Args:
            q: list of qubit indices to include at beginning of past light cone

        Raises:
            RuntimeError: QrackCircuit C++ library raised an exception.
        """
        return QrackCircuit(clone_cid=self.cid, is_inverse=False, past_light_cone=q)

    def get_qubit_count(self):
        """Get count of qubits in circuit

        Raises:
            RuntimeError: QrackCircuit C++ library raised an exception.
        """
        return Qrack.qrack_lib.get_qcircuit_qubit_count(self.cid)

    def swap(self, q1, q2):
        """Add a 'Swap' gate to the circuit

        Args:
            q1: qubit index #1
            q2: qubit index #2

        Raises:
            RuntimeError: QrackCircuit C++ library raised an exception.
        """
        Qrack.qrack_lib.qcircuit_swap(self.cid, q1, q2)

    def mtrx(self, m, q):
        """Operation from matrix.

        Applies arbitrary operation defined by the given matrix.

        Args:
            m: row-major complex list representing the operator.
            q: the qubit number on which the gate is applied to.

        Raises:
            ValueError: 2x2 matrix 'm' in QrackCircuit.mtrx() must contain at least 4 elements.
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(m) < 4:
            raise ValueError(
                "2x2 matrix 'm' in QrackCircuit.mtrx() must contain at least 4 elements."
            )
        Qrack.qrack_lib.qcircuit_append_1qb(self.cid, self._complex_byref(m), q)

    def ucmtrx(self, c, m, q, p):
        """Multi-controlled single-target-qubit gate

        Specify a controlled gate by its control qubits, its single-qubit
        matrix "payload," the target qubit, and the permutation of qubits
        that activates the gate.

        Args:
            c: list of controlled qubits
            m: row-major complex list representing the operator.
            q: target qubit
            p: permutation of target qubits

        Raises:
            ValueError: 2x2 matrix 'm' in QrackCircuit.ucmtrx() must contain at least 4 elements.
            RuntimeError: QrackSimulator raised an exception.
        """
        if len(m) < 4:
            raise ValueError(
                "2x2 matrix 'm' in QrackCircuit.ucmtrx() must contain at least 4 elements."
            )
        Qrack.qrack_lib.qcircuit_append_mc(
            self.cid, self._complex_byref(m), len(c), self._ulonglong_byref(c), q, p
        )

    def run(self, qsim):
        """Run circuit on simulator

        Run the encoded circuit on a specific simulator. The
        result will remain in this simulator.

        Args:
            qsim: QrackSimulator on which to run circuit

        Raises:
            RuntimeError: QrackCircuit raised an exception.
        """
        qb_count = self.get_qubit_count()
        sim_qb_count = qsim.num_qubits()
        if sim_qb_count < qb_count:
            for i in range(sim_qb_count, qb_count):
                qsim.allocate_qubit(i)
        Qrack.qrack_lib.qcircuit_run(self.cid, qsim.sid)
        qsim._throw_if_error()

    def out_to_file(self, filename):
        """Output optimized circuit to file

        Outputs the (optimized) circuit to a file named
        according to the "filename" parameter.

        Args:
            filename: Name of file
        """
        Qrack.qrack_lib.qcircuit_out_to_file(self.cid, filename.encode("utf-8"))

    def in_from_file(filename):
        """Read in optimized circuit from file

        Reads in an (optimized) circuit from a file named
        according to the "filename" parameter.

        Args:
            filename: Name of file
        """
        out = QrackCircuit()
        Qrack.qrack_lib.qcircuit_in_from_file(out.cid, filename.encode("utf-8"))

        return out

    def out_to_string(self):
        """Output optimized circuit to string

        Outputs the (optimized) circuit to a string.
        """
        string_length = Qrack.qrack_lib.qcircuit_out_to_string_length(self.cid)
        out = ctypes.create_string_buffer(string_length)
        Qrack.qrack_lib.qcircuit_out_to_string(self.cid, out)

        return out.value.decode("utf-8")

    def file_gate_count(filename):
        """File gate count

        Return the count of gates in a QrackCircuit file

        Args:
            filename: Name of file
        """
        tokens = []
        with open(filename, "r") as file:
            tokens = file.read().split()
        return int(tokens[1])

    def to_qiskit_circuit(self):
        """Convert to a Qiskit circuit

        Outputs a Qiskit circuit from a QrackCircuit.

        Raises:
            RuntimeErorr: Before trying to string_to_qiskit_circuit() with
                QrackCircuit, you must install Qiskit, numpy, and math!
        """
        if not _IS_QISKIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to_qiskit_circuit() with QrackCircuit, you must install Qiskit, numpy, and math!"
            )

        return QrackCircuit.string_to_qiskit_circuit(self.out_to_string())

    def file_to_qiskit_circuit(filename):
        """Convert an output file to a Qiskit circuit

        Reads in an (optimized) circuit from a file named
        according to the "filename" parameter and outputs
        a Qiskit circuit.

        Args:
            filename: Name of file

        Raises:
            RuntimeErorr: Before trying to file_to_qiskit_circuit() with
                QrackCircuit, you must install Qiskit, numpy, and math!
        """
        if not _IS_QISKIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to file_to_qiskit_circuit() with QrackCircuit, you must install Qiskit, numpy, and math!"
            )

        tokens = []
        with open(filename, "r") as file:
            return QrackCircuit.string_to_qiskit_circuit(file.read())

    def string_to_qiskit_circuit(circ_string):
        """Convert an output string to a Qiskit circuit

        Reads in an (optimized) circuit from a string
        parameter and outputs a Qiskit circuit.

        Args:
            circ_string: String representation of circuit

        Raises:
            RuntimeErorr: Before trying to string_to_qiskit_circuit() with
                QrackCircuit, you must install Qiskit, numpy, and math!
        """
        if not _IS_QISKIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to string_to_qiskit_circuit() with QrackCircuit, you must install Qiskit, numpy, and math!"
            )

        tokens = circ_string.split()

        i = 0
        num_qubits = int(tokens[i])
        i += 1
        circ = QuantumCircuit(num_qubits)

        num_gates = int(tokens[i])
        i += 1

        identity = np.eye(2, dtype=complex)
        for g in range(num_gates):
            target = int(tokens[i])
            i += 1

            control_count = int(tokens[i])
            i += 1
            controls = []
            for j in range(control_count):
                controls.append(int(tokens[i]))
                i += 1

            payload_count = int(tokens[i])
            i += 1
            payloads = {}
            for j in range(payload_count):
                key = int(tokens[i])
                i += 1

                mtrx = []
                for _ in range(4):
                    amp = tokens[i].replace("(", "").replace(")", "").split(",")
                    mtrx.append(float(amp[0]) + float(amp[1]) * 1j)
                    i += 1

                mtrx = QrackCircuit._make_mtrx_unitary(mtrx)

                op = np.eye(2, dtype=complex)
                op[0][0] = mtrx[0]
                op[0][1] = mtrx[1]
                op[1][0] = mtrx[2]
                op[1][1] = mtrx[3]

                payloads[key] = op

            gate_list = []
            for j in range(1 << control_count):
                if j in payloads:
                    gate_list.append(payloads[j])
                else:
                    gate_list.append(identity)
            circ.append(UCGate(gate_list), [target] + controls)

        return circ

    def in_from_qiskit_circuit(circ):
        """Read a Qiskit circuit into a QrackCircuit

        Reads in a circuit from a Qiskit `QuantumCircuit`

        Args:
            circ: Qiskit circuit

        Raises:
            RuntimeErorr: Before trying to file_to_qiskit_circuit() with
                QrackCircuit, you must install Qiskit, numpy, and math!
        """
        if not _IS_QISKIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to file_to_qiskit_circuit() with QrackCircuit, you must install Qiskit, numpy, and math!"
            )

        out = QrackCircuit()
        basis_gates = ["x", "y", "z", "u", "cx", "cy", "cz", "cu"]
        circ = transpile(circ, basis_gates=basis_gates, optimization_level=0)
        for gate in circ.data:
            o = gate.operation

            op = []
            if o.name in ["x", "cx"]:
                op = [0, 1, 1, 0]
            elif o.name in ["y", "cy"]:
                op = [0, -1j, 1j, 0]
            elif o.name in ["z", "cz"]:
                op = [1, 0, 0, -1]
            else:
                op = QrackCircuit._u3_to_mtrx(o.params)

            if o.name in ["x", "y", "z", "u"]:
                out.mtrx(op, circ.find_bit(gate.qubits[0])[0])
            else:
                out.ucmtrx(
                    [circ.find_bit(gate.qubits[0])[0]],
                    op,
                    circ.find_bit(gate.qubits[1])[0],
                    1,
                )

        return out

    def file_to_quimb_circuit(
        filename,
        circuit_type=QuimbCircuitType.Circuit,
        psi0=None,
        gate_opts=None,
        tags=None,
        psi0_dtype="complex128",
        psi0_tag="PSI0",
        bra_site_ind_id="b{}",
    ):
        """Convert an output file to a Quimb circuit

        Reads in an (optimized) circuit from a file named
        according to the "filename" parameter and outputs
        a Quimb circuit.

        Args:
            filename: Name of file
            circuit_type: "QuimbCircuitType" enum value specifying type of Quimb circuit
            psi0: The initial state, assumed to be |00000....0> if not given. The state is always copied and the tag PSI0 added
            gate_opts: Default keyword arguments to supply to each gate_TN_1D() call during the circuit
            tags: Tag(s) to add to the initial wavefunction tensors (whether these are propagated to the rest of the circuit’s tensors
            psi0_dtype: Ensure the initial state has this dtype.
            psi0_tag: Ensure the initial state has this tag.
            bra_site_ind_id: Use this to label ‘bra’ site indices when creating certain (mostly internal) intermediate tensor networks.

        Raises:
            RuntimeErorr: Before trying to file_to_quimb_circuit() with
                QrackCircuit, you must install quimb, Qiskit, numpy, and math!
        """
        if not _IS_QUIMB_AVAILABLE:
            raise RuntimeError(
                "Before trying to file_to_quimb_circuit() with QrackCircuit, you must install quimb, Qiskit, numpy, and math!"
            )

        qcirc = QrackCircuit.file_to_qiskit_circuit(filename)
        basis_gates = ["u", "cx"]
        qcirc = transpile(qcirc, basis_gates=basis_gates, optimization_level=3)

        tcirc = (
            qtn.Circuit(
                N=qcirc.num_qubits,
                psi0=psi0,
                gate_opts=gate_opts,
                tags=tags,
                psi0_dtype=psi0_dtype,
                psi0_tag=psi0_tag,
                bra_site_ind_id=bra_site_ind_id,
            )
            if circuit_type == QuimbCircuitType.Circuit
            else (
                qtn.CircuitDense(
                    N=qcirc.num_qubits, psi0=psi0, gate_opts=gate_opts, tags=tags
                )
                if circuit_type == QuimbCircuitType.CircuitDense
                else qtn.CircuitMPS(
                    N=qcirc.num_qubits,
                    psi0=psi0,
                    gate_opts=gate_opts,
                    tags=tags,
                    psi0_dtype=psi0_dtype,
                    psi0_tag=psi0_tag,
                    bra_site_ind_id=bra_site_ind_id,
                )
            )
        )
        for gate in qcirc.data:
            o = gate.operation
            if o.name == "u":
                th = float(o.params[0])
                ph = float(o.params[1])
                lm = float(o.params[2])

                tcirc.apply_gate("U3", th, ph, lm, qcirc.find_bit(gate.qubits[0])[0])
            else:
                tcirc.apply_gate(
                    "CNOT",
                    qcirc.find_bit(gate.qubits[0])[0],
                    qcirc.find_bit(gate.qubits[1])[0],
                )

        return tcirc

    def file_to_tensorcircuit(
        filename, inputs=None, circuit_params=None, binding_params=None
    ):
        """Convert an output file to a TensorCircuit circuit

        Reads in an (optimized) circuit from a file named
        according to the "filename" parameter and outputs
        a TensorCircuit circuit.

        Args:
            filename: Name of file
            inputs: pass-through to tensorcircuit.Circuit.from_qiskit
            circuit_params: pass-through to tensorcircuit.Circuit.from_qiskit
            binding_params: pass-through to tensorcircuit.Circuit.from_qiskit

        Raises:
            RuntimeErorr: Before trying to file_to_quimb_circuit() with
                QrackCircuit, you must install TensorCircuit, Qiskit, numpy, and math!
        """
        if not _IS_TENSORCIRCUIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to file_to_tensorcircuit() with QrackCircuit, you must install TensorCircuit, Qiskit, numpy, and math!"
            )

        qcirc = QrackCircuit.file_to_qiskit_circuit(filename)
        basis_gates = ["u", "cx"]
        qcirc = transpile(qcirc, basis_gates=basis_gates, optimization_level=3)

        return tc.Circuit.from_qiskit(
            qcirc, qcirc.num_qubits, inputs, circuit_params, binding_params
        )

    def in_from_tensorcircuit(tcirc, enable_instruction=False, enable_inputs=False):
        """Convert a TensorCircuit circuit to a QrackCircuit

        Accepts a TensorCircuit circuit and outputs an equivalent QrackCircuit

        Args:
            tcirc: TensorCircuit circuit
            enable_instruction: whether to also export measurement and reset instructions
            enable_inputs: whether to also export the inputs

        Raises:
            RuntimeErorr: Before trying to in_from_tensorcircuit() with
                QrackCircuit, you must install TensorCircuit, Qiskit, numpy, and math!
        """
        if not _IS_TENSORCIRCUIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to in_from_tensorcircuit() with QrackCircuit, you must install TensorCircuit, Qiskit, numpy, and math!"
            )

        # Convert from TensorCircuit to Qiskit
        qcirc = tcirc.to_qiskit(enable_instruction, enable_inputs)

        # Convert to QrackCircuit
        return QrackCircuit.in_from_qiskit_circuit(qcirc)
