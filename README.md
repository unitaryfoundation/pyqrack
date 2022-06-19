# pyqrack
[![Downloads](https://pepy.tech/badge/pyqrack)](https://pepy.tech/project/pyqrack) [![Downloads](https://pepy.tech/badge/pyqrack/month)](https://pepy.tech/project/pyqrack)

Pure Python bindings for the pure C++11/OpenCL Qrack quantum computer simulator library

(**PyQrack** is just pure Qrack.)

**IMPORTANT**: You must build and install [vm6502q/qrack](https://github.com/vm6502q/qrack) to use this `main` branch. The `pypi_package` branch, however, comes with pre-compiled Qrack binaries, and that is the form published on PyPi.

Import and instantiate [`QrackSimulator`](https://github.com/vm6502q/pyqrack/blob/main/pyqrack/qrack_simulator.py) instances. This simulator can perform arbitrary single qubit and controlled-single-qubit gates, as well as other specific gates like `SWAP`.

Any 2x2 bit operator matrix is represented by a list of 4 `complex` floating point numbers, in [**row-major order**](https://en.wikipedia.org/wiki/Row-_and_column-major_order).

Single and array "`b`" parameters represent [**Pauli operator bases**](https://en.wikipedia.org/wiki/Pauli_matrices). They are specified according to the enumeration of the [`Pauli`](https://github.com/vm6502q/pyqrack/blob/main/pyqrack/pauli.py) class.

`MC[x]` and `MAC[x]` methods are controlled single bit gates, with as many control qubits as you specify via Python list `c` argument. `MCX` is multiply-controlled Pauli X, and `MACX` is "anti-"controlled Pauli X, i.e. "anti-control" activates the gate if all control bits are specifically **off**, as opposed to **on**.

To load the required **vm6502q/qrack** libraries from a different location, set the `PYQRACK_SHARED_LIB_PATH` environment variable.

PyQrack v0.4.6 adds experimental support for [PyZX](https://github.com/Quantomatic/pyzx) `Circuit` definitions as an intermediate representation for `QrackSimulator`. To try this, load a `Circuit` in PyZX, (use that module to optimize your circuit, as you like,) and create a `QrackSimulator()` instance using the `pyzxCircuit` named argument of the constructor, like so:

```python
sim = QrackSimulator(pyzxCircuit=c)
```

where `c` is a PyZX circuit object. The circuit will automatically be simulated in the constructed `QrackSimulator` instance. This also allows loading from QASM and other intermediate representations supported by PyZX.

See [https://pyqrack.readthedocs.io/en/latest/](https://pyqrack.readthedocs.io/en/latest/) for an API reference.

Please feel welcome to open an issue, if you'd like help. 😃

**Special thanks go to Zeeshan Ahmed, for bug fixes and design suggestions, Ashish Panigrahi, for documentation and design suggestions, WingCode, for documentation, and to the broader community of Qrack contributors, for years of happy Qracking! You rock!**
