# pyqrack
[![Downloads](https://pepy.tech/badge/pyqrack-cpu)](https://pepy.tech/project/pyqrack-cpu) [![Downloads](https://pepy.tech/badge/pyqrack-cpu/month)](https://pepy.tech/project/pyqrack-cpu) [![Downloads](https://static.pepy.tech/badge/pyqrack-cpu/week)](https://pepy.tech/project/pyqrack-cpu)

Pure Python bindings for the pure C++11 Qrack quantum computer simulator library

(**PyQrack** is just pure Qrack.)

This is the **purely CPU-based** variant of PyQrack (with module name `pyqrack-cpu`). This alternate build of PyQrack is provided so that systems that won't use OpenCL or CUDA anyway (such as because they lack any GPUs or other accelerators like the Intel HD) do not need to install the OpenCL library in order to use PyQrack. In this form, both the PyQrack module and the underlying C++ Qrack library are entirely external **dependency-free.**

**If you're looking for Mac ARM support, use the package `pyqrack`, not `pyqrack-cpu`.** Mac officially "deprecated" OpenCL years ago. Hence, accelerator support is not included in ARM-based Mac wheels, and OpenCL installation is **not** required on these systems, but, if you have a CUDA accelerator on ARM-based Mac, you could try the package `pyqrack-cuda` instead.

(**If you installing from source,** it doesn't matter whether you use `pyqrack` or `pyqrack-cpu`, because the build version of C++ Qrack you intend to use must already be installed locally.)

**If you're looking for Mac ARM support, use the package `pyqrack`, not `pyqrack-cpu`.** Mac officially "deprecated" OpenCL years ago. Hence, accelerator support is not included in ARM-based Mac wheels, and OpenCL installation is **not** required on these systems, but, if you have a CUDA accelerator on ARM-based Mac, you could try the package `pyqrack-cuda` instead.

**Performance can benefit greatly from following the [Qrack repository "Quick Start" and "Power user considerations."](https://github.com/unitaryfund/qrack/blob/main/README.md#quick-start)**

Import and instantiate [`QrackSimulator`](https://github.com/unitaryfund/pyqrack/blob/main/pyqrack/qrack_simulator.py) instances. This simulator can perform arbitrary single qubit and controlled-single-qubit gates, as well as other specific gates like `SWAP`.

Any 2x2 bit operator matrix is represented by a list of 4 `complex` floating point numbers, in [**row-major order**](https://en.wikipedia.org/wiki/Row-_and_column-major_order).

Single and array "`b`" parameters represent [**Pauli operator bases**](https://en.wikipedia.org/wiki/Pauli_matrices). They are specified according to the enumeration of the [`Pauli`](https://github.com/unitaryfund/pyqrack/blob/main/pyqrack/pauli.py) class.

`MC[x]` and `MAC[x]` methods are controlled single bit gates, with as many control qubits as you specify via Python list `c` argument. `MCX` is multiply-controlled Pauli X, and `MACX` is "anti-"controlled Pauli X, i.e. "anti-control" activates the gate if all control bits are specifically **off**, as opposed to **on**.

To load the required **unitaryfund/qrack** libraries from a different location, set the `PYQRACK_SHARED_LIB_PATH` environment variable.

PyQrack has experimental support for [PyZX](https://github.com/Quantomatic/pyzx) `Circuit` definitions as an intermediate representation for `QrackSimulator`. To try this, load a `Circuit` in PyZX, (use that module to optimize your circuit, as you like,) and create a `QrackSimulator()` instance using the `pyzxCircuit` named argument of the constructor, like so:

```python
sim = QrackSimulator(pyzxCircuit=c)
```

where `c` is a PyZX circuit object. The circuit will automatically be simulated in the constructed `QrackSimulator` instance. This also allows loading from QASM and other intermediate representations supported by PyZX.

See [https://pyqrack.readthedocs.io/en/latest/](https://pyqrack.readthedocs.io/en/latest/) for an API reference.

For custom Qrack build floating-point precision, where options are `half`, `float`, `double`, and `quad`, set an environment variable via `export QRACK_FPPOW=[n]` (or as appropriate to your shell) where `[n]` is the logarithm base 2 of the number of bits in the systemic floating point type (`4`, `5`, `6`, or `7`, with `5` or `float` as default, i.e. `2**5=32` for 32-bit `float`). Your Qrack installation floating-point build option must match this specific value, which might require a custom Qrack build.

Please feel welcome to open an issue, if you'd like help. 😃

**Special thanks go to Zeeshan Ahmed, for bug fixes and design suggestions, Ashish Panigrahi, for documentation and design suggestions, WingCode, for documentation, and to the broader community of Qrack contributors, for years of happy Qracking! You rock!**
