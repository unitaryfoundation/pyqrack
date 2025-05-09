# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Pauli operators are specified for "b" (or "basis") parameters.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

from enum import IntEnum


class QuimbCircuitType(IntEnum):
    # Class for simulating quantum circuits using tensor networks.
    Circuit = 0
    # Quantum circuit simulation keeping the state in full dense form.
    CircuitDense = 1
    # Quantum circuit simulation keeping the state always in a MPS form.
    CircuitMPS = 3
