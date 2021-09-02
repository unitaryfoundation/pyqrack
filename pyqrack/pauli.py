# (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
#
# Pauli operators are specified for "b" (or "basis") paramters.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

from enum import Enum


class Pauli(Enum):
    # Pauli Identity operator. Corresponds to Q# constant "PauliI."
    PauliI = 0,
    # Pauli X operator. Corresponds to Q# constant "PauliX."
    PauliX = 1,
    # Pauli Y operator. Corresponds to Q# constant "PauliY."
    PauliY = 3,
    # Pauli Z operator. Corresponds to Q# constant "PauliZ."
    PauliZ = 2
