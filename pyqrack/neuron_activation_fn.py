# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Pauli operators are specified for "b" (or "basis") parameters.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

from enum import IntEnum


class NeuronActivationFn(IntEnum):
    # Default
    Sigmoid = 0
    # Rectified linear
    ReLU = 1
    # Gaussian linear
    GeLU = 2
    # Version of (default) "Sigmoid" with tunable sharpness
    Generalized_Logistic = 3
    # Leaky rectified linear
    LeakyReLU = 4
