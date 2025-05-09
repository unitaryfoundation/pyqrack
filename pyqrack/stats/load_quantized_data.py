# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Load quantized data into a QrackSimulator
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.


def load_data(sim, index_bits, value_bits, data):
    """
    Take discretized features from quantize_by_range and load into a QrackSimulator

    Args:
        sim (QrackSimulator): Simulator into which to load data
        index_bits (list[int]): List of index bits, least-to-most significant
        value_bits (list[int]): List of value bits, least-to-most significant
        data (list[list[bool]]): Data to load, row-major

    Raises:
        Value error: Length of value_bits does not match data feature column count!
    """
    if (len(data) > 0) and (len(value_bits) != len(data[0])):
        raise ValueError("Length of value_bits does not match data feature column count!")

    for i in range(index_bits):
        if sim.m(i):
            sim.x(i)
        sim.h(i)

    if len(data) == 0:
        return

    sim.lda(list(range(index_bits)), list(range(value_bits)), data)
