# (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
#
# "QrackSystem" is a central access point, in Python, to the vm6502q/qrack
# shared library, in C++11. For ease, it wraps all Qrack shared library
# functions with native Python method signatures.
#
# The "QrackSystem" references underlying distinct Qrack simulator instances
# via integer IDs that are created upon request to allocate a new simulator.
# While you can directly use QrackSystem to manage all your simulators, we
# suggest that you instead instantiate an instance of "QrackSimulator", which
# requests its own new simulator ID and supplies it to "QrackSystem" for all
# Qrack shared library calls, therefore acting as an independent Qrack
# simulator object.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

from ctypes import *
from sys import platform


class QrackSystem:

    def __init__(self):
        shared_lib_path = "/usr/local/lib/libqrack_pinvoke.so"
        if platform.startswith('win32'):
            shared_lib_path = "C:\\Program Files\\Qrack\\bin\\qrack_pinvoke.dll"
        try:
            self.qrack_lib = CDLL(shared_lib_path)
        except Exception as e:
            print(e)

        #Define function signatures, up front
        self.qrack_lib.init.restype = c_uint
        self.qrack_lib.init.argTypes = []

        self.qrack_lib.init_count.restype = c_uint
        self.qrack_lib.init_count.argTypes = [c_uint]

        self.qrack_lib.init_clone.restype = c_uint
        self.qrack_lib.init_clone.argTypes = [c_uint]

        self.qrack_lib.destroy.restype = None
        self.qrack_lib.destroy.argTypes = [c_uint]

        self.qrack_lib.seed.restype = None
        self.qrack_lib.seed.argTypes = [c_uint, c_uint]

        self.qrack_lib.set_concurrency.restype = None
        self.qrack_lib.set_concurrency.argTypes = [c_uint, c_uint]

        self.qrack_lib.Prob.restype = c_double
        self.qrack_lib.Prob.argTypes = [c_uint, c_uint]

        self.qrack_lib.PermutationExpectation.restype = c_double
        self.qrack_lib.PermutationExpectation.argTypes = [c_uint, c_uint, POINTER(c_uint)]

        self.qrack_lib.JointEnsembleProbability.resType = c_double
        self.qrack_lib.JointEnsembleProbability.argTypes = [c_uint, c_uint, POINTER(c_int), c_uint]

Qrack = QrackSystem()
