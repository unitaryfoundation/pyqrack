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

import os
from ctypes import *
from sys import platform as _platform


class QrackSystem:
    def __init__(self):
        shared_lib_path = "/usr/local/lib/libqrack_pinvoke.so"
        if os.environ.get('PYQRACK_SHARED_LIB_PATH') != None:
            shared_lib_path = os.environ.get('PYQRACK_SHARED_LIB_PATH')
        elif _platform == "darwin":
            shared_lib_path = "/usr/local/lib/libqrack_pinvoke.dylib"
        elif _platform == "win32":
            shared_lib_path = "C:\\Program Files\\Qrack\\bin\\qrack_pinvoke.dll"
        elif _platform != "linux" and _platform != "linux2":
            print(
                "No Qrack binary for your platform, attempting to use /usr/local/lib/libqrack_pinvoke.so"
            )
            print(
                "You can choose the binary file to load with the environment variable: PYQRACK_SHARED_LIB_PATH"
            )

        try:
            self.qrack_lib = CDLL(shared_lib_path)
        except Exception as e:
            print(e)

        self.fppow = 5
        if "QRACK_FPPOW" in os.environ:
            self.fppow = int(os.environ.get('QRACK_FPPOW'))
        if self.fppow < 4 or self.fppow > 7:
            raise ValueError(
                "QRACK_FPPOW environment variable must be an integer >3 and <8. (Qrack builds from 4 for fp16/half, up to 7 for fp128/quad."
            )

        # Define function signatures, up front

        # non-quantum

        self.qrack_lib.DumpIds.restype = None
        self.qrack_lib.DumpIds.argTypes = [c_ulonglong, CFUNCTYPE(None, c_ulonglong)]

        self.qrack_lib.Dump.restype = None
        self.qrack_lib.Dump.argTypes = [
            c_ulonglong,
            CFUNCTYPE(c_ulonglong, c_double, c_double),
        ]

        # These next two methods need to have c_double pointers, if PyQrack is built with fp64.
        self.qrack_lib.InKet.restype = None
        self.qrack_lib.OutKet.restype = None

        if self.fppow == 5:
            self.qrack_lib.InKet.argTypes = [c_ulonglong, POINTER(c_float)]
            self.qrack_lib.OutKet.argTypes = [c_ulonglong, POINTER(c_float)]
        if self.fppow == 6:
            self.qrack_lib.InKet.argTypes = [c_ulonglong, POINTER(c_double)]
            self.qrack_lib.OutKet.argTypes = [c_ulonglong, POINTER(c_double)]

        self.qrack_lib.init.restype = c_ulonglong
        self.qrack_lib.init.argTypes = []

        self.qrack_lib.get_error.restype = c_int
        self.qrack_lib.get_error.argTypes = [c_ulonglong]

        self.qrack_lib.init_count.restype = c_ulonglong
        self.qrack_lib.init_count.argTypes = [c_ulonglong, c_bool]

        self.qrack_lib.init_count_pager.restype = c_ulonglong
        self.qrack_lib.init_count_pager.argTypes = [c_ulonglong, c_bool]

        self.qrack_lib.init_count_type.restype = c_ulonglong
        self.qrack_lib.init_count_type.argTypes = [
            c_ulonglong,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
        ]

        self.qrack_lib.init_count_type.restype = c_ulonglong
        self.qrack_lib.init_count_type.argTypes = [
            c_ulonglong,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
        ]

        self.qrack_lib.init_clone.restype = c_ulonglong
        self.qrack_lib.init_clone.argTypes = [c_ulonglong]

        self.qrack_lib.destroy.restype = None
        self.qrack_lib.destroy.argTypes = [c_ulonglong]

        self.qrack_lib.seed.restype = None
        self.qrack_lib.seed.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.set_concurrency.restype = None
        self.qrack_lib.set_concurrency.argTypes = [c_ulonglong, c_ulonglong]

        # pseudo-quantum

        self.qrack_lib.Prob.restype = c_double
        self.qrack_lib.Prob.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.PermutationExpectation.restype = c_double
        self.qrack_lib.PermutationExpectation.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.JointEnsembleProbability.resType = c_double
        self.qrack_lib.JointEnsembleProbability.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_int),
            c_ulonglong,
        ]

        self.qrack_lib.PhaseParity.resType = None
        self.qrack_lib.PhaseParity.argTypes = [
            c_ulonglong,
            c_double,
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.ResetAll.resType = None
        self.qrack_lib.ResetAll.argTypes = [c_ulonglong]

        # allocate and release

        self.qrack_lib.allocateQubit.resType = None
        self.qrack_lib.allocateQubit.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.release.resType = c_bool
        self.qrack_lib.release.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.num_qubits.resType = c_ulonglong
        self.qrack_lib.num_qubits.argTypes = [c_ulonglong]

        # single-qubit gates

        self.qrack_lib.X.resType = None
        self.qrack_lib.X.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.Y.resType = None
        self.qrack_lib.Y.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.Z.resType = None
        self.qrack_lib.Z.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.H.resType = None
        self.qrack_lib.H.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.S.resType = None
        self.qrack_lib.S.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.T.resType = None
        self.qrack_lib.T.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.AdjS.resType = None
        self.qrack_lib.AdjS.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.AdjT.resType = None
        self.qrack_lib.AdjT.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.U.resType = None
        self.qrack_lib.U.argTypes = [
            c_ulonglong,
            c_ulonglong,
            c_double,
            c_double,
            c_double,
        ]

        self.qrack_lib.Mtrx.resType = None
        self.qrack_lib.Mtrx.argTypes = [c_ulonglong, POINTER(c_double), c_ulonglong]

        # multi-controlled single-qubit gates

        self.qrack_lib.MCX.resType = None
        self.qrack_lib.MCX.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MCY.resType = None
        self.qrack_lib.MCY.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MCZ.resType = None
        self.qrack_lib.MCZ.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MCH.resType = None
        self.qrack_lib.MCH.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MCS.resType = None
        self.qrack_lib.MCS.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MCT.resType = None
        self.qrack_lib.MCT.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MCAdjS.resType = None
        self.qrack_lib.MCAdjS.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MCAdjT.resType = None
        self.qrack_lib.MCAdjT.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MCU.resType = None
        self.qrack_lib.MCU.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            c_double,
            c_double,
            c_double,
        ]

        self.qrack_lib.MCMtrx.resType = None
        self.qrack_lib.MCMtrx.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_double),
            c_ulonglong,
        ]

        # multi-anti-controlled single-qubit gates

        self.qrack_lib.MACX.resType = None
        self.qrack_lib.MACX.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MACY.resType = None
        self.qrack_lib.MACY.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MACZ.resType = None
        self.qrack_lib.MACZ.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MACH.resType = None
        self.qrack_lib.MACH.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MACS.resType = None
        self.qrack_lib.MACS.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MACT.resType = None
        self.qrack_lib.MACT.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MACAdjS.resType = None
        self.qrack_lib.MACAdjS.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MACAdjT.resType = None
        self.qrack_lib.MACAdjT.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        self.qrack_lib.MACU.resType = None
        self.qrack_lib.MACU.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            c_double,
            c_double,
            c_double,
        ]

        self.qrack_lib.MACMtrx.resType = None
        self.qrack_lib.MACMtrx.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_double),
            c_ulonglong,
        ]

        self.qrack_lib.Multiplex1Mtrx.resType = None
        self.qrack_lib.Multiplex1Mtrx.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_double),
        ]

        # coalesced single qubit gates

        self.qrack_lib.MX.resType = None
        self.qrack_lib.MX.argTypes = [c_ulonglong, c_ulonglong, POINTER(c_ulonglong)]

        self.qrack_lib.MY.resType = None
        self.qrack_lib.MY.argTypes = [c_ulonglong, c_ulonglong, POINTER(c_ulonglong)]

        self.qrack_lib.MZ.resType = None
        self.qrack_lib.MZ.argTypes = [c_ulonglong, c_ulonglong, POINTER(c_ulonglong)]

        # rotations

        self.qrack_lib.R.resType = None
        self.qrack_lib.R.argTypes = [c_ulonglong, c_ulonglong, c_double, c_ulonglong]

        self.qrack_lib.MCR.resType = None
        self.qrack_lib.MCR.argTypes = [
            c_ulonglong,
            c_ulonglong,
            c_double,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
        ]

        # exponential of Pauli operators

        self.qrack_lib.Exp.resType = None
        self.qrack_lib.Exp.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_int),
            c_double,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.MCExp.resType = None
        self.qrack_lib.MCExp.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_int),
            c_double,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
        ]

        # measurements

        self.qrack_lib.M.resType = c_ulonglong
        self.qrack_lib.M.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.ForceM.resType = c_ulonglong
        self.qrack_lib.ForceM.argTypes = [c_ulonglong, c_ulonglong, c_bool]

        self.qrack_lib.MAll.resType = c_ulonglong
        self.qrack_lib.MAll.argTypes = [c_ulonglong]

        self.qrack_lib.Measure.resType = c_ulonglong
        self.qrack_lib.Measure.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_int),
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.MeasureShots.resType = None
        self.qrack_lib.MeasureShots.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        # swap

        self.qrack_lib.SWAP.resType = None
        self.qrack_lib.SWAP.argTypes = [c_ulonglong, c_ulonglong, c_ulonglong]

        self.qrack_lib.ISWAP.resType = None
        self.qrack_lib.ISWAP.argTypes = [c_ulonglong, c_ulonglong, c_ulonglong]

        self.qrack_lib.AdjISWAP.resType = None
        self.qrack_lib.AdjISWAP.argTypes = [c_ulonglong, c_ulonglong, c_ulonglong]

        self.qrack_lib.FSim.resType = None
        self.qrack_lib.FSim.argTypes = [
            c_ulonglong,
            c_double,
            c_double,
            c_ulonglong,
            c_ulonglong,
        ]

        self.qrack_lib.CSWAP.resType = None
        self.qrack_lib.CSWAP.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            c_ulonglong,
        ]

        self.qrack_lib.ACSWAP.resType = None
        self.qrack_lib.ACSWAP.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            c_ulonglong,
        ]

        # Schmidt decomposition

        self.qrack_lib.Compose.resType = None
        self.qrack_lib.Compose.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.Decompose.resType = c_ulonglong
        self.qrack_lib.Decompose.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.Dispose.resType = None
        self.qrack_lib.Dispose.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        # (quasi-)Boolean gates

        self.qrack_lib.AND.resType = None
        self.qrack_lib.AND.argTypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
        ]

        self.qrack_lib.OR.resType = None
        self.qrack_lib.OR.argTypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
        ]

        self.qrack_lib.XOR.resType = None
        self.qrack_lib.XOR.argTypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
        ]

        self.qrack_lib.NAND.resType = None
        self.qrack_lib.NAND.argTypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
        ]

        self.qrack_lib.NOR.resType = None
        self.qrack_lib.NOR.argTypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
        ]

        self.qrack_lib.XNOR.resType = None
        self.qrack_lib.XNOR.argTypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
        ]

        # half classical (quasi-)Boolean gates

        self.qrack_lib.CLAND.resType = None
        self.qrack_lib.CLAND.argTypes = [c_ulonglong, c_bool, c_ulonglong, c_ulonglong]

        self.qrack_lib.CLOR.resType = None
        self.qrack_lib.CLOR.argTypes = [c_ulonglong, c_bool, c_ulonglong, c_ulonglong]

        self.qrack_lib.CLXOR.resType = None
        self.qrack_lib.CLXOR.argTypes = [c_ulonglong, c_bool, c_ulonglong, c_ulonglong]

        self.qrack_lib.CLNAND.resType = None
        self.qrack_lib.CLNAND.argTypes = [c_ulonglong, c_bool, c_ulonglong, c_ulonglong]

        self.qrack_lib.CLNOR.resType = None
        self.qrack_lib.CLNOR.argTypes = [c_ulonglong, c_bool, c_ulonglong, c_ulonglong]

        self.qrack_lib.CLXNOR.resType = None
        self.qrack_lib.CLXNOR.argTypes = [c_ulonglong, c_bool, c_ulonglong, c_ulonglong]

        # Fourier transform

        self.qrack_lib.QFT.resType = None
        self.qrack_lib.QFT.argTypes = [c_ulonglong, c_ulonglong, POINTER(c_ulonglong)]

        self.qrack_lib.IQFT.resType = None
        self.qrack_lib.IQFT.argTypes = [c_ulonglong, c_ulonglong, POINTER(c_ulonglong)]

        # Arithmetic-Logic-Unit (ALU)

        self.qrack_lib.ADD.resType = None
        self.qrack_lib.ADD.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.SUB.resType = None
        self.qrack_lib.SUB.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.ADDS.resType = None
        self.qrack_lib.ADDS.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.SUBS.resType = None
        self.qrack_lib.SUBS.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.MUL.resType = None
        self.qrack_lib.MUL.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.DIV.resType = None
        self.qrack_lib.DIV.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.MULN.resType = None
        self.qrack_lib.MULN.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.DIVN.resType = None
        self.qrack_lib.DIVN.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.POWN.resType = None
        self.qrack_lib.POWN.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.MCADD.resType = None
        self.qrack_lib.MCADD.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.MCSUB.resType = None
        self.qrack_lib.MCSUB.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.MCMUL.resType = None
        self.qrack_lib.MCMUL.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.MCDIV.resType = None
        self.qrack_lib.MCDIV.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.MCMULN.resType = None
        self.qrack_lib.MCMULN.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.MCDIVN.resType = None
        self.qrack_lib.MCDIVN.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.MCPOWN.resType = None
        self.qrack_lib.MCPOWN.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
        ]

        self.qrack_lib.LDA.resType = None
        self.qrack_lib.LDA.argType = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ubyte),
        ]

        self.qrack_lib.ADC.resType = None
        self.qrack_lib.ADC.argType = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ubyte),
        ]

        self.qrack_lib.SBC.resType = None
        self.qrack_lib.SBC.argType = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ubyte),
        ]

        self.qrack_lib.Hash.resType = None
        self.qrack_lib.Hash.argType = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ubyte),
        ]

        # miscellaneous

        self.qrack_lib.TrySeparate1Qb.resType = c_bool
        self.qrack_lib.TrySeparate1Qb.argTypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.TrySeparate2Qb.resType = c_bool
        self.qrack_lib.TrySeparate2Qb.argTypes = [c_ulonglong, c_ulonglong, c_ulonglong]

        self.qrack_lib.TrySeparateTol.resType = c_bool
        self.qrack_lib.TrySeparateTol.argTypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_double,
        ]

        self.qrack_lib.SetReactiveSeparate.resType = c_bool
        self.qrack_lib.SetReactiveSeparate.argTypes = [c_ulonglong, c_bool]
        self.qrack_lib.SetTInjection.resType = c_bool
        self.qrack_lib.SetTInjection.argTypes = [c_ulonglong, c_bool]
