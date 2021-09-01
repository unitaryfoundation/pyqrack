# (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

from .qrack_system import QrackSystem

# Global entry-point for Qrack shared library
Qrack = QrackSystem()
