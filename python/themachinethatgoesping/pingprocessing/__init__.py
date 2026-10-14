# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

import sys as _sys

# Import the C++ module (which, at import time, pulls in its dependency modules so their
# types are registered). We intentionally do NOT `import *` here: that would pull the C++
# 'watercolumn' submodule into this package namespace and shadow the python 'watercolumn'
# subpackage imported below.
from themachinethatgoesping import pingprocessing_nanopy

# Expose the C++ 'datatypes' submodule (the format-independent ping interfaces + calibration
# classes) as a proper, importable submodule: themachinethatgoesping.pingprocessing.datatypes.
from themachinethatgoesping.pingprocessing_nanopy import datatypes  # flake8: noqa
_sys.modules['themachinethatgoesping.pingprocessing.datatypes'] = datatypes
del pingprocessing_nanopy, _sys

# Python folders
from . import core
from . import filter_pings
from . import split_pings
from . import group_pings
from . import overview
from . import watercolumn  # flake8: noqa
from . import testing

__version__ = "@PROJECT_VERSION@"

