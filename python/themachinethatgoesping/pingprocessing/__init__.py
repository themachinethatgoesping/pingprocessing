# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

# Import all symbols from the C++ module
#from themachinethatgoesping.pingprocessing_nanopy import *  # flake8: noqa

# Dynamically expose all C++ submodules
#from themachinethatgoesping.tools._submodule_helper import expose_submodules
from themachinethatgoesping import pingprocessing_nanopy

#globals().update(expose_submodules(pingprocessing_nanopy, 'themachinethatgoesping.pingprocessing'))
#del pingprocessing_nanopy, expose_submodules

# Python folders
from . import core
from . import filter_pings
from . import split_pings
from . import group_pings
from . import overview
from . import watercolumn #as watercolumn_ext  # flake8: noqa
from . import widgets
from . import testing

# overwrite watercolumn module using the loaded python extension
#watercolumn = watercolumn_ext

__version__ = "@PROJECT_VERSION@"

