# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

# folders
from . import watercolumn as watercolumn_ext  # flake8: noqa

# modules

#cpp modules
from themachinethatgoesping.pingprocessing_cppy import *  # flake8: noqa

# overwrite timeconv module using the loaded python extension
watercolumn = watercolumn_ext

__version__ = "@PROJECT_VERSION@"
