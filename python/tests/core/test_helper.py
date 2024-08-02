# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0


from themachinethatgoesping.pingprocessing.core import helper
from tqdm.auto import tqdm


class TestHelper:
    """
    This class contains unit tests for helper functions.
    """

    def test_clear_memory_does_not_crash(self):
        helper.clear_memory()
