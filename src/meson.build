# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: CC0-1.0

# -- setup variables --
base_path = meson.current_source_dir()

# -- C++ projects --
subdir('themachinethatgoesping')

# -- catch2 tests --
if get_option('build_tests').enabled()
    catch2 = subproject('catch2', default_options: ['tests=false'])
    catch2 = catch2.get_variable('catch2_with_main_dep')
    
    subdir('tests')
endif


# -- python modules --
if get_option('build_pythonmodule').enabled()
    subdir('pymodule')
endif
