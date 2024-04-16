# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: CC0-1.0

# -- module name
module_name = 'pingprocessing_cppy'
        
# -- sources
sources = [
  'module.cpp',
  'watercolumn/module.cpp',
  'watercolumn/image/module.cpp',
]

# define python module
py_pingprocessing = pymod.extension_module(
    module_name,
    sources: sources,
    dependencies : [
        tools_pybind_dep,
        pingprocessing_dep,
        pybind11_dep,
        python3_dep],
    link_language : 'cpp',
    override_options: [
        #'cpp_rtti=true',
    ],
    cpp_args : ['-DMODULE_NAME='+module_name, '-DMODULE_VERSION=' + '"' + meson.project_version() + '"'],
    install : true,
    subdir : projectnamespace,
    install_tag : 'python-runtime'
)



        
