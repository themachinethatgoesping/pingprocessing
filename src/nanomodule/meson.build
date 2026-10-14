# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: CC0-1.0

# -- module name
module_name = 'pingprocessing_nanopy'
        
# -- sources
sources = [
  'module.cpp',
  'watercolumn/module.cpp',
  'watercolumn/image/module.cpp',
  'py_datatypes/c_i_ping.cpp',
  'py_datatypes/c_i_pingbottom.cpp',
  'py_datatypes/c_i_pingcommon.cpp',
  'py_datatypes/c_i_pingfiledata.cpp',
  'py_datatypes/c_i_pingwatercolumn.cpp',
  'py_datatypes/c_pinginfo.cpp',
  'py_datatypes/module.cpp',
  'py_datatypes/py_calibration/c_amplitudecalibration.cpp',
  'py_datatypes/py_calibration/c_multisectorwatercolumncalibration.cpp',
  'py_datatypes/py_calibration/c_watercolumncalibration.cpp',
]

# define python module
py_pingprocessing = pymod.extension_module(
    module_name,
    sources: sources,
    dependencies : [
        tools_nanobind_dep,
        pingprocessing_dep,
        nanobind_dep,
        python3_dep],
    link_language : 'cpp',
    override_options: [
        #'cpp_rtti=true',
    ],
    cpp_args : ['-DMODULE_NAME='+module_name, '-DMODULE_VERSION=' + '"' + meson.project_version() + '"'],
    install : true,
    subdir : projectnamespace,
    install_tag : 'python-runtime',
    limited_api : nanobind_limited_api,
    override_options: ['unity=off']
)

# Expose the nanobind binding-helper headers in this nanomodule so that the echosounders
# nanomodule (which now depends on pingprocessing) can reuse shared template helpers such as
# py_datatypes/py_calibration/c_multisectorwatercolumncalibration.hpp (init_multisectorcalibration)
# to bind its format-specific derived calibration types.
pingprocessing_nanobind_dep = declare_dependency(
  include_directories : include_directories('.')
  )



        
