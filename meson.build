# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: CC0-1.0

# --- Project ---
# Define project meta data
project(
  'themachinethatgoesping_pingprocessing',
  'cpp',
  license: 'MPL-2.0',
  version: '0.11.6',
  default_options: ['warning_level=0', 'buildtype=release', 'cpp_std=c++20'],
  meson_version: '>=1.8.1' #there is a problem with meson 1.8.0 so just use a higher version
)

projectnamespace = 'themachinethatgoesping'

# --- dependencies ---
# boost
boost_dep = dependency('boost', static: get_option('dynamic_boost').disabled())

# ext libraries


# themachinethatgoesping
# tools
tools_dep = dependency('themachinethatgoesping_tools', required: false)
echosounders_dep = dependency('themachinethatgoesping_echosounders', required: false)

if not tools_dep.found()
    tools_dep = subproject('tools', default_options: ['cpp_std=' + get_option('cpp_std')]).get_variable('tools_dep')
endif

if not echosounders_dep.found()
    tools_dep = subproject('echosounders', default_options: ['cpp_std=' + get_option('cpp_std')]).get_variable('echosounders_dep')
endif

# -- python modules --
if get_option('build_pythonmodule').enabled()
  pymod = import('python').find_installation(get_option('python_path'), pure: false)

  python3_dep = pymod.dependency()
  pybind11_dep = dependency('pybind11',  version : '>=3.0.1')
  
  tools_pybind_dep = dependency('themachinethatgoesping_tools_pybind', required: false)

  if not tools_pybind_dep.found()
      tools_pybind_dep = subproject('tools').get_variable('tools_pybind_dep')
  endif

  # set path where to install sources (pythonpath/site-packages/projectnamespace)
  message('python module install directory:', pymod.get_install_dir(subdir: projectnamespace))
endif

#themachinethatgoesping meta info
meta_dep = dependency('themachinethatgoesping_meta', required: false)
if not meta_dep.found()
    meta_dep = subproject('meta').get_variable('meta_dep')
endif


# add src
subdir('src')
 
# add python files
subdir('python')
