# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: CC0-1.0

# -- library setup --
libname   = projectnamespace + '_pingprocessing'
headerdir = projectnamespace + '/pingprocessing'

# -- source files --
sources = [
]

headers = [
  'watercolumn/wci/make_image.hpp',
  'watercolumn/wci/.docstrings/make_image.doc.hpp',
]

# -- targets --
# for now force static linking on windows
if build_machine.system() != 'linux'
  pingprocessing_lib = static_library(libname,sources,
                        dependencies : [
                          boost_dep, 
                          echosounders_dep,
                          tools_dep
                          ],
                        override_options: [
                          'b_lundef=false'
                      ],
                        install : true, 
                        install_tag : 'library'
                        )
else  
  pingprocessing_lib = library(libname,sources,
                        dependencies : [
                          boost_dep, 
                          echosounders_dep,
                          tools_dep
                          ],
                        override_options: [
                          'b_lundef=false'
                      ],
                        install : true, 
                        install_tag : 'library'
                        )
endif
                      

# -- internal dependency --
pingprocessing_dep = declare_dependency(
  dependencies : [
    boost_dep,
    echosounders_dep,
    tools_dep
  ],
  link_with : [pingprocessing_lib],
  include_directories : ['../../']
  )


# -- installation setup --

# install headers
if headers != ['']
    install_headers(headers, subdir: headerdir+'/', preserve_path: true)
endif

# create pkgconfig file
pkg = import('pkgconfig')
pkg.generate(pingprocessing_lib,
             description : 'Library for processing themachinethatgoesping pings',
             url : 'https://github.com/themachinethatgoesping/pingprocessing')
