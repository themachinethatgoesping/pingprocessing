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
  'watercolumn/echograms/echogrambase.hpp',
  'watercolumn/echograms/.docstrings/echogrambase.doc.hpp',
  'watercolumn/echograms/.docstrings/pingindexconverter.doc.hpp',
  'watercolumn/image/make_wci.hpp',
  'watercolumn/image/.docstrings/make_wci.doc.hpp',
]

# -- check if there are source files to compile --
if sources.length() > 0
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
     
  link_with = [pingprocessing_lib]

  # create pkgconfig file
  pkg = import('pkgconfig')
  pkg.generate(pingprocessing_lib,
              description : 'Library for processing themachinethatgoesping pings',
              url : 'https://github.com/themachinethatgoesping/pingprocessing')
else
  link_with = []
endif                   

# -- internal dependency --
pingprocessing_dep = declare_dependency(
  dependencies : [
    boost_dep,
    echosounders_dep,
    tools_dep
  ],
  link_with : link_with,
  include_directories : ['../../']
  )


# -- installation setup --

# install headers
if headers != ['']
    install_headers(headers, subdir: headerdir+'/', preserve_path: true)
endif

