# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: CC0-1.0

# -- library setup --
libname   = projectnamespace + '_pingprocessing'
headerdir = projectnamespace + '/pingprocessing'

# -- source files --
sources = [
  'datatypes/i_ping.cpp',
  'datatypes/i_pingbottom.cpp',
  'datatypes/i_pingcommon.cpp',
  'datatypes/i_pingfiledata.cpp',
  'datatypes/i_pingwatercolumn.cpp',
  'datatypes/pingcontainer.cpp',
  'datatypes/calibration/amplitudecalibration.cpp',
  'datatypes/calibration/multisectorwatercolumncalibration.cpp',
  'datatypes/calibration/watercolumncalibration.cpp',
]

headers = [
  'watercolumn/echograms/echogrambase.hpp',
  'watercolumn/echograms/.docstrings/echogrambase.doc.hpp',
  'watercolumn/image/make_wci.hpp',
  'watercolumn/image/.docstrings/make_wci.doc.hpp',
  'datatypes/i_ping.hpp',
  'datatypes/i_pingbottom.hpp',
  'datatypes/i_pingcommon.hpp',
  'datatypes/i_pingfiledata.hpp',
  'datatypes/i_pingwatercolumn.hpp',
  'datatypes/pingcontainer.hpp',
  'datatypes/calibration/amplitudecalibration.hpp',
  'datatypes/calibration/calibrationinfo.hpp',
  'datatypes/calibration/multisectorwatercolumncalibration.hpp',
  'datatypes/calibration/watercolumncalibration.hpp',
  'datatypes/calibration/.docstrings/amplitudecalibration.doc.hpp',
  'datatypes/calibration/.docstrings/calibrationinfo.doc.hpp',
  'datatypes/calibration/.docstrings/multisectorwatercolumncalibration.doc.hpp',
  'datatypes/calibration/.docstrings/watercolumncalibration.doc.hpp',
  'datatypes/.docstrings/i_ping.doc.hpp',
  'datatypes/.docstrings/i_pingbottom.doc.hpp',
  'datatypes/.docstrings/i_pingcommon.doc.hpp',
  'datatypes/.docstrings/i_pingfiledata.doc.hpp',
  'datatypes/.docstrings/i_pingwatercolumn.doc.hpp',
  'datatypes/.docstrings/pingcontainer.doc.hpp',
]

# -- check if there are source files to compile --
if sources.length() > 0
  # -- targets --
  # for now force static linking on windows
  if build_machine.system() != 'linux'
    pingprocessing_lib = static_library(libname,sources,
                          dependencies : [
                            boost_dep, 
                            algorithms_dep,
                            navigation_dep,
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
                            algorithms_dep,
                            navigation_dep,
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
    algorithms_dep,
                            navigation_dep,
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

