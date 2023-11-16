# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: CC0-1.0

# -- python sources --
sources = [
    'themachinethatgoesping/pingprocessing/__init__.py',
    'themachinethatgoesping/pingprocessing/watercolumn/helper/__init__.py',
    'themachinethatgoesping/pingprocessing/watercolumn/helper/make_image_helper.py',
    'themachinethatgoesping/pingprocessing/watercolumn/make_image/__init__.py',
    'themachinethatgoesping/pingprocessing/watercolumn/make_image/make_wci.py',

    'themachinethatgoesping/pingprocessing/core/__init__.py',
    'themachinethatgoesping/pingprocessing/core/progress.py',

    'themachinethatgoesping/pingprocessing/filter_pings/__init__.py',
    'themachinethatgoesping/pingprocessing/filter_pings/by_region.py',
    'themachinethatgoesping/pingprocessing/filter_pings/by_time.py',

    'themachinethatgoesping/pingprocessing/overview/__init__.py',
    'themachinethatgoesping/pingprocessing/overview/pingoverview.py',
    'themachinethatgoesping/pingprocessing/overview/nav_plot.py',

    'themachinethatgoesping/pingprocessing/split_pings/__init__.py',
    'themachinethatgoesping/pingprocessing/split_pings/by_distance.py',
    'themachinethatgoesping/pingprocessing/split_pings/by_time.py',
]

if get_option('build_pythonmodule').enabled()
    
    if get_option('pydev_install').disabled() #don't install if pydev_install is activated
        foreach source : sources
            #create install path for file
            fields = []
            foreach field : source.split('/')
                if not field.endswith('.py')
                    fields += [field]
                endif
            endforeach
            pth = '/'.join(fields)

            pymod.install_sources(source, subdir: pth, install_tag: 'python-runtime')
        endforeach

    endif
endif
