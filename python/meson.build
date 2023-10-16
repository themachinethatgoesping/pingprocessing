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
