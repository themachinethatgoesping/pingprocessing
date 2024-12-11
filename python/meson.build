# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: CC0-1.0

# -- python sources --
sources = [
  'themachinethatgoesping/pingprocessing/__init__.py',
  'themachinethatgoesping/pingprocessing/widgets/__init__.py',
  'themachinethatgoesping/pingprocessing/widgets/echogramviewer.py',
  'themachinethatgoesping/pingprocessing/widgets/tqdmwidget.py',
  'themachinethatgoesping/pingprocessing/widgets/wciviewer.py',
  'themachinethatgoesping/pingprocessing/overview/__init__.py',
  'themachinethatgoesping/pingprocessing/overview/nav_plot.py',
  'themachinethatgoesping/pingprocessing/overview/pingoverview.py',
  'themachinethatgoesping/pingprocessing/filter_pings/__init__.py',
  'themachinethatgoesping/pingprocessing/filter_pings/by_features.py',
  'themachinethatgoesping/pingprocessing/filter_pings/by_files.py',
  'themachinethatgoesping/pingprocessing/filter_pings/by_region.py',
  'themachinethatgoesping/pingprocessing/filter_pings/by_time.py',
  'themachinethatgoesping/pingprocessing/filter_pings/by_time_list.py',
  'themachinethatgoesping/pingprocessing/split_pings/__init__.py',
  'themachinethatgoesping/pingprocessing/split_pings/by_channel_id.py',
  'themachinethatgoesping/pingprocessing/split_pings/by_distance.py',
  'themachinethatgoesping/pingprocessing/split_pings/by_file.py',
  'themachinethatgoesping/pingprocessing/split_pings/by_function_return.py',
  'themachinethatgoesping/pingprocessing/split_pings/by_time.py',
  'themachinethatgoesping/pingprocessing/split_pings/by_time_blocks.py',
  'themachinethatgoesping/pingprocessing/split_pings/into_ping_blocks.py',
  'themachinethatgoesping/pingprocessing/split_pings/into_time_blocks.py',
  'themachinethatgoesping/pingprocessing/watercolumn/__init__.py',
  'themachinethatgoesping/pingprocessing/watercolumn/echograms/__init__.py',
  'themachinethatgoesping/pingprocessing/watercolumn/echograms/echodata.py',
  'themachinethatgoesping/pingprocessing/watercolumn/echograms/echogrambuilder.py',
  'themachinethatgoesping/pingprocessing/watercolumn/echograms/echolayer.py',
  'themachinethatgoesping/pingprocessing/watercolumn/image/__init__.py',
  'themachinethatgoesping/pingprocessing/watercolumn/image/imagebuilder.py',
  'themachinethatgoesping/pingprocessing/watercolumn/image/make_wci.py',
  'themachinethatgoesping/pingprocessing/watercolumn/helper/__init__.py',
  'themachinethatgoesping/pingprocessing/watercolumn/helper/make_image_helper.py',
  'themachinethatgoesping/pingprocessing/watercolumn/helper/select_get_wci_image.py',
  'themachinethatgoesping/pingprocessing/core/__init__.py',
  'themachinethatgoesping/pingprocessing/core/asserts.py',
  'themachinethatgoesping/pingprocessing/core/helper.py',
  'themachinethatgoesping/pingprocessing/core/progress.py',
  'themachinethatgoesping/pingprocessing/testing/__init__.py',
  'themachinethatgoesping/pingprocessing/testing/find_test_files.py',
  'themachinethatgoesping/pingprocessing/group_pings/__init__.py',
  'themachinethatgoesping/pingprocessing/group_pings/dual_head.py',
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
