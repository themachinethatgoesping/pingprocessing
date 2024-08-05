import os
import themachinethatgoesping as Ping
from collections import defaultdict

def find_test_files():
    test_files_per_ending = {}
    
    dirname = os.path.dirname(__file__)
    test_folders = os.path.join(dirname, "../../../../../echosounders/unittest_data/")
    if not os.path.exists(test_folders):
        test_folders = os.path.join(dirname, "../../../../subprojects/echosounders-main/unittest_data/")
    print(test_folders)
    print(__file__)
    print(os.path.abspath(test_folders))
    assert os.path.exists(test_folders)

    for ending in ['.all', '.wcd', '.all,.wcd', 'raw']:
        test_files_per_ending[ending] = Ping.echosounders.index_functions.find_files(test_folders, ending.split(','))

    test_files_per_folder = {}
    for ending,file_list in test_files_per_ending.items():
        test_files_per_folder[ending] = defaultdict(list)
        for file in file_list:
            root = os.path.split(file)[0]
            test_files_per_folder[ending]['_'].append(file)
            test_files_per_folder[ending][root].append(file)
    
    for ending, folder in test_files_per_folder.items():
        for folder_name, files in folder.items():
           assert len(files) > 0
            
    return test_files_per_folder