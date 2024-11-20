import os
import themachinethatgoesping as theping
from collections import defaultdict

def find_test_files(base_path):
    test_files_per_ending = {}
    
    dirname = os.path.dirname(os.path.abspath(base_path))
    test_folders1 = os.path.abspath(os.path.join(dirname, "../echosounders/unittest_data/"))
    test_folders2 = os.path.abspath(os.path.join(dirname, "./subprojects/echosounders-main/unittest_data/"))

    if os.path.exists(test_folders1):
        test_folders = test_folders1
    else :
        test_folders = test_folders2

    assert os.path.exists(test_folders), f"ERROR finding paths!\nTest folders {test_folders1} and \nTest folders {test_folders2} not found \n(base_path:  {base_path})"

    for ending in ['.all', '.wcd', '.all,.wcd', 'raw']:
        test_files_per_ending[ending] = theping.echosounders.index_functions.find_files(test_folders, ending.split(','))

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