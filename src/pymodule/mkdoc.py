#!/bin/env python3

import os
import shutil
import sys
import subprocess
from tqdm import tqdm
from copy import deepcopy
import hashlib

# --arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--regenerate", help="Regenerate all docstrings", action="store_true")
parser.add_argument("--renew", help="Delete all docstring folders first", action="store_true")
args = parser.parse_args()

FORCE_REGENERATE = args.regenerate
FORCE_RENEW = args.renew

# source: https://www.debugpointer.com/python/create-sha256-hash-of-a-file-in-python
def get_hash(file_name):
    hash_sha256 = hashlib.sha256()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


# does not work because pybind11_mkdoc only provides a module script
#from pybind11_mkdoc import mkdoc
ignore_files = [
    # 'classhelper.hpp'
]

# read modified headers
new_header = ""
with open("new_doc_header.hpp", 'r', encoding="utf-8") as ifi:
    new_header = ifi.read()


def modify_doc(doc):
    new_doc = deepcopy(new_header)

    # get ignore doc lines
    ignore_doc = get_ignore_doc(header)
    if ignore_doc:
        print(ignore_doc)

    start = False
    ignore = False
    for line in doc.split('\n'):
        if not start:
            if "#define DOC(...)" in line:
                start = True
            continue

        for id in ignore_doc:
            if id in line:
                ignore = True
                break

        if not ignore:
            new_doc += line + "\n"

        if ignore:
            if ";" in line:
                ignore = False

    return new_doc


def add_doc_line(header, doc_path):

    include_string = '#include ".docstrings/' + doc_path.split("docstrings/")[-1] + '"'

    # chech if dockline exists
    file = ""
    found_pragma = False
    with open(header, 'r', encoding="utf-8") as ifi:
        for line in ifi:
            if include_string in line:
                return

            file += line
            if "#pragma once" in line:
                found_pragma = True
                file += f"\n/* generated doc strings */\n{include_string}\n"

    if not found_pragma:
        print(f"WARNING: did not find #pragma once in {header}")

    with open(header, 'w', encoding="utf-8") as ofi:
        ofi.write(file)


def get_ignore_doc(header):

    # chech if dockline exists
    ignore_doc = []
    with open(header, 'r', encoding="utf-8") as ifi:
        for line in ifi:
            if "IGNORE_DOC:" in line:
                ignore_doc.append(line.split("IGNORE_DOC:")[-1].strip() + " ")

    return ignore_doc


headers = []
for r, d, f in os.walk('../themachinethatgoesping/'):
    for file in f:
        if file.endswith(".doc.hpp"):
            continue

        if file.endswith('.hpp'):
            if file not in ignore_files:
                headers.append(r + '/' + file)
headers.sort()


with open('mkdoc_log.log', 'w', encoding="utf-8") as ofi_log:
    prg = tqdm(headers)
    for header in prg:
        if len(header) > 53:
            prg.set_postfix_str(f"...{header[-50:]}")
        else:
            prg.set_postfix_str(header)

        filename = header.split("/")[-1]
        output_path = "/".join(header.split("/")[:-1]) + "/.docstrings/" + \
            filename.replace(filename.split('.')[-1], "doc.hpp")

        if FORCE_RENEW:
            shutil.rmtree("/".join(output_path.split('/')[:-1]))
        os.makedirs("/".join(output_path.split('/')[:-1]), exist_ok=True)

        add_doc_line(header, output_path)

        # get file hash
        hash_new = get_hash(header)

        if not FORCE_REGENERATE and not FORCE_RENEW:  # False means: ignore hash and force update
            # check old hash (written into doc file)
            if os.path.exists(output_path):
                hash_old = "INVALID"
                with open(output_path, 'r', encoding="utf-8") as ifi:
                    line = ifi.readline()
                    if "//sourcehash:" in line:
                        hash_old = line.split("//sourcehash:")[1].strip()

                # only recreate file if hash changed
                if hash_new == hash_old:
                    continue

        #print(p,doc_paths, output_path)

        docstrings = subprocess.check_output(
            [sys.executable, '-m', 'pybind11_mkdoc', header], stderr=ofi_log).decode('utf8')
        new_doc = modify_doc(docstrings)

        with open(output_path, 'w', encoding="utf-8") as ofi:
            #prg.set_postfix_str(f"HASH ...{header[-50:]}")
            ofi.write(f"//sourcehash: {hash_new}\n\n")
            ofi.write(new_doc)
            print("updated:", output_path)
