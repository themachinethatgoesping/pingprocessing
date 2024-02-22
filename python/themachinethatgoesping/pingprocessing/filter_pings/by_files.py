from typing import List
from tqdm.auto import tqdm
from pathlib import Path
import os

from themachinethatgoesping.echosounders import filetemplates
Ping = filetemplates.I_Ping

def by_folders(pings: List[Ping], folders: List[str], progress_delay: int = 5) -> List[Ping]:
    """
    Filter pings by folders.

    Args:
        pings (List[Ping]): List of pings to filter.
        folders (List[str]): List of folder paths to filter by.
        progress_delay (int): Delay in seconds for the progress bar.

    Returns:
        List[Ping]: List of filtered pings.
    """

    folders = [Path(f) for f in folders]
    outputpings = []
    for ping in tqdm(pings, delay=progress_delay):
        for path in ping.file_data.get_file_paths():
            if Path(os.path.split(path)[0]) in folders:
                outputpings.append(ping)
                break

    return outputpings

def by_files(pings: List[Ping], files: List[str], progress_delay: int = 5) -> List[Ping]:
    """
    Filter pings by files.

    Args:
        pings (List[Ping]): List of pings to filter.
        files (List[str]): List of file paths to filter by.
        progress_delay (int): Delay in seconds for the progress bar.

    Returns:
        List[Ping]: List of filtered pings.
    """
    
    files = [Path(f) for f in files]
    outputpings = []
    for ping in tqdm(pings, delay=progress_delay):
        for path in ping.file_data.get_file_paths():
            if Path(path) in files:
                outputpings.append(ping)
                break

    return outputpings
