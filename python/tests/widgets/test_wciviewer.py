import os
import logging
import themachinethatgoesping as Ping
from themachinethatgoesping.echosounders import kongsbergall, simradraw

LOGGER = logging.getLogger(__name__)


class TestWCIViewer:
    test_files_per_ending_per_folder = None

    def find_files(self):
        if self.test_files_per_ending_per_folder is not None:
            return self.test_files_per_ending_per_folder

        self.test_files_per_ending_per_folder = Ping.pingprocessing.testing.find_test_files(
            os.path.join(os.path.dirname(__file__), "../../")
        )
        return self.test_files_per_ending_per_folder

    def clean_cache(self):
        LOGGER.info(f"Cleaning up cache files")
        for file in self.cache_files:
            os.remove(file)

    def get_pings(self, files, cache=True):
        endings = set()
        for file in files:
            ending = "." + file.split(".")[-1]
            if ending == ".wcd":
                ending = ".all"
            endings.add(ending)

        if len(endings) > 1:
            raise ValueError(f"Can only open files with the same type of datagrams!, got {endings}")

        match list(endings)[0]:
            case ".all":
                FileHandler = kongsbergall.KongsbergAllFileHandler
            case ".raw":
                FileHandler = simradraw.SimradRawFileHandler
            case _:
                raise ValueError(f"Unknown file ending {list(endings)[0]}")

        if cache:
            file_cache_paths = Ping.echosounders.index_functions.get_cache_file_paths(files)
            fm = FileHandler(files, file_cache_paths=file_cache_paths, show_progress=False)
        else:
            fm = FileHandler(files, show_progress=False)

        pings = fm.get_pings()
        del fm
        Ping.pingprocessing.core.clear_memory()

        assert len(pings) > 0

        return pings

    def test_viewing_pings_in_wci_viewer_should_not_crash(self):
        for ending, subfolders in self.find_files().items():
            for folder, files in subfolders.items():
                LOGGER.info(f"Testing {ending} files in {folder}")
                pings = self.get_pings(files)
                #viewer = Ping.pingprocessing.widgets.WCIViewer(pings)

        #viewer = Ping.pingprocessing.widgets.WCIViewer(self.get_pings(self.files_all+self.files_wcd))
