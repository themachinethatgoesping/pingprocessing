import os
import logging
import themachinethatgoesping as theping
from themachinethatgoesping.echosounders import kongsbergall, simradraw

LOGGER = logging.getLogger(__name__)


class TestWCIViewer:
    test_files_per_ending_per_folder = None

    def find_files(self):
        if self.test_files_per_ending_per_folder is not None:
            return self.test_files_per_ending_per_folder

        self.test_files_per_ending_per_folder = theping.pingprocessing.testing.find_test_files(
            os.path.join(os.path.dirname(__file__), "../../../")
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
            file_cache_paths = theping.echosounders.index_functions.get_cache_file_paths(files)
            fm = FileHandler(files, file_cache_paths=file_cache_paths, show_progress=False)
        else:
            fm = FileHandler(files, show_progress=False)

        pings = fm.get_pings()
        del fm
        theping.pingprocessing.core.clear_memory()

        assert len(pings) > 0

        return pings

    def test_creating_echograms_should_not_crash(self):
        for ending, items in self.find_files().items():
            for folder, files in items.items():
                try:
                    self.get_pings(files)
                    pings = self.get_pings(files)
                    pings = theping.pingprocessing.filter_pings.by_features(pings, ["watercolumn.av"])
                    if len(pings) == 0:
                        continue

                    echodata = theping.pingprocessing.watercolumn.echograms.EchoData.from_pings(pings)

                    # / samplenr pingnr case
                    echodata.set_y_axis_sample_nr(max_samples=100)
                    echodata.set_x_axis_ping_nr(max_steps=500)
                    image, extent = echodata.build_image()

                    # depth / ping time case
                    echodata.set_y_axis_depth(max_samples=100)
                    echodata.set_x_axis_date_time(max_steps=500)
                    image, extent = echodata.build_image()
                except:
                    LOGGER.error(f"Failed to create echograms from {files} in {folder}")
                    raise
