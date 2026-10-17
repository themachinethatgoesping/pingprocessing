import os
import logging
import themachinethatgoesping as theping

LOGGER = logging.getLogger(__name__)

try:
    from themachinethatgoesping.echosounders import kongsbergall, simradraw
except ImportError:
    # skip the tests in this file, create clear message for this
    LOGGER.warning("Skipping tests in this file because echosounders library is not available.")
    import pytest
    pytest.skip("Skipping tests in this file because echosounders library is not available.", allow_module_level=True)


class TestWCIViewer:
    test_files_per_ending_per_folder = None

    def find_files(self):
        if self.test_files_per_ending_per_folder is not None:
            return self.test_files_per_ending_per_folder

        self.test_files_per_ending_per_folder = theping.pingprocessing.testing.find_test_files(
            os.path.join(os.path.dirname(__file__), "../../")
        )
        return self.test_files_per_ending_per_folder

    def clean_index(self):
        LOGGER.info(f"Cleaning up cache files")
        for file in self.index_files:
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
            index_paths = theping.echosounders.index_functions.get_index_paths(files)
            fm = FileHandler(files, index_paths=index_paths, show_progress=False)
        else:
            fm = FileHandler(files, show_progress=False)

        pings = fm.get_pings()
        del fm
        theping.pingprocessing.core.clear_memory()

        assert len(pings) > 0

        return pings

    def test_viewing_pings_in_wci_viewer_should_not_crash(self):
        for ending, subfolders in self.find_files().items():
            for folder, files in subfolders.items():
                LOGGER.info(f"Testing {ending} files in {folder}")
                pings = self.get_pings(files)
                #viewer = theping.widgets.WCIViewer(pings)

        #viewer = theping.widgets.WCIViewer(self.get_pings(self.files_all+self.files_wcd))

    def test_wci_build_with_bottom_range_pss_should_not_crash(self):
        """Regression test: a PingSampleSelector that limits the water column to a
        bottom-range percentage over the full swath includes outer beams without a
        valid bottom detection. Those beams have range 0, which previously made the
        backtracer build a degenerate {0, 0} range->sample interpolator and abort
        the whole image build (ERROR[Interpolation::_check_XY]). The water-column
        viewer swallows that exception, so the image silently stops updating.
        """
        import numpy as np
        from themachinethatgoesping.pingprocessing.watercolumn.image.make_wci import (
            make_wci_dual_head,
        )

        pss = theping.pingprocessing.pingtools.PingSampleSelector()
        pss.select_max_bottom_range_percent(102)
        pss.select_beam_range_by_angles(-90, 90, 2)

        n_tested = 0
        for ending, subfolders in self.find_files().items():
            if ending not in (".all", ".wcd"):
                continue
            for folder, files in subfolders.items():
                pings = [p for p in self.get_pings(files) if p.has_watercolumn()]
                for ping in pings:
                    # Must not raise even when some selected beams have no bottom.
                    wci, extent = make_wci_dual_head(
                        ping,
                        horizontal_pixels=256,
                        wci_value="sv/av/pv/rv",
                        linear_mean=True,
                        apply_pss_to_bottom=False,
                        ping_sample_selector=pss,
                        mp_cores=1,
                    )
                    assert np.isfinite(wci).any()
                    n_tested += 1

        if n_tested == 0:
            import pytest

            pytest.skip("no kongsberg water-column test files available")

        #viewer = theping.widgets.WCIViewer(self.get_pings(self.files_all+self.files_wcd))
