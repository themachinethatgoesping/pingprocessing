import themachinethatgoesping as theping
import rasterio as rio
import os

class TestNavPlot:
    """
    This class contains unit tests for navigation plot functions.
    """

    def test_create_figure_background_image_projection(self):
        src_utm = '../../../unittest_data/background_maps/test_utm.tiff' # map in UTM projection EPSG:32631
        src_latlon = '../../../unittest_data/background_maps/test_latlon.tiff' # map in latlon projection EPSG:4326

        src_utm = os.path.join(os.path.dirname(__file__), src_utm)
        src_latlon = os.path.join(os.path.dirname(__file__), src_latlon)

        # make sure this does not crash with no figure provided
        _,_,crs = theping.pingprocessing.overview.nav_plot.create_figure('nav', return_crs=True)
        _,_ = theping.pingprocessing.overview.nav_plot.create_figure('nav', return_crs=False)
        _,_,crs = theping.pingprocessing.overview.nav_plot.create_figure('nav', return_crs=True, dst_crs = None)
        _,_ = theping.pingprocessing.overview.nav_plot.create_figure('nav', return_crs=False, dst_crs = None)

        # default projection should be EPSG:4326
        _,_,crs = theping.pingprocessing.overview.nav_plot.create_figure('nav', background_image_path=src_latlon, return_crs=True)
        assert crs == rio.crs.CRS.from_epsg(4326)

        # utm should be converted to default projection EPSG:4326
        _,_,crs = theping.pingprocessing.overview.nav_plot.create_figure('nav', background_image_path=src_utm, return_crs=True)
        assert crs == rio.crs.CRS.from_epsg(4326)

        # if dst_crs is set to None, default projecttion should be preserved
        _,_,crs = theping.pingprocessing.overview.nav_plot.create_figure('nav', background_image_path=src_latlon, return_crs=True, dst_crs = None)
        assert crs == rio.crs.CRS.from_epsg(4326)

        # if dst_crs is set to None, default projecttion should be preserved
        _,_,crs = theping.pingprocessing.overview.nav_plot.create_figure('nav', background_image_path=src_utm, return_crs=True, dst_crs = None)
        assert crs == rio.crs.CRS.from_epsg(32631)

        # if dst_crs is set to another projection, this projection should be used
        _,_,crs = theping.pingprocessing.overview.nav_plot.create_figure('nav', background_image_path=src_latlon, return_crs=True, dst_crs = 'EPSG:32631')
        assert crs == rio.crs.CRS.from_epsg(32631)

        # if dst_crs is set to another projection, this projection should be used
        _,_,crs = theping.pingprocessing.overview.nav_plot.create_figure('nav', background_image_path=src_utm, return_crs=True, dst_crs = 'EPSG:32631')
        assert crs == rio.crs.CRS.from_epsg(32631)