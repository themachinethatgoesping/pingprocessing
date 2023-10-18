# import numpy as np
# from themachinethatgoesping.pingprocessing.watercolumn.make_image import make_wci
# from themachinethatgoesping.echosounders import filetemplates
# I_Ping = filetemplates.I_Ping

# # created by github copilot
# def test_make_wci():
#     # create a mock ping object with some test data
#     class MockPing():
#         def __init__(self):

#             self.watercolumn = MockWaterColumn()
#             self.geolocation = MockGeolocation()
#             self.sensor_configuration = MockSensorConfiguration()
        
#         def get_watercolumn(self):
#             return self.watercolumn
        
#         def get_geolocation(self):
#             return self.geolocation
        
#         def get_sensor_configuration(self):
#             return self.sensor_configuration
    
#     class MockWaterColumn:
#         def get_amplitudes(self):
#             return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
#         def get_number_of_samples_per_beam(self):
#             return 3
    
#     class MockGeolocation:
#         z = 10.0
    
#     class MockSensorConfiguration:
#         def get_target(self, name):
#             return MockTarget()
        
#         def get_position_source(self):
#             return MockPositionSource()
    
#     class MockTarget:
#         x = 0.0
#         y = 0.0
    
#     class MockPositionSource:
#         pass
    
#     # call the make_wci function with the mock ping object and some test parameters
#     horizontal_res = 3
#     from_bottom_xyz = True
#     wci, extent = make_wci(MockPing(), horizontal_res, from_bottom_xyz)
    
#     # check that the output is correct
#     expected_wci = np.array([1, 5, 9])
#     expected_extent = (-0.05, 0.05, 11.0, 9.0)
#     assert np.allclose(wci, expected_wci)
#     assert np.allclose(extent, expected_extent)