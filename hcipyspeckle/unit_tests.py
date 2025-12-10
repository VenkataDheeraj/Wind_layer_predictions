import unittest
import h5py
import main
import os
import numpy as np


class TestDataGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print ("Running Data Generation code")
        self.filename = main.main()
        print ("Testing", self.filename)

    @classmethod
    def tearDownClass(self):
        print ("Removing ", self.filename)
        os.remove (self.filename)
        print ("Tests completed")

    def test_data_exists(self):
        self.assertEqual (os.path.isfile (self.filename), True, 'H5 data not saved')

    def test_check_frames(self):
        with h5py.File(self.filename, "r") as f:
           self.assertEqual (f.attrs['number_of_frames'], 2000, 'Frames not defaulting to 2000') 

    def test_default_parameters_saved(self):
        with h5py.File(self.filename, "r") as f:
            self.assertEqual (f.attrs['diameter'], 1, 'Diameter not saved correctly')
            self.assertEqual (f.attrs['pupil_min_sampling'], 4e-2, 'Pupil Sampling not saved correctly')
            self.assertEqual (f.attrs['exposure_time'], 0.005, 'Exposure Time not saved correctly')
            self.assertEqual (f.attrs['angular_distance_zenith'], 0.296705972839036, 'Angular Distance from Zenith not saved correctly')
            self.assertEqual (f.attrs['elevation'], 2400, 'Elevation not saved correctly')
            self.assertEqual (f.attrs['number_of_layers'], 4, 'Num. Layers not saved correctly')
            self.assertEqual (f.attrs['propagation_distance'], 30e3, 'Propagation Distance not saved correctly')
            self.assertEqual (f.attrs['resolution'], 0.06875, 'Resolution not saved correctly')
            self.assertEqual (f.attrs['min_observing_wavelengths'], 550.0e-9, 'Min. Wavelength not saved correctly')
            self.assertEqual (f.attrs['max_observing_wavelengths'], 550.0e-9, 'Max. Wavelength not saved correctly')
            self.assertEqual (f.attrs['magnitude'], 4, 'Magnitude not saved correctly')
            self.assertEqual (f.attrs['pupil_support_size'], 256, 'Pupil Support Size not saved correctly')

    def test_star_data(self):
        with h5py.File(self.filename, "r") as f:
            self.assertEqual (f['star_images'].shape, (2000,256,256), 'Convolved data not shaped correctly')


    def test_zernike_data(self):
        with h5py.File(self.filename, "r") as f:
            self.assertEqual (f['zernike_coefficients'].shape, (10, 2000), 'Zernike coefficients not shaped correctly')


if __name__ == '__main__':
    unittest.main()
