import unittest
import numpy
import numpy.testing as npt
from crm_solver.beamlet import Beamlet


class BeamletTest(unittest.TestCase):

    DECIMALS_PRECISSION = 1
    lithium_testcase = 'beamlet/D_plasma_Li_beam@60keV.xml'
    sodium_testcase = 'beamlet/H_plasma_Na_beam@60keV.xml'
    hydrogen_testcase = 'beamlet/He_plasma_H_beam@60keV.xml'

    def test_lithium_beamlet(self):
        actual = Beamlet(data_path=self.lithium_testcase)
        levels = actual.profiles.filter(like='RENATE', axis=1)
        nr_levels = len(levels.keys())
        #for level in range(nr_levels):
        #    npt.assert_almost_equal(actual.profiles['level ' + str(level)],
        #                            actual.profiles['level ' + str(level)], self.DECIMALS_PRECISSION)

    def test_sodium_beamlet(self):
        actual = Beamlet(data_path=self.sodium_testcase)
        levels = actual.profiles.filter(like='RENATE', axis=1)
        nr_levels = len(levels.keys())

    def test_hydrogen_beamlet(self):
        actual = Beamlet(data_path=self.hydrogen_testcase)
        levels = actual.profiles.filter(like='RENATE', axis=1)
        nr_levels = len(levels.keys())
