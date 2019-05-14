import unittest
from crm_solver.beamlet import Beamlet


class BeamletTest(unittest.TestCase):

    DECIMALS_PRECISSION = 0
    lithium_testcase = 'beamlet/D_plasma_Li_beam@60keV.xml'
    sodium_testcase = 'beamlet/H_plasma_Na_beam@60keV.xml'
    hydrogen_testcase = 'beamlet/He_plasma_H_beam@60keV.xml'

    def test_lithium_beamlet(self):
        actual = Beamlet(data_path=self.lithium_testcase)
        levels = actual.profiles.filter(like='RENATE', axis=1)
        nr_levels = len(levels.keys())
        for level in range(nr_levels):
            renate = actual.profiles['RENATE level ' + str(level)]
            renate_od = actual.profiles['level ' + str(level)]/actual.profiles['level 0'].max()
            for index in range(len(renate)-1):
                self.assertAlmostEqual(renate[index], renate_od[index], self.DECIMALS_PRECISSION)


    def test_sodium_beamlet(self):
        actual = Beamlet(data_path=self.sodium_testcase)
        levels = actual.profiles.filter(like='RENATE', axis=1)
        nr_levels = len(levels.keys())
        for level in range(nr_levels):
            renate = actual.profiles['RENATE level ' + str(level)]
            renate_od = actual.profiles['level ' + str(level)]/actual.profiles['level 0'].max()
            for index in range(len(renate)-1):
                self.assertAlmostEqual(renate[index], renate_od[index], self.DECIMALS_PRECISSION)

    def test_hydrogen_beamlet(self):
        actual = Beamlet(data_path=self.hydrogen_testcase)
        levels = actual.profiles.filter(like='RENATE', axis=1)
        nr_levels = len(levels.keys())
        for level in range(nr_levels):
            renate = actual.profiles['RENATE level ' + str(level)]
            renate_od = actual.profiles['level ' + str(level)]/actual.profiles['level 0'].max()
            for index in range(len(renate)-1):
                self.assertAlmostEqual(renate[index], renate_od[index], self.DECIMALS_PRECISSION)
