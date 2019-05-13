import unittest
import numpy
from crm_solver.beamlet import Beamlet


class BeamletTest(unittest.TestCase):

    DECIMALS_6 = 6
    Lithium_testcase = 'D_plasma_Li_beam@60keV.xml'
    Sodium_testcase = 'H_plasma_Na_beam@60keV.xml'
    Hydrogen_testcase = 'He_plasma_H_beam@60keV.xml'

    def test_lithium_beamlet(self):
        # TODO: Implement code to benchmark lithium beam evolution: RENATE with RENATE-OD
        pass

    def test_sodium_beamlet(self):
        # TODO: Implement code to benchmark sodium beam evolution: RENATE with RENATE-OD
        pass

    def test_hydrogen_beamlet(self):
        # TODO: Implement code to benchmark hydrogen beam evolution: RENATE with RENATE-OD
        pass
