import unittest
from crm_solver.beamlet import Beamlet


class BeamletTest(unittest.TestCase):

    EXPECTED_PRECISION = [0.15, 0.001, 0.3]
    INPUT_TESTCASES = ['beamlet/D_plasma_Li_beam@60keV.xml',
                       'beamlet/H_plasma_Na_beam@60keV.xml',
                       'beamlet/He_plasma_H_beam@60keV.xml']

    def test_renate_rod_benchmark(self):
        for testcase in range(len(self.INPUT_TESTCASES)):
            actual = Beamlet(data_path=self.INPUT_TESTCASES[testcase])
            for level in actual.atomic_db.atomic_dict.keys():
                renate = actual.profiles['RENATE level ' + str(actual.atomic_db.atomic_dict[level])]
                renate_od = actual.profiles['level ' + level] / actual.profiles['level ' +
                                                                                actual.atomic_db.inv_atomic_dict[0]][0]
                for index in range(1, len(renate)-1):
                    self.assertTrue(abs(renate_od[index] - renate[index])/renate_od[index] <=
                                    self.EXPECTED_PRECISION[testcase])
