import unittest
from crm_solver.beamlet import Beamlet
from crm_solver.crmtestcase import CrmTestCase
from copy import deepcopy


class CrmSystemTest(CrmTestCase):
    path = 'test_dataset/crm_systemtests/renate_idl/scenario-standard_plasma-H_energy-100_beam-Na_profile.xml'

    def test(self):
        reference = Beamlet(data_path=self.path, solver='disregard')
        actual_source = deepcopy(reference)
        actual = Beamlet(param=actual_source.param, profiles=actual_source.profiles,
                         components=actual_source.components, atomic_db=actual_source.atomic_db, solver='numerical')
        self.assertAlmostEqualRateEvolution(actual, reference)


class CrmAcceptanceTest(unittest.TestCase):

    RELATIVE_PRECISION = 0.001
    lithium_testcase = 'beamlet/acceptancetest/scenario-standard_plasma-H_energy-100_beam-Li_profile.xml'
    sodium_testcase = 'beamlet/acceptancetest/scenario-standard_plasma-H_energy-100_beam-Na_profile.xml'
    hydrogen_testcase = 'beamlet/acceptancetest/scenario-standard_plasma-H_energy-100_beam-H_profile.xml'
    deuterium_testcase = 'beamlet/acceptancetest/scenario-standard_plasma-H_energy-100_beam-D_profile.xml'
    tritium_testcase = 'beamlet/acceptancetest/scenario-standard_plasma-H_energy-100_beam-T_profile.xml'

    def test_lithium_beamlet(self):
        actual = Beamlet(data_path=self.lithium_testcase)
        for level in range(actual.atomic_db.atomic_levels):
            renate = actual.profiles['RENATE level ' + str(level)]
            renate_od = actual.profiles['level ' + actual.atomic_db.inv_atomic_dict[level]] / \
                actual.profiles['level '+actual.atomic_db.inv_atomic_dict[0]].max()
            for index in range(len(renate)-1):
                if renate[index] != 0:
                    self.assertTrue(abs(renate[index] - renate_od[index])/renate[index] <= self.RELATIVE_PRECISION)

    def test_sodium_beamlet(self):
        actual = Beamlet(data_path=self.sodium_testcase)
        for level in range(actual.atomic_db.atomic_levels):
            renate = actual.profiles['RENATE level ' + str(level)]
            renate_od = actual.profiles['level ' + actual.atomic_db.inv_atomic_dict[level]] / \
                actual.profiles['level '+actual.atomic_db.inv_atomic_dict[0]].max()
            for index in range(len(renate)-1):
                if renate[index] != 0:
                    self.assertTrue(abs(renate[index] - renate_od[index])/renate[index] <= self.RELATIVE_PRECISION)

    def test_hydrogen_beamlet(self):
        actual = Beamlet(data_path=self.hydrogen_testcase)
        for level in range(actual.atomic_db.atomic_levels):
            renate = actual.profiles['RENATE level ' + str(level)]
            renate_od = actual.profiles['level ' + actual.atomic_db.inv_atomic_dict[level]] /\
                actual.profiles['level '+actual.atomic_db.inv_atomic_dict[0]].max()
            for index in range(len(renate)-1):
                if renate[index] != 0:
                    self.assertTrue(abs(renate[index] - renate_od[index])/renate[index] <= self.RELATIVE_PRECISION)

    def test_deuterium_beamlet(self):
        actual = Beamlet(data_path=self.deuterium_testcase)
        for level in range(actual.atomic_db.atomic_levels):
            renate = actual.profiles['RENATE level ' + str(level)]
            renate_od = actual.profiles['level ' + actual.atomic_db.inv_atomic_dict[level]] / \
                actual.profiles['level '+actual.atomic_db.inv_atomic_dict[0]].max()
            for index in range(len(renate)-1):
                if renate[index] != 0:
                    self.assertTrue(abs(renate[index] - renate_od[index])/renate[index] <= self.RELATIVE_PRECISION)

    def test_tritium_beamlet(self):
        actual = Beamlet(data_path=self.tritium_testcase)
        for level in range(actual.atomic_db.atomic_levels):
            renate = actual.profiles['RENATE level ' + str(level)]
            renate_od = actual.profiles['level ' + actual.atomic_db.inv_atomic_dict[level]] /\
                actual.profiles['level '+actual.atomic_db.inv_atomic_dict[0]].max()
            for index in range(len(renate)-1):
                if renate[index] != 0:
                    self.assertTrue(abs(renate[index] - renate_od[index])/renate[index] <= self.RELATIVE_PRECISION)
