from crm_solver.beamlet import Beamlet
from crm_solver.crmtestcase import CrmTestCase
from copy import deepcopy
from shutil import rmtree
import os


class CrmRegressionTest(CrmTestCase):

    EXPECTED_PRECISION = 1E-2

    def setUp(self):
        self.test_cases = ['H_test_case', 'D_test_case', 'T_test_case', 'Li_test_case', 'Na_test_case']

    def tearDown(self):
        public_folder = os.path.join(os.getcwd(), 'data', 'dummy', 'test_dataset')
        private_folder = os.path.join(os.getcwd(), 'data', 'test_dataset')
        rmtree(public_folder)
        rmtree(private_folder)

    def test_actual_to_previous_release(self):
        for test_case in self.test_cases:
            path = 'test_dataset/crm_systemtests/actual/'+test_case+'.xml'
            reference = Beamlet(data_path=path, solver='disregard')
            actual_source = deepcopy(reference)
            actual = Beamlet(param=actual_source.param, profiles=actual_source.profiles,
                             components=actual_source.components, atomic_db=actual_source.atomic_db, solver='numerical')
            msg = 'Failure for following test case: '+test_case+'\n'
            self.assertAlmostEqualRateEvolution(actual, reference, precision=self.EXPECTED_PRECISION, msg=msg)
            self.assertAlmostEqualBeamAttenuation(actual, reference, precision=self.EXPECTED_PRECISION, msg=msg)
            self.assertAlmostEqualEmissionDensity(actual, reference, precision=self.EXPECTED_PRECISION, msg=msg)
            self.assertAlmostEqualRelativePopulation(actual, reference, precision=self.EXPECTED_PRECISION, msg=msg)


class CrmAcceptanceTest(CrmTestCase):

    def setUp(self):
        self.test_cases = ['scenario-standard_plasma-H_energy-100_beam-H_profile',
                           'scenario-standard_plasma-H_energy-100_beam-D_profile',
                           'scenario-standard_plasma-H_energy-100_beam-T_profile',
                           'scenario-standard_plasma-H_energy-100_beam-Li_profile',
                           'scenario-standard_plasma-H_energy-100_beam-Na_profile']

    def tearDown(self):
        public_folder = os.path.join(os.getcwd(), 'data', 'dummy', 'test_dataset')
        private_folder = os.path.join(os.getcwd(), 'data', 'test_dataset')
        rmtree(public_folder)
        rmtree(private_folder)

    def test_actual_to_renate_idl(self):
        for test_case in self.test_cases:
            path = 'test_dataset/crm_systemtests/archive/renate_idl/'+test_case+'.xml'
            reference = Beamlet(data_path=path, solver='disregard')
            actual_source = deepcopy(reference)
            actual = Beamlet(param=actual_source.param, profiles=actual_source.profiles,
                             components=actual_source.components, atomic_db=actual_source.atomic_db, solver='numerical')
            msg = 'Failure for following test case: '+test_case+'\n'
            self.assertAlmostEqualRateEvolution(actual, reference, precision=self.EXPECTED_PRECISION, msg=msg)
