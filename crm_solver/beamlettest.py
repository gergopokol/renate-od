from crm_solver.beamlet import Beamlet
import unittest
import numpy
import scipy
import utility.convert as uc
import pandas


class BeamletTest(unittest.TestCase):
    EXPECTED_ATTR = ['param', 'components', 'profiles', 'coefficient_matrix', 'atomic_db', 'initial_condition']
    EXPECTED_INITIAL_CONDITION = [4832583711.839067, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_attributes(self):
        actual = Beamlet()
        for attr in self.EXPECTED_ATTR:
            assert hasattr(actual, attr)

    def test_initial_conditions(self):
        actual = Beamlet()
        self.assertIsInstance(actual.initial_condition, list, msg='Initial condition is of wrong type. '
                                                                  'Expected type: list')
        self.assertEqual(len(actual.initial_condition), actual.atomic_db.atomic_levels, msg='Initial conditions must '
                         'match number of atomic levels.')
        for element in range(len(actual.initial_condition)):
            self.assertIsInstance(actual.initial_condition[element], float, msg='Expected type for initial'
                                                                                ' conditions is float.')
            self.assertEqual(actual.initial_condition[element], self.EXPECTED_INITIAL_CONDITION[element],
                             msg='Computed Init conditions do not match expected init conditions.')

    def test_param_xml(self):
        pass
