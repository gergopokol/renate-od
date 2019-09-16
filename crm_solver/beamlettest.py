from crm_solver.beamlet import Beamlet
import unittest
import numpy
import scipy
import utility.convert as uc
import pandas


class BeamletTest(unittest.TestCase):
    EXPECTED_ATTR = ['param', 'components', 'profiles', 'coefficient_matrix', 'atomic_db', 'initial_condition']

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
