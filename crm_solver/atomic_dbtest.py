from crm_solver.atomic_db import AtomicDB
import unittest
import numpy
import scipy


class AtomicDBTest(unittest.TestCase):
    EXPECTED_ATTR = ['energy', 'param', 'species', 'electron_impact_loss', 'ion_impact_loss',
                     'atomic_dict', 'rate_type', 'electron_impact_trans', 'ion_impact_trans',
                     'spontaneous_trans', 'atomic_levels', 'charged_states', 'inv_atomic_dict']
    EXPECTED_ATOM = ['dummy', 'Li', 'Na', 'T', 'H', 'D']
    EXPECTED_ATOMIC_LEVELS = [3, 9, 8, 6, 6, 6]
    EXPECTED_ENERGY = '60'
    EXPECTED_ATOMIC_DICT = [{'1': 1, '0': 0, '2': 2},
                            {'2s': 0, '2p': 1, '3s': 2, '3p': 3, '3d': 4, '4s': 5, '4p': 6, '4d': 7, '4f': 8},
                            {'3s': 0, '3p': 1, '3d': 2, '4s': 3, '4p': 4, '4d': 5, '4f': 6, '5s': 7},
                            {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5},
                            {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5},
                            {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5}]

    def test_all_attributes(self):
        actual = AtomicDB()
        for attr in self.EXPECTED_ATTR:
            assert hasattr(actual, attr)

    def test_atomic_species(self):
        actual = AtomicDB()
        self.assertIsInstance(actual.species, str)
        self.assertIn(actual.species, self.EXPECTED_ATOM)
        self.assertIsInstance(actual.species, str)

    def test_atomic_levels(self):
        actual = AtomicDB()
        for index in range(len(self.EXPECTED_ATOM)):
            actual.param.getroot().find('body').find('beamlet_species').text = self.EXPECTED_ATOM[index]
            atom = AtomicDB(param=actual.param)
            self.assertIsInstance(atom.atomic_levels, int)
            self.assertEqual(atom.atomic_levels, self.EXPECTED_ATOMIC_LEVELS[index])
            self.assertIsInstance(atom.atomic_dict, dict)
            self.assertDictEqual(atom.atomic_dict, self.EXPECTED_ATOMIC_DICT[index])
            self.assertIsInstance(atom.inv_atomic_dict, dict)

    def test_spontaneous_trans(self):
        actual = AtomicDB()
        self.assertIsInstance(actual.spontaneous_trans, numpy.ndarray)
        self.assertEqual(actual.spontaneous_trans.ndim, 2)
        self.assertEqual(actual.atomic_levels, int(actual.spontaneous_trans.size ** 0.5))
        for to_level in range(actual.atomic_levels):
            for from_level in range(actual.atomic_levels):
                if from_level <= to_level:
                    self.assertEqual(actual.spontaneous_trans[to_level, from_level],
                                     0.0, msg='Spontaneous transition levels set wrong!!')

    def test_electron_impact_loss_terms(self):
        actual = AtomicDB()
        self.assertEqual(len(actual.electron_impact_loss), actual.atomic_levels)
        self.assertIsInstance(actual.electron_impact_loss, list)
        for index in range(actual.atomic_levels):
            self.assertIsInstance(actual.electron_impact_loss[index], scipy.interpolate.interp1d)

