from crm_solver.atomic_db import AtomicDB
import unittest
from lxml import etree


class AtomicDBTest(unittest.TestCase):
    EXPECTED_ATTR = ['energy', 'param', 'species', 'electron_impact_loss', 'ion_impact_loss',
                     'atomic_dict', 'rate_type', 'electron_impact_trans', 'ion_impact_trans',
                     'spontaneous_trans', 'atomic_levels', 'charged_states', 'inv_atomic_dict']
    EXPECTED_ATOM = ['dummy', 'Li', 'Na', 'T', 'H', 'D']
    EXPECTED_ATOMIC_LEVELS = [3, 9, 8, 6, 6, 6]
    EXPECTED_ENERGY = '60'

    def test_all_attributes(self):
        actual = AtomicDB()
        for attr in self.EXPECTED_ATTR:
            assert hasattr(actual, attr)

    def test_atomic_species(self):
        actual = AtomicDB()
        assert isinstance(actual.species, str)
        assert actual.species in self.EXPECTED_ATOM
        assert isinstance(actual.species, str)

    def test_atomic_levels(self):
        actual = AtomicDB()
        assert isinstance(actual.atomic_dict, dict)
        assert isinstance(actual.inv_atomic_dict, dict)
        assert isinstance(actual.atomic_levels, int)
        for index in range(len(self.EXPECTED_ATOM)):
            actual.param.getroot().find('body').find('beamlet_species').text = self.EXPECTED_ATOM[index]
            atom = AtomicDB(param=actual.param)
            self.assertEqual(atom.atomic_levels, self.EXPECTED_ATOMIC_LEVELS[index])
