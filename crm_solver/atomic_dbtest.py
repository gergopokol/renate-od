from crm_solver.atomic_db import AtomicDB
import unittest


class AtomicDBTest(unittest.TestCase):
    EXPECTED_ATTR = ['energy', 'param', 'species', 'electron_impact_loss', 'ion_impact_loss',
                     'atomic_dict', 'rate_type', 'electron_impact_trans', 'ion_impact_trans',
                     'spontaneous_trans', 'atomic_levels', 'charged_states', 'inv_atomic_dict']
    EXPECTED_ATOM = ['dummy', 'Li', 'Na', 'T', 'H', 'D']

    def test_all_attributes(self):
        actual = AtomicDB()
        for attr in self.EXPECTED_ATTR:
            assert hasattr(actual, attr)

    def test_atomic_species(self):
        actual = AtomicDB()
        assert isinstance(actual.species, str)
        assert actual.species in self.EXPECTED_ATOM

