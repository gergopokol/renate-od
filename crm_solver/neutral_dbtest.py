import unittest
from utility.input import AtomicInput
from crm_solver.atomic_db import NeutralDB


class NeutralDBTest(unittest.TestCase):

    INPUT_ENERGY = 50
    INPUT_PROJECTILE = 'H'
    INPUT_CURRENT = 0.001
    INPUT_SOURCE = 'Internal'
    INPUT_PARAM_NAME = 'NeutralDB test'
    INPUT_H = [0, 1, 1, None]
    INPUT_H2 = [0, 1, 1, 'H2']

    EXPECTED_KEY_2 = 2
    EXPECTED_KEY_1 = 1
    EXPECTED_KEY_NAME_1 = 'neutral1'
    EXPECTED_KEY_NAME_2 = ['neutral1', 'neutral2']

    def setUp(self):
        self.input = AtomicInput(energy=self.INPUT_ENERGY, projectile=self.INPUT_PROJECTILE, current=self.INPUT_CURRENT,
                                 param_name=self.INPUT_PARAM_NAME, source=self.INPUT_SOURCE)

    def tearDown(self):
        del self.input

    def test_neutral_atom_addition(self):
        self.input.add_target_component(charge=self.INPUT_H[0], atomic_number=self.INPUT_H[1],
                                        mass_number=self.INPUT_H[2], molecule_name=self.INPUT_H[3])
        actual_param, actual_components = self.input.get_atomic_db_input()
        actual_db = NeutralDB(param=actual_param, components=actual_components)
        self.assertEqual(actual_db.neutral_target_count, self.EXPECTED_KEY_1,
                         msg='Only a single neutral component was expected.')
        self.assertEqual(list(actual_db.neutral_cross_sections.keys())[0], self.EXPECTED_KEY_NAME_1,
                         msg='Expected <neutral1> for single neutral component addition to the Neutral DB.')

    def test_neutral_molecule_addition(self):
        self.input.add_target_component(charge=self.INPUT_H2[0], atomic_number=self.INPUT_H2[1],
                                        mass_number=self.INPUT_H2[2], molecule_name=self.INPUT_H2[3])
        actual_param, actual_components = self.input.get_atomic_db_input()
        actual_db = NeutralDB(param=actual_param, components=actual_components)
        self.assertEqual(actual_db.neutral_target_count, self.EXPECTED_KEY_1,
                         msg='Only a single neutral component was expected.')
        self.assertEqual(list(actual_db.neutral_cross_sections.keys())[0], self.EXPECTED_KEY_NAME_1,
                         msg='Expected <neutral1> for single neutral component addition to the Neutral DB.')

    def test_multiple_neutral_addition(self):
        self.input.add_target_component(charge=self.INPUT_H[0], atomic_number=self.INPUT_H[1],
                                        mass_number=self.INPUT_H[2], molecule_name=self.INPUT_H[3])
        self.input.add_target_component(charge=self.INPUT_H2[0], atomic_number=self.INPUT_H2[1],
                                        mass_number=self.INPUT_H2[2], molecule_name=self.INPUT_H2[3])
        actual_param, actual_components = self.input.get_atomic_db_input()
        actual_db = NeutralDB(param=actual_param, components=actual_components)
        self.assertEqual(actual_db.neutral_target_count, self.EXPECTED_KEY_2,
                         msg='Only a single neutral component was expected.')
        self.assertListEqual(list(actual_db.neutral_cross_sections.keys()), self.EXPECTED_KEY_NAME_2,
                             msg='Expected <neutral1> for single neutral component addition to the Neutral DB.')
