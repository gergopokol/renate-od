from crm_solver.beamlet import Beamlet
import unittest
from lxml import etree
from crm_solver.atomic_db import AtomicDB
import pandas
import numpy


class BeamletTest(unittest.TestCase):
    EXPECTED_ATTR = ['param', 'components', 'profiles', 'coefficient_matrix', 'atomic_db', 'initial_condition']
    EXPECTED_INITIAL_CONDITION = [4832583711.839067, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    EXPECTED_PARAM_ATTR = ['beamlet_source', 'beamlet_energy', 'beamlet_species', 'beamlet_current']
    EXPECTED_COMPONENTS_KEYS = ['q', 'Z', 'A']
    EXPECTED_COMPONENTS_SPECIES = ['electron', 'ion1', 'ion2']
    EXPECTED_PROFILES_LENGTH = 101
    EXPECTED_PROFILES_KEYS = [('beamlet grid', 'distance', 'm'),
                              ('electron', 'density', 'm-3'),
                              ('electron', 'temperature', 'eV'),
                              ('ion1', 'density', 'm-3'),
                              ('ion1', 'temperature', 'eV'),
                              ('ion2', 'density', 'm-3'),
                              ('ion2', 'temperature', 'eV')]

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
        actual = Beamlet()
        self.assertIsInstance(actual.param, etree._ElementTree, msg='Expected type for param input is xml elementtree.')
        for param_attribute in self.EXPECTED_PARAM_ATTR:
            self.assertIsInstance(actual.param.getroot().find('body').find(param_attribute).text, str,
                                  msg='Failed to load or find attribut '+param_attribute+' in xml file.')

    def test_atomic_db(self):
        actual = Beamlet()
        self.assertIsInstance(actual.atomic_db, AtomicDB)

    def test_components(self):
        actual = Beamlet()
        self.assertIsInstance(actual.components, pandas.core.frame.DataFrame)
        description = actual.components.keys()
        species = actual.components.T.keys()
        self.assertEqual(len(description), 3, msg='')
        for key in range(len(description)):
            self.assertEqual(description[key], self.EXPECTED_COMPONENTS_KEYS[key], msg='The index: ' +
                             description[key] + ' is a mismatch for '+self.EXPECTED_COMPONENTS_KEYS[key])
        self.assertEqual(len(species), len(actual.atomic_db.ion_impact_loss[0])+1)
        for key in range(len(species)):
            self.assertEqual(species[key], self.EXPECTED_COMPONENTS_SPECIES[key],
                             msg='Mismatch of expected plasma species.')
        for component in species:
            for coordinate in description:
                self.assertIsInstance(actual.components[coordinate][component], numpy.int64,
                                      msg='Type mismatch of pandas components content.')

    def test_profiles(self):
        actual = Beamlet(solver='disregard')
        self.assertIsInstance(actual.profiles, pandas.core.frame.DataFrame,
                              msg='Expected data type of profiles is: pandas DataFrame.')
        self.assertEqual(len(actual.profiles), self.EXPECTED_PROFILES_LENGTH)
        self.assertTupleEqual(actual.profiles.shape, (self.EXPECTED_PROFILES_LENGTH,
                                                      2*len(self.EXPECTED_COMPONENTS_KEYS)+1))
        self.assertIsInstance(actual.profiles.axes[0], pandas.Int64Index,
                              msg='Expected data type of X - axis for profiles is Int64Index.')
        self.assertIsInstance(actual.profiles.axes[1], pandas.MultiIndex,
                              msg='Expected data type of Y - axis for profiles is MultiIndex.')
        for key in range(len(actual.profiles.keys())):
            self.assertTupleEqual(actual.profiles.keys()[key], self.EXPECTED_PROFILES_KEYS[key])

    def test_analytical_solver(self):
        with self.assertRaises(NotImplementedError):
            actual = Beamlet(solver='analytical')

    def test_numerical_solver(self):
        actual = Beamlet()
        self.assertTupleEqual(actual.profiles.shape, (self.EXPECTED_PROFILES_LENGTH,
                              2*len(self.EXPECTED_COMPONENTS_KEYS)+1+actual.atomic_db.atomic_levels))
        for key_index in range(2*len(self.EXPECTED_COMPONENTS_KEYS)+1,
                               len(actual.profiles.keys())):
            self.assertEqual(actual.profiles.keys()[key_index][0], 'level '+actual.atomic_db.inv_atomic_dict[key_index -
                             2*len(self.EXPECTED_COMPONENTS_KEYS)-1])
        for level in range(actual.atomic_db.atomic_levels):
            self.assertIsInstance(actual.profiles['level '+actual.atomic_db.inv_atomic_dict[level]],
                                  pandas.core.series.Series)
            self.assertEqual(len(actual.profiles['level '+actual.atomic_db.inv_atomic_dict[level]]),
                             self.EXPECTED_PROFILES_LENGTH)

    def test_not_supported_solver(self):
        with self.assertRaises(Exception):
            actual = Beamlet(solver='not-supported')

    def test_attenuation_caluculator(self):
        pass

    def test_emission_calculator(self):
        pass

    def test_relative_population_calculator(self):
        pass
