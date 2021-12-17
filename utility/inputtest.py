import pandas
import unittest
import numpy
from lxml import etree
from utility.input import AtomicInput, BeamletInput


class AtomicInputTest(unittest.TestCase):
    INPUT_ENERGY = 50
    INPUT_PROJECTILE = 'Li'
    INPUT_CURRENT = 0.001
    INPUT_SOURCE = 'test_input'
    INPUT_PARAM_NAME = 'Li_atomic_testcase'
    INPUT_CHARGE = [-1, 0, 0, 1, 2]
    INPUT_ATOMIC_NUMBER = [0, 1, 1, 1, 2]
    INPUT_MASS_NUMBER = [0, 1, 2, 2, 4]
    INPUT_MOLECULE_NAME = [None, None, 'D2', None, None]

    EXPECTED_COL = ['q', 'Z', 'A', 'Molecule']
    EXPECTED_INDEX = ['electron', 'neutral1', 'neutral2', 'ion1', 'ion2']

    def setUp(self):
        self.input = AtomicInput(energy=self.INPUT_ENERGY, projectile=self.INPUT_PROJECTILE, source=self.INPUT_SOURCE,
                                 current=self.INPUT_CURRENT, param_name=self.INPUT_PARAM_NAME)
        self.add_components()

    def tearDown(self):
        del self.input

    def test_return_atomic_db_input(self):
        param, components = self.input.get_atomic_db_input()
        self.assertIsInstance(param, etree._ElementTree, msg='The expected parameters as input for the '
                                                             'AtomicDB is of <etree._Element> type.')
        self.assertIsInstance(components, pandas.DataFrame, msg='The expected type for the plasma '
                                                                'components is <pandas.Dataframe>')

    def test_param_content(self):
        param, components = self.input.get_atomic_db_input()
        self.assertEqual(param.getroot().find('head').find('id').text, self.INPUT_PARAM_NAME,
                         msg='The param xml is expected to hold the beamlet name.')
        self.assertEqual(int(param.getroot().find('body').find('beamlet_energy').text), self.INPUT_ENERGY,
                         msg='The param xml is expected to hold the beamlet energy values.')
        self.assertEqual(param.getroot().find('body').find('beamlet_species').text, self.INPUT_PROJECTILE,
                         msg='The param xml is expected to hold the beamlet projectile type values.')
        self.assertEqual(float(param.getroot().find('body').find('beamlet_current').text), self.INPUT_CURRENT,
                         msg='The param xml is expected to hold the beamlet current type values.')
        self.assertEqual(param.getroot().find('body').find('beamlet_source').text, self.INPUT_SOURCE,
                         msg='The param xml is expected to hold the beamlet source type values.')

    def test_components_content(self):
        param, components = self.input.get_atomic_db_input()
        self.assertListEqual(list(components.keys()), self.EXPECTED_COL,
                             msg='The components dataframe does not have the expected columns.')
        self.assertListEqual(list(components.T.keys()), self.EXPECTED_INDEX,
                             msg='The components dataframe does not have the expected indexes.')

    def add_components(self):
        for index in range(len(self.INPUT_CHARGE)):
            self.input.add_target_component(charge=self.INPUT_CHARGE[index],
                                            atomic_number=self.INPUT_ATOMIC_NUMBER[index],
                                            mass_number=self.INPUT_MASS_NUMBER[index],
                                            molecule_name=self.INPUT_MOLECULE_NAME[index])


class BeamletInputTest(unittest.TestCase):
    INPUT_ENERGY = 50
    INPUT_PROJECTILE = 'Li'
    INPUT_CURRENT = 0.001
    INPUT_SOURCE = 'test_input'
    INPUT_PARAM_NAME = 'Li_atomic_testcase'
    INPUT_CHARGE = [-1, 0, 0, 1, 2]
    INPUT_ATOMIC_NUMBER = [0, 1, 1, 1, 2]
    INPUT_MASS_NUMBER = [0, 1, 2, 2, 4]
    INPUT_MOLECULE_NAME = [None, None, 'D2', None, None]

    INPUT_GRID = numpy.linspace(0, 1, num=50)
    INPUT_DENSITY_PROFILES = [numpy.ones(len(INPUT_GRID))*1E19]*len(INPUT_CHARGE)
    INPUT_TEMPERATURE_PROFILES = [numpy.ones(len(INPUT_GRID))*1E3]*len(INPUT_CHARGE)

    EXPECTED_TYPE = ['beamlet grid', 'electron', 'ion1', 'ion2', 'neutral1', 'neutral2']
    EXPECTED_PROPERTY = ['density', 'distance', 'temperature']
    EXPECTED_UNIT = ['eV', 'm', 'm-3']
    EXPECTED_NAMES = ['type', 'property', 'unit']

    def add_profiles(self):
        self.input.add_grid(self.INPUT_GRID)
        for index in range(len(self.INPUT_CHARGE)):
            self.input.add_target_profiles(charge=self.INPUT_CHARGE[index],
                                           atomic_number=self.INPUT_ATOMIC_NUMBER[index],
                                           mass_number=self.INPUT_MASS_NUMBER[index],
                                           molecule_name=self.INPUT_MOLECULE_NAME[index],
                                           density=self.INPUT_DENSITY_PROFILES[index],
                                           temperature=self.INPUT_TEMPERATURE_PROFILES[index])

    def setUp(self):
        self.input = BeamletInput(energy=self.INPUT_ENERGY, projectile=self.INPUT_PROJECTILE, source=self.INPUT_SOURCE,
                                  current=self.INPUT_CURRENT, param_name=self.INPUT_PARAM_NAME)
        self.add_profiles()

    def tearDown(self):
        del self.input

    def test_atomic_db_features(self):
        self.assertIsInstance(self.input, AtomicInput, msg='The BeamletInput is expected to be a child of AtomicInput.')

    def test_return_beamlet_input(self):
        param, components, profiles = self.input.get_beamlet_input()
        self.assertIsInstance(param, etree._ElementTree, msg='The expected parameters as input for the '
                                                             'AtomicDB is of <etree._Element> type.')
        self.assertIsInstance(components, pandas.DataFrame, msg='The expected type for the plasma '
                                                                'components is <pandas.Dataframe>')
        self.assertIsInstance(profiles, pandas.DataFrame, msg='The expected type for the plasma '
                                                              'profiles is <pandas.Dataframe>')

    def test_profiles_content(self):
        param, components, profiles = self.input.get_beamlet_input()
        self.assertIsInstance(profiles.keys(), pandas.MultiIndex, msg='The expected type for profile '
                                                                      'keys is <pandas. MultiIndex>')
        names = [profiles.keys().levels[0].name, profiles.keys().levels[1].name, profiles.keys().levels[2].name]
        self.assertListEqual(names, self.EXPECTED_NAMES, msg='The actual names of the multi index properties do not '
                                                             'match with the expected multi index names.')
        self.assertListEqual(list(profiles.keys().levels[0]), self.EXPECTED_TYPE,
                             msg='The actual profile type designations to not match the expected type designations.')
        self.assertListEqual(list(profiles.keys().levels[1]), self.EXPECTED_PROPERTY, msg='The actual profile '
                             'properties designations do not match expected property designations.')
        self.assertListEqual(list(profiles.keys().levels[2]), self.EXPECTED_UNIT, msg='The actual profile unit '
                             'designations do not match expected unit designations.')
