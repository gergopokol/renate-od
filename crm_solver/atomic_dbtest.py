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
    INTERPOLATION_TEST_TEMPERATURE = [0, 1, 2, 2.5, 3, 8, 10]
    EXPECTED_ELECTRON_IMPACT_LOSS = [[11., 111., 211., 261., 311., 811., 1011.],
                                     [21., 121., 221., 271., 321., 821., 1021.],
                                     [31., 131., 231., 281., 331., 831., 1031.]]
    EXPECTED_ELECTRON_IMPACT_TRANS = [[[0.,    0.,   0.,   0.,   0.,   0.,    0.],
                                       [12., 112., 212., 262., 312., 812., 1012.],
                                       [13., 113., 213., 263., 313., 813., 1013.]],
                                      [[21., 121., 221., 271., 321., 821., 1021.],
                                       [0.,    0.,   0.,   0.,   0.,   0.,    0.],
                                       [23., 123., 223., 273., 323., 823., 1023.]],
                                      [[31., 131., 231., 281., 331., 831., 1031.],
                                       [32., 132., 232., 282., 332., 832., 1032.],
                                       [0.,    0.,   0.,   0.,   0.,   0.,    0.]]]
    EXPECTED_ION_IMPACT_LOSS = [[[12., 112., 212., 262., 312., 812., 1012.],
                                 [13., 113., 213., 263., 313., 813., 1013.],
                                 [14., 114., 214., 264., 314., 814., 1014.],
                                 [15., 115., 215., 265., 315., 815., 1015.],
                                 [16., 116., 216., 266., 316., 816., 1016.]],
                                [[22., 122., 222., 272., 322., 822., 1022.],
                                 [23., 123., 223., 273., 323., 823., 1023.],
                                 [24., 124., 224., 274., 324., 824., 1024.],
                                 [25., 125., 225., 275., 325., 825., 1025.],
                                 [26., 126., 226., 276., 326., 826., 1026.]],
                                [[32., 132., 232., 282., 332., 832., 1032.],
                                 [33., 133., 233., 283., 333., 833., 1033.],
                                 [34., 134., 234., 284., 334., 834., 1034.],
                                 [35., 135., 235., 285., 335., 835., 1035.],
                                 [36., 136., 236., 286., 336., 836., 1036.]]]

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
        self.assertIsInstance(actual.electron_impact_loss, list)
        self.assertEqual(len(actual.electron_impact_loss), actual.atomic_levels)
        for index in range(actual.atomic_levels):
            self.assertIsInstance(actual.electron_impact_loss[index], scipy.interpolate.interp1d)

    def test_electron_impact_transition_terms(self):
        actual = AtomicDB()
        self.assertIsInstance(actual.electron_impact_trans, list)
        self.assertEqual(len(actual.electron_impact_trans), actual.atomic_levels)
        for from_level in range(actual.atomic_levels):
            self.assertIsInstance(actual.electron_impact_trans[from_level], list)
            self.assertEqual(len(actual.electron_impact_trans[from_level]), actual.atomic_levels)
            for to_level in range(actual.atomic_levels):
                self.assertIsInstance(actual.electron_impact_trans[from_level][to_level], scipy.interpolate.interp1d)

    def test_charged_state_library(self):
        actual = AtomicDB()
        self.assertIsInstance(actual.charged_states, list)
        for state in range(len(actual.charged_states)):
            self.assertIsInstance(actual.charged_states[state], str)
            self.assertEqual(actual.charged_states[state], 'charge-'+str(state+1))

    def test_ion_impact_loss_terms(self):
        actual = AtomicDB()
        self.assertIsInstance(actual.ion_impact_loss, list)
        self.assertEqual(len(actual.ion_impact_loss), actual.atomic_levels)
        for from_level in range(actual.atomic_levels):
            self.assertIsInstance(actual.ion_impact_loss[from_level], list)
            self.assertEqual(len(actual.ion_impact_loss[from_level]), len(actual.charged_states))
            for charge in range(len(actual.charged_states)):
                self.assertIsInstance(actual.ion_impact_loss[from_level][charge], scipy.interpolate.interp1d)

    def test_ion_impact_transition_terms(self):
        actual = AtomicDB()
        self.assertIsInstance(actual.ion_impact_trans, list)
        self.assertEqual(len(actual.ion_impact_trans), actual.atomic_levels)
        for from_level in range(actual.atomic_levels):
            self.assertIsInstance(actual.ion_impact_trans[from_level], list)
            self.assertEqual(len(actual.ion_impact_trans[from_level]), actual.atomic_levels)
            for to_level in range(actual.atomic_levels):
                self.assertIsInstance(actual.ion_impact_trans[from_level][to_level], list)
                self.assertEqual(len(actual.ion_impact_trans[from_level][to_level]), len(actual.charged_states))
                for charge in range(len(actual.charged_states)):
                    self.assertIsInstance(actual.ion_impact_trans[from_level][to_level][charge],
                                          scipy.interpolate.interp1d)

    def test_electron_impact_loss_interpolator(self):
        actual = AtomicDB(data_path='beamlet/dummy0001.xml')
        for level in range(actual.atomic_levels):
            rates = actual.electron_impact_loss[level](self.INTERPOLATION_TEST_TEMPERATURE)
            for element_index in range(len(rates)):
                self.assertEqual(rates[element_index], self.EXPECTED_ELECTRON_IMPACT_LOSS[level][element_index])

    def test_electron_impact_transition_interpolator(self):
        actual = AtomicDB(data_path='beamlet/dummy0001.xml')
        for from_level in range(actual.atomic_levels):
            for to_level in range(actual.atomic_levels):
                rates = actual.electron_impact_trans[from_level][to_level](self.INTERPOLATION_TEST_TEMPERATURE)
                for element_index in range(len(rates)):
                    self.assertEqual(rates[element_index],
                                     self.EXPECTED_ELECTRON_IMPACT_TRANS[from_level][to_level][element_index])

    def test_ion_impact_loss_interpolator(self):
        actual = AtomicDB(data_path='beamlet/dummy0001.xml')
        for level in range(actual.atomic_levels):
            for charge in range(len(actual.charged_states)):
                rates = actual.ion_impact_loss[level][charge](self.INTERPOLATION_TEST_TEMPERATURE)
                for element_index in range(len(rates)):
                    self.assertEqual(rates[element_index], self.EXPECTED_ION_IMPACT_LOSS[level][charge][element_index])

    def test_ion_impact_transition_interpolator(self):
        pass
