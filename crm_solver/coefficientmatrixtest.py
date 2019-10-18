import unittest
from crm_solver.atomic_db import AtomicDB
from crm_solver.coefficientmatrix import CoefficientMatrix
from utility.getdata import GetData
import numpy
import pandas


class CoefficientMatrixTest(unittest.TestCase):

    elements = ['1H+', '1D+', '3He+', '4He++', '9Be++++']
    q = [1, 1, 1, 2, 4]
    z = [1, 1, 2, 2, 4]
    a = [1, 2, 3, 4, 9]
    COMPONENTS = pandas.DataFrame([[-1] + q, [0] + z, [0] + a], index=['q', 'Z', 'A'],
                                  columns=['electron'] + ['ion' + str(i + 1) for i in
                                  range(len(elements))]).transpose()

    grid = numpy.array([1., 2., 3., 4., 5., 6., 7.])
    temperature = numpy.array([0, 1, 2, 2.5, 3, 8, 10])
    density = numpy.array([1]*len(grid))
    profile = numpy.stack([grid, density, temperature])

    comp = ['beamlet grid', 'electron', 'electron']
    properties = ['distance', 'density', 'temperature']
    units = ['m', 'm-3', 'eV']

    for item in range(len(a)):
        profile = numpy.vstack([profile, numpy.array([a[item]]*len(grid)), temperature])
        comp = comp + ['ion' + str(item + 1)] * 2
        properties = properties + ['density'] + ['temperature']
        units = units + ['m-3'] + ['eV']

    PROFILES = pandas.DataFrame(profile, pandas.MultiIndex.from_arrays([comp, properties, units],
                                                                       names=['type', 'property', 'unit'])).transpose()
    INPUT_DUMMY_PATH = 'beamlet/dummy0001.xml'
    ATOMIC_DB = AtomicDB(data_path=INPUT_DUMMY_PATH, components=COMPONENTS)
    BEAMLET_PARAM = GetData(data_path_name=INPUT_DUMMY_PATH).data

    EXPECTED_DECIMAL_PRECISION_6 = 6
    EXPECTED_ATTRIBUTES = ['matrix', 'electron_terms', 'ion_terms', 'photon_terms', 'beamlet_profiles',
                           'electron_impact_trans_np', 'electron_impact_loss_np', 'ion_impact_trans_np',
                           'ion_impact_loss_np']
    EXPECTED_ELECTRON_IMPACT_TERMS = [[[0.,     0.,     0.,     0.,     0.,     0.,     0.    ],
                                       [0.0012, 0.0112, 0.0212, 0.0262, 0.0312, 0.0812, 0.1012],
                                       [0.0013, 0.0113, 0.0213, 0.0263, 0.0313, 0.0813, 0.1013]],
                                      [[0.0021, 0.0121, 0.0221, 0.0271, 0.0321, 0.0821, 0.1021],
                                       [0.,     0.,     0.,     0.,     0.,     0.,     0.    ],
                                       [0.0023, 0.0123, 0.0223, 0.0273, 0.0323, 0.0823, 0.1023]],
                                      [[0.0031, 0.0131, 0.0231, 0.0281, 0.0331, 0.0831, 0.1031],
                                       [0.0032, 0.0132, 0.0232, 0.0282, 0.0332, 0.0832, 0.1032],
                                       [0.,     0.,     0.,     0.,     0.,     0.,     0.    ]]]
    EXPECTED_ELECTRON_LOSS_TERMS = [[0.0011, 0.0111, 0.0211, 0.0261, 0.0311, 0.0811, 0.1011],
                                    [0.0021, 0.0121, 0.0221, 0.0271, 0.0321, 0.0821, 0.1021],
                                    [0.0031, 0.0131, 0.0231, 0.0281, 0.0331, 0.0831, 0.1031]]
    EXPECTED_ION_LOSS_TERMS = [[[0.0012, 0.0112, 0.0212, 0.0262, 0.0312, 0.0812, 0.1012],
                                [0.0022, 0.0122, 0.0222, 0.0272, 0.0322, 0.0822, 0.1022],
                                [0.0032, 0.0132, 0.0232, 0.0282, 0.0332, 0.0832, 0.1032]],
                               [[0.0012, 0.0062, 0.0112, 0.0137, 0.0162, 0.0412, 0.0512],
                                [0.0022, 0.0072, 0.0122, 0.0147, 0.0172, 0.0422, 0.0522],
                                [0.0032, 0.0082, 0.0132, 0.0157, 0.0182, 0.0432, 0.0532]],
                               [[0.0012, 0.00453333, 0.00786667, 0.00953333, 0.0112, 0.02786667, 0.03453333],
                                [0.0022, 0.00553333, 0.00886667, 0.01053333, 0.0122, 0.02886667, 0.03553333],
                                [0.0032, 0.00653333, 0.00986667, 0.01153333, 0.0132, 0.02986667, 0.03653333]],
                               [[0.0013, 0.0113, 0.0213, 0.0263, 0.0313, 0.0813, 0.1013],
                                [0.0023, 0.0123, 0.0223, 0.0273, 0.0323, 0.0823, 0.1023],
                                [0.0033, 0.0133, 0.0233, 0.0283, 0.0333, 0.0833, 0.1033]],
                               [[0.0015, 0.0115, 0.0215, 0.0265, 0.0315, 0.0815, 0.1015],
                                [0.0025, 0.0125, 0.0225, 0.0275, 0.0325, 0.0825, 0.1025],
                                [0.0035, 0.0135, 0.0235, 0.0285, 0.0335, 0.0835, 0.1035]]]

    def setUp(self):
        self.RATE_MATRIX = CoefficientMatrix(self.BEAMLET_PARAM, self.PROFILES, self.COMPONENTS, self.ATOMIC_DB)

    def tearDown(self):
        del self.RATE_MATRIX

    def test_all_attributes(self):
        for attr in self.EXPECTED_ATTRIBUTES:
            assert hasattr(self.RATE_MATRIX, attr)

    def test_electron_impact_transition(self):
        self.assertIsInstance(self.RATE_MATRIX.electron_impact_trans_np, numpy.ndarray,
                              msg='The electron impact transition terms are not in the expected format.')
        self.assertTupleEqual(self.RATE_MATRIX.electron_impact_trans_np.shape,
                              (self.ATOMIC_DB.atomic_levels, self.ATOMIC_DB.atomic_levels,
                               self.PROFILES['beamlet grid'].size), msg='The electron impact transition '
                                                                        'terms is not dimensionally accurate.')
        numpy.testing.assert_almost_equal(self.RATE_MATRIX.electron_impact_trans_np,
                                          self.EXPECTED_ELECTRON_IMPACT_TERMS, self.EXPECTED_DECIMAL_PRECISION_6,
                                          err_msg='Interpolation failure for electron impact transitions.')

    def test_electron_impact_loss(self):
        self.assertIsInstance(self.RATE_MATRIX.electron_impact_loss_np, numpy.ndarray, msg='The electron impact '
                              'ionization terms are not in the expected format.')
        self.assertTupleEqual(self.RATE_MATRIX.electron_impact_loss_np.shape, (self.ATOMIC_DB.atomic_levels,
                              self.PROFILES['beamlet grid'].size), msg='The electron impact ionization term is '
                                                                       'not dimensionally accurate.')
        numpy.testing.assert_almost_equal(self.RATE_MATRIX.electron_impact_loss_np, self.EXPECTED_ELECTRON_LOSS_TERMS,
                                          self.EXPECTED_DECIMAL_PRECISION_6, err_msg='Interpolation failure for '
                                                                                     'electron impact loss.')

    def test_ion_impact_loss(self):
        self.assertIsInstance(self.RATE_MATRIX.electron_impact_loss_np, numpy.ndarray, msg='The ion impact '
                              'ionization terms are not in the expected format.')
        self.assertTupleEqual(self.RATE_MATRIX.ion_impact_loss_np.shape, (len(self.COMPONENTS.T.keys())-1,
                              self.ATOMIC_DB.atomic_levels, self.PROFILES['beamlet grid'].size), msg='The ion impact '
                              'ionization term is not dimensionally accurate.')
        print(self.RATE_MATRIX.ion_impact_loss_np)
        numpy.testing.assert_almost_equal(self.RATE_MATRIX.ion_impact_loss_np, self.EXPECTED_ION_LOSS_TERMS,
                                          self.EXPECTED_DECIMAL_PRECISION_6, err_msg='Interpolation failure '
                                                                                     'for ion impact loss.')
