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

    EXPECTED_ATTRIBUTES = ['matrix', 'electron_terms', 'ion_terms', 'photon_terms', 'beamlet_profiles',
                           'electron_impact_trans_np', 'electron_impact_loss_np', 'ion_impact_trans_np',
                           'ion_impact_loss_np']

    def setUp(self):
        self.RATE_MATRIX = CoefficientMatrix(self.BEAMLET_PARAM, self.PROFILES, self.COMPONENTS, self.ATOMIC_DB)

    def tearDown(self):
        pass

    def test_all_attributes(self):
        for attr in self.EXPECTED_ATTRIBUTES:
            assert hasattr(self.RATE_MATRIX, attr)
