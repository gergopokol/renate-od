import  unittest
from crm_solver.atomic_db import AtomicDB
from crm_solver.coefficientmatrix import CoefficientMatrix
from utility.getdata import GetData
from crm_solver.beamlet import Beamlet
import numpy
import  pandas
from scipy.interpolate import interp1d


class CoefficientMatrixTest(unittest.TestCase):
    FROM_LEVEL_0 = 0
    ATOMIC_DB = AtomicDB(data_path='dummy_data_path')
    BEAMLET_PARAM = GetData(data_path_name="beamlet/dummy_param.xml").data
    HDF5_PATH = BEAMLET_PARAM.getroot().find('body').find('beamlet_source').text
    BEAMLET_PROFILES = GetData(data_path_name=HDF5_PATH, data_key=['profiles']).data
    PLASMA_COMPONENTS = GetData(data_path_name=HDF5_PATH, data_key=['components']).data

    def test_spontaneous_trans(self):
        coeff_matrix = CoefficientMatrix(beamlet_param=self.BEAMLET_PARAM, beamlet_profiles=self.BEAMLET_PROFILES,
                                         plasma_components=self.PLASMA_COMPONENTS, atomic_db=self.ATOMIC_DB)
        self.assertEqual(coeff_matrix.spontaneous_trans_np.shape, self.ATOMIC_DB.spontaneous_trans.shape)
        for i in range(self.ATOMIC_DB.atomic_levels):
            for j in range(self.ATOMIC_DB.atomic_levels):
                if i < j+1:
                    self.assertEqual(coeff_matrix.spontaneous_trans_np[i, j], 0)

    def test_electron_loss(self):
        coeff_matrix = CoefficientMatrix(beamlet_param=self.BEAMLET_PARAM,
                                         beamlet_profiles=self.BEAMLET_PROFILES,
                                         plasma_components=self.PLASMA_COMPONENTS, atomic_db=self.ATOMIC_DB)
        self.assertEqual(coeff_matrix.electron_impact_trans_np.shape.size, 2)
        self.assertEqual(coeff_matrix.electron_impact_loss_np.shape[0], self.ATOMIC_DB.electron_impact_loss.shape[1])
        for i in range(self.ATOMIC_DB.atomic_levels):
            for k in range(self.BEAMLET_PROFILES['beamlet_grid'].size):
                self.assertEqual(coeff_matrix.electron_impact_loss_np[i, k], self.ATOMIC_DB.electron_impact_loss[i][0](
                    self.BEAMLET_PROFILES['electron']['temperature']['eV'][k]))

    def test_electron_trans(self):
        coeff_matrix = CoefficientMatrix(beamlet_param=self.BEAMLET_PARAM,
                                         beamlet_profiles=self.BEAMLET_PROFILES,
                                         plasma_components=self.PLASMA_COMPONENTS, atomic_db=self.ATOMIC_DB)
        self.assertEqual(coeff_matrix.electron_impact_trans_np.shape.size, 3)
        self.assertEqual(coeff_matrix.electron_impact_trans_np.shape, self.ATOMIC_DB.electron_impact_trans.shape[:2])
        for i in range(self.ATOMIC_DB.atomic_levels):
            for j in range(self.ATOMIC_DB.atomic_levels):
                for k in range(self.BEAMLET_PROFILES['beamlet_grid'].size):
                    self.assertEqual(coeff_matrix.electron_impact_trans_np[i, j, k], self.ATOMIC_DB.electron_impact_trans[i][j](
                        self.BEAMLET_PROFILES['electron']['temperature']['eV'][:]))






