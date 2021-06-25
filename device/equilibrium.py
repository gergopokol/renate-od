import numpy as np
from scipy.interpolate import interp2d
from utility.getdata import GetData
import matplotlib.pyplot as plt


class Equilibrium(object):
    def __init__(self, source='renate', data_id=None, data_path='device_data/test/test_idl_data.sav'):
        self.data_source = source
        self.data_path = data_path
        self.data_id = data_id

    def load_poloidal_flux_map(self):
        if self.data_source == 'renate':
            self.__load_renate_IDL_flux_map_data()
        elif self.data_source == 'ids':
            self.__load_ids_flux_map_data()
        else:
            raise ValueError('The data source requested: ' + str(self.data_source) + ' is not supported. ')
        self.__generate_point_to_flux_interpolator()

    def __load_renate_IDL_flux_map_data(self):
        idl_flux_map_data = GetData(data_path_name=self.data_path).data
        self.R = idl_flux_map_data['psi_rz_matrix'][0][1]
        self.Z = idl_flux_map_data['psi_rz_matrix'][0][2]
        self.Psi = idl_flux_map_data['psi_rz_matrix'][0][0]

    def __generate_point_to_flux_interpolator(self):
        self.point2flux = interp2d(self.R, self.Z, self.Psi)

    def __load_ids_flux_map_data(self):
        pass

    def show_rzflux_map(self, levels=40, lcfs=False):
        plt.contour(self.R, self.Z, self.Psi, levels)
        if lcfs:
            plt.contour(self.R, self.Z, self.Psi, [1], colors='r')
        plt.title('2D flux surface map')
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.axis('equal')
        plt.show()
