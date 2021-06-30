from scipy.interpolate import interp2d
from utility.getdata import GetData
import matplotlib.pyplot as plt

try:
    import imas
    from imas_utility.idsequilibrium import EquilibriumIds
    IMAS_FLAG = True
except ImportError:
    IMAS_FLAG = False


class Equilibrium(object):
    def __init__(self, source='renate', data_id=None, data_path='device_data/test/test_idl_data.sav'):
        self.data_source = source
        self.data_path = data_path
        self.imas_flag = IMAS_FLAG
        if data_id is not None:
            self.__unpack_data_id(data_id)

    def __unpack_data_id(self, data_id):
        self.shot = data_id[0]
        self.run = data_id[1]
        self.user = data_id[2]
        self.machine = data_id[3]
        self.time = data_id[4]

    def load_poloidal_flux_map(self):
        if self.data_source == 'renate':
            self.__load_renate_idl_flux_map_data()
        elif self.data_source == 'ids' and self.imas_flag:
            self.__load_ids_flux_map_data()
        else:
            raise ValueError('The data source requested: ' + str(self.data_source) + ' is not supported. ')
        self.__generate_point_to_flux_interpolator()

    def __load_renate_idl_flux_map_data(self):
        idl_flux_map_data = GetData(data_path_name=self.data_path).data
        self.R = idl_flux_map_data['psi_rz_matrix'][0][1]
        self.Z = idl_flux_map_data['psi_rz_matrix'][0][2]
        self.Psi = idl_flux_map_data['psi_rz_matrix'][0][0]

    def __generate_point_to_flux_interpolator(self):
        self.point2flux = interp2d(self.R, self.Z, self.Psi)

    def __load_ids_flux_map_data(self):
        if not hasattr(self, 'shot'):
            raise AttributeError('Requested attribute: shot, is missing. Check existence of data_id.')
        imas_equilibrium = EquilibriumIds(shot=self.shot, run=self.run, machine=self.machine, user=self.user)
        self.R, self.Z = imas_equilibrium.get_2d_equilibrium_grid(self.time)
        self.Psi = imas_equilibrium.get_normalized_2d_flux(self.time)

    def show_rzflux_map(self, levels=40, lcfs=False):
        plt.contour(self.R, self.Z, self.Psi, levels)
        if lcfs:
            plt.contour(self.R, self.Z, self.Psi, [1], colors='r')
        plt.title('2D flux surface map')
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.axis('equal')
        plt.show()
