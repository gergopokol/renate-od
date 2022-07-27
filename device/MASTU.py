import numpy as np
from scipy.interpolate import RegularGridInterpolator
import h5py


# 2Dprofile[r,z]


class PSFforMASTU:

    def __init__(self, efit_path):
        self.efit_data = self.read_MASTU_EFIT(efit_path)

    def read_MASTU_EFIT(self, path):
        f = h5py.File(path)
        data_dict = {}
        data_dict['time'] = f['epm']['time'][...]
        print('time'+': '+str(data_dict['time'].shape))
        for key in list(f['epm']['output']['profiles2D'].keys()):
            data_dict[key] = f['epm']['output']['profiles2D'][key][...]
            print(key+': '+str(data_dict[key].shape))
        f.close()
        return data_dict

    def make_interpolators_at_time(self, time):
        time_ind = np.where(self.efit_data['time'] <= time)[-1][-1]
        self.Br_interpolator = RegularGridInterpolator((self.efit_data['r'], self.efit_data['z']),
                                                       self.efit_data['Br'][time_ind])
        self.Bz_interpolator = RegularGridInterpolator((self.efit_data['r'], self.efit_data['z']),
                                                       self.efit_data['Bz'][time_ind])
        self.Bphi_interpolator = RegularGridInterpolator((self.efit_data['r'], self.efit_data['z']),
                                                         self.efit_data['Bphi'][time_ind])
