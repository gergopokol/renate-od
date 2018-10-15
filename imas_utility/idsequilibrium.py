from imas_utility.idsinstance import ImasObject
import numpy as np
from scipy.interpolate import interp2d


class EquilibriumIds(ImasObject):
    def __init__(self, shot, run, user=None, machine=None):
        super().__init__(shot, run, user, machine)
        self.load_equilibrium_ids()

    def load_equilibrium_ids(self):
        try:
            self.equilibrium = self.imas_pointer.get('equilibrium')
        except:
            print('Equilibrium IDS not found in shot ' + str(self.shot) + ' at run ' + str(self.run))
            exit()

    def get_time_index(self, time):
        try:
            time_array = self.equilibrium.time
        except:
            print('No time array available for the requested Shot: '+str(self.shot)+' and Run: '+str(self.run))
            exit()

        if (time_array[time_array.argmin()] <= time) and (time_array[time_array.argmax()] >= time):
            return (np.abs(time_array - time)).argmin()
        else:
            print('Time value : '+str(time)+' is out of bound. Please select new time instance.')
            print('Min time instance is '+str(time_array[time_array.argmin]) +
                  '. Max time instance is : '+str(time_array[time_array.argmax]))
            exit()

    def get_2d_equilibrium_grid(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.equilibrium.time_slice[time_index].profiles_2d[0].r, \
                   self.equilibrium.time_slice[time_index].profiles_2d[0].z
        except:
            print('There is no R grid data in equilibrium IDS @ Shot: ' + str(self.shot) + ' Run: ' + str(self.run))
            exit()

    def get_2d_psi_values(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.equilibrium.time_slice[time_index].profiles_2d[0].psi
        except:
            print('There is no Psi grid data in equilibrium IDS @ Shot: ' + str(self.shot) + ' Run: ' + str(self.run))
            exit()

    def get_lcfs_boundary(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.equilibrium.time_slice[time_index].boundary.outline.r,\
                   self.equilibrium.time_slice[time_index].boundary.outline.z
        except:
            print('No RZ coordinates are available for Shot: ' + str(self.shot) + ' Run: ' + str(self.run))
            exit()

    def get_normalized_2d_flux(self, time):
        r_grid, z_grid = self.get_2d_equilibrium_grid(time)
        r_lcfs, z_lcfs = self.get_lcfs_boundary(time)
        flux = self.get_2d_psi_values(time)
        flux_function = interp2d(r_grid, z_grid, flux, kind='cubic')
        return flux / flux_function(r_lcfs[0], z_lcfs[0])
