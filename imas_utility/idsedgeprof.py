from imas_utility.idsinstance import ImasObject
import numpy as np


class EdgeprofIds(ImasObject):
    def __init__(self, shot, run, user=None, machine=None):
        super().__init__(shot, run, user, machine)
        self.load_edge_profiles_ids()

    def load_edge_profiles_ids(self):
        try:
            self.edge_profiles = self.imas_pointer.get('edge_profiles')
        except:
            print('The edge_profiles IDS is absent for shot ' + str(self.shot) + ' at run ' + str(self.run))
            exit()

    def get_time_index(self, time):
        try:
            time_array = self.edge_profiles.time
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

    def get_grid_in_rho_tor_norm(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.edge_profiles.profiles_1d[time_index].grid.rho_tor_norm
        except:
            print('There is no available rho tor norm based grid for Shot:' +
                  str(self.shot) + ' at Run: ' + str(self.run))
            exit()

    def get_grid_in_psi(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.edge_profiles.profiles_1d[time_index].grid.psi
        except:
            print('There is no available psi based grid for Shot:' + str(self.shot) + ' at Run: ' + str(self.run))
            exit()
