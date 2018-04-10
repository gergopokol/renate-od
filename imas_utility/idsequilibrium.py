from imas_utility.idsinstance import ImasObject
import numpy as np


class EquilibriumIds(ImasObject):
    def __init__(self, shot, run, user=None, machine=None):
        super().__init__(shot, run, user, machine)
        self.load_equilibrium_ids()

    def load_equilibrium_ids(self):
        try:
            self.equilibrium = self.imas_pointer.get('equilibrium')
        except:
            print('The equilibrium IDS is absent from data file. Muhaha.')
            print('Please get more information about shot ' + str(self.shot) + ' at run ' + str(self.run))
            exit()

    def get_time_index(self, time):
        try:
            time_array = self.equilibrium.time
        except:
            print('No time array available for the requested Shot: '+str(self.shot)+' and Run: '+str(self.run))
            print('Aborting calculations')
            exit()

        if (time_array[time_array.argmin()] <= time) and (time_array[time_array.argmax()] >= time) :
            return (np.abs(time_array - time)).argmin()
        else:
            print('Time value : '+str(time)+' is out of bound. Please select new time instance.')
            print('Min time instance is '+str(time_array[time_array.argmin]) +
                  '. Max time instance is : '+str(time_array[time_array.argmax]))
            exit()

    def get_2d_equilibrium_grid_r(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.equilibrium.time_slice[time_index].profiles_2d[0].r
        except:
            print('There is no R grid data in equilibrium IDS @ Shot: ' + str(self.shot) + ' Run: ' + str(self.run))
            print('The requested time instance was: ' + str(time) + ' ABORTING run.')
            exit()

    def get_2d_equilibrium_grid_z(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.equilibrium.time_slice[time_index].profiles_2d[0].z
        except:
            print('There is no Z grid data in equilibrium IDS @ Shot: ' + str(self.shot) + ' Run: ' + str(self.run))
            print('The requested time instance was: ' + str(time) + ' ABORTING run.')
            exit()

    def get_2d_psi_values(self, time):

        time_index = self.get_time_index(time)
        try:
            return self.equilibrium.time_slice[time_index].profiles_2d[0].psi
        except:
            print('There is no Psi grid data in equilibrium IDS @ Shot: ' + str(self.shot) + ' Run: ' + str(self.run))
            print('The requested time instance was: ' + str(time) + ' ABORTING run.')
            exit()