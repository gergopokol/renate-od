import numpy
from utility import get_data_from_hdf5
from scipy.interpolate import interpn


class Inputs:
    def __init__(self, filename='/marconi_work/eufus_gw/work/g2bszond/renate-od.git/trunk/data/Profiles/SOL_profiles/EAST_412.h5', time_index=50):
        self.filename = filename
        self.time_index = time_index
        self.initial_condition = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.number_of_levels = len(self.initial_condition)
        self.step_interval = 0.08
        self.step_number = 100
        self.beam_energy = 60
        self.beam_species = 'Li'
        self.profiles = Profiles(filename, time_index)
        self.profiles.calculate_new_r_axis() # Shift coordinate

        self.vertical = numpy.zeros(self.step_number)
        self.steps = numpy.linspace(0, self.step_interval, self.step_number)
        self.electron_temperature = self.interpolate_profile(self.profiles.electron_temperature_2d)
        self.ion_temperature = self.interpolate_profile(self.profiles.ion_temperature_2d)
        self.density = self.interpolate_profile(self.profiles.density_2d)

    def interpolate_profile(self, profile):
        print("profile " + str(profile.shape))
        print ('r '+ str(len(self.profiles.r_axis)))
        print ('z '+ str(len(self.profiles.z_axis)))
        new_profile = interpn((self.profiles.z_axis, self.profiles.r_axis), profile, (self.vertical, self.steps))
        return new_profile

class Inputs1:
    initial_condition = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    step_interval = 0.1
    step_number = 100
    beam_energy = 60
    beam_species = 'Li'
    electron_temperature = numpy.linspace(1,6,step_number)*1e3
    proton_temperature = numpy.linspace(1,6,step_number)*1e3
    density = numpy.linspace(1,6,step_number)*1e20

    number_of_levels = len(initial_condition)
    steps = numpy.linspace(0, step_interval, step_number)


class Profiles:
    def __init__(self, filename, time_index):
        self.time_index = time_index
        self.filename = filename
        self.r_axis = numpy.array([0])
        self.z_axis = numpy.array([0])
        self.t_axis = numpy.array([0])
        self.density_2d = numpy.array([0])
        self.electron_temperature_2d = numpy.array([0])
        self.ion_temperature_2d = numpy.array([0])
        self.get_grid()
        self.get_density_2d()
        self.get_electron_temperature_2d()
        self.get_ion_temperature_2d()




    def get_grid(self):
        self.r_axis = get_data_from_hdf5.get_data_from_hdf5(self.filename, 'grid/xAxis')
        self.z_axis = get_data_from_hdf5.get_data_from_hdf5(self.filename, 'grid/yAxis')
        self.t_axis = get_data_from_hdf5.get_data_from_hdf5(self.filename, 'grid/tAxis')
        print(self.z_axis)


    def get_density_2d(self):
        time_density_2d = get_data_from_hdf5.get_data_from_hdf5(self.filename, 'fields/density')
        self.density_2d = time_density_2d[self.time_index, :, :] # [t, z, r]
        del time_density_2d

    def get_electron_temperature_2d(self):
        time_temperature_2d = get_data_from_hdf5.get_data_from_hdf5(self.filename, 'fields/electronTemperature')
        print(time_temperature_2d.shape)
        self.electron_temperature_2d = time_temperature_2d[self.time_index, :, :]
        del time_temperature_2d

    def get_ion_temperature_2d(self):
        time_temperature_2d = get_data_from_hdf5.get_data_from_hdf5(self.filename, 'fields/ionTemperature')
        self.ion_temperature_2d = time_temperature_2d[self.time_index, :, :]
        del time_temperature_2d

    def calculate_new_r_axis(self):
        r = self.r_axis
        shifted_r = abs(r - max(r))
        self.r_axis = shifted_r[::-1]


