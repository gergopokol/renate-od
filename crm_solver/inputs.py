import numpy
from utility import get_data_from_hdf5
from scipy.interpolate import interpn
#import imas
import sys

class Input_IMAS:
    def __init__(self):
        self.initial_condition = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.number_of_levels = len(self.initial_condition)
        self.beam_energy = 60
        self.beam_species = 'Li'
        self.beam_current = 0.0002
        self.observation_profile = [0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26]
        self.detector_size = 0.02
        self.aperture_diamater = 0.05
        self.observation_point = [0.9, 0.8]

        self.shot = 1123
        self.run = 12

        self.steps = numpy.linspace(0, self.step_interval, self.step_number)
        self.step_interval = 0.08
        self.step_number = 100

    def get_coreprof_profile(self, time_index):
        self.density

class Inputs:
    def __init__(self,filename='C:\\Users\\asztalos\\Desktop\\Projects\\BES\\Codes\\RENATE v2\\Trunk\\Device data\\EAST\\SOL profiles\\EAST_#412.h5',time_index=50):
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
        new_profile = interpn((self.profiles.z_axis, self.profiles.r_axis), profile, (self.vertical, self.steps))
        return new_profile

class Constant_Plasma_Inputs:
    def __init__(self):
        self.initial_condition = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.number_of_levels = len(self.initial_condition)
        self.step_interval = 0.3
        self.step_number = 300
        self.beam_energy = 60
        self.beam_species = 'Li'
        self.electron_temperature = numpy.full(self.step_number,2)*1e3
        self.ion_temperature = numpy.full(self.step_number,1.5)*1e3
        self.density = numpy.full(self.step_number,2)*1e19
        self.steps = numpy.linspace(0, self.step_interval, self.step_number)
        self.beam_current = 0.002
        self.observation_profile = numpy.array([0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26])
        self.detector_size = 0.02
        self.aperture_diamater = 0.05
        self.observation_point = [0.9,0.8]

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

    def get_density_2d(self):
        time_density_2d = get_data_from_hdf5.get_data_from_hdf5(self.filename, 'fields/density')
        self.density_2d = time_density_2d[self.time_index, :, :] # [t, z, r]
        del time_density_2d

    def get_electron_temperature_2d(self):
        time_temperature_2d = get_data_from_hdf5.get_data_from_hdf5(self.filename, 'fields/electronTemperature')
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