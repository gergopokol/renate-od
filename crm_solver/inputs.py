import numpy
import h5py
from scipy.interpolate import interp2d


class Inputs:
    def __init__(self, filename, time_index):
        self.filename = filename
        self.time_index = time_index
        self.initial_condition = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.number_of_levels = len(self.initial_condition)
        self.step_interval = 0.1
        self.step_number = 100
        self.beam_energy = 60
        self.beam_species = 'Li'
        self.profiles = Profiles(filename, time_index)
        self.profiles.calculate_new_r_axis() # Shift coordinate

        self.vertical = numpy.array([0])
        self.steps = numpy.linspace(0, self.step_interval, self.step_number)
        self.electron_temperature = self.interpolate_profile(self.profiles.electron_temperature_2d)
        self.proton_temperature = self.interpolate_profile(self.profiles.ion_temperature_2d)
        self.density = self.interpolate_profile(self.profiles.density_2d)

    def interpolate_profile(self, profile):
        f = interp2d(self.profiles.r_axis, self.profiles.z_axis, profile)
        new_profile = f(self.steps, self.vertical)
        return new_profile


class Profiles:
    def __init__(self, filename, time_index):
        self.time_index=time_index
        self.filename=filename
        self.get_grid()
        self.get_density_2d()
        self.get_electron_temperature_2d()

    def get_grid(self):
        self.r_axis = self.get_data_from_hdf5(self.filename, 'grid/xAxis')
        self.z_axis = self.get_data_from_hdf5(self.filename, 'grid/yAxis')
        self.t_axis = self.get_data_from_hdf5(self.filename, 'grid/tAxis')

    def get_density_2d(self):
        time_density_2d = self.get_data_from_hdf5(self.filename, 'fields/density')
        self.density_2d = time_density_2d[:, :, self.time_index]

    def get_electron_temperature_2d(self):
        time_temperature_2d = self.get_data_from_hdf5(self.filename, 'fields/electronTemperature')
        self.electron_temperature_2d = time_temperature_2d[:, :, self.time_index]

    def get_ion_temperature_2d(self):
        time_temperature_2d = self.get_data_from_hdf5(self.filename, 'fields/ionTemperature')
        self.ion_temperature_2d = time_temperature_2d[:, :, self.time_index]

    def calculate_new_r_axis(self):
        r = self.r_axis
        self.r_axis = abs(r - max(r))

    @staticmethod
    def get_data_from_hdf5(name, source):
        try:
            hdf5_id = h5py.File(name, 'r')
        except IOError:
            print("File not found!", name)
            quit()
        else:
            data = hdf5_id[source].value
            hdf5_id.close()
            return data

