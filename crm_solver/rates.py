import h5py
import os
import numpy
import math
from scipy.interpolate import interp1d
from crm_solver import inputs
from utility import get_data_from_hdf5


class Rates:
    # Get rate coefficients from hdf5 files:
    def __init__(self):
        self.inputs = inputs.Constant_Plasma_Inputs()
        rate_coefficients = self.setup_rate_coeff_arrays()
        temperature_array = rate_coefficients[0]
        electron_neutral_collisions_array = rate_coefficients[1]
        proton_neutral_collisions_array = rate_coefficients[2]
        electron_loss_collisions_array = rate_coefficients[4]
        einstein_coeffs_array = rate_coefficients[5]

        # Interpolate rate coeffs to new grid:
        electron_neutral_collisions_array_new = numpy.zeros((self.inputs.number_of_levels, self.inputs.number_of_levels,
                                                             len(self.inputs.steps)))
        proton_neutral_collisions_array_new = numpy.zeros((self.inputs.number_of_levels, self.inputs.number_of_levels,
                                                           len(self.inputs.steps)))
        electron_loss_collisions_array_new = numpy.zeros((2, self.inputs.number_of_levels, len(self.inputs.steps)))
        for from_level in range(self.inputs.number_of_levels):
            for to_level in range(self.inputs.number_of_levels):
                for step in range(len(self.inputs.steps)):
                    if to_level != from_level:
                        x = temperature_array
                        y = electron_neutral_collisions_array[from_level, to_level, :], \
                            proton_neutral_collisions_array[from_level, to_level, :]
                        f = interp1d(x, y)
                        electron_neutral_collisions_array_new[from_level, to_level, step] =\
                            f(self.inputs.electron_temperature[step])[0]
                        proton_neutral_collisions_array_new[from_level, to_level, step]\
                            = f(self.inputs.ion_temperature[step])[1]
                    else:
                        continue
        for from_level in range(self.inputs.number_of_levels):
            for step in range(len(self.inputs.steps)):
                x = temperature_array
                y = electron_loss_collisions_array[0, from_level, :], electron_loss_collisions_array[1, from_level, :]
                f = interp1d(x, y)
                electron_loss_collisions_array_new[0, from_level, step] = f(self.inputs.electron_temperature[step])[0]
                electron_loss_collisions_array_new[1, from_level, step] = f(self.inputs.ion_temperature[step])[1]

        self.electron_neutral_collisions = electron_neutral_collisions_array_new

        self.proton_neutral_collisions = proton_neutral_collisions_array_new
        self.electron_loss_collisions = electron_loss_collisions_array_new
        self.einstein_coeffs = einstein_coeffs_array
        self.mass = self.get_mass(beam_species=self.inputs.beam_species)
        self.velocity = math.sqrt(2 * self.inputs.beam_energy * 1.602176487e-16 / self.mass)

    def setup_rate_coeff_arrays(self):
        local_dir = os.getcwd()
        file_name = 'rate_coeffs_' + str(self.inputs.beam_energy) + '_' + self.inputs.beam_species + '.h5'
        filename = self.locate_h5_dir(local_dir, self.inputs.beam_species) + '/' + file_name
        temperature_array = get_data_from_hdf5.get_data_from_hdf5(filename, 'Temperature axis')
        # \1e4 - changing of units
        electron_neutral_collisions_array =\
            get_data_from_hdf5.get_data_from_hdf5(filename, 'Collisional Coeffs/Electron Neutral Collisions') / 1e4
        proton_neutral_collisions_array\
            = get_data_from_hdf5.get_data_from_hdf5(filename, 'Collisional Coeffs/Proton Neutral Collisions') / 1e4
        impurity_neutral_collisions_array =\
            get_data_from_hdf5.get_data_from_hdf5(filename, 'Collisional Coeffs/Impurity Neutral Collisions') / 1e4
        electron_loss_collisions_array =\
            get_data_from_hdf5.get_data_from_hdf5(filename, 'Collisional Coeffs/Electron Loss Collisions') / 1e4
        einstein_coeffs_array = get_data_from_hdf5.get_data_from_hdf5(filename, 'Einstein Coeffs')
        impurity_collisions_array = get_data_from_hdf5.get_data_from_hdf5(filename,
                                                           'Impurity Collisions') / 1e4
        rate_coeff_arrays = [temperature_array, electron_neutral_collisions_array, proton_neutral_collisions_array,
                             impurity_neutral_collisions_array, electron_loss_collisions_array,
                             einstein_coeffs_array, impurity_collisions_array]
        return rate_coeff_arrays

    def get_mass(self, local_dir=os.getcwd(), beam_species='Li'):
        mass_kg = numpy.loadtxt(self.locate_h5_dir(local_dir, beam_species) + '/Mass/' + beam_species + '_m.txt')
        return mass_kg

    def locate_h5_dir(self, cwd, atom):
        rod_loc = (str.find(cwd, 'trunk'))
        atom_folder = self.choose_atom_folder(atom)
        return cwd[0:rod_loc] + 'trunk/data/' + atom_folder

    @staticmethod
    def choose_atom_folder(atom):
        if atom == 'H':
            return 'H bundled-n'
        elif atom == 'D':
            return 'D bundled-n'
        else:
            return atom