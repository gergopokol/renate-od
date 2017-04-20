import h5py
import os
import numpy
import math
from scipy.interpolate import interp1d

class Rates:
    # Get rate coefficients from hdf5 files:
    def __init__(self, inputs):
        rate_coefficients=self.setup_rate_coeff_arrays()
        temperature_array=rate_coefficients[0]
        electron_neutral_collisions_array=rate_coefficients[1]
        proton_neutral_collisions_array=rate_coefficients[2]
        electron_loss_collisions_array=rate_coefficients[4]
        einstein_coeffs_array=rate_coefficients[5]

        # Interpolate rate coeffs to new grid:
        electron_neutral_collisions_array_new = numpy.zeros((inputs.number_of_levels, inputs.number_of_levels, len(inputs.steps)))
        proton_neutral_collisions_array_new = numpy.zeros((inputs.number_of_levels, inputs.number_of_levels, len(inputs.steps)))
        electron_loss_collisions_array_new = numpy.zeros((2, inputs.number_of_levels, len(inputs.steps)))
        for i in range(inputs.number_of_levels):
            for j in range(inputs.number_of_levels):
                for k in range(len(inputs.steps)):
                    if j != i:
                        x = temperature_array
                        y = electron_neutral_collisions_array[i, j, :], proton_neutral_collisions_array[i, j, :]
                        f = interp1d(x, y)
                        electron_neutral_collisions_array_new[i, j, k] = f(inputs.electron_temperature)[0]
                        proton_neutral_collisions_array_new[i, j, k] = f(inputs.proton_temperature)[1]
                    else:
                        continue
        for i in range(inputs.number_of_levels):
            for k in range(len(inputs.steps)):
                x = temperature_array
                y = electron_loss_collisions_array[0, i, :], electron_loss_collisions_array[1, i, :]
                f = interp1d(x, y)
                electron_loss_collisions_array_new[0, i, k] = f(inputs.electron_temperature)[0]
                electron_loss_collisions_array_new[1, i, k] = f(inputs.proton_temperature)[1]

        self.electron_neutral_collisions=electron_neutral_collisions_array_new
        self.proton_neutral_collisions=proton_neutral_collisions_array_new
        self.electron_loss_collisions=electron_loss_collisions_array_new
        self.einstein_coeffs=einstein_coeffs_array
        self.inputs=inputs
        self.velocity=math.sqrt(2 * inputs.beam_energy * 1.602176487e-16 / self.get_mass().mass_kg)


    def setup_rate_coeff_arrays(self):
        local_dir = os.getcwd()
        print(self.locate_h5_dir(local_dir, self.inputs.beam_species))
        file_name = 'rate_coeffs_' + str(self.inputs.beam_energy) + '_' + self.inputs.beam_species + '.h5'
        temperature_array = self.get_data_from_hdf5(self.locate_h5_dir(local_dir, self.inputs.beam_species) + '\\' + file_name, 'Temperature axis')
        electron_neutral_collisions_array = self.get_data_from_hdf5(self.locate_h5_dir(local_dir, self.inputs.beam_species) + '\\' + file_name,
                                                               'Collisional Coeffs/Electron Neutral Collisions') / 1e4  # changing of units
        proton_neutral_collisions_array = self.get_data_from_hdf5(self.locate_h5_dir(local_dir, self.inputs.beam_species) + '\\' + file_name,
                                                             'Collisional Coeffs/Proton Neutral Collisions') / 1e4
        impurity_neutral_collisions_array = self.get_data_from_hdf5(self.locate_h5_dir(local_dir, self.inputs.beam_species) + '\\' + file_name,
                                                               'Collisional Coeffs/Impurity Neutral Collisions') / 1e4
        electron_loss_collisions_array = self.get_data_from_hdf5(self.locate_h5_dir(local_dir, self.inputs.beam_species) + '\\' + file_name,
                                                            'Collisional Coeffs/Electron Loss Collisions') / 1e4
        einstein_coeffs_array = self.get_data_from_hdf5(self.locate_h5_dir(local_dir, self.inputs.beam_species) + '\\' + file_name, 'Einstein Coeffs')
        impurity_collisions_array = self.get_data_from_hdf5(self.locate_h5_dir(local_dir, self.inputs.beam_species) + '\\' + file_name,
                                                       'Impurity Collisions') / 1e4
        rate_coeff_arrays=[temperature_array,electron_neutral_collisions_array,proton_neutral_collisions_array,\
               impurity_neutral_collisions_array, electron_loss_collisions_array,\
               einstein_coeffs_array, impurity_collisions_array]
        return rate_coeff_arrays

    def get_mass(self, local_dir=os.getcwd(), beam_species='D'):
        mass_kg = numpy.loadtxt(self.locate_h5_dir(local_dir, beam_species) + '\\Mass\\' + beam_species + '_m.txt')
        return mass_kg

    def locate_h5_dir(self, cwd, atom):
        rod_loc = (str.find(cwd, 'renate-od'))
        atom_folder = self.choose_atom_folder(atom)
        return cwd[0:rod_loc] + 'renate-od/data/' + atom_folder

    @staticmethod
    def choose_atom_folder(atom):
        if atom == 'H':
            return 'H bundled-n'
        elif atom == 'D':
            return 'D bundled-n'
        else:
            return atom

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
    
