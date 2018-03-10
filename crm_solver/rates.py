import numpy
import math
import utility
import pandas
from lxml import etree
from scipy.interpolate import interp1d


class Rates:
    # Get rate coefficients from hdf5 files:
    def __init__(self, beamlet_param, beamlet_profiles, rate_type='default'):

        assert isinstance(beamlet_param, etree._ElementTree)
        self.beamlet_energy = beamlet_param.getroot().find('body').find('beamlet_energy')
        self.beamlet_species = beamlet_param.getroot().find('body').find('beamlet_species')

        assert isinstance(beamlet_profiles, pandas.DataFrame)
        self.beamlet_profiles = beamlet_profiles
        self.number_of_levels = from_beamlet_profiles
        self.number_of_steps = from_beamlet_profiles

        self.rate_type = rate_type

        rate_coefficients = self.setup_rate_coeff_arrays()
        temperature_array = rate_coefficients[0]
        electron_neutral_collisions_array = rate_coefficients[1]
        proton_neutral_collisions_array = rate_coefficients[2]
        electron_loss_collisions_array = rate_coefficients[4]
        einstein_coeffs_array = rate_coefficients[5]

        # Interpolate rate coeffs to new grid:
        electron_neutral_collisions_array_new = numpy.zeros((self.beamlet_param.number_of_levels,
                                                             self.beamlet_param.number_of_levels,
                                                             len(self.beamlet_param.steps)))
        proton_neutral_collisions_array_new = numpy.zeros((self.beamlet_param.number_of_levels,
                                                           self.beamlet_param.number_of_levels,
                                                           len(self.beamlet_param.steps)))
        electron_loss_collisions_array_new = numpy.zeros((2, self.beamlet_param.number_of_levels,
                                                          len(self.beamlet_param.steps)))
        for from_level in range(self.beamlet_param.number_of_levels):
            for to_level in range(self.beamlet_param.number_of_levels):
                for step in range(len(self.beamlet_param.steps)):
                    if to_level != from_level:
                        x = temperature_array
                        y = electron_neutral_collisions_array[from_level, to_level, :], \
                            proton_neutral_collisions_array[from_level, to_level, :]
                        f = interp1d(x, y)
                        electron_neutral_collisions_array_new[from_level, to_level, step] =\
                            f(self.beamlet_param.profiles.electron_temperature[step])[0]
                        proton_neutral_collisions_array_new[from_level, to_level, step]\
                            = f(self.beamlet_param.profiles.ion_temperature[step])[1]
                    else:
                        continue
        for from_level in range(self.beamlet_param.number_of_levels):
            for step in range(len(self.beamlet_param.steps)):
                x = temperature_array
                y = electron_loss_collisions_array[0, from_level, :], electron_loss_collisions_array[1, from_level, :]
                f = interp1d(x, y)
                electron_loss_collisions_array_new[0, from_level, step] = \
                    f(self.beamlet_param.profiles.electron_temperature[step])[0]
                electron_loss_collisions_array_new[1, from_level, step] = \
                    f(self.beamlet_param.profiles.ion_temperature[step])[1]

        self.electron_neutral_collisions = electron_neutral_collisions_array_new
        self.proton_neutral_collisions = proton_neutral_collisions_array_new
        self.electron_loss_collisions = electron_loss_collisions_array_new
        self.einstein_coeffs = einstein_coeffs_array
        self.mass = self.get_mass()
        self.velocity = math.sqrt(2 * self.beamlet_energy * 1.602176487e-16 / self.mass)

    def setup_rate_coeff_arrays(self):
        file_name = 'rate_coeffs_' + str(self.beamlet_energy) + '_' + \
                    self.beamlet_species + '.h5'
        data_path_name = self.locate_h5_dir() + file_name
        temperature_array = utility.getdata.GetData(data_path_name, 'Temperature axis', data_format='array').data
        electron_neutral_collisions_array = \
            utility.getdata.GetData(data_path_name, 'Collisional Coeffs/Electron Neutral Collisions',
                                    data_format="array").data
        proton_neutral_collisions_array = \
            utility.getdata.GetData(data_path_name, 'Collisional Coeffs/Proton Neutral Collisions',
                                    data_format="array").data
        impurity_neutral_collisions_array = \
            utility.getdata.GetData(data_path_name, 'Collisional Coeffs/Impurity Neutral Collisions',
                                    data_format="array").data
        electron_loss_collisions_array = \
            utility.getdata.GetData(data_path_name, 'Collisional Coeffs/Electron Loss Collisions',
                                    data_format="array").data
        einstein_coeffs_array = utility.getdata.GetData(data_path_name, 'Einstein Coeffs',
                                                        data_format="array").data
        impurity_collisions_array = utility.getdata.GetData(data_path_name, 'Impurity Collisions',
                                                            data_format="array").data
        # This is to be removed when input file is in SI
        electron_neutral_collisions_array = utility.convert.convert_from_cm2_to_m2(electron_neutral_collisions_array)
        proton_neutral_collisions_array = utility.convert.convert_from_cm2_to_m2(proton_neutral_collisions_array)
        impurity_neutral_collisions_array = utility.convert.convert_from_cm2_to_m2(impurity_neutral_collisions_array)
        electron_loss_collisions_array = utility.convert.convert_from_cm2_to_m2(electron_loss_collisions_array)
        impurity_collisions_array = utility.convert.convert_from_cm2_to_m2(impurity_collisions_array)
        # To-be-removed end

        rate_coeff_arrays = [temperature_array, electron_neutral_collisions_array, proton_neutral_collisions_array,
                             impurity_neutral_collisions_array, electron_loss_collisions_array,
                             einstein_coeffs_array, impurity_collisions_array]
        return rate_coeff_arrays

    def get_mass(self):
        data_path_name = 'atomic_data/' + self.beamlet_species + '/supplementary_data/default/' + \
                         self.beamlet_species + '_m.txt'
        mass_str = utility.getdata.GetData(data_path_name = data_path_name, data_format="array").data
        try:
            mass = float(mass_str)
        except ValueError:
            print('Unexpected data in file: ' + data_path_name + '(Expecting single float!)')
            raise ValueError
        return mass

    def locate_h5_dir(self):
        return 'atomic_data/' + self.beamlet_species + '/rates/' + self.rate_type + '/'
