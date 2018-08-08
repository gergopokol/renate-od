import numpy
import utility
from utility import getdata
import pandas
from lxml import etree
from scipy.interpolate import interp1d


class Rates:
    # Get rate coefficients from hdf5 files:
    def __init__(self, beamlet_param, beamlet_profiles, rate_type='default'):

        assert isinstance(beamlet_param, etree._ElementTree)
        self.beamlet_energy = beamlet_param.getroot().find('body').find('beamlet_energy').text
        self.beamlet_species = beamlet_param.getroot().find('body').find('beamlet_species').text

        assert isinstance(beamlet_profiles, pandas.DataFrame)
        self.beamlet_profiles = beamlet_profiles
        self.number_of_steps = self.beamlet_profiles['beamlet_grid'].size
        self.rate_type = rate_type

        rate_coefficients = getdata.setup_rate_coeff_arrays(self.beamlet_energy, self.beamlet_species, self.rate_type)
        temperature_array = rate_coefficients[0]
        electron_neutral_collisions_array = rate_coefficients[1]
        proton_neutral_collisions_array = rate_coefficients[2]
        electron_loss_collisions_array = rate_coefficients[4]
        einstein_coeffs_array = rate_coefficients[5]
        self.number_of_levels = int(einstein_coeffs_array.size ** 0.5)

        # Interpolate rate coeffs to new grid:
        electron_neutral_collisions_array_new = numpy.zeros((self.number_of_levels,
                                                             self.number_of_levels,
                                                             self.number_of_steps))
        proton_neutral_collisions_array_new = numpy.zeros((self.number_of_levels,
                                                           self.number_of_levels,
                                                           self.number_of_steps))
        electron_loss_collisions_array_new = numpy.zeros((2, self.number_of_levels,
                                                          self.number_of_steps))
        for from_level in range(self.number_of_levels):
            for to_level in range(self.number_of_levels):
                for step in range(self.number_of_steps):
                    if to_level != from_level:
                        x = temperature_array
                        y = electron_neutral_collisions_array[from_level, to_level, :], \
                            proton_neutral_collisions_array[from_level, to_level, :]
                        f = interp1d(x, y)
                        electron_neutral_collisions_array_new[from_level, to_level, step] =\
                            f(self.beamlet_profiles['beamlet_electron_temp'][step])[0]
                        proton_neutral_collisions_array_new[from_level, to_level, step]\
                            = f(self.beamlet_profiles['beamlet_ion_temp'][step])[1]
                    else:
                        continue
        for from_level in range(self.number_of_levels):
            for step in range(self.number_of_steps):
                x = temperature_array
                y = electron_loss_collisions_array[0, from_level, :], electron_loss_collisions_array[1, from_level, :]
                f = interp1d(x, y)
                electron_loss_collisions_array_new[0, from_level, step] = \
                    f(self.beamlet_profiles['beamlet_electron_temp'][step])[0]
                electron_loss_collisions_array_new[1, from_level, step] = \
                    f(self.beamlet_profiles['beamlet_ion_temp'][step])[1]

        self.electron_neutral_collisions = electron_neutral_collisions_array_new
        self.proton_neutral_collisions = proton_neutral_collisions_array_new
        self.electron_loss_collisions = electron_loss_collisions_array_new
        self.einstein_coeffs = einstein_coeffs_array
        self.mass = float(beamlet_param.getroot().find('body').find('mass').text)
        self.velocity = float(beamlet_param.getroot().find('body').find('velocity').text)
        


