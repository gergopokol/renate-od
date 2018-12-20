import numpy
import utility
from utility import getdata
import pandas
from lxml import etree
from scipy.interpolate import interp1d


class Rates:
    # Get rate coefficients from hdf5 files:
    def __init__(self, beamlet_param, beamlet_profiles, plasma_components, rate_type='default'):

        assert isinstance(beamlet_param, etree._ElementTree)
        self.beamlet_energy = beamlet_param.getroot().find('body').find('beamlet_energy').text
        self.beamlet_species = beamlet_param.getroot().find('body').find('beamlet_species').text
        self.mass = float(beamlet_param.getroot().find('body').find('beamlet_mass').text)
        self.velocity = float(beamlet_param.getroot().find('body').find('beamlet_velocity').text)

        assert isinstance(beamlet_profiles, pandas.DataFrame)
        self.beamlet_profiles = beamlet_profiles
        self.number_of_steps = self.beamlet_profiles['beamlet_grid'].size
        self.rate_type = rate_type

        self.file_name = 'rate_coeffs_' + str(self.beamlet_energy) + '_' + \
                    self.beamlet_species + '.h5'
        self.data_path_name = getdata.locate_h5_dir(self.beamlet_species, rate_type) + self.file_name
        self.temperature_array = getdata.GetData(data_path_name=self.data_path_name,
                                    data_key=['Temperature axis'],
                                    data_format='array').data

        if isinstance(plasma_components, pandas.DataFrame):
            neutral_collisions_zeros = numpy.zeros(self.number_of_levels, self.number_of_levels, self.number_of_steps)
            self.electron_neutral_collisions = neutral_collisions_zeros
            self.plasma_components=plasma_components
            self.number_of_ions = int(self.plasma_components['Z'][self.plasma_components['Z'] == 1].count())
            self.ion_neutral_collisions = numpy.concatenate([[neutral_collisions_zeros]*self.number_of_ions])
            self.number_of_impurities = int(self.plasma_components['Z'][self.plasma_components['Z'] > 1].count())
            self.imp_neutral_collisions = numpy.concatenate([[neutral_collisions_zeros]*self.number_of_impurities])
            self.interpolate_rate_coeffs()
        else:
            self.interpolate_rate_coeffs_old()


    def interpolate_rate_coeffs(self):
        # Interpolate rate coeffs to new grid:
        for i in range(2):
            ('imp' + str(i+3) + '_collisions_array')

        self.number_of_levels = 9
        electron_neutral_collisions_array=getdata.GetData(data_path_name=self.data_path_name,
                                     data_key=['Collisional Coeffs/Electron Neutral Collisions'],
                                     data_format="array").data
        self.electron_neutral_collisions = numpy.zeros((self.number_of_levels, self.number_of_levels,
                                                        self.number_of_steps))
        for ion in range(self.number_of_ions):
            array = 'ion' + str(ion+1) + '_neutral_collisions_array'
            vars()[array] = getdata.GetData(data_path_name=self.data_path_name,
                                            data_key=['Collisional Coeffs/Proton Neutral Collisions'],
                                            data_format="array").data
            array = 'self.ion' + str(ion+1) + '_neutral_collisions'
            vars()[array] = numpy.zeros((self.number_of_levels, self.number_of_levels, self.number_of_steps))

        for imp in range(self.number_of_impurities):
            array = 'imp' + str(imp+1) + '_neutral_collisions_array'
            vars()[array] = getdata.GetData(data_path_name=self.data_path_name,
                                            data_key=['Collisional Coeffs/Impurity Neutral Collisions'],
                                            data_format="array").data[int(self.plasma_components['q']['imp' + str(imp+1)])]
            array = 'self.imp' + str(imp+1) + '_neutral_collisions'
            vars()[array] = numpy.zeros((self.number_of_levels, self.number_of_levels, self.number_of_steps))
        for from_level in range(self.number_of_levels):
            for to_level in range(self.number_of_levels):
                for step in range(self.number_of_steps):
                    if to_level != from_level:
                        x = self.temperature_array
                        y = electron_neutral_collisions_array[from_level, to_level, :]
                        f = interp1d(x, y)
                        self.electron_neutral_collisions[from_level, to_level, step] = \
                            f(self.beamlet_profiles['electron']['temperature'][step])
                    else:
                        continue
        for from_level in range(self.number_of_levels):
            continue


    def interpolate_rate_coeffs_old(self):
        self.rate_coefficients = getdata.setup_rate_coeff_arrays(self.beamlet_energy, self.beamlet_species,
                                                                 self.rate_type)
        # Interpolate rate coeffs to new grid:
        temperature_array = self.rate_coefficients[0]
        electron_neutral_collisions_array = self.rate_coefficients[1]
        proton_neutral_collisions_array = self.rate_coefficients[2]
        electron_loss_collisions_array = self.rate_coefficients[4]
        einstein_coeffs_array = self.rate_coefficients[5]
        self.number_of_levels = int(einstein_coeffs_array.size ** 0.5)

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

        


