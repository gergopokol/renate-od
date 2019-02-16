import numpy
import utility
from utility import getdata
import pandas
from lxml import etree
from scipy.interpolate import interp1d
from utility import convert


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
        self.file_name = 'rate_coeffs_' + str(self.beamlet_energy) + '_' + self.beamlet_species + '.h5'
        self.data_path_name = getdata.locate_h5_dir(self.beamlet_species, rate_type) + self.file_name
        self.einstein_coeffs = getdata.GetData(data_path_name=self.data_path_name,
                                               data_key=['Einstein Coeffs'], data_format="array").data
        self.number_of_levels = int(self.einstein_coeffs.size ** 0.5)
        self.number_of_charges = 12
        self.temperature_array = getdata.GetData(data_path_name=self.data_path_name,
                                                 data_key=['Temperature axis'], data_format='array').data

        if isinstance(plasma_components, pandas.DataFrame):
            neutral_collisions_zeros = numpy.zeros((self.number_of_levels, self.number_of_levels, self.number_of_steps))
            loss_collisions_zeros = numpy.zeros((self.number_of_levels, self.number_of_steps))
            self.electron_neutral_collisions = neutral_collisions_zeros
            self.electron_loss_collisions = loss_collisions_zeros
            self.plasma_components=plasma_components
            self.number_of_ions = int(self.plasma_components['Z'][self.plasma_components['Z'] == 1].count())
            self.ion_neutral_collisions = numpy.concatenate([[neutral_collisions_zeros]*self.number_of_ions])
            self.electron_loss_ion_collisions = numpy.concatenate([[loss_collisions_zeros]*self.number_of_ions])
            self.number_of_impurities = int(self.plasma_components['Z'][self.plasma_components['Z'] > 1].count())
            self.imp_neutral_collisions = numpy.concatenate([[neutral_collisions_zeros]*self.number_of_impurities])
            self.electron_loss_imp_collisions = numpy.concatenate([[loss_collisions_zeros]*self.number_of_impurities])
            self.interpolate_rate_coeffs()
            self.convert_rate_coefficients()
        else:
            print('Plasma component data is missing.')

    def interpolate_rate_coeffs(self):
        # Interpolate rate coeffs to new grid:
        for i in range(2):
            ('imp' + str(i+3) + '_collisions_array')

        electron_neutral_collisions_array = getdata.GetData(data_path_name=self.data_path_name,
                                                          data_key=['Collisional Coeffs/Electron Neutral Collisions'],
                                                          data_format="array").data
        electron_loss_collisions_array = getdata.GetData(data_path_name=self.data_path_name,
                                                         data_key=['Collisional Coeffs/Electron Neutral Collisions'],
                                                         data_format="array").data
        ion_array = getdata.GetData(data_path_name=self.data_path_name,
                                    data_key=['Collisional Coeffs/Proton Neutral Collisions'],
                                    data_format="array").data
        imp_data = getdata.GetData(data_path_name=self.data_path_name,
                                   data_key=['Collisional Coeffs/Impurity Neutral Collisions'],
                                   data_format="array").data
        for ion in range(self.number_of_ions):
            if ion == 0:
                ion_neutral_collisions_array = [ion_array]
            else:
                ion_neutral_collisions_array = ion_neutral_collisions_array.append([ion_array])
        for imp in range(self.number_of_impurities):
            imp_charge = int(self.plasma_components['q']['imp' + str(imp+1)])
            if imp_charge == 1:
                imp_array = ion_array
            elif imp_charge > 1:
                imp_array = imp_data[imp_charge]
            else:
                print('Data is only available for positively charged impurities.')
            if imp == 0:
                imp_neutral_collisions_array = [imp_array]
            else:
                imp_neutral_collisions_array = numpy.concatenate([imp_neutral_collisions_array, [imp_array]])
        for from_level in range(self.number_of_levels):
            for to_level in range(self.number_of_levels):
                for step in range(self.number_of_steps):
                    if to_level != from_level:
                        x = self.temperature_array
                        y = electron_neutral_collisions_array[from_level, to_level, :]
                        f = interp1d(x, y)
                        self.electron_neutral_collisions[from_level, to_level, step] = \
                            f(self.beamlet_profiles['electron']['temperature'][step])
                        for ion in range(self.number_of_ions):
                            x = self.atomic_mass_correction(self.plasma_components['A']['ion' + str(ion + 1)])
                            y = ion_neutral_collisions_array[ion][from_level, to_level, :]
                            f = interp1d(x, y)
                            self.ion_neutral_collisions[ion][from_level, to_level, step] = \
                                f(self.beamlet_profiles['ion' + str(ion+1)]['temperature'][step])
                        for imp in range(self.number_of_impurities):
                            x = self.atomic_mass_correction(int(self.plasma_components['A']['imp' + str(imp+1)]))
                            y = imp_neutral_collisions_array[imp][from_level, to_level, :]
                            f = interp1d(x, y)
                            self.imp_neutral_collisions[imp][from_level, to_level, step] = \
                                f(self.beamlet_profiles['imp' + str(imp+1)]['temperature'][step])
                    else:
                        continue
        for from_level in range(self.number_of_levels):
            for step in range(self.number_of_steps):
                x = self.temperature_array
                y = electron_loss_collisions_array[0, from_level, :]
                f = interp1d(x, y)
                self.electron_loss_collisions[from_level, step] = \
                    f(self.beamlet_profiles['electron']['temperature'][step])
            for ion in range(self.number_of_ions):
                x = self.atomic_mass_correction(int(self.plasma_components['A']['ion' + str(ion + 1)]))
                y = electron_loss_collisions_array[1, from_level, :]
                f = interp1d(x, y)
                self.electron_loss_ion_collisions[ion][from_level, step] = \
                    f(self.beamlet_profiles['ion' + str(ion + 1)]['temperature'][step])
            for imp in range(self.number_of_impurities):
                x = self.atomic_mass_correction(int(self.plasma_components['A']['imp' + str(imp+1)]))
                y = electron_loss_collisions_array[int(self.plasma_components['q']['imp' + str(imp+1)]), from_level, :]
                f = interp1d(x, y)
                self.electron_loss_imp_collisions[imp][from_level, step] = \
                    f(self.beamlet_profiles['imp' + str(imp + 1)]['temperature'][step])
            return

    def convert_rate_coefficients(self):
        self.electron_neutral_collisions = convert.convert_from_cm2_to_m2(self.electron_neutral_collisions)
        self.ion_neutral_collisions = convert.convert_from_cm2_to_m2(self.ion_neutral_collisions)
        self.imp_neutral_collisions = convert.convert_from_cm2_to_m2(self.imp_neutral_collisions)
        self.electron_loss_collisions = convert.convert_from_cm2_to_m2(self.electron_loss_collisions)
        self.electron_loss_ion_collisions = convert.convert_from_cm2_to_m2(self.electron_loss_ion_collisions)
        self.electron_loss_imp_collisions = convert.convert_from_cm2_to_m2(self.electron_loss_imp_collisions)

    def atomic_mass_correction(self, scale):
        return self.temperature_array*scale
