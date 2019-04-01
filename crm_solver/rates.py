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
        self.number_of_steps = self.beamlet_profiles['beamlet grid'].size
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
            self.number_of_ions = int(self.plasma_components['Z'][self.plasma_components['Z'] > 0].count())
            self.ion_neutral_collisions = numpy.concatenate([[neutral_collisions_zeros]*self.number_of_ions])
            self.electron_loss_ion_collisions = numpy.concatenate([[loss_collisions_zeros]*self.number_of_ions])
            self.interpolate_rate_coeffs()
            self.convert_rate_coefficients()
        else:
            print('Plasma component data is missing.')

    def interpolate_rate_coeffs(self):
        # Interpolate rate coeffs to new grid:
        electron_neutral_collisions_array = getdata.GetData(data_path_name=self.data_path_name,
                                                          data_key=['Collisional Coeffs/Electron Neutral Collisions'],
                                                          data_format="array").data
        electron_loss_collisions_array = getdata.GetData(data_path_name=self.data_path_name,
                                                         data_key=['Collisional Coeffs/Electron Loss Collisions'],
                                                         data_format="array").data
        proton_array = getdata.GetData(data_path_name=self.data_path_name,
                                    data_key=['Collisional Coeffs/Proton Neutral Collisions'],
                                    data_format="array").data
        imp_data = getdata.GetData(data_path_name=self.data_path_name,
                                   data_key=['Collisional Coeffs/Impurity Neutral Collisions'],
                                   data_format="array").data
        for ion in range(self.number_of_ions):
            ion_charge = int(self.plasma_components['q']['ion' + str(ion+1)])
            if ion_charge == 1:
                ion_array = proton_array
            elif ion_charge > 1:
                ion_array = imp_data[ion_charge-2]
            else:
                print('Data is only available for positively charged ions.')
            if ion == 0:
                ion_neutral_collisions_array = [ion_array]
            else:
                ion_neutral_collisions_array = numpy.concatenate([ion_neutral_collisions_array, [ion_array]])
        for from_level in range(self.number_of_levels):
            for to_level in range(self.number_of_levels):
                if to_level != from_level:
                    x = self.temperature_array
                    y = electron_neutral_collisions_array[from_level, to_level, :]
                    f = interp1d(x, y)
                    self.electron_neutral_collisions[from_level, to_level, :] = \
                        f(self.beamlet_profiles['electron']['temperature']['eV'][:])
                    for ion in range(self.number_of_ions):
                        x = self.atomic_mass_correction(self.plasma_components['A']['ion' + str(ion + 1)])
                        y = ion_neutral_collisions_array[ion][from_level, to_level, :]
                        f = interp1d(x, y)
                        self.ion_neutral_collisions[ion][from_level, to_level, :] = \
                            f(self.beamlet_profiles['ion' + str(ion+1)]['temperature']['eV'][:])
                else:
                    continue
            x = self.temperature_array
            y = electron_loss_collisions_array[0, from_level, :]
            f = interp1d(x, y)
            self.electron_loss_collisions[from_level, :] = \
                f(self.beamlet_profiles['electron']['temperature']['eV'][:])
            for ion in range(self.number_of_ions):
                x = self.atomic_mass_correction(int(self.plasma_components['A']['ion' + str(ion+1)]))
                y = electron_loss_collisions_array[int(self.plasma_components['q']['ion' + str(ion+1)]), from_level, :]
                f = interp1d(x, y)
                self.electron_loss_ion_collisions[ion][from_level, :] = \
                    f(self.beamlet_profiles['ion' + str(ion + 1)]['temperature']['eV'][:])
        return

    def convert_rate_coefficients(self):
        self.electron_neutral_collisions = convert.convert_from_cm2_to_m2(self.electron_neutral_collisions)
        self.ion_neutral_collisions = convert.convert_from_cm2_to_m2(self.ion_neutral_collisions)
        self.electron_loss_collisions = convert.convert_from_cm2_to_m2(self.electron_loss_collisions)
        self.electron_loss_ion_collisions = convert.convert_from_cm2_to_m2(self.electron_loss_ion_collisions)

    def atomic_mass_correction(self, scale):
        return self.temperature_array*scale
