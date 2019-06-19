import numpy
from lxml import etree
from utility import getdata
import pandas
from scipy.interpolate import interp1d


class AtomicDB:
    def __init__(self, param=None, rate_type='default', data_path='beamlet/testimp0001.xml'):
        self.param = param
        if not isinstance(self.param, etree._ElementTree):
            self.param = getdata.GetData(data_path_name=data_path).data
        assert isinstance(self.param, etree._ElementTree)
        self.energy = self.param.getroot().find('body').find('beamlet_energy').text
        self.species = self.param.getroot().find('body').find('beamlet_species').text
        self.__set_atomic_dictionary()
        self.__set_rates_path(rate_type)
        self.__generate_rate_function_db()

    def set_default_atomic_levels(self):
        if self.species in ['H', 'D', 'T']:
            return '3n', '2n', '1n', '3n-2n'
        elif self.species == 'Li':
            return '2p', '2s', '2s', '2p-2s'
        elif self.species == 'Na':
            return '3p', '3s', '3s', '3p-3s'
        else:
            raise ValueError('The atomic species: ' + self.atomic_db.species + ' is not supported')

    def __generate_rate_function_db(self):
        self.temperature_axis = self.load_rate_data(self.rates_path, 'Temperature axis')
        self.__set_einstein_coefficient_db()
        self.__set_impact_loss_functions()
        self.__set_electron_impact_transition_functions()
        self.__set_ion_impact_transition_functions()

    def __set_impact_loss_functions(self):
        raw_impact_loss_transition = self.load_rate_data(self.rates_path,
                                                         'Collisional Coeffs/Electron Loss Collisions')
        self.__set_charge_state_lib(raw_impact_loss_transition.shape[0]-1)
        self.electron_impact_loss = pandas.DataFrame(columns=self.atomic_dict.keys(), index=['electron'])
        self.electron_impact_loss.columns.name = 'from'
        self.ion_impact_loss = pandas.DataFrame(columns=self.atomic_dict.keys(), index=self.charged_states)
        self.ion_impact_loss.columns.name = 'from'
        self.ion_impact_loss.index.name = 'q'
        for from_level in range(self.atomic_levels):
            self.electron_impact_loss[self.inv_atomic_dict[from_level]] = \
                interp1d(self.temperature_axis, raw_impact_loss_transition[0, from_level, :],
                         fill_value='extrapolate')
            for charged_state in range(raw_impact_loss_transition.shape[0]-1):
                self.ion_impact_loss[self.inv_atomic_dict[from_level]][self.charged_states[charged_state]] = \
                    interp1d(self.temperature_axis, raw_impact_loss_transition[charged_state+1, from_level, :],
                             fill_value='extrapolate')

    def __set_electron_impact_transition_functions(self):
        raw_electron_transition = self.load_rate_data(self.rates_path,
                                                           'Collisional Coeffs/Electron Neutral Collisions')
        self.electron_impact_trans = pandas.DataFrame(index=self.atomic_dict.keys(), columns=self.atomic_dict.keys())
        for from_level in range(self.atomic_levels):
            for to_level in range(self.atomic_levels):
                self.electron_impact_trans[self.inv_atomic_dict[from_level]][self.inv_atomic_dict[to_level]] = \
                    interp1d(self.temperature_axis, raw_electron_transition[from_level, to_level, :],
                             fill_value='extrapolate')

    def __set_ion_impact_transition_functions(self):
        raw_proton_transition = self.load_rate_data(self.rates_path,
                                                         'Collisional Coeffs/Proton Neutral Collisions')
        raw_impurity_transition = self.load_rate_data(self.rates_path,
                                                           'Collisional Coeffs/Impurity Neutral Collisions')

        multiindex = pandas.MultiIndex.from_product([self.atomic_dict.keys(), self.charged_states], names=['to', 'q'])
        self.ion_impact_trans = pandas.DataFrame(columns=self.atomic_dict.keys(), index=multiindex)
        self.ion_impact_trans.columns.name = 'from'
        for from_level in range(self.atomic_levels):
            for to_level in range(self.atomic_levels):
                self.ion_impact_trans[self.inv_atomic_dict[from_level]][self.inv_atomic_dict[to_level], self.charged_states[0]] = \
                    interp1d(self.temperature_axis, raw_proton_transition[from_level, to_level, :],
                             fill_value='extrapolate')
                for charged_state in range(raw_impurity_transition.shape[0]-1):
                    self.ion_impact_trans[self.inv_atomic_dict[from_level]][self.inv_atomic_dict[to_level],
                                                                            self.charged_states[charged_state+1]] =\
                        interp1d(self.temperature_axis, raw_impurity_transition[charged_state, from_level, to_level, :],
                                 fill_value='extrapolate')

    def __set_einstein_coefficient_db(self):
        raw_einstein_coefficient = self.load_rate_data(self.rates_path, 'Einstein Coeffs')
        if self.atomic_levels != int(raw_einstein_coefficient.size ** 0.5):
            raise Exception('Loaded atomic database is inconsistent with atomic data dictionary. Wrong data loaded.')
        self.spontaneous_trans = pandas.DataFrame(raw_einstein_coefficient,
                                                  columns=self.atomic_dict.keys(), index=self.atomic_dict.keys())
        self.spontaneous_trans.transpose()
        self.spontaneous_trans.columns.name = 'from'
        self.spontaneous_trans.index.name = 'to'

    def __set_atomic_dictionary(self):
        assert isinstance(self.species, str)
        if self.species not in ['H', 'D', 'T', 'Li', 'Na']:
            raise Exception(self.species + ' beam atomic data not supported')
        if self.species in ['H', 'D', 'T']:
            self.atomic_dict = {'1n': 0, '2n': 1, '3n': 2, '4n': 3, '5n': 4, '6n': 5}
            self.atomic_levels = 6
        if self.species == 'Li':
            self.atomic_dict = {'2s': 0, '2p': 1, '3s': 2, '3p': 3, '3d': 4, '4s': 5, '4p': 6, '4d': 7, '4f': 8}
            self.atomic_levels = 9
        if self.species == 'Na':
            self.atomic_dict = {'3s': 0, '3p': 1, '3d': 2, '4s': 3, '4p': 4, '4d': 5, '4f': 6, '5s': 7}
            self.atomic_levels = 8
        self.inv_atomic_dict = {index: name for name, index in self.atomic_dict.items()}

    def __set_rates_path(self, rate_type):
        self.rate_type = rate_type
        self.file_name = 'rate_coeffs_' + str(self.energy) + '_' + self.species + '.h5'
        self.rates_path = getdata.locate_rates_dir(self.species, rate_type) + self.file_name

    def __set_charge_state_lib(self, nr_charged_states):
        self.charged_states = []
        for state in range(nr_charged_states):
            self.charged_states.append('charge-'+str(state+1))

    @staticmethod
    def load_rate_data(path, tag_name):
        return getdata.GetData(data_path_name=path, data_key=[tag_name], data_format='array').data
