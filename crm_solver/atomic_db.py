import numpy
from lxml import etree
from utility import getdata
import pandas
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


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
            return '3', '2', '1', '3-2'
        elif self.species == 'Li':
            return '2p', '2s', '2s', '2p-2s'
        elif self.species == 'Na':
            return '3p', '3s', '3s', '3p-3s'
        elif self.species == 'dummy':
            return '1', '0', '0', '1-0'
        else:
            raise ValueError('The atomic species: ' + self.species + ' is not supported')

    def __generate_rate_function_db(self):
        self.temperature_axis = self.load_rate_data(self.rates_path, 'Temperature axis')
        self.__set_einstein_coefficient_db()
        self.__set_impact_loss_functions()
        self.__set_electron_impact_transition_functions()
        self.__set_ion_impact_transition_functions()

    def __set_impact_loss_functions(self):
        '''''
        Contains beam atom impact ionization (ion + charge exchange) data for loaded atomic type.
        Indexing convention: data[from_level][charge] for ion impact electron loss interaction.
        Indexing convention: data[from_level] for electron impact electron loss interactions. 
        '''''
        raw_impact_loss_transition = self.load_rate_data(self.rates_path,
                                                         'Collisional Coeffs/Electron Loss Collisions')
        self.__set_charge_state_lib(raw_impact_loss_transition.shape[0]-1)
        self.electron_impact_loss, self.ion_impact_loss = [], []
        for from_level in range(self.atomic_levels):
            from_level_functions = []
            self.electron_impact_loss.append(interp1d(self.temperature_axis, raw_impact_loss_transition
                                                      [0, from_level, :], fill_value='extrapolate'))
            for charged_state in range(raw_impact_loss_transition.shape[0]-1):
                from_level_functions.append(interp1d(self.temperature_axis, raw_impact_loss_transition
                                                     [charged_state+1, from_level, :], fill_value='extrapolate'))
            self.ion_impact_loss.append(from_level_functions)

    def __set_electron_impact_transition_functions(self):
        '''''
        Contains electron impact transition data for loaded atomic type.
        Indexing convention: data[from_level][to_level]
        '''''
        raw_electron_transition = self.load_rate_data(self.rates_path,
                                                           'Collisional Coeffs/Electron Neutral Collisions')
        self.electron_impact_trans = []
        for from_level in range(self.atomic_levels):
            from_level_functions = []
            for to_level in range(self.atomic_levels):
                from_level_functions.append(interp1d(self.temperature_axis, raw_electron_transition[from_level,
                                                     to_level, :], fill_value='extrapolate'))
            self.electron_impact_trans.append(from_level_functions)

    def __set_ion_impact_transition_functions(self):
        '''''
        Contains spontanous transition data for loaded atomic type.
        Indexing convention: data[from_level, to_level, charge]
        '''''
        raw_proton_transition = self.load_rate_data(self.rates_path,
                                                         'Collisional Coeffs/Proton Neutral Collisions')
        raw_impurity_transition = self.load_rate_data(self.rates_path,
                                                           'Collisional Coeffs/Impurity Neutral Collisions')
        self.ion_impact_trans = []
        for from_level in range(self.atomic_levels):
            from_level_functions = []
            for to_level in range(self.atomic_levels):
                to_level_functions = []
                for charged_state in range(raw_impurity_transition.shape[0]+1):
                    if charged_state == 0:
                        to_level_functions.append(interp1d(self.temperature_axis, raw_proton_transition
                                                           [from_level, to_level, :], fill_value='extrapolate'))
                    else:
                        to_level_functions.append(interp1d(self.temperature_axis, raw_impurity_transition
                                                           [charged_state-1, from_level, to_level, :],
                                                           fill_value='extrapolate'))
                from_level_functions.append(to_level_functions)
            self.ion_impact_trans.append(from_level_functions)

    def __set_einstein_coefficient_db(self):
        '''''
        Contains spontanous transition data for loaded atomic type.
        Indexing convention: data[to_level, from_level]
        '''''
        self.spontaneous_trans = self.load_rate_data(self.rates_path, 'Einstein Coeffs')
        if self.atomic_levels != int(self.spontaneous_trans.size ** 0.5):
            raise Exception('Loaded atomic database is inconsistent with atomic data dictionary. Wrong data loaded.')

    def __set_atomic_dictionary(self):
        assert isinstance(self.species, str)
        if self.species not in ['H', 'D', 'T', 'Li', 'Na', 'dummy']:
            raise Exception(self.species + ' beam atomic data not supported')
        if self.species in ['H', 'D', 'T']:
            self.atomic_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5}
            self.atomic_levels = 6
        if self.species == 'Li':
            self.atomic_dict = {'2s': 0, '2p': 1, '3s': 2, '3p': 3, '3d': 4, '4s': 5, '4p': 6, '4d': 7, '4f': 8}
            self.atomic_levels = 9
        if self.species == 'Na':
            self.atomic_dict = {'3s': 0, '3p': 1, '3d': 2, '4s': 3, '4p': 4, '4d': 5, '4f': 6, '5s': 7}
            self.atomic_levels = 8
        if self.species == 'dummy':
            self.atomic_dict = {'1': 1, '0': 0, '2': 2}
            self.atomic_levels = 3
        self.inv_atomic_dict = {index: name for name, index in self.atomic_dict.items()}

    def __set_rates_path(self, rate_type):
        self.rate_type = rate_type
        self.file_name = 'rate_coeffs_' + str(self.energy) + '_' + self.species + '.h5'
        self.rates_path = getdata.locate_rates_dir(self.species, rate_type) + self.file_name

    def __set_charge_state_lib(self, nr_charged_states):
        self.charged_states = []
        for state in range(nr_charged_states):
            self.charged_states.append('charge-'+str(state+1))

    def plot_rates(self, *args, temperature=None):
        if temperature is None:
            temperature = self.temperature_axis
        elif not (isinstance(temperature, numpy.ndarray) or isinstance(temperature, list)):
            raise TypeError('The give temperature axis is not of list or numpy.ndarray type.')

        for arg in args:
            if not isinstance(arg, tuple):
                raise TypeError('Rate coordinates are expected to be of type tuples. Each coordinate is expected '
                                'to contain following information (transition type {ion, trans, spont}, '
                                'interaction with {e, p}, from level {2s,3s,1,...}, to level {2s,2p,2,...}, '
                                'charge {-1,1,2,...} )')
            assert isinstance(arg[0], str)
            if arg[0] not in ['trans', 'ion', 'spont']:
                raise ValueError(arg[0] + ' is not a supported transition. Try: trans, ion or spont keywords.')
            if arg[0] is 'spont':
                assert (isinstance(arg[1], str) and isinstance(arg[2], str))
                plt.plot(temperature, self.spontaneous_trans[self.atomic_dict[arg[2]], self.atomic_dict[arg[1]]] *
                         numpy.ones(len(temperature)), label='Spont. trans.: '+arg[1]+'-->'+arg[2])
            assert isinstance(arg[1], str)
            if arg[1] not in ['e', 'p']:
                raise ValueError('Expected impact interactions are: e or p')
            if arg[1] is 'e':
                assert isinstance(arg[2], str)
                if arg[0] is 'ion':
                    plt.plot(temperature, self.electron_impact_loss[self.atomic_dict[arg[2]]](temperature),
                             label='e impact ion: '+self.atomic_dict[arg[2]]+'-->i')
                else:
                    assert isinstance(arg[3], str)
                    plt.plot(temperature, self.electron_impact_trans[self.atomic_dict[arg[2]]]
                             [self.atomic_dict[arg[3]]](temperature), label='e impact trans: '+arg[2]+'-->'+arg[3])
            else:
                assert isinstance(arg[2], str)
                assert isinstance(arg[-1], int)
                if arg[-1] > len(self.charged_states):
                    raise ValueError('There are no rates available for atom impact with charged state: q='+str(arg[-1]))
                if arg[-1] < 1:
                    raise ValueError('There are supported charged for or below: q='+str(arg[-1]))
                if arg[0] is 'ion':
                    plt.plot(temperature, self.ion_impact_loss[self.atomic_dict[arg[2]]][arg[-1]](temperature),
                             label='p impact ion (q='+str(arg[-1])+'): '+arg[2]+'-->i')
                else:
                    assert isinstance(arg[3], str)
                    plt.plot(temperature, self.ion_impact_trans[self.atomic_dict[arg[2]]][self.atomic_dict[arg[3]]]
                             [arg[-1]-1](temperature), label='p impact trans (q='+str(arg[-1])+'): '+arg[2]+'-->'+arg[3])
        plt.title('Reduced rates for '+self.species+' projectiles at '+str(self.energy)+' keV impact energy.')
        plt.xlabel('Temperature [keV]')
        plt.ylabel('Rates [m^2]')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.show()

    @staticmethod
    def load_rate_data(path, tag_name):
        return getdata.GetData(data_path_name=path, data_key=[tag_name], data_format='array').data
