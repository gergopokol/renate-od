import numpy
from lxml import etree
from utility import getdata
import utility.convert as uc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas


class RenateDB:
    def __init__(self, param, rate_type, data_path):
        self.param = param
        if not isinstance(self.param, etree._ElementTree):
            self.param = getdata.GetData(data_path_name=data_path).data
        assert isinstance(self.param, etree._ElementTree)
        self.__set_impurity_mass_scaling_dictionary()
        self.__projectile_parameters()
        self.__set_atomic_dictionary()
        self.__set_rates_path(rate_type)

    def __set_impurity_mass_scaling_dictionary(self):
        self.impurity_mass_normalization = {'charge-1': 1,
                                            'charge-2': 4,
                                            'charge-3': 7,
                                            'charge-4': 9,
                                            'charge-5': 11,
                                            'charge-6': 12,
                                            'charge-7': 14,
                                            'charge-8': 16,
                                            'charge-9': 19,
                                            'charge-10': 20,
                                            'charge-11': 23}

    def __projectile_parameters(self):
        self.energy = self.param.getroot().find('body').find('beamlet_energy').text
        self.species = self.param.getroot().find('body').find('beamlet_species').text
        self.__get_atomic_mass()
        self.__get_projectile_velocity()

    def __get_atomic_mass(self):
        data_path_name = 'atomic_data/' + self.param.getroot().find('body').find('beamlet_species').text + \
                         '/supplementary_data/default/' + \
                         self.param.getroot().find('body').find('beamlet_species').text + '_m.txt'
        mass_str = getdata.GetData(data_path_name=data_path_name, data_format="array").data
        try:
            self.mass = float(mass_str)
        except ValueError:
            print('Unexpected data in file: ' + data_path_name + '(Expecting single float!)')
            raise ValueError

    def __get_projectile_velocity(self):
        self.velocity = uc.calculate_velocity_from_energy(uc.convert_keV_to_eV(float(self.energy)), self.mass)

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

    @staticmethod
    def load_rate_data(path, tag_name):
        return getdata.GetData(data_path_name=path, data_key=[tag_name], data_format='array').data

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

    def get_from_renate_atomic(self, source):
        assert isinstance(source, str)
        if source is 'electron_transition':
            return self.load_rate_data(self.rates_path, 'Collisional Coeffs/Electron Neutral Collisions')
        elif source is 'ion_transition':
            return self.load_rate_data(self.rates_path, 'Collisional Coeffs/Proton Neutral Collisions')
        elif source is 'impurity_transition':
            return self.load_rate_data(self.rates_path, 'Collisional Coeffs/Impurity Neutral Collisions')
        elif source is 'ionization_terms':
            return self.load_rate_data(self.rates_path, 'Collisional Coeffs/Electron Loss Collisions')
        elif source is 'spontaneous_transition':
            return self.load_rate_data(self.rates_path, 'Einstein Coeffs')
        elif source is 'temperature':
            return self.load_rate_data(self.rates_path, 'Temperature axis')
        else:
            raise ValueError('Data ' + source + ' is not located and supported in the Renate rate library.')


class AtomicDB(RenateDB):
    def __init__(self, atomic_source='renate', param=None, rate_type='default',
                 data_path='beamlet/testimp0001.xml', components=None):
        assert isinstance(atomic_source, str)
        assert isinstance(components, pandas.core.frame.DataFrame)
        self.components = components
        if atomic_source is 'renate':
            RenateDB.__init__(self, param, rate_type, data_path)
            self.__generate_rate_function_db()
        else:
            raise ValueError('Currently the requested atomic DB: ' + atomic_source + ' is not supported')

    def __generate_rate_function_db(self):
        self.__set_temperature_axis()
        self.__set_einstein_coefficient_db()
        self.__set_impact_loss_functions()
        self.__set_electron_impact_transition_functions()
        self.__set_ion_impact_transition_functions()

    def __set_temperature_axis(self):
        self.temperature_axis = self.get_from_renate_atomic('temperature')

    def __set_einstein_coefficient_db(self):
        '''''
        Contains spontanous transition data for loaded atomic type.
        Indexing convention: data[to_level, from_level]
        '''''
        self.spontaneous_trans = self.get_from_renate_atomic('spontaneous_transition')
        if self.atomic_levels != int(self.spontaneous_trans.size ** 0.5):
            raise Exception('Loaded atomic database is inconsistent with atomic data dictionary. Wrong data loaded.')

    def __set_impact_loss_functions(self):
        '''''
        Contains beam atom impact ionization (ion + charge exchange) data for loaded atomic type.
        Indexing convention: data[from_level][target] for ion impact electron loss interaction.
        Indexing convention: data[from_level] for electron impact electron loss interactions. 
        '''''
        raw_impact_loss_transition = self.get_from_renate_atomic('ionization_terms')
        self.__set_charge_state_lib(raw_impact_loss_transition.shape[0]-1)
        self.electron_impact_loss, self.ion_impact_loss = [], []
        for from_level in range(self.atomic_levels):
            from_level_functions = []
            self.electron_impact_loss.append(interp1d(self.temperature_axis, uc.convert_from_cm2_to_m2(
                raw_impact_loss_transition[0, from_level, :]), fill_value='extrapolate'))
            for target in self.components.T.keys():
                if target is not 'electron':
                    from_level_functions.append(self.__interp1d_scaled_ion(uc.convert_from_cm2_to_m2(
                        raw_impact_loss_transition[self.components['q'][target], from_level, :]), target))
            self.ion_impact_loss.append(tuple(from_level_functions))
        self.electron_impact_loss, self.ion_impact_loss = tuple(self.electron_impact_loss), tuple(self.ion_impact_loss)

    def __set_electron_impact_transition_functions(self):
        '''''
        Contains electron impact transition data for loaded atomic type.
        Indexing convention: data[from_level][to_level]
        '''''
        raw_electron_transition = self.get_from_renate_atomic('electron_transition')
        self.electron_impact_trans = []
        for from_level in range(self.atomic_levels):
            from_level_functions = []
            for to_level in range(self.atomic_levels):
                from_level_functions.append(interp1d(self.temperature_axis, uc.convert_from_cm2_to_m2(
                    raw_electron_transition[from_level, to_level, :]), fill_value='extrapolate'))
            self.electron_impact_trans.append(tuple(from_level_functions))
        self.electron_impact_trans = tuple(self.electron_impact_trans)

    def __set_charge_state_lib(self, nr_charged_states):
        self.charged_states = []
        for state in range(nr_charged_states):
            self.charged_states.append('charge-'+str(state+1))
        self.charged_states = tuple(self.charged_states)

    def __set_ion_impact_transition_functions(self):
        '''''
        Contains spontanous transition data for loaded atomic type.
        Indexing convention: data[from_level, to_level, target]
        '''''
        raw_proton_transition = self.get_from_renate_atomic('ion_transition')
        raw_impurity_transition = self.get_from_renate_atomic('impurity_transition')
        self.ion_impact_trans = []
        for from_level in range(self.atomic_levels):
            from_level_functions = []
            for to_level in range(self.atomic_levels):
                to_level_functions = []
                for target in self.components.T.keys():
                    if (target is not 'electron') and (self.components['Z'][target] == 1):
                        to_level_functions.append(self.__interp1d_scaled_ion(uc.convert_from_cm2_to_m2(
                            raw_proton_transition[from_level, to_level, :]), target))
                    elif (target is not 'electron') and (self.components['q'] >= 2):
                        to_level_functions.append(self.__interp1d_scaled_ion(uc.convert_from_cm2_to_m2(
                            raw_impurity_transition[self.components['q'][target]-2, from_level, to_level, :]), target))
                from_level_functions.append(tuple(to_level_functions))
            self.ion_impact_trans.append(tuple(from_level_functions))
        self.ion_impact_trans = tuple(self.ion_impact_trans)

    def __interp1d_scaled_ion(self, rates, target):
        scaling_mass_ratio = float(self.components['A'][target]) /\
                             self.impurity_mass_normalization['charge-'+str(self.components['q'][target])]
        return interp1d(self.temperature_axis/scaling_mass_ratio, rates, fill_value='extrapolate')

    def plot_rates(self, *args, temperature=None, external_density=1.):
        if temperature is None:
            temperature = self.temperature_axis
        elif not (isinstance(temperature, numpy.ndarray) or isinstance(temperature, list)):
            raise TypeError('The give temperature axis is not of list or numpy.ndarray type.')
        spont_flag = False
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
                spont_flag = True
                if external_density is 1.:
                    raise ValueError('In case spontaneous emission terms are being compared an external density '
                                     'correction is required for the rates. Please apply a realistic density value.')
                plt.plot(temperature, self.spontaneous_trans[self.atomic_dict[arg[2]], self.atomic_dict[arg[1]]] *
                         numpy.ones(len(temperature)) / self.velocity, label='Spont. trans.: '+arg[1]+'-->'+arg[2])
            else:
                assert isinstance(arg[1], str)
                if arg[1] not in ['e', 'p']:
                    raise ValueError('Expected impact interactions are: e or p')
                if arg[1] is 'e':
                    assert isinstance(arg[2], str)
                    if arg[0] is 'ion':
                        plt.plot(temperature, self.electron_impact_loss[self.atomic_dict[arg[2]]](temperature) *
                                 external_density, label='e impact ion: '+self.atomic_dict[arg[2]]+'-->i')
                    else:
                        assert isinstance(arg[3], str)
                        plt.plot(temperature, external_density * self.electron_impact_trans[self.atomic_dict[arg[2]]]
                                 [self.atomic_dict[arg[3]]](temperature), label='e impact trans: '+arg[2]+'-->'+arg[3])
                else:
                    assert isinstance(arg[2], str)
                    assert isinstance(arg[-1], int)
                    if arg[-1] > len(self.charged_states):
                        raise ValueError('There are no rates available for atom impact with charged state: q='+str(arg[-1]))
                    if arg[-1] < 1:
                        raise ValueError('There are supported charged for or below: q='+str(arg[-1]))
                    if arg[0] is 'ion':
                        plt.plot(temperature, self.ion_impact_loss[self.atomic_dict[arg[2]]][arg[-1]](temperature) *
                                 external_density, label='p impact ion (q='+str(arg[-1])+'): '+arg[2]+'-->i')
                    else:
                        assert isinstance(arg[3], str)
                        plt.plot(temperature, self.ion_impact_trans[self.atomic_dict[arg[2]]][self.atomic_dict[arg[3]]]
                                 [arg[-1]-1](temperature) * external_density, label='p impact trans (q=' +
                                                                                    str(arg[-1])+'): '+arg[2]+'-->'+arg[3])
        plt.title('Reduced rates for '+self.species+' projectiles at '+str(self.energy)+' keV impact energy.')
        plt.xlabel('Temperature [eV]')
        if spont_flag:
            plt.ylabel('Reduced rate [1/m]')
        else:
            plt.ylabel('Reduces rate coefficient  [m^2]')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.show()
