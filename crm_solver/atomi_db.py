import numpy
from lxml import etree
from utility import getdata


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

    def __generate_rate_function_db(self):
        self.temperature_axis = self.load_rate_data(self.rates_path, 'Temperature axis')

    def __set_einstein_coefficient_db(self):
        raw_einstein_coefficient = self.load_rate_data(self.rates_path, 'Einstein Coeffs')
        if self.atomic_levels != int(raw_einstein_coefficient.size ** 0.5):
            raise Exception('Loaded atomic database is inconsistent with atomic data dictionary. Wrong data loaded.')

    def __set_atomic_dictionary(self):
        assert isinstance(self.species, str)
        if self.species not in ['H', 'D', 'T', 'Li', 'Na']:
            raise Exception(self.species + ' beam atomic data not supported')
        if self.species in ['H', 'D', 'T']:
            self.atomic_dict = {'1n': 0, '2n': 1, '3n': 2, '4n': 3, '5n': 4, '6n': 5}
            self.atomic_levels = 6
        if self.species is 'Li':
            self.atomic_dict = {'2s': 0, '2p': 1, '3s': 2, '3p': 3, '3d': 4, '4s': 5, '4p': 6, '4d': 7, '4f': 8}
            self.atomic_levels = 9
        if self.species is 'Na':
            self.atomic_dict = {'3s': 0, '3p': 1, '3d': 2, '4s': 3, '4p': 4, '4d': 5, '4f': 6, '5s': 7}
            self.atomic_levels = 8

    def __set_rates_path(self, rate_type):
        self.rate_type = rate_type
        self.file_name = 'rate_coeffs_' + str(self.energy) + '_' + self.species + '.h5'
        self.rates_path = getdata.locate_rates_dir(self.species, rate_type) + self.file_name

    @staticmethod
    def load_rate_data(path, tag_name):
        return getdata.GetData(data_path_name=path, data_key=[tag_name], data_format='array').data
