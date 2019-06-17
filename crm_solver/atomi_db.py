import numpy
from lxml import etree
from utility.getdata import GetData


class AtomicDB:
    def __init__(self, param=None, data_path='beamlet/testimp0001.xml'):
        self.param = param
        if not isinstance(self.param, etree._ElementTree):
            self.param = GetData(data_path_name=data_path).data
        assert isinstance(self.param, etree._ElementTree)
        self.beam_energy = self.param.getroot().find('body').find('beamlet_energy').text
        self.beam_species = self.param.getroot().find('body').find('beamlet_species').text

    @staticmethod
    def set_atomic_dictionary(atom):
        assert isinstance(atom, str)
        if atom not in ['H', 'D', 'T', 'Li', 'Na']:
            raise Exception(atom + ' beam atomic data not supported')
        if atom is 'H' or atom is 'D' or atom is 'T':
            return {'1n': 0, '2n': 1, '3n': 2, '4n': 3, '5n': 4, '6n': 5}
        if atom is 'Li':
            return {'2s': 0, '2p': 1, '3s': 2, '3p': 3, '3d': 4, '4s': 5, '4p': 6, '4d': 7, '4f': 8}
        if atom is 'Na':
            return {'3s': 0, '3p': 1, '3d': 2, '4s': 3, '4p': 4, '4d': 5, '4f': 6, '5s': 7}
