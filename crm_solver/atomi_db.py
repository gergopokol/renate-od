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
