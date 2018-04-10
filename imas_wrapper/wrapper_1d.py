from lxml import etree
from utility.getdata import GetData


class BeamletFromIds:
    def __init__(self, input_source='beamlet/test_imas.xml'):
        self.access_path = input_source
        self.read_imas_input_xml()
        #self.param.getroot().find('body').find('beamlet_energy').text

    def read_imas_input_xml(self):
        self.param = GetData(data_path_name=self.access_path).data
        assert isinstance(self.param, etree._ElementTree)
        print('ElementTree read to dictionary from: ' + self.access_path)
