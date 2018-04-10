from lxml import etree

class BeamletFromIds:
    def __init__(self, input_source='beamlet/test_imas.xml'):
        self.access_path = input_source
        pass

    def read_imas_input_xml(self):
        self.data = etree.parse(self.access_path)
        assert isinstance(self.data, etree._ElementTree)
        print('ElementTree read to dictionary from: ' + self.access_path)
