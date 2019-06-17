import numpy
from lxml import etree
from utility.getdata import GetData


class AtomicDB:
    def __init__(self, param=None, data_path='beamlet/testimp0001.xml'):
        self.param = param
        if not isinstance(self.param, etree._ElementTree):
            self.param = GetData(data_path_name=data_path).data
        assert isinstance(self.param, etree._ElementTree)
