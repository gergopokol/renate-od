import numpy as np
import pandas as pd
from lxml import etree

try:
    import imas
except ImportError:
    IMAS_FLAG = False


class Beam(object):
    def __init__(self, beam=None, source=None, data_id=None, data_path='device_data/test/test_beam.xml'):
        self.beam_source = source
        self.data_path = data_path
        self.imas_flag = IMAS_FLAG
        if data_id is not None:
            self.__unpack_data_id(data_id)

    def __unpack_data_id(self, data_id):
        self.shot = data_id[0]
        self.run = data_id[1]
        self.user = data_id[2]
        self.machine = data_id[3]
        self.time = data_id[4]

    def build_beam_data(self):
        pass

    def generate_beamlets(self):
        pass
