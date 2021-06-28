import numpy as np
import pandas as pd


class Beam(object):
    def __init__(self, beam_profile=None, source=None, data_path='device_data/test/test_beam.xml'):
        self.beam_profile = beam_profile
        self.beam_source = source
        self.data_path = data_path

    def get_beam_data(self):
        pass

    def generate_beamlets(self):
        pass
