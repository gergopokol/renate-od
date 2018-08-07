from lxml import etree
import pandas
import numpy
import math
from utility.constants import Constants
from utility import convert
from utility import getdata
from pdb import set_trace as bp


class Obs1d:
    def __init__(self, input_obj, data_path, beam_current):
        self.input_param = input_obj.param
        self.input_profiles = input_obj.profiles
        self.constants = Constants()
        self.velocity = convert.calculate_velocity_from_energy(self.input_param.getroot().find('body').find('beamlet_energy').text,
                                                               getdata.get_mass(
                                                                   self.input_param.getroot().find('body').find('beamlet_species').text))
        self.photon_fraction = beam_current/self.constants.charge_electron/self.velocity
        self.obs_param = self.read_observation_param(data_path)
        self.obs_profile = self.read_observation_profile(data_path=self.obs_param.getroot().find('body').find('observation_profile_path').text)

    def calculate_light_profile(self):
        light_profile = numpy.full(self.obs_profile.size, 0)
        for detector_index in range(len(self.obs_profile)):
            loc_up = self.obs_profile[0][detector_index] + float(self.obs_param.getroot().find('body').find('detector_size').text)/ 2
            loc_down = self.obs_profile[0][detector_index] - float(self.obs_param.getroot().find('body').find('detector_size').text)/ 2
            position_index_1 = numpy.where((self.input_profiles.beamlet_grid < loc_up))
            position_index_2 = numpy.where((self.input_profiles.beamlet_grid > loc_down))
            low = list(position_index_1)
            up = list(position_index_2)
            for step_index in range(up[0][0], low[0][-1]):
                light_profile[detector_index] = light_profile[detector_index] + self.input_profiles['level 2'][step_index] \
                                                                                * self.photon_fraction
        return light_profile

    @staticmethod
    def read_observation_param(data_path):
        obs_param = getdata.GetData(data_path_name=data_path).data
        assert isinstance(obs_param, etree._ElementTree)
        print('obs_param read from file: ' + data_path)
        return obs_param

    @staticmethod
    def read_observation_profile(data_path):
        obs_profile = getdata.GetData(data_path_name=data_path).data
        assert isinstance(obs_profile, pandas.DataFrame)
        print('obs_profile read from file: ' + data_path)
        return obs_profile