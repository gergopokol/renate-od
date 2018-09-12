from lxml import etree
import pandas
import numpy
import math
from utility.constants import Constants
from utility import convert
from utility import getdata
from pdb import set_trace as bp


class Obs1d:
    def __init__(self, beamlet, data_path, beam_current):
        self.beamlet = beamlet
        self.constants = Constants()
        self.photon_fraction = beam_current/self.constants.charge_electron\
                               / float(beamlet.param.getroot().find('body').find('beamlet_velocity').text)
        self.obs_param = self.read_observation_param(data_path)
        self.obs_profile = self.read_observation_profile(data_path=self.obs_param.getroot().find('body').find('observation_profile_path').text)

    def calculate_light_profile(self):
        light_profile = numpy.zeros(self.obs_profile.size)
        for detector_index in range(len(self.obs_profile)):
            detector_steps = numpy.where(abs(self.beamlet.profiles.beamlet_grid-self.obs_profile[0][detector_index])
                                         < float(self.obs_param.getroot().find('body').find('detector_size').text) / 2)
            for step_index in detector_steps[0]:
                light_profile[detector_index] = light_profile[detector_index] + \
                                                self.beamlet.profiles[
                                                    self.obs_param.getroot().find('body')
                                                        .find('observed_level').text][step_index] * self.photon_fraction
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