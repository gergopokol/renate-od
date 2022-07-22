import numpy as np
import pandas as pd
from lxml import etree
from utility.getdata import GetData
from utility.geometrical_objects import Point, Vector, Line

try:
    import imas
    from imas_utility.idsnbi import NbiIds
except ImportError:
    IMAS_FLAG = False


class Beam(object):
    def __init__(self, beam=None, source='external', data_id=None, data_path='device_data/test/test_beam.xml'):
        self.beam_source = source
        self.data_path = data_path
        self.imas_flag = IMAS_FLAG
        if data_id is not None:
            self.__unpack_data_id(data_id)
        self.__get_beam_data()

    def __unpack_data_id(self, data_id):
        self.shot = data_id[0]
        self.run = data_id[1]
        self.user = data_id[2]
        self.machine = data_id[3]
        self.time = data_id[4]

    def __get_beam_data(self):
        if self.beam_source == 'ids' and self.imas_flag:
            self.__load_beam_from_ids()
        elif self.beam_source == 'external':
            self.__load_beam_external_source()

    def __load_beam_external_source(self):
        external_beam = GetData(data_path_name=self.data_path).data
        if not isinstance(external_beam, etree._ElementTree):
            raise TypeError('The external data source is expected to be in an .xml data format.')
        self.beam_energy = external_beam.getroot().find('body').find('beam_energy').text
        self.beam_species = external_beam.getroot().find('body').find('beam_species').text
        self.beam_current = float(external_beam.getroot().find('body').find('beam_current').text)
        self.beam_fwhm = float(external_beam.getroot().find('body').find('beam_fwhm').text)
        self.beam_resolution = float(external_beam.getroot().find('body').find('beam_resolution').text)
        self.start = Point(cartesian=[float(external_beam.getroot().find('body').find('beam_start').find('x').text),
                                      float(external_beam.getroot().find('body').find('beam_start').find('y').text),
                                      float(external_beam.getroot().find('body').find('beam_start').find('z').text)])
        self.end = Point(cartesian=[float(external_beam.getroot().find('body').find('beam_end').find('x').text),
                                    float(external_beam.getroot().find('body').find('beam_end').find('y').text),
                                    float(external_beam.getroot().find('body').find('beam_end').find('z').text)])
        self.centerline = Line(self.start, self.end, 2)

    def __load_beam_from_ids(self):
        imas_beam = NbiIds(shot=self.shot, run=self.run, machine=self.machine, user=self.user)

    def generate_current_distribution(self, X, Y, shape_function):
        self.current_distribution = np.zeros((len(X), len(Y)))
        self.grid_x = X
        self.grid_y = Y
        for i, x in enumerate(self.grid_x):
            for j, y in enumerate(self.grid_y):
                self.current_distribution[i, j] = shape_function(x, y)
        self.current_distribution /= np.sum(self.current_distribution)
        self.current_distribution *= self.current

    def generate_beamlet_lines(self):
        self.beamlet_lines = np.empty(self.current_distribution.shape, dtype=BeamletLine)
        XY_length = np.sqrt(self.centerline.vector.x**2+self.centerline.vector.y**2)
        cos_theta = XY_length/self.centerline.length
        sin_phi = -self.centerline.vector.x/XY_length
        cos_phi = self.centerline.vector.y/XY_length
        for i, x in enumerate(self.grid_x):
            for j, y in enumerate(self.grid_y):
                line_start = (self.start.x+x*cos_phi, self.start.y+x*sin_phi, self.start.z+y*cos_theta)
                line_end = (self.end.x+x*cos_phi, self.end.y+x*sin_phi, self.end.z+y*cos_theta)
                self.beamlet_lines[i, j] = BeamletLine(line_start, line_end, resolution=self.beam_resolution)

    def generate_beamlets(self, singular=False, points_along_beamlet=None):
        if points_along_beamlet is None:
            self.points_along_beamlet = self.end.distance(self.start)/self.beam_resolution
        else:
            self.points_along_beamlet = points_along_beamlet
        if singular:
            self.__generate_1d_beam()

    def __generate_1d_beam(self):
        pass


class BeamletLine(Line):

    def __init__(self, rootPoint, endPoint, current, number_of_points=1000, resolution=None):
        super().__init__(self, rootPoint, endPoint, number_of_points, resolution)
        self.current = current
