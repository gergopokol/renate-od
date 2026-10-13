import numpy as np
import pandas as pd
from lxml import etree
from utility.getdata import GetData
from utility.geometrical_objects import Point, Vector, Line, Plane
from scipy.interpolate import LinearNDInterpolator

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
        self.base_plane = Plane(self.start, self.centerline.vector)
        self.end_plane = Plane(self.end, self.centerline.vector)

    def add_base_grid(self, x, y):
        self.base_grid = np.array(list(zip(x, y)), dtype=[('x', 'float'), ('y', 'float')])

    def add_end_grid(self, x, y):
        self.end_grid = np.array(list(zip(x, y)), dtype=[('x', 'float'), ('y', 'float')])

    def __load_beam_from_ids(self):
        imas_beam = NbiIds(shot=self.shot, run=self.run, machine=self.machine, user=self.user)

    def add_current_distribution(self, currents):
        self.current_distribution = (currents/np.sum(currents))*self.beam_current

    def generate_beamlet_lines(self):
        self.beamlet_lines = []
        for i, c in enumerate(self.current_distribution):
            root = self.base_plane.transform_to_world([self.base_grid[i]['x'],
                                                       self.base_grid[i]['y'],
                                                       0.0])
            end = self.end_plane.transform_to_world([self.end_grid[i]['x'],
                                                     self.end_grid[i]['y'],
                                                     0.0])
            self.beamlet_lines.append(BeamletLine(root, end, c, resolution=self.beam_resolution))

    def generate_current_interpolator(self):
        points = self.beamlet_lines[0].points.view((float, 3))
        values = np.full(self.beamlet_lines[0].points.shape, self.beamlet_lines[0].current)
        for i in range(1, len(self.beamlet_lines)):
            points = np.vstack((points, self.beamlet_lines[i].points.view((float, 3))))
            values = np.hstack((values, np.full(self.beamlet_lines[i].points.shape,
                                                self.beamlet_lines[i].current)))
        self.current_interpolator = LinearNDInterpolator(points, values)
        return self.current_interpolator

    def generate_beamlets(self, singular=False, points_along_beamlet=None):
        if points_along_beamlet is None:
            self.points_along_beamlet = self.end.distance(self.start)/self.beam_resolution
        else:
            self.points_along_beamlet = points_along_beamlet
        if singular:
            self.__generate_1d_beam()

    def __generate_1d_beam(self):
        pass

    def intersection_with_Line(self, line):
        pass


class BeamletLine(Line):

    def __init__(self, rootPoint, endPoint, current, number_of_points=1000, resolution=None):
        super().__init__(rootPoint, endPoint, number_of_points, resolution)
        self.current = current
