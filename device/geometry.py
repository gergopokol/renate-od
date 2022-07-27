import numpy as np
import scipy as sc
from utility.geometrical_objects import Line


class LineOfSight(Line):

    def __init__(self, rootPoint, endPoint, number_of_points=1000, resolution=None):
        super().__init__(rootPoint, endPoint, number_of_points, resolution)

    def interpolate_points(self, interpolator):
        result = interpolator(self.points.view((float, 3)))
        nan_loc = np.isnan(result)
        return self.points[~nan_loc], result[~nan_loc]


class ObservationGeometry(object):
    def __init__(self):
        pass
