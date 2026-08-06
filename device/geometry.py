import numpy as np
from utility.geometrical_objects import Line, Vector


class LineOfSight(Line):

    def __init__(self, rootPoint, endPoint, number_of_points=1000, resolution=None, extension=None):
        if isinstance(extension, float) or isinstance(extension, int):
            root = Vector(rootPoint)
            end = Vector(endPoint)
            vector = end-root
            ext_end = (root+vector.normalized() *
                       (vector.length+extension)).cartesians
            super().__init__(rootPoint, ext_end, number_of_points, resolution)
        else:
            super().__init__(rootPoint, endPoint, number_of_points, resolution)
        self.detector_position = Vector(endPoint)

    def interpolate_points(self, interpolator):
        result = interpolator(self.points.view((float, 3)))
        nan_loc = np.isnan(result)
        return self.points[~nan_loc], result[~nan_loc]


class ObservationGeometry(object):
    def __init__(self):
        pass
