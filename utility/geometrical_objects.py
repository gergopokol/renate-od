import numpy as np
import math


class Point(object):
    def __init__(self, cartesian=None, cylindrical=None, equilibrium_interpolator=None):
        if cartesian is not None and cylindrical is None:
            self.add_cartesian_point(cartesian)
        elif cylindrical is not None and cartesian is None:
            self.add_cylindrical_point(cylindrical)
        elif cylindrical is not None and cartesian is not None:
            self.add_cartesian_point(cartesian)
            if self.r != cylindrical[0] or self.phi != cylindrical[1] or cartesian[2] != cylindrical[2]:
                raise ValueError('The cylindrical and cartesian input data does not match! Point object invalid.')
        if equilibrium_interpolator is not None:
            self.add_flux_surface_value(equilibrium_interpolator)
        else:
            self.psi = '-'

    def add_cartesian_point(self, point):
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]
        self.from_cartesian_to_cylindrical()

    def add_cylindrical_point(self, point):
        self.r = point[0]
        self.phi = point[1]
        self.z = point[2]
        self.from_cylindrical_to_cartesian()

    def add_flux_surface_value(self, interpolator):
        self.psi = interpolator(self.r, self.z)

    def from_cartesian_to_cylindrical(self):
        self.r = math.sqrt(self.x**2 + self.y**2)
        self.phi = math.atan2(self.y, self.x)

    def from_cylindrical_to_cartesian(self):
        self.x = self.r * math.cos(self.phi)
        self.y = self.r * math.sin(self.phi)
