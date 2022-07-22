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

    def __repr__(self):
        if hasattr(self, 'x'):
            return 'Point coordinates:\n' \
                   'Cartesian: (x,y,z) [m,m,m] = (' + str(self.x) + '; ' + str(self.y) + '; ' + str(self.z) + ')\n' \
                   'Cylindrical: (r,phi,z) [m,rad,m]= (' + str(self.r) + '; ' + str(self.phi) + '; ' + str(self.z) + \
                   ')\nNormalized flux value: Psi [-] = ' + str(self.psi)
        else:
            return 'Point coordinates: NOT defined.'

    def __iter__(self):
        yield from [self.x, self.y, self.z]

    def add_cartesian_point(self, point):
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]
        self.cartesians = np.array((self.x, self.y, self.z))
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

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


ORIGIN = Point((0, 0, 0))


class Vector(Point):

    def __init__(self, point, pointtype='cartesian'):
        if isinstance(point, Point) or isinstance(point, Vector):
            super().__init__((point.x, point.y, point.z))
            self.psi = point.psi
        elif pointtype == 'cartesian':
            super().__init__(point)
        elif pointtype == 'cylindrical':
            super().__init__(cylindrical=point)
        self.length = self.distance(ORIGIN)

    def __repr__(self):
        if hasattr(self, 'x'):
            return 'Vector coordinates:\n' \
                   'Cartesian: (x,y,z) [m,m,m] = (' + str(self.x) + '; ' + str(self.y) + '; ' + str(self.z) + ')\n' \
                   'Cylindrical: (r,phi,z) [m,rad,m]= (' + str(self.r) + '; ' + str(self.phi) + '; ' + str(self.z) + \
                   ')\nLength [m]: '+str(self.length)+'\nNormalized flux value: Psi [-] = ' + str(self.psi)
        else:
            return 'Point coordinates: NOT defined.'

    def __add__(self, other):
        return Vector((self.x+other.x, self.y+other.y, self.z+other.z))

    def __sub__(self, other):
        return Vector((self.x-other.x, self.y-other.y, self.z-other.z))

    def __truediv__(self, numeric):
        try:
            float(numeric)
        except:
            raise TypeError('A Vector can only be divided by a number.')
        return Vector((self.x/numeric, self.y/numeric, self.z/numeric))

    def __mul__(self, numeric):
        try:
            float(numeric)
        except:
            raise TypeError('The * operator is used to multiply by scalar.')
        return Vector((self.x*numeric, self.y*numeric, self.z*numeric))

    def dot(self, other):
        return self.x*other.x+self.y*other.y+self.z*other.z

    def normalized(self,):
        return self/self.length


class Line():

    def __init__(self, rootPoint, endPoint, number_of_points=1000, resolution=None):
        self.root = Vector(rootPoint)
        self.end = Vector(endPoint)
        self.vector = self.end-self.root
        self.length = self.vector.length
        if resolution == None:
            try:
                self.pointsnum = int(number_of_points)
            except:
                raise TypeError('Number of points must be an integer.')
            self.x = np.linspace(self.root.x, self.end.x, self.pointsnum)
            self.y = np.linspace(self.root.y, self.end.y, self.pointsnum)
            self.z = np.linspace(self.root.z, self.end.z, self.pointsnum)
            self.resolution = Point((self.x[1], self.y[1], self.z[1])).distance(self.root)
        else:
            try:
                self.resolution = float(resolution)
            except:
                raise TypeError('Resolution must be a number.')
            self.pointsnum = int(self.length/self.resolution)
            dv = self.vector/self.pointsnum
            self.x = np.arange(self.root.x, self.end.x, dv.x)
            self.y = np.arange(self.root.y, self.end.y, dv.y)
            self.z = np.arange(self.root.z, self.end.z, dv.z)

    def project(self, plane):
        root = plane.project_point(self.root)
        end = plane.project_point(self.end)
        return Line(root, end, number_of_points=self.pointsnum)


class Plane:

    def __init__(self, point, normal):
        self.point = Vector(point)
        self.normal = Vector(normal).normalized()

    def project_point(self, point):
        r_point = Vector(point)
        r_projected = r_point+self.normal*(self.normal.dot(self.point-r_point))
        return Point(r_projected.cartesians)
