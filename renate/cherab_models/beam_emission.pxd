
from numpy cimport ndarray
from raysect.optical cimport Node, World, Primitive, Ray, Spectrum, SpectralFunction, Point3D, Vector3D, AffineMatrix3D

from cherab.core cimport Species, Plasma, Beam, Line, AtomicData, BeamCXPEC
from cherab.core.math cimport Function1D
from cherab.core.beam cimport BeamModel


cdef class BeamEmissionLine(BeamModel):

    cdef:
        Line _line
        double _wavelength
        Function1D _emissivity

    cdef int _populate_cache(self) except -1
