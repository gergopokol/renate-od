
from cherab.core cimport Line
from cherab.core.math cimport Function1D
from cherab.core.beam cimport BeamModel


cdef class RenateBeamEmissionLine(BeamModel):

    cdef:
        Line _line
        double _wavelength
        Function1D _emissivity

    cdef int _populate_cache(self) except -1
