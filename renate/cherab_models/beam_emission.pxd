
from cherab.core cimport Line
from cherab.core.math cimport Function1D, Function2D
from cherab.core.beam cimport BeamModel
from cherab.core.model.lineshape cimport BeamLineShapeModel


cdef class RenateBeamEmissionLine(BeamModel):

    cdef:
        Line _line
        double _wavelength
        Function1D _emissivity
        BeamLineShapeModel _lineshape
        Function2D _sigma_to_pi
        Function1D _sigma1_to_sigma0, _pi2_to_pi3, _pi4_to_pi3
        bint _using_stark_splitting

    cdef int _populate_cache(self) except -1
