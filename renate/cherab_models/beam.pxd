
from cherab.core.math cimport Function1D
from cherab.core.beam.node cimport Beam, ModelManager


cdef class RenetaModelManager(ModelManager):
    pass


cdef class RenateBeam(Beam):

    cdef:
        object _renate_wrapper
        Function1D _density
        readonly double _tanxdiv, _tanydiv, _clamp_sigma_sqr
        readonly bint clamp_to_zero
