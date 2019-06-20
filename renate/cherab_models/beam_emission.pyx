# cython: language_level=3


from libc.math cimport exp, sqrt, M_PI
cimport cython

from cherab.core cimport Plasma, Beam, Element
from cherab.core.model.lineshape cimport doppler_shift, thermal_broadening, add_gaussian_line
from cherab.core.utility.constants cimport RECIP_4_PI, ELEMENTARY_CHARGE, ATOMIC_MASS


cdef double RECIP_ELEMENTARY_CHARGE = 1 / ELEMENTARY_CHARGE
cdef double RECIP_ATOMIC_MASS = 1 / ATOMIC_MASS


cdef double evamu_to_ms(double x):
    return sqrt(2 * x * ELEMENTARY_CHARGE * RECIP_ATOMIC_MASS)


cdef double ms_to_evamu(double x):
    return 0.5 * (x ** 2) * RECIP_ELEMENTARY_CHARGE * ATOMIC_MASS


cdef double amu_to_kg(double x):
    return x * ATOMIC_MASS


cdef double kg_to_amu(double x):
    return x * RECIP_ATOMIC_MASS


cdef double ev_to_j(double x):
    return x * ELEMENTARY_CHARGE


cdef double j_to_ev(double x):
    return x * RECIP_ELEMENTARY_CHARGE


cdef class RenateBeamEmissionLine(BeamModel):
    """Calculates the line emission for a beam.

    :param line:
    :param step: integration step in meters
    :return:
    """

    def __init__(self, Line line not None, Beam beam=None, Plasma plasma=None, AtomicData atomic_data=None):

        super().__init__(beam, plasma, atomic_data)

        self._line = line

        # initialise cache to empty
        self._wavelength = 0.0
        self._emissivity = None

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, Line value not None):
        # the data cache depends on the line configuration
        self._line = value
        self._change()

    @property
    def emissivity(self):
        return self._emissivity

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef Spectrum emission(self, Point3D beam_point, Point3D plasma_point, Vector3D beam_direction,
                            Vector3D observation_direction, Spectrum spectrum):

        cdef:
            double x, y, z
            double temperature, emissivity
            double sigma_x, sigma_y, norm_radius_sqr, gaussian_sample
            Vector3D velocity
            double natural_wavelength, central_wavelength, width_squared, radiance, sigma

        # cache data on first run
        if self._emissivity is None:
            self._populate_cache()

        # abort calculation if temperature is zero
        temperature = self._beam._temperature
        if temperature == 0:
            # print('beam temperature was zero')
            return spectrum

        # extract for more compact code
        x = beam_point.x
        y = beam_point.y
        z = beam_point.z

        # calculate beam width
        sigma_x = self._beam.get_sigma() + z * self._beam._tanxdiv
        sigma_y = self._beam.get_sigma() + z * self._beam._tanydiv

        # normalised radius squared
        norm_radius_sqr = ((x / sigma_x)**2 + (y / sigma_y)**2)

        # clamp low densities to zero (beam models can skip their calculation if density is zero)
        # comparison is done using the squared radius to avoid a costly square root
        if self._beam.clamp_to_zero:
            if norm_radius_sqr > self._beam._clamp_sigma_sqr:
                return spectrum

        # bi-variate Gaussian distribution (normalised)
        gaussian_sample = exp(-0.5 * norm_radius_sqr) / (2 * M_PI * sigma_x * sigma_y)

        # spectral line emission in W/m^3/str
        # print('emissivity:', z, self._emissivity.evaluate(z), gaussian_sample)
        emissivity = self._emissivity.evaluate(z) * gaussian_sample

        velocity = beam_direction.normalise().mul(evamu_to_ms(self._beam.get_energy()))

        # calculate emission line central wavelength, doppler shifted along observation direction
        natural_wavelength = self._wavelength
        central_wavelength = doppler_shift(natural_wavelength, observation_direction, velocity)

        beam_ion_mass = self._beam.get_element().atomic_weight

        sigma = thermal_broadening(natural_wavelength, temperature, beam_ion_mass)

        return add_gaussian_line(emissivity, central_wavelength, sigma, spectrum)

    cdef int _populate_cache(self) except -1:

        cdef:
            Element beam_element
            int charge
            tuple transition

        # sanity checks
        if self._beam is None or self._plasma is None:
            raise RuntimeError("The emission model is not connected to a beam object.")

        if self._line is None:
            raise RuntimeError("The emission line has not been set.")

        if not self._line.element == self._beam.element:
            raise ValueError("This composition of this beam ({}) is not compatible with the line specified ({})."
                             "".format(self._beam.element, self._line.element))

        beam_element = self._beam.element
        charge = self._line.charge
        transition = self._line.transition

        # obtain wavelength for specified line
        self._wavelength = self._atomic_data.wavelength(beam_element, charge, transition)

        # get emission function from Renate-OD
        renate_wrapper = self._beam.renate_wrapper
        self._emissivity = renate_wrapper.beam_emission_intensity(self._line.transition)

    def _change(self):

        # clear cache to force regeneration on first use
        self._wavelength = 0.0
        self._emissivity = None
