import pyximport
pyximport.install()

import numpy as np

from raysect.core import Point3D, Vector3D
from raysect.core.math.function.vector3d.function1d.autowrap import _autowrap_function1d
from raysect.core.math.function.vector3d.function2d.autowrap import _autowrap_function2d
from raysect.core.math.function.vector3d.function2d.autowrap import PythonFunction2D
from raysect.optical import Spectrum

from cherab.core import Plasma, Beam, Element, AtomicData
from cherab.core.atomic import hydrogen
from cherab.core.model.lineshape import doppler_shift, thermal_broadening, add_gaussian_line, BeamEmissionMultiplet
from cherab.core.model.beam.beam_emission import SIGMA_TO_PI, SIGMA1_TO_SIGMA0, PI2_TO_PI3, PI4_TO_PI3
#from cherab.core.utility.constants import RECIP_4_PI, ELEMENTARY_CHARGE, ATOMIC_MASS

from cherab.core.beam.model import BeamModel

RECIP_4_PI=1/(4*np.pi)
ELEMENTARY_CHARGE=1.602176634e-19
ATOMIC_MASS=1.66053906660e-27

RECIP_ELEMENTARY_CHARGE = 1 / ELEMENTARY_CHARGE
RECIP_ATOMIC_MASS = 1 / ATOMIC_MASS


def evamu_to_ms(x):
    return np.sqrt(2 * x * ELEMENTARY_CHARGE * RECIP_ATOMIC_MASS)


def ms_to_evamu(x):
    return 0.5 * (x ** 2) * RECIP_ELEMENTARY_CHARGE * ATOMIC_MASS


def amu_to_kg(x):
    return x * ATOMIC_MASS


def kg_to_amu(x):
    return x * RECIP_ATOMIC_MASS


def ev_to_j(x):
    return x * ELEMENTARY_CHARGE


def j_to_ev(x):
    return x * RECIP_ELEMENTARY_CHARGE


class RenateBeamEmissionLine(BeamModel):
    """Calculates the line emission for a beam.

    :param line:
    :param step: integration step in meters
    :return:
    """

    def __init__(self, line, beam=None, plasma=None, atomic_data=None,
                 sigma_to_pi=SIGMA_TO_PI, sigma1_to_sigma0=SIGMA1_TO_SIGMA0,
                 pi2_to_pi3=PI2_TO_PI3, pi4_to_pi3=PI4_TO_PI3):

        super().__init__(beam, plasma, atomic_data)

        self._line = line
        self._beam=beam
        self._plasma=plasma
        self._atomic_data=atomic_data

        self._sigma_to_pi = sigma_to_pi
        self._sigma1_to_sigma0 = sigma1_to_sigma0
        self._pi2_to_pi3 = pi2_to_pi3
        self._pi4_to_pi3 = pi4_to_pi3

        # initialise cache to empty
        self._wavelength = 0.0
        self._emissivity = None

    @property
    def beam(self):
        return self._beam

    @beam.setter
    def beam(self, value):

        # disconnect from previous beam's notifications
        if self._beam:
            self._beam.notifier.remove(self._change)

        # attach to beam to inform model of changes to beam properties
        self._beam = value
        self._beam.notifier.add(self._change)

        # inform model source data has changed
        self._change()

    def get_beam(self):
        return self._beam

    @property
    def atomic_data(self):
        return self._atomic_data

    @atomic_data.setter
    def atomic_data(self, value):

        self._atomic_data = value

        # inform model source data has changed
        self._change()

    @property
    def plasma(self):
        return self._plasma

    @plasma.setter
    def plasma(self, value):

        # disconnect from previous plasma's notifications
        if self._plasma:
            self._plasma.notifier.remove(self._change)

        # attach to plasma to inform model of changes to plasma properties
        self._plasma = value
        self._plasma.notifier.add(self._change)

        # inform model source data has changed
        self._change()

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, value):
        # the data cache depends on the line configuration
        self._line = value
        self._change()

    @property
    def emissivity(self):
        return self._emissivity


    def emission(self, beam_point, plasma_point, beam_direction,
                            observation_direction, spectrum):

        # cache data on first run
        if self._emissivity is None:
            self._populate_cache()

        # abort calculation if temperature is zero
        temperature = self._beam._temperature
        if temperature == 0:
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
        gaussian_sample = np.exp(-0.5 * norm_radius_sqr/(sigma_x*sigma_y)) / np.sqrt(2 * np.pi * sigma_x * sigma_y)

        # spectral line emission in W/m^3/str
        emissivity = self._emissivity(z) * gaussian_sample
        
        if False:#self._using_stark_splitting:

            return self._lineshape.add_line(emissivity, beam_point, plasma_point,
                                            beam_direction, observation_direction, spectrum)

        else:

            velocity = beam_direction.normalise()*(evamu_to_ms(self._beam.get_energy()))

            # calculate emission line central wavelength, doppler shifted along observation direction
            natural_wavelength = self._wavelength
            central_wavelength = doppler_shift(natural_wavelength, observation_direction, velocity)

            beam_ion_mass = self._beam.get_element().atomic_weight

            sigma = thermal_broadening(natural_wavelength, temperature, beam_ion_mass)

            return add_gaussian_line(emissivity, central_wavelength, sigma, spectrum)

    def _populate_cache(self):

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

        if self._line.element == hydrogen and self._line.transition == (3, 2):
            self._using_stark_splitting = True
        else:
            self._using_stark_splitting = False

        # TODO - add a simple guassian beam emission feature
        # instance line shape renderer
        self._lineshape = BeamEmissionMultiplet(self._line, self._wavelength, self._beam, self._sigma_to_pi,
                                                self._sigma1_to_sigma0, self._pi2_to_pi3, self._pi4_to_pi3)

    def _change(self):

        # clear cache to force regeneration on first use
        self._wavelength = 0.0
        self._emissivity = None
        self._lineshape = None
