# cython: language_level=3


"""Calculate Beam emissivity with RENATE-OD atomic modelling"""

from scipy import constants

from libc.math cimport exp, sqrt, M_PI
from numpy cimport ndarray
cimport cython
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

from cherab.core cimport Species, Plasma, Beam, Element, BeamPopulationRate
from cherab.core.math cimport Interpolate1DLinear
from cherab.core.model.lineshape cimport doppler_shift, thermal_broadening, add_gaussian_line
from cherab.core.utility.constants cimport RECIP_4_PI, ELEMENTARY_CHARGE, ATOMIC_MASS


# RENATE-OD imports
import pandas as pd
import numpy as np
from lxml import etree
from crm_solver.beamlet import Beamlet


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


cdef class BeamEmissionLine(BeamModel):
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

        # print('running beam emission model')

        # cache data on first run
        if self._emissivity is None:
            print('populating cache')
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
        sigma_x = self._beam.get_sigma() + z * self._beam._attenuator._tanxdiv
        sigma_y = self._beam.get_sigma() + z * self._beam._attenuator._tanydiv

        # normalised radius squared
        norm_radius_sqr = ((x / sigma_x)**2 + (y / sigma_y)**2)

        # clamp low densities to zero (beam models can skip their calculation if density is zero)
        # comparison is done using the squared radius to avoid a costly square root
        if self._beam._attenuator.clamp_to_zero:
            if norm_radius_sqr > self._beam._attenuator._clamp_sigma_sqr:
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

        # run renate model
        beamlet_grid, beam_emission = _calculate_renate_emissivity(self._plasma, self._beam)

        # print()
        # print(beam_emission.min(), beam_emission.max(), beam_emission.mean())
        # print()

        # a tiny degree of extrapolation is permitted to handle numerical accuracy issues with the end of the array
        self._emissivity = Interpolate1DLinear(beamlet_grid, beam_emission, extrapolate=True, extrapolation_range=1e-4)

    def _change(self):

        # clear cache to force regeneration on first use
        self._wavelength = 0.0
        self._emissivity = None


def _sample_along_beam_axis(function, beam_axis, beam_to_world, debug=False):

    if debug:
        print(beam_axis)
        samples = []
        for i, z in enumerate(beam_axis):
            p = Point3D(0, 0, z).transform(beam_to_world)
            if i == 50:
                print(z, p)
                print(function(p.x, p.y, p.z))
            samples.append(function(p.x, p.y, p.z))
    else:
        samples = []
        for z in beam_axis:
            p = Point3D(0, 0, z).transform(beam_to_world)
            samples.append(function(p.x, p.y, p.z))

    return samples


def _calculate_renate_emissivity(plasma, beam):

    # build species specifications, starting with electrons
    charges = [-1]
    charges.extend([s.charge for s in plasma.composition if not s.charge == 0])
    nuclear_charges = [0]
    nuclear_charges.extend([s.element.atomic_number for s in plasma.composition if not s.charge == 0])
    atomic_weights = [0]
    atomic_weights.extend([int(s.element.atomic_weight) for s in plasma.composition if not s.charge == 0])
    index = ['electrons']
    index.extend(['ion{}'.format(i+1) for i in range(len(atomic_weights) - 1)])
    components = pd.DataFrame(data={'q': charges, 'Z': nuclear_charges, 'A': atomic_weights}, index=index)

    # sample plasma parameters along the beam axis
    beam_axis = np.linspace(0, beam.length, num=500)
    beam_to_world = beam.to_root()

    num_params = 1 + 2 + len(plasma.composition) * 2  # *2 since every species has density and temperature
    profiles = np.zeros((num_params, 500))
    type_labels = []
    property_labels = []
    unit_labels = []
    profiles[0, :] = beam_axis
    type_labels.append('beamlet grid')
    property_labels.append('distance')
    unit_labels.append('m')
    profiles[1, :] = _sample_along_beam_axis(plasma.electron_distribution.density, beam_axis, beam_to_world, debug=True)
    type_labels.append('electron')
    property_labels.append('density')
    unit_labels.append('m-3')
    profiles[2, :] = _sample_along_beam_axis(plasma.electron_distribution.effective_temperature, beam_axis, beam_to_world)
    type_labels.append('electron')
    property_labels.append('temperature')
    unit_labels.append('eV')
    for i, species in enumerate(plasma.composition):
        profiles[i*2 + 3, :] = _sample_along_beam_axis(species.distribution.density, beam_axis, beam_to_world)
        type_labels.append('ion{}'.format(i))
        property_labels.append('density')
        unit_labels.append('m-3')
        profiles[i*2 + 4, :] = _sample_along_beam_axis(species.distribution.effective_temperature, beam_axis, beam_to_world)
        type_labels.append('ion{}'.format(i))
        property_labels.append('temperature')
        unit_labels.append('eV')

    profiles = np.swapaxes(profiles, 0, 1)
    row_index = [i for i in range(500)]
    column_index = pd.MultiIndex.from_arrays([type_labels, property_labels, unit_labels], names=['type', 'property', 'unit'])

    profiles = pd.DataFrame(data=profiles, columns=column_index, index=row_index)


    # construct beam param specification
    xml = etree.Element('xml')
    head = etree.SubElement(xml, 'head')
    id_tag = etree.SubElement(head, 'id')
    id_tag.text = 'beamlet_test'
    body_tag = etree.SubElement(xml, 'body')
    beamlet_energy = etree.SubElement(body_tag, 'beamlet_energy', {'unit': 'keV'})
    beamlet_energy.text = str(int(beam.energy/1000))
    beamlet_species = etree.SubElement(body_tag, 'beamlet_species')
    beamlet_species.text = beam.element.symbol
    beamlet_source = etree.SubElement(body_tag, 'beamlet_source')
    beamlet_source.text = 'beamlet/test_impurity.h5'
    beamlet_current = etree.SubElement(body_tag, 'beamlet_current', {'unit': 'A'})
    beamlet_current.text = '0.001'
    beamlet_mass = etree.SubElement(body_tag, 'beamlet_mass', {'unit': 'kg'})
    beamlet_mass.text = '1.15258e-026'
    beamlet_velocity = etree.SubElement(body_tag, 'beamlet_velocity', {'unit': 'm/s'})
    beamlet_velocity.text = '1291547.1348855693'
    beamlet_profiles = etree.SubElement(body_tag, 'beamlet_profiles', {})
    beamlet_profiles.text = './beamlet_test.h5'
    param = etree.ElementTree(element=xml)

    b = Beamlet(param=param, profiles=profiles, components=components)

    print(b.profiles['beamlet grid']['distance']['m'])
    print(b.profiles['electron']['density']['m-3'])
    print(b.profiles['ion1']['temperature']['eV'])

    b.compute_linear_emission_density()
    b.compute_linear_density_attenuation()
    b.compute_relative_populations()

    beamlet_grid = np.squeeze(np.array(b.profiles['beamlet grid']))
    beam_emission = np.squeeze(np.array(b.profiles['linear_emission_density']))

    return beamlet_grid, beam_emission
