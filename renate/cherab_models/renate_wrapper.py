

import pandas as pd
import numpy as np
from lxml import etree
from raysect.core import Point3D

from cherab.core import Plasma, Beam
from cherab.core.math import Interpolate1DLinear

from crm_solver.beamlet import Beamlet


class RenateCRMWrapper:

    def __init__(self, plasma, beam):

        # TODO - input variable validation

        # build species specifications, starting with electrons
        charges = [-1]
        charges.extend([s.charge for s in plasma.composition if not s.charge == 0])
        nuclear_charges = [0]
        nuclear_charges.extend([s.element.atomic_number for s in plasma.composition if not s.charge == 0])
        atomic_weights = [0]
        atomic_weights.extend([int(s.element.atomic_weight) for s in plasma.composition if not s.charge == 0])
        index = ['electron']
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

        # move this outside
        # from crm_solver.atomic_db import AtomicDB
        # renata_ad = AtomicDB(param=param)

        b = Beamlet(param=param, profiles=profiles, components=components)
        b.compute_linear_density_attenuation()
        b.compute_relative_populations()

        self.renate_beamlet = b

    def beam_emission_intensity(self, transition):

        b = self.renate_beamlet

        b.compute_linear_emission_density(to_level=str(transition[1]), from_level=str(transition[0]))

        from_level, to_level, ground_level, transition = b.atomic_db.set_default_atomic_levels()

        beamlet_grid = np.squeeze(np.array(b.profiles['beamlet grid']))
        beam_emission = np.squeeze(np.array(b.profiles[transition]))

        return Interpolate1DLinear(beamlet_grid, beam_emission, extrapolate=True, extrapolation_range=1e-4)

    def beam_density(self):

        b = self.renate_beamlet

        beamlet_grid = np.squeeze(np.array(b.profiles['beamlet grid']))
        beam_density = np.squeeze(np.array(b.profiles['linear_density_attenuation']))

        return Interpolate1DLinear(beamlet_grid, beam_density, extrapolate=True, extrapolation_range=1e-4)


def _sample_along_beam_axis(function, beam_axis, beam_to_world, debug=False):

    if debug:
        samples = []
        for i, z in enumerate(beam_axis):
            p = Point3D(0, 0, z).transform(beam_to_world)
            samples.append(function(p.x, p.y, p.z))
    else:
        samples = []
        for z in beam_axis:
            p = Point3D(0, 0, z).transform(beam_to_world)
            samples.append(function(p.x, p.y, p.z))

    return samples

