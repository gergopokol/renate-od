
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import electron_mass, atomic_mass

from raysect.core import Point3D, Vector3D, translate, rotate_basis
from raysect.primitive import Box
from raysect.optical import World

from cherab.core.math import Interpolate1DCubic, sample1d, sample3d, ConstantVector3D
from cherab.core.math.mappers import Xto3D
from cherab.core import Species, Maxwellian, Plasma, Beam
from cherab.core.atomic import hydrogen
from cherab.core.model import SingleRayAttenuator
from cherab.tools.plasmas.slab import build_slab_plasma

from cherab.openadas import OpenADAS


# create atomic data source
adas = OpenADAS(permit_extrapolation=True)


world = World()


# PLASMA ----------------------------------------------------------------------
plasma = build_slab_plasma(peak_density=5e19, world=world)


integration_step = 0.0025
beam_transform = translate(-0.000001, 0.0, 0) * rotate_basis(Vector3D(1, 0, 0), Vector3D(0, 0, 1))

beam_full = Beam(parent=world, transform=beam_transform)
beam_full.plasma = plasma
beam_full.atomic_data = adas
beam_full.energy = 100000
beam_full.power = 3e6
beam_full.element = hydrogen
beam_full.sigma = 0.05
beam_full.divergence_x = 0.5
beam_full.divergence_y = 0.5
beam_full.length = 3.0
beam_full.attenuator = SingleRayAttenuator(clamp_to_zero=True)
beam_full.integrator.step = integration_step
beam_full.integrator.min_samples = 10


############################
# Try converting to Renate #

import pandas as pd
from lxml import etree
from crm_solver.beamlet import Beamlet


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


beam_axis = np.linspace(0, 5, num=500)
e_densities = [plasma.electron_distribution.density(x, 0, 0) for x in beam_axis]
e_temps = [plasma.electron_distribution.effective_temperature(x, 0, 0) for x in beam_axis]
h1 = plasma.composition[hydrogen, 1]
h1_densities = [h1.distribution.density(x, 0, 0) for x in beam_axis]
h1_temps = [h1.distribution.effective_temperature(x, 0, 0) for x in beam_axis]


profiles_data = np.zeros((5, 500))
profiles_data[0, :] = beam_axis
profiles_data[1, :] = e_densities
profiles_data[2, :] = e_temps
profiles_data[3, :] = h1_densities
profiles_data[4, :] = h1_temps
profiles_data = np.swapaxes(profiles_data, 0, 1)
row_index = [i for i in range(500)]
column_index = pd.MultiIndex.from_arrays([['beamlet grid', 'electron', 'electron', 'ion1', 'ion1'],
                                          ['distance', 'density', 'temperature', 'density', 'temperature'],
                                          ['m', 'm-3', 'eV', 'm-3', 'eV']],
                                         names=['type', 'property', 'unit'])
profiles = pd.DataFrame(data=profiles_data, columns=column_index, index=row_index)

# construct beam param specification

xml = etree.Element('xml')
head = etree.SubElement(xml, 'head')
id_tag = etree.SubElement(head, 'id')
id_tag.text = 'beamlet_test'
body_tag = etree.SubElement(xml, 'body')
beamlet_energy = etree.SubElement(body_tag, 'beamlet_energy', {'unit': 'keV'})
beamlet_energy.text = '100'
beamlet_species = etree.SubElement(body_tag, 'beamlet_species')
beamlet_species.text = 'H'  # Li
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

b.compute_linear_emission_density()
b.compute_linear_density_attenuation()
b.compute_relative_populations()

plt.plot(b.profiles['beamlet grid'], b.profiles['linear_emission_density'])
