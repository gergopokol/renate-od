
import numpy as np
import matplotlib.pyplot as plt

from raysect.core import Point3D, Vector3D, translate, rotate_basis
from raysect.primitive import Box
from raysect.optical import World
from raysect.optical.observer import PinholeCamera,  SightLine, PowerPipeline0D, SpectralPowerPipeline0D


from cherab.core import Species, Maxwellian, Plasma, Beam
from cherab.core.atomic import lithium, Line
from cherab.core.model import SingleRayAttenuator
from cherab.tools.plasmas.slab import build_slab_plasma

from cherab.openadas import OpenADAS

from renate.cherab_models import BeamEmissionLine


# create atomic data source
adas = OpenADAS(permit_extrapolation=True)


world = World()


# PLASMA ----------------------------------------------------------------------
plasma = build_slab_plasma(peak_density=5e19, world=world)


# BEAM SETUP ------------------------------------------------------------------
integration_step = 0.0025
beam_transform = translate(-0.000001, 0.0, 0) * rotate_basis(Vector3D(1, 0, 0), Vector3D(0, 0, 1))
line = Line(lithium, 0, ('2p', '2s'))

beam = Beam(parent=world, transform=beam_transform)
beam.plasma = plasma
beam.atomic_data = adas
beam.energy = 60000
beam.power = 1e5
beam.element = lithium
beam.temperature = 30
beam.sigma = 0.03
beam.divergence_x = 0.5
beam.divergence_y = 0.5
beam.length = 3.0
beam.attenuator = SingleRayAttenuator(clamp_to_zero=True)
beam.models = [BeamEmissionLine(line, renate_atomic_data=temp)]
beam.integrator.step = integration_step
beam.integrator.min_samples = 10


# OBSERVATIONS ----------------------------------------------------------------
camera = PinholeCamera((128, 128), parent=world, transform=translate(1.25, -3.5, 0) * rotate_basis(Vector3D(0, 1, 0), Vector3D(0, 0, 1)))
camera.spectral_rays = 1
camera.spectral_bins = 15
camera.pixel_samples = 5

# turning off parallisation because this causes issues with the way RENATE currently loads atomic data
from raysect.core.workflow import SerialEngine
camera.render_engine = SerialEngine()

plt.ion()
camera.observe()

power = PowerPipeline0D(accumulate=False)
spectral_power = SpectralPowerPipeline0D()
los = SightLine(pipelines=[power, spectral_power], min_wavelength=668, max_wavelength=672,
                parent=world, transform=translate(0.25, -0.25, 0) * rotate_basis(Vector3D(0, 1, 0), Vector3D(0, 0, 1)))
los.pixel_samples = 1
los.spectral_bins = 200
los.observe()



