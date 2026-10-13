
import numpy as np
import matplotlib.pyplot as plt

from raysect.core import Point3D, Vector3D, translate, rotate_basis
from raysect.optical import World
from raysect.optical.observer import PinholeCamera,  SightLine, PowerPipeline0D, SpectralPowerPipeline0D

from cherab.openadas import OpenADAS
from cherab.core.math import ConstantVector3D
from cherab.core.atomic import hydrogen, Line
from cherab.tools.plasmas.slab import build_slab_plasma
from cherab.core.beam import Beam
from cherab.core.model import BeamEmissionLine, SingleRayAttenuator
from cherab.core.model.beam.beam_emission import SIGMA_TO_PI, SIGMA1_TO_SIGMA0, PI2_TO_PI3, PI4_TO_PI3

from renate.cherab_models import RenateBeamEmissionLine
from renate.cherab_models import RenateBeam


world = World()


# PLASMA ----------------------------------------------------------------------
plasma = build_slab_plasma(peak_density=5e19, parent=world)
plasma.b_field = ConstantVector3D(Vector3D(0, 0.6, 0))


# BEAM SETUP ------------------------------------------------------------------
integration_step = 0.0025
beam_transform = translate(-0.5, 0.0, 0) * rotate_basis(Vector3D(1, 0, 0), Vector3D(0, 0, 1))
line = Line(hydrogen, 0, (3, 2))

beam = RenateBeam(parent=world, transform=beam_transform)
beamline=RenateBeamEmissionLine(line)
beamline.beam=beam
beam.plasma = plasma
beam.energy = 100000
beam.power = 3e7
beam.element = hydrogen
beam.temperature = 30
beam.sigma = 0.15
beam.divergence_x = 0.
beam.divergence_y = 0.
beam.length = 3.0
beam.models = [beamline]
beam.integrator.step = integration_step
beam.integrator.min_samples = 10

bes2_model = BeamEmissionLine(Line(hydrogen, 0, (3, 2)),
                                  sigma_to_pi=SIGMA_TO_PI, sigma1_to_sigma0=SIGMA1_TO_SIGMA0,
                                  pi2_to_pi3=PI2_TO_PI3, pi4_to_pi3=PI4_TO_PI3)

beam2 = Beam(parent=world, transform=beam_transform)
beam2.atomic_data = OpenADAS(permit_extrapolation=True, missing_rates_return_null=True)
beam2.plasma = plasma
beam2.energy = 100000
beam2.power = 3e6
beam2.element = hydrogen
beam2.temperature = 30
beam2.sigma = 0.1
beam2.divergence_x = 0.
beam2.divergence_y = 0.
beam2.length = 3.0
beam2.attenuator = SingleRayAttenuator(clamp_to_zero=True)
#beam2.models = [bes2_model]
beam2.integrator.step = integration_step
beam2.integrator.min_samples = 10


# line of sight settings
los_start = Point3D(1.5, -1, 0)
los_target = Point3D(0, 0, 0)
los_direction = los_start.vector_to(los_target).normalise()


beam_density = np.empty((200, 200))
beam_density2 = np.empty((200, 200))
xpts = np.linspace(-1, 2, 200)
ypts = np.linspace(-1, 1, 200)
for i, xpt in enumerate(xpts):
    for j, ypt in enumerate(ypts):
        pt = Point3D(xpt, ypt, 0).transform(beam.to_local())
        beam_density[i, j] = beam.density(pt.x, pt.y, pt.z)
        beam_density2[i, j] = beam2.density(pt.x, pt.y, pt.z)

plt.figure()
plt.imshow(np.transpose(np.squeeze(beam_density2)), extent=[-1, 2, -1, 1], origin='lower')
plt.plot([los_start.x, los_target.x], [los_start.y, los_target.y], 'k')
plt.colorbar()
plt.axis('equal')
plt.xlabel('x axis (beam coords)')
plt.ylabel('z axis (beam coords)')
plt.title("Beam full energy density profile in r-z plane")


z = np.linspace(-1, 2, 200)
beam_densities = [beam.density(0, 0, zz) for zz in z]
beam2_densities = [beam2.density(0, 0, zz) for zz in z]
plt.figure()
plt.plot(z, beam_densities, label="ROD")
plt.plot(z, beam2_densities, label="CHERAB")
plt.xlabel('z axis (beam coords)')
plt.ylabel('beam component density [m^-3]')
plt.title("Beam attenuation by energy component")
plt.legend()


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
los = SightLine(pipelines=[power, spectral_power], min_wavelength=640, max_wavelength=665,
                parent=world, transform=translate(*los_start) * rotate_basis(los_direction, Vector3D(0, 0, 1)))
los.pixel_samples = 10
los.spectral_bins = 2000
los.observe()

plt.ioff()
plt.show()
