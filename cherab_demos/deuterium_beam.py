
import numpy as np
import matplotlib.pyplot as plt

from raysect.core import Point3D, Vector3D, translate, rotate_basis
from raysect.optical import World
from raysect.optical.observer import PinholeCamera,  SightLine, PowerPipeline0D, SpectralPowerPipeline0D


from cherab.core.math import ConstantVector3D
from cherab.core.atomic import hydrogen, Line
from cherab.tools.plasmas.slab import build_slab_plasma

from renate.cherab_models import RenateBeamEmissionLine, RenateBeam


world = World()


# PLASMA ----------------------------------------------------------------------
plasma = build_slab_plasma(peak_density=5e19, world=world)
plasma.b_field = ConstantVector3D(Vector3D(0, 1.5, 0))


# BEAM SETUP ------------------------------------------------------------------
integration_step = 0.0025
beam_transform = translate(-0.5, 0.0, 0) * rotate_basis(Vector3D(1, 0, 0), Vector3D(0, 0, 1))
line = Line(hydrogen, 0, (3, 2))

beam = RenateBeam(parent=world, transform=beam_transform)
beam.plasma = plasma
beam.energy = 100000
beam.power = 3e6
beam.element = hydrogen
beam.temperature = 30
beam.sigma = 0.05
beam.divergence_x = 0.5
beam.divergence_y = 0.5
beam.length = 3.0
beam.models = [RenateBeamEmissionLine(line)]
beam.integrator.step = integration_step
beam.integrator.min_samples = 10


# line of sight settings
los_start = Point3D(1.5, -1, 0)
los_target = Point3D(0.5, 0, 0)
los_direction = los_start.vector_to(los_target).normalise()


beam_density = np.empty((200, 200))
xpts = np.linspace(-1, 2, 200)
ypts = np.linspace(-1, 1, 200)
for i, xpt in enumerate(xpts):
    for j, ypt in enumerate(ypts):
        pt = Point3D(xpt, ypt, 0).transform(beam.to_local())
        beam_density[i, j] = beam.density(pt.x, pt.y, pt.z)

plt.figure()
plt.imshow(np.transpose(np.squeeze(beam_density)), extent=[-1, 2, -1, 1], origin='lower')
plt.plot([los_start.x, los_target.x], [los_start.y, los_target.y], 'k')
plt.colorbar()
plt.axis('equal')
plt.xlabel('x axis (beam coords)')
plt.ylabel('z axis (beam coords)')
plt.title("Beam full energy density profile in r-z plane")


z = np.linspace(0, 3, 200)
beam_full_densities = [beam.density(0, 0, zz) for zz in z]
plt.figure()
plt.plot(z, beam_full_densities, label="full energy")
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
los.pixel_samples = 1
los.spectral_bins = 2000
los.observe()
