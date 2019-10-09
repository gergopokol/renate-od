import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import electron_mass, atomic_mass
from raysect.core import Point3D, Vector3D, translate, rotate_basis
from raysect.optical import World
from crm_solver.beamlet import Beamlet
from cherab.core.math import Interpolate1DCubic
from cherab.core.math import ConstantVector3D
from cherab.core import Maxwellian, Plasma
from cherab.core.atomic.elements import deuterium
from raysect.primitive import Box
from raysect.optical.observer import PinholeCamera,  SightLine, PowerPipeline0D, SpectralPowerPipeline0D


from cherab.core.atomic import sodium, Line

from renate.cherab_models import RenateBeamEmissionLine, RenateBeam

renate = Beamlet(data_path='beamlet/acceptancetest/scenario-standard_plasma-H_energy-100_beam-Na_profile.xml',
                 solver='disregard')
grid = renate.profiles['beamlet grid']['distance']['m']
density = renate.profiles['electron']['density']['m-3']
temperature = renate.profiles['electron']['temperature']['ev']

f1d_electron = Interpolate1DCubic(grid, density)
f1t_electron = Interpolate1DCubic(grid, temperature)

f1d_ion = Interpolate1DCubic(grid, renate.profiles['ion1']['density']['m-3'])
f1t_ion = Interpolate1DCubic(grid, renate.profiles['ion1']['temperature']['eV'])

zero_velocity = ConstantVector3D(Vector3D(0, 0, 0))
#define electron
e_distribution = Maxwellian(f1d_electron, f1t_electron, zero_velocity, electron_mass)

#define ion
ion_distribution = Maxwellian(f1d_ion, f1t_ion, zero_velocity, deuterium.atomic_weight*atomic_mass)

world = World()
plasma = Plasma(parrent=world)
plasma.electron_distribution = e_distribution
plasma.composition = [ion_distribution]

#plasma geometry

height = width = length = 0.55
plasma.geometry = Box(Point3D(0, -width/2, -height/2), Point3D(length, width/2, height/2))

# BEAM SETUP ------------------------------------------------------------------
integration_step = 0.0025
beam_transform = translate(-0.5, 0.0, 0) * rotate_basis(Vector3D(1, 0, 0), Vector3D(0, 0, 1))
line = Line(sodium, 0, ('3p', '3s'))

beam = RenateBeam(parent=world, transform=beam_transform)
beam.plasma = plasma
beam.energy = 100000
beam.power = 1e5
beam.element = sodium
beam.temperature = 30
beam.sigma = 0.03
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
camera = PinholeCamera((128, 128), parent=world, transform=translate(0.25, -1.25, 0) * rotate_basis(Vector3D(0, 1, 0), Vector3D(0, 0, 1)))
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
los = SightLine(pipelines=[power, spectral_power], min_wavelength=582, max_wavelength=586,
                parent=world, transform=translate(*los_start) * rotate_basis(los_direction, Vector3D(0, 0, 1)))
los.pixel_samples = 1
los.spectral_bins = 2000
los.observe()

plt.ioff()
plt.show()
