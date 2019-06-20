
from libc.math cimport tan, exp, M_PI

from raysect.core import Vector3D
from raysect.primitive import Cylinder
from raysect.optical cimport World, AffineMatrix3D, Primitive, Ray, new_vector3d
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator

from cherab.core.beam.model cimport BeamModel, BeamAttenuator
from cherab.core.beam.material cimport BeamMaterial
from cherab.core.atomic cimport AtomicData, Element
from cherab.core.utility import Notifier

from renate.cherab_models.wavelength_data import RenateWavelengthData
from renate.cherab_models.beam_emission import RenateBeamEmissionLine
from renate.cherab_models.renate_wrapper import RenateCRMWrapper


cdef double DEGREES_TO_RADIANS = (M_PI / 180)


cdef class RenetaModelManager(ModelManager):

    cpdef object set(self, object models):

        # copy models and test it is an iterable
        models = list(models)

        # check contents of list are RENATE beam emission models
        for model in models:
            if not isinstance(model, RenateBeamEmissionLine):
                raise TypeError('The Renate Beam model only works with Renate specific BeamEmissionLine objects.')

        self._models = models
        self.notifier.notify()

    cpdef object add(self, BeamModel model):

        if not isinstance(model, RenateBeamEmissionLine):
            raise TypeError('The Renate Beam model only works with Renate specific BeamEmissionLine objects.')

        self._models.append(model)
        self.notifier.notify()


cdef class RenateBeam(Beam):
    """
    A beam using the Renate-OD CRM models.

    :param Node parent: The parent node in the Raysect scene-graph.
      See the Raysect documentation for more guidance.
    :param AffineMatrix3D transform: The transform defining the spatial position
      and orientation of this beam. See the Raysect documentation if you need
      guidance on how to use AffineMatrix3D transforms.
    :param str name: The name for this beam object.

    :ivar float divergence_x: The beam profile divergence in the x dimension in beam
      coordinates (degrees).
    :ivar float divergence_y: The beam profile divergence in the y dimension in beam
      coordinates (degrees).
    :ivar Element element: The element of which this beam is composed.
    :ivar float energy: The beam energy in eV/amu.
    :ivar VolumeIntegrator integrator: The configurable method for doing
      volumetric integration through the beam along a Ray's path. Defaults to
      a numerical integrator with 1mm step size, NumericalIntegrator(step=0.001).
    :ivar float length: The approximate length of this beam from source to extinction
      in the plasma. This is used for setting the bounding geometry over which calculations
      will occur. Units of m.
    :ivar ModelManager models: The manager class that sets and provides access to the
      emission models for this beam.
    :ivar Plasma plasma: The plasma instance with which this beam interacts.
    :ivar float power: The total beam power in W.
    :ivar float sigma: The guassian beam width at the origin in m.
    :ivar float temperature: The broadening of the beam (eV).

    .. code-block:: pycon

       >>> # This example shows how to initialise and populate a basic beam
       >>>
       >>> from raysect.core.math import Vector3D, translate, rotate
       >>> from raysect.optical import World
       >>>
       >>> from cherab.core.atomic import carbon, deuterium, Line
       >>> from cherab.core.model import BeamCXLine
       >>> from cherab.openadas import OpenADAS
       >>>
       >>>
       >>> world = World()
       >>>
       >>> beam = Beam(parent=world, transform=translate(1.0, 0.0, 0) * rotate(90, 0, 0))
       >>> beam.plasma = plasma  # put your plasma object here
       >>> beam.atomic_data = OpenADAS()
       >>> beam.energy = 60000
       >>> beam.power = 1e4
       >>> beam.element = deuterium
       >>> beam.sigma = 0.025
       >>> beam.divergence_x = 0.5
       >>> beam.divergence_y = 0.5
       >>> beam.length = 3.0
       >>> beam.models = [BeamCXLine(Line(carbon, 5, (8, 7)))]
       >>> beam.integrator.step = 0.001
       >>> beam.integrator.min_samples = 5
    """

    def __init__(self, object parent=None, AffineMatrix3D transform=None, str name=None,
                 bint clamp_to_zero=False, double clamp_sigma=5.0):

        super().__init__(parent, transform, name)

        # change reporting and tracking
        self.notifier = Notifier()

        # beam properties
        self.BEAM_AXIS = Vector3D(0.0, 0.0, 1.0)
        self._energy = 0.0                         # eV/amu
        self._power = 0.0                          # total beam power, W
        self._temperature = 0.0                    # Broadening of the beam (eV)
        self._element = element = None             # beam species, an Element object
        self._divergence_x = 0.0                   # beam divergence x (degrees)
        self._divergence_y = 0.0                   # beam divergence y (degrees)
        self._length = 1.0                         # m
        self._sigma = 0.1                          # m (gaussian beam width at origin)

        # external data dependencies
        self._plasma = None
        self._atomic_data = None

        # setup emission model handler and trigger geometry rebuilding if the models change
        self._models = RenetaModelManager()
        self._models.notifier.add(self._configure_geometry)

        # beam attenuation model
        self._attenuator = None

        # beam geometry
        self._geometry = None
        self._tanxdiv = 0.0
        self._tanydiv = 0.0

        # emission model integrator
        self._integrator = NumericalIntegrator(step=0.001)

        # setup property change notifications for plasma
        if self._plasma:
            self._plasma.notifier.add(self._plasma_changed)

        self._renate_wrapper = None
        self._density = None

        # beam density clamping optimisation settings
        if clamp_sigma <= 0.0:
            raise ValueError("The value of clamp_sigma must be greater than zero.")
        self.clamp_to_zero = clamp_to_zero
        self._clamp_sigma_sqr = clamp_sigma**2

    cpdef double density(self, double x, double y, double z) except? -1e999:
        """
        Returns the bean density at the specified position in beam coordinates.
        
        Note: this function is only defined over the domain 0 < z < beam_length.
        Outside of this range the density is clamped to zero.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters.
        :return: Beam density in m^-3
        """

        cdef double sigma_x, sigma_y, norm_radius_sqr, gaussian_sample

        if z < 0 or z > self._length:
            return 0

        if not self._renate_wrapper:
            print('loading wrapper')
            self._configure_renate()
            print('wrapper loaded')

        print('# calculate beam width')
        # calculate beam width
        sigma_x = self.sigma + z * self._tanxdiv
        sigma_y = self.sigma + z * self._tanydiv

        print('# normalised radius squared')
        # normalised radius squared
        norm_radius_sqr = ((x / sigma_x)**2 + (y / sigma_y)**2)

        print('# clamp low densities')
        # clamp low densities to zero (beam models can skip their calculation if density is zero)
        # comparison is done using the squared radius to avoid a costly square root
        if self.clamp_to_zero:
            if norm_radius_sqr > self._clamp_sigma_sqr:
                return 0.0

        print('# bi-variate Gaussian')
        # bi-variate Gaussian distribution (normalised)
        gaussian_sample = exp(-0.5 * norm_radius_sqr) / (2 * M_PI * sigma_x * sigma_y)

        print('returning')
        return self._density.evaluate(z) * gaussian_sample

    @property
    def renate_wrapper(self):
        if not self._renate_wrapper:
            self._configure_renate()
        return self._renate_wrapper

    @property
    def atomic_data(self):
        return self._atomic_data

    @atomic_data.setter
    def atomic_data(self, AtomicData value not None):
        raise NotImplementedError("The RENATE-OD beam models do not use a AtomicDataProvider class"
                                  "because they use a custom CRM model to calculate a number of different"
                                  "rate types that don't exist in CHERAB.")

    @property
    def attenuator(self):
        return self._attenuator
    
    @attenuator.setter
    def attenuator(self, BeamAttenuator value not None):
        raise NotImplementedError("The RENATE-OD beam models do not use a BeamAttenuator class"
                                  "because they use a custom CRM model to calculate a number of different"
                                  "rate types that don't exist in CHERAB.")

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, object values):

        # check necessary data is available
        if not self._plasma:
            raise ValueError('The beam must have a reference to a plasma object to be used with an emission model.')

        # setting the emission models causes ModelManager to notify the Beam object to configure geometry
        # so no need to explicitly rebuild here
        self._models.set(values)

    def _configure_geometry(self):

        # detach existing geometry
        # take a copy of self.children as it will be modified when unparenting
        children = self.children.copy()
        for child in children:
            child.parent = None

        # no further work if there are no emission models
        if not list(self._models):
            return

        # check necessary data is available
        if not self._plasma:
            raise ValueError('The beam must have a reference to a plasma object to be used with an emission model.')

        # build geometry to fit beam
        self._geometry = self._generate_geometry()

        # attach geometry to the beam
        self._geometry.parent = self
        self._geometry.name = 'Beam Geometry'

        # build plasma material
        atomic_data = RenateWavelengthData()
        self._geometry.material = BeamMaterial(self, self._plasma, atomic_data, list(self._models), self.integrator)

        self._tanxdiv = tan(DEGREES_TO_RADIANS * self.divergence_x)
        self._tanydiv = tan(DEGREES_TO_RADIANS * self.divergence_y)

    def _generate_geometry(self):

        # the beam bounding envelope is a cylinder aligned with the beam axis, sharing the same coordinate space
        # the cylinder radius is set to 5 sigma around the widest section of the gaussian beam
        radius = 5.0 * (self.sigma + self.length * tan(DEGREES_TO_RADIANS * max(self._divergence_x, self._divergence_y)))
        return Cylinder(radius=radius, height=self.length)

    def _configure_attenuator(self):
        # RENATE-OD does not use a BeamAttenuator class
        pass

    def _plasma_changed(self):
        self._renate_wrapper = None

    def _configure_renate(self):
        self._renate_wrapper = RenateCRMWrapper(self._plasma, self)
        self._density = self._renate_wrapper.beam_density()

    cdef int _modified(self) except -1:
        """
        Called when a scene-graph change occurs that modifies this Node's root
        transforms. This will occur if the Node's transform is modified, a
        parent node transform is modified or if the Node's section of scene-
        graph is re-parented.
        """

        # beams section of the scene-graph has been modified, alert dependents
        self.notifier.notify()
