from utility.constants import Constants
from utility.exceptions import InputError

CONST = Constants()


class Particle(object):
    def __init__(self, label=str, charge=0, mass=None):
        if isinstance(label, str):
            self.label = label
        else:
            raise InputError('Label is expected to be of string type.')
        if isinstance(charge, int) and (charge >= -1):
            self.charge = charge
        else:
            raise InputError('The charge '+str(charge) + ' of the particle must exceed -1 and be an integer.')
        if mass is None and self.label == 'e':
            self.mass = CONST.electron_mass
        else:
            self.mass = mass

    def update_mass(self, mass):
        self.mass = mass

    def update_charge(self, charge):
        self.charge = charge

    def __str__(self):
        return str(self.label)

    def __repr__(self):
        return 'Particle: ' + str(self.label) + '\t charge = ' + str(self.charge)


class Ion(Particle):
    def __init__(self, label=str, charge=int, mass=None, atomic_number=int, mass_number=int):
        Particle.__init__(self, label=label, mass=mass, charge=charge)
        if (atomic_number >= 0) and isinstance(atomic_number, int) and (atomic_number >= self.charge):
            self.atomic_number = atomic_number
        else:
            raise InputError('The atomic number is expected to be a positive integer '
                             'and larger or equal to the charge.')
        if (mass_number >= 0) and isinstance(mass_number, int) and (self.atomic_number <= mass_number):
            self.mass_number = mass_number
        else:
            raise InputError('The mass number must be a positive integer and larger than the atomic number.')
        if mass is None:
            self.mass = CONST.proton_mass * self.atomic_number + \
                        CONST.neutron_mass * (self.mass_number - self.atomic_number)
        else:
            self.mass = mass

    def __repr__(self):
        return 'Ion: ' + str(self.label) + '\t Z = ' + str(self.atomic_number) + \
                ' N = ' + str(self.mass_number) + ' charge = ' + str(self.charge)


class Atom(Ion):
    def __init__(self, label=str, mass_number=int, atomic_number=int, mass=None):
        Ion.__init__(self, label=label, mass_number=mass_number, mass=mass, atomic_number=atomic_number, charge=0)

    def __repr__(self):
        return 'Atom: ' + str(self.label) + '\t Z = ' + str(self.atomic_number) + ' N = ' + str(self.mass_number)


class Transition(object):
    def __init__(self, projectile=Particle, target=Particle, from_level=str, to_level=str, trans=str):
        self.projectile = projectile
        self.target = target
        if isinstance(from_level, str):
            self.from_level = from_level
