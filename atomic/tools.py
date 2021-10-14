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

    def __add__(self, other):
        self.atomic_number += other.atomic_number
        self.mass_number += other.mass_number
        self.label += other.label
        self.mass += other.mass

    def __mul__(self, other):
        self.atomic_number *= other
        self.mass_number *= other
        self.label = str(other) + self.label
        self.mass *= other


class Molecule(Atom):
    def __init__(self, atoms=list, mass=None):
        Atom.__init__(self, label='', mass_number=0, atomic_number=0, mass=mass)
        for atom in atoms:
            self.mass += atom.mass
            self.mass_number += atom.mass_number
            self.atomic_number += atom.atomic_number
            self.label += atom.label

    def __repr__(self):
        return 'Molecule: ' + str(self.label) + '\t Protons = ' + str(self.atomic_number) + \
               ' Neutrons = ' + str(self.mass_number - self.atomic_number)


class IonizedMolecule(Molecule):
    def __init__(self, atoms=list, mass=None, charge=int):
        Molecule.__init__(self, atoms=atoms, mass=mass)
        self.update_charge(charge=charge)

    def __repr__(self):
        return 'Ionized Molecule: ' + str(self.label) + '\t Protons = ' + str(self.atomic_number) + \
               ' Nucleons = ' + str(self.mass_number) + ' Charge = ' + str(self.charge)


class Transition(object):
    def __init__(self, projectile=Particle, target=Particle, from_level=str, to_level=None, trans=str):
        self.projectile = projectile
        self.target = target
        if isinstance(from_level, str) and isinstance(trans, str):
            self.from_level = from_level
            if trans in ['ex', 'de-ex', 'eloss', 'ion', 'cx']:
                self.name = trans
            else:
                InputError('The provided transition is not supported. '
                           'Supported transitions are: ex, de-ex, eloss, ion and cx.')
            if (to_level is None) or isinstance(to_level, str):
                self.to_level = to_level
            else:
                InputError('The provided end state for the electron transition is not valid. Str or None is expected.')
        else:
            InputError('Expected input data format for <from_level>, <to_level> and <trans> to be of str type.')

    def __str__(self):
        if self.name in ['cx', 'eloss', 'ion']:
            return self.from_level + '-i'
        else:
            return self.from_level + '-' + self.to_level

    def __repr__(self):
        return 'Collision of: '+str(self.projectile)+' + '+str(self.target)+' with transition: ' + \
               self.name+' | from level: '+self.from_level+' | to_level: '+str(self.to_level)
