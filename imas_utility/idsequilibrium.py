from imas_utility.idsinstance import ImasObject
import numpy as np


class EquilibriumIds(ImasObject):
    def __init__(self, shot, run, user=None, machine=None):
        super().__init__(shot, run, user, machine)
        self.load_equilibrium_ids()

    def load_equilibrium_ids(self):
        try:
            self.equilibrium = self.imas_pointer.get('equilibrium')
        except:
            print('The equilibrium IDS is absent from data file. Muhaha.')
            print('Please get more information about shot ' + str(self.shot) + ' at run ' + str(self.run))
            exit()
