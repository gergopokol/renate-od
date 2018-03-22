from imas_utility.idsinstance import ImasObject
import numpy as np


class EquilibriumIds:
    def __init__(self, shot, run, user=None, machine=None):
        self.ids = ImasObject(shot, run, user=user, machine=machine)
        self.load_equilibrium_ids()

    def load_equilibrium_ids(self):
        try:
            self.equilibrium = self.ids.imas_pointer.get('equilibrium')
        except:
            print, 'The equilibrium IDS is absent from data file. Muhaha.'
            print, 'Please get more information about shot ' + self.shot + ' at run ' + self.run
            exit()
