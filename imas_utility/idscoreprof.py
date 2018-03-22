from imas_utility.idsinstance import ImasObject
import numpy as np
import os


class CoreprofIds(ImasObject):
    def __init__(self, shot, run, user=None, machine=None):
        super().__init__(shot, run, user, machine)
        self.load_core_profiles_ids()

    def load_core_profiles_ids(self):
        try:
            self.core_profiles = self.imas_pointer.get('core_profiles')
        except:
            print('The core_profiles IDS is absent from data file. Muhaha.')
            print('Please get more information about shot ' + str(self.shot) + ' at run ' + str(self.run))
            exit()
