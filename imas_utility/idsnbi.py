from imas_utility.idsinstance import ImasObject
import numpy as np


class NbiIds(ImasObject):
    def __init__(self, shot, run, user=None, machine=None):
        super().__init__(shot, run, user, machine)
        self.load_nbi_ids()

    def load_nbi_ids(self):
        try:
            self.nbi = self.imas_pointer.get('nbi')
        except:
            print('No nbi IDS found in shot ' + str(self.shot) + ' at run ' + str(self.run))
            exit()
