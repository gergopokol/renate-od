from imas_utility.idsinstance import ImasObject
import numpy as np


class EdgeprofIds(ImasObject):
    def __init__(self, shot, run, user=None, machine=None):
        super().__init__(shot, run, user, machine)
        self.load_edge_profiles_ids()

    def load_edge_profiles_ids(self):
        try:
            self.edge_profiles = self.imas_pointer.get('edge_profiles')
        except:
            print('The edge_profiles IDS is absent for shot ' + str(self.shot) + ' at run ' + str(self.run))
            exit()
