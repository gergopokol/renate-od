from imas_utility.idsinstance import ImasObject
import numpy as np
import os


class CoreprofIds(ImasObject):
    def __init__(self, shot, run, user=None, machine=None):
        super().__init__(self, shot, run, user, machine)
        self.load_coreprof_ids()

    def load_coreprof_ids(self):
        try:
            self.coreprof = self.imas_pointer.get('coreprof')
        except:
            print, 'The coreprof IDS is absent from data file. Muhaha.'
            print, 'Please get more information about shot ' + self.shot + ' at run ' + self.run
            exit()
