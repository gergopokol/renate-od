from imas_utility.idsinstance import ImasObject
import numpy as np
import os


class CoreprofIds:
    def __init__(self, shot, run, user=None, machine=None):
        self.ids = ImasObject(shot, run, user=user, machine=machine)
        self.load_coreprof_ids()

    def load_coreprof_ids(self):
        try:
            self.coreprof = self.ids.imas_pointer.get('coreprof')
        except:
            print, 'The coreprof IDS is basent from data file. Muhaha.'
            print, 'Please get more information about shot '+ self.shot + ' at run '+ self.run
            exit()
