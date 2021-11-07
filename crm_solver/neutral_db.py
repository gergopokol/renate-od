import numpy as np
from utility.getdata import GetData


class NeutralDB(object):
    def __init__(self, path=None, param=None, components=None, resolved='bundled-n'):
        self.param = param
        self.resolved = resolved
        if (components['q'] == 0).any():
            self.__load_neutral_cross_section()

    def __load_neutral_cross_section(self):
        pass
