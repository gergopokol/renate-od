import scipy.constants as c


class Constants:
    def __init__(self):
        try:
            import imas
        except ImportError:
            self.__setup_standalone_constants_db()

    def __setup_standalone_constants_db(self):
        self.charge_electron = c.elementary_charge
        self.speed_of_light = c.speed_of_light
        self.Boltzmann = c.Boltzmann

    def __setup_imas_constants_db(self):
        pass
