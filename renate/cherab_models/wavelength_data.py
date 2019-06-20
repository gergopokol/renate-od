
from cherab.core import AtomicData
from cherab.core.atomic.elements import Isotope, hydrogen, lithium, sodium


RENATE_WAVELENGTHS = {
    hydrogen: {
        (3, 2): 656.19
    },
    lithium: {
        ("2p", "2s"): 670.78
    },
    sodium: {
        ("3p", "3s"): 588.995
    }
}


class RenateWavelengthData(AtomicData):

    def wavelength(self, ion, charge, transition):
        """
        :param ion: Element object defining the ion type.
        :param charge: Charge state of the ion.
        :param transition: Tuple containing (initial level, final level)
        :return: Wavelength in nanometers.
        """

        if isinstance(ion, Isotope):
            ion = ion.element

        if not charge == 0:
            raise ValueError("Renate only models neutral emission.")

        try:
            return RENATE_WAVELENGTHS[ion][transition]

        except KeyError:
            raise ValueError("The specified transition ({}, '{}') is not supported by Renate-OD.".format(ion, transition))
