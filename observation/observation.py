import numpy
import warnings


class Obs1d:
    def __init__(self, detector_positions, detector_size=None, from_level=None, to_level=None):
        self.detector_positions = self.__validate_detector_positions(detector_positions)
        self.detector_size, self.detector_ranges = self.__compute_detector_ranges(self.detector_positions,
                                                                                  detector_size)
        if (from_level is None) != (to_level is None):
            raise ValueError('Both <from_level> and <to_level> must be given to set a default transition, '
                             'or neither to use the default transition of the beam species.')
        if from_level is not None and not (isinstance(from_level, str) and isinstance(to_level, str)):
            raise TypeError('The expected data type for <from_level> and <to_level> is str.')
        self.from_level = from_level
        self.to_level = to_level
        for array in (self.detector_positions, self.detector_size, self.detector_ranges):
            array.flags.writeable = False

    @staticmethod
    def __validate_detector_positions(detector_positions):
        positions = numpy.array(detector_positions, dtype=float)
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError('<detector_positions> must be a non-empty 1D array.')
        if not numpy.all(numpy.isfinite(positions)):
            raise ValueError('<detector_positions> must contain finite values only.')
        if numpy.any(positions < 0):
            raise ValueError('<detector_positions> are distances along the beam measured from the beam starting '
                             'point and must be non-negative.')
        if numpy.any(numpy.diff(positions) <= 0):
            raise ValueError('<detector_positions> must be strictly increasing.')
        return positions

    @staticmethod
    def __compute_detector_ranges(positions, detector_size):
        if detector_size is None:
            if positions.size < 2:
                raise ValueError('At least two <detector_positions> are needed to derive detector ranges '
                                 'when <detector_size> is None.')
            midpoints = (positions[1:] + positions[:-1]) / 2.
            lower_edges = numpy.concatenate(([positions[0] - (midpoints[0] - positions[0])], midpoints))
            upper_edges = numpy.concatenate((midpoints, [positions[-1] + (positions[-1] - midpoints[-1])]))
            sizes = upper_edges - lower_edges
        else:
            sizes = numpy.array(detector_size, dtype=float)
            if sizes.ndim == 0:
                sizes = numpy.full(positions.shape, float(sizes))
            elif sizes.shape != positions.shape:
                raise ValueError('<detector_size> must be None, a single number or an array of the same size '
                                 'as <detector_positions>.')
            if not numpy.all(numpy.isfinite(sizes)) or numpy.any(sizes <= 0):
                raise ValueError('<detector_size> must contain finite, positive values only.')
            lower_edges = positions - sizes / 2.
            upper_edges = positions + sizes / 2.
        return sizes, numpy.column_stack((lower_edges, upper_edges))

    def _resolve_transition(self, beamlet, from_level=None, to_level=None):
        if (from_level is None) != (to_level is None):
            raise ValueError('Both <from_level> and <to_level> must be given to override the default transition, '
                             'or neither.')
        if from_level is None:
            if self.from_level is not None:
                from_level, to_level = self.from_level, self.to_level
            else:
                from_level, to_level, _, label = beamlet.atomic_db.set_default_atomic_levels()
                return from_level, to_level, label
        return from_level, to_level, from_level + '-->' + to_level

    @staticmethod
    def _get_linear_emission_density(beamlet, from_level, to_level, label):
        try:
            emission = beamlet.profiles[label].to_numpy()
        except KeyError:
            beamlet.compute_linear_emission_density(from_level=from_level, to_level=to_level)
            try:
                emission = beamlet.profiles[label].to_numpy()
            except KeyError:
                raise RuntimeError('Linear emission density for transition ' + label + ' could not be computed. '
                                   'Run the beam evolution solver first.') from None
        grid = beamlet.profiles[('beamlet grid', 'distance', 'm')].to_numpy()
        return grid, emission

    @staticmethod
    def _integrate_over_detectors(grid, emission, detector_ranges):
        steps = numpy.diff(grid)
        slopes = numpy.diff(emission) / steps
        cumulative = numpy.concatenate(([0.], numpy.cumsum((emission[1:] + emission[:-1]) / 2. * steps)))
        index = numpy.clip(numpy.searchsorted(grid, detector_ranges, side='right') - 1, 0, grid.size - 2)
        offset = detector_ranges - grid[index]
        integral = cumulative[index] + offset * (emission[index] + slopes[index] * offset / 2.)
        return integral[:, 1] - integral[:, 0]

    def _flag_detectors_outside_grid(self, grid, photon_flux):
        outside = (self.detector_ranges[:, 0] < grid[0]) | (self.detector_ranges[:, 1] > grid[-1])
        if outside.any():
            photon_flux[outside] = numpy.nan
            warnings.warn('Detectors ' + str(numpy.flatnonzero(outside).tolist()) + ' extend beyond the beamlet grid '
                          '(' + str(grid[0]) + ' m - ' + str(grid[-1]) + ' m); their emission is set to NaN.')
        return photon_flux

    def compute_photon_flux(self, beamlets, from_level=None, to_level=None, store=False):
        single = not isinstance(beamlets, (list, tuple))
        if single:
            beamlets = [beamlets]
        photon_flux = numpy.empty((len(beamlets), self.detector_positions.size))
        transitions = []
        for index, beamlet in enumerate(beamlets):
            beamlet_from, beamlet_to, label = self._resolve_transition(beamlet, from_level, to_level)
            grid, emission = self._get_linear_emission_density(beamlet, beamlet_from, beamlet_to, label)
            photon_flux[index] = self._flag_detectors_outside_grid(
                grid, self._integrate_over_detectors(grid, emission, self.detector_ranges))
            transitions.append(label)
        if single:
            photon_flux = photon_flux[0]
        if store:
            self.photon_flux = photon_flux
            self.observed_transitions = transitions
        return self.detector_positions, photon_flux
