import pandas
import numpy


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

    @staticmethod
    def __validate_detector_positions(detector_positions):
        positions = numpy.asarray(detector_positions, dtype=float)
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
            sizes = numpy.asarray(detector_size, dtype=float)
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
        beamlet.compute_linear_emission_density(from_level=from_level, to_level=to_level)
        try:
            emission = beamlet.profiles[label].to_numpy()
        except KeyError:
            raise RuntimeError('Linear emission density for transition ' + label + ' could not be computed. '
                               'Run the beam evolution solver first.') from None
        grid = beamlet.profiles[('beamlet grid', 'distance', 'm')].to_numpy()
        return grid, emission

    def calculate_photon_emission_profile(self, interpolate=False):
        if interpolate:
            new_resolution = int((max(self.beamlet.profiles['beamlet grid']['distance']['m']) -
                                  min(self.beamlet.profiles['beamlet grid']['distance']['m'])) /
                                 self.detector_size * 10)
            self.beamlet_grid_int = numpy.linspace(min(self.beamlet.profiles['beamlet grid']['distance']['m']),
                                                   max(self.beamlet.profiles['beamlet grid']['distance']['m']),
                                                   new_resolution)
            self.observed_profile_int = numpy.interp(self.beamlet_grid_int, self.beamlet.profiles['beamlet grid'][
                'distance']['m'], self.beamlet.profiles[self.observed_level])
        else:
            self.beamlet_grid_int = self.beamlet.profiles['beamlet grid']['distance']['m']
            self.observed_profile_int = self.beamlet.profiles[self.observed_level]
        step_size = (max(self.beamlet_grid_int)-min(self.beamlet_grid_int))/len(self.beamlet_grid_int)
        for detector_index in range(len(self.obs_profile)):
            if self.obs_profile[0][detector_index] + self.detector_size / 2. < max(self.beamlet_grid_int):
                detector_steps = [i for i in range(self.beamlet_grid_int.size) if abs(
                    self.beamlet_grid_int[i] - self.obs_profile[0][detector_index])
                                       < (self.detector_size/2)]
                self.photon_emission_profile[detector_index] = sum([self.observed_profile_int[step]*step_size
                                                                    for step in detector_steps])
        self.emission_profile = pandas.concat([self.obs_profile, pandas.DataFrame(self.photon_emission_profile)],
                                              axis=1, keys=['Location', 'Emission'])
        return
