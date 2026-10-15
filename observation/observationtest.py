import unittest
import numpy
import pandas
from observation.observation import Obs1D


class _StubAtomicDB:
    DEFAULT_TRANSITION = ('9x', '8x', '1x', '9x-->8x')

    def set_default_atomic_levels(self):
        return self.DEFAULT_TRANSITION


class _StubBeamlet:
    GRID = [0., 0.5, 1.]
    EMISSION = [1., 3., 2.]

    def __init__(self, emission_stored=False, solved=True):
        self.atomic_db = _StubAtomicDB()
        self.solved = solved
        self.compute_calls = []
        self.profiles = pandas.DataFrame({('beamlet grid', 'distance', 'm'): self.GRID})
        if emission_stored:
            self.profiles[('9x-->8x', '', '')] = self.EMISSION

    def compute_linear_emission_density(self, to_level=None, from_level=None):
        self.compute_calls.append({'from_level': from_level, 'to_level': to_level})
        if self.solved:
            self.profiles[(from_level + '-->' + to_level, '', '')] = self.EMISSION


class Obs1DInitTest(unittest.TestCase):
    INPUT_POSITIONS = [0.25, 0.5, 1.0]
    INPUT_SIZE_SCALAR = 0.125
    INPUT_SIZE_ARRAY = [0.125, 0.25, 0.5]
    INPUT_INVALID_POSITIONS = [[[0.25, 0.5]], [], [0.25, numpy.nan], [0.25, numpy.inf], [-0.25, 0.5], [0.5, 0.25],
                               [0.25, 0.25]]
    INPUT_INVALID_SIZES = [[0.125, 0.25], [[0.125, 0.25, 0.5]], 0., -0.125, numpy.nan, numpy.inf, [0.125, 0., 0.5]]
    INPUT_INVALID_TRANSITIONS = [(('3n', None), ValueError), ((None, '2n'), ValueError), ((3, 2), TypeError),
                                 (('3n', 2), TypeError)]
    EXPECTED_ATTRIBUTES = ['detector_positions', 'detector_size', 'detector_ranges']

    def test_positions(self):
        obs = Obs1D(self.INPUT_POSITIONS)
        numpy.testing.assert_array_equal(obs.detector_positions, self.INPUT_POSITIONS,
                                         err_msg='Detector positions are not stored as given.')
        self.assertEqual(obs.detector_positions.dtype, numpy.float64, msg='Detector positions are expected as floats.')

    def test_invalid_positions(self):
        for positions in self.INPUT_INVALID_POSITIONS:
            with self.subTest(positions=positions):
                with self.assertRaises(ValueError, msg='Invalid detector positions are expected to be rejected.'):
                    Obs1D(positions)

    def test_size_none_single_position(self):
        with self.assertRaises(ValueError, msg='A single position without size is expected to be rejected.'):
            Obs1D(self.INPUT_POSITIONS[:1])

    def test_size_modes(self):
        n = len(self.INPUT_POSITIONS)
        for size in (None, self.INPUT_SIZE_SCALAR, self.INPUT_SIZE_ARRAY):
            with self.subTest(size=size):
                obs = Obs1D(self.INPUT_POSITIONS, size)
                self.assertEqual(obs.detector_positions.shape, (n,), msg='Wrong detector position shape.')
                for attribute, shape in (('detector_size', (n,)), ('detector_ranges', (n, 2))):
                    value = getattr(obs, attribute)
                    self.assertIsInstance(value, numpy.ndarray, msg='<' + attribute + '> is expected as numpy array.')
                    self.assertEqual(value.dtype, numpy.float64, msg='<' + attribute + '> is expected as floats.')
                    self.assertEqual(value.shape, shape, msg='Wrong <' + attribute + '> shape.')

    def test_invalid_sizes(self):
        for size in self.INPUT_INVALID_SIZES:
            with self.subTest(size=size):
                with self.assertRaises(ValueError, msg='Invalid detector sizes are expected to be rejected.'):
                    Obs1D(self.INPUT_POSITIONS, size)

    def test_read_only(self):
        obs = Obs1D(self.INPUT_POSITIONS, self.INPUT_SIZE_ARRAY)
        for attribute in self.EXPECTED_ATTRIBUTES:
            with self.subTest(attribute=attribute):
                with self.assertRaises(ValueError, msg='<' + attribute + '> is expected to be read-only.'):
                    getattr(obs, attribute)[0] = 0.

    def test_input_copied(self):
        positions = numpy.array(self.INPUT_POSITIONS)
        sizes = numpy.array(self.INPUT_SIZE_ARRAY)
        obs = Obs1D(positions, sizes)
        stored_sizes = obs.detector_size.copy()
        self.assertTrue(positions.flags.writeable and sizes.flags.writeable,
                        msg='Input arrays of the caller are expected to remain writeable.')
        positions[0], sizes[0] = 0., 1.
        numpy.testing.assert_array_equal(obs.detector_positions, self.INPUT_POSITIONS,
                                         err_msg='Detector positions are expected to be a copy of the input.')
        numpy.testing.assert_array_equal(obs.detector_size, stored_sizes,
                                         err_msg='Detector sizes are expected to be independent of the input.')

    def test_default_transition_not_given(self):
        obs = Obs1D(self.INPUT_POSITIONS)
        self.assertIsNone(obs.from_level, msg='Default <from_level> is expected to be None.')
        self.assertIsNone(obs.to_level, msg='Default <to_level> is expected to be None.')

    def test_default_transition_given(self):
        obs = Obs1D(self.INPUT_POSITIONS, from_level='3n', to_level='2n')
        self.assertEqual((obs.from_level, obs.to_level), ('3n', '2n'), msg='Default transition is not stored.')

    def test_invalid_default_transition(self):
        for (from_level, to_level), error in self.INPUT_INVALID_TRANSITIONS:
            with self.subTest(from_level=from_level, to_level=to_level):
                with self.assertRaises(error, msg='Invalid default transition is expected to be rejected.'):
                    Obs1D(self.INPUT_POSITIONS, from_level=from_level, to_level=to_level)


class Obs1DTransitionTest(unittest.TestCase):
    INPUT_POSITIONS = [0.25, 0.5, 1.0]
    INPUT_INSTANCE_DEFAULT = ('3n', '2n')
    INPUT_CALL_OVERRIDE = ('4n', '2n')
    INPUT_PRIORITY_CASES = [((None, None), (None, None), ('9x', '8x')),
                            ((None, None), ('4n', '2n'), ('4n', '2n')),
                            (('3n', '2n'), (None, None), ('3n', '2n')),
                            (('3n', '2n'), ('4n', '2n'), ('4n', '2n'))]
    INPUT_SINGLE_LEVEL_OVERRIDES = [('4n', None), (None, '2n')]

    def setUp(self):
        self.beamlet = _StubBeamlet()

    def tearDown(self):
        del self.beamlet

    def test_priority(self):
        for instance_default, call_levels, expected in self.INPUT_PRIORITY_CASES:
            with self.subTest(instance_default=instance_default, call_levels=call_levels):
                obs = Obs1D(self.INPUT_POSITIONS, from_level=instance_default[0], to_level=instance_default[1])
                actual = obs._resolve_transition(self.beamlet, *call_levels)
                self.assertEqual(actual[:2], expected, msg='Transition levels resolved from the wrong source.')
                self.assertEqual(actual[2], expected[0] + '-->' + expected[1],
                                 msg='Transition label is expected in the <from>--><to> format.')

    def test_override_keeps_instance_default(self):
        obs = Obs1D(self.INPUT_POSITIONS, from_level=self.INPUT_INSTANCE_DEFAULT[0],
                    to_level=self.INPUT_INSTANCE_DEFAULT[1])
        obs._resolve_transition(self.beamlet, *self.INPUT_CALL_OVERRIDE)
        self.assertEqual((obs.from_level, obs.to_level), self.INPUT_INSTANCE_DEFAULT,
                         msg='A call override is not expected to change the instance default transition.')

    def test_single_level_override(self):
        obs = Obs1D(self.INPUT_POSITIONS)
        for from_level, to_level in self.INPUT_SINGLE_LEVEL_OVERRIDES:
            with self.subTest(from_level=from_level, to_level=to_level):
                with self.assertRaises(ValueError, msg='Overriding a single level is expected to be rejected.'):
                    obs._resolve_transition(self.beamlet, from_level, to_level)


class Obs1DEmissionDensityTest(unittest.TestCase):
    INPUT_TRANSITION = ('9x', '8x', '9x-->8x')
    EXPECTED_COMPUTE_CALL = {'from_level': '9x', 'to_level': '8x'}

    def test_stored_emission_reused(self):
        beamlet = _StubBeamlet(emission_stored=True)
        _, emission = Obs1D._get_linear_emission_density(beamlet, *self.INPUT_TRANSITION)
        self.assertEqual(beamlet.compute_calls, [], msg='Stored emission is not expected to be recomputed.')
        numpy.testing.assert_array_equal(emission, _StubBeamlet.EMISSION,
                                         err_msg='Stored emission is expected to be returned.')

    def test_missing_emission_computed(self):
        beamlet = _StubBeamlet()
        _, emission = Obs1D._get_linear_emission_density(beamlet, *self.INPUT_TRANSITION)
        self.assertEqual(beamlet.compute_calls, [self.EXPECTED_COMPUTE_CALL],
                         msg='Missing emission is expected to be computed once with matching levels.')
        numpy.testing.assert_array_equal(emission, _StubBeamlet.EMISSION,
                                         err_msg='Computed emission is expected to be returned.')

    def test_unsolved_beamlet(self):
        with self.assertRaises(RuntimeError, msg='Unsolved beamlet is expected to raise RuntimeError.'):
            Obs1D._get_linear_emission_density(_StubBeamlet(solved=False), *self.INPUT_TRANSITION)

    def test_return_types(self):
        grid, emission = Obs1D._get_linear_emission_density(_StubBeamlet(emission_stored=True),
                                                            *self.INPUT_TRANSITION)
        for name, value, reference in (('grid', grid, _StubBeamlet.GRID),
                                       ('emission', emission, _StubBeamlet.EMISSION)):
            with self.subTest(output=name):
                self.assertIsInstance(value, numpy.ndarray, msg='<' + name + '> is expected as numpy array.')
                numpy.testing.assert_array_equal(value, reference, err_msg='<' + name + '> is returned incorrectly.')


class Obs1DIntegratorTest(unittest.TestCase):
    pass


class Obs1DOutsideGridTest(unittest.TestCase):
    pass


class Obs1DPhotonFluxTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
