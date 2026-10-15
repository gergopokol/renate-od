import unittest
import numpy
from observation.observation import Obs1D


class _StubAtomicDB:
    DEFAULT_TRANSITION = ('9x', '8x', '1x', '9x-->8x')

    def set_default_atomic_levels(self):
        return self.DEFAULT_TRANSITION


class _StubBeamlet:
    def __init__(self):
        self.atomic_db = _StubAtomicDB()


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
        obs = Obs1D(self.INPUT_POSITIONS, *self.INPUT_INSTANCE_DEFAULT)
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
    pass


class Obs1DIntegratorTest(unittest.TestCase):
    pass


class Obs1DOutsideGridTest(unittest.TestCase):
    pass


class Obs1DPhotonFluxTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
