"""Smoke test the architecture using synthetic input only."""

import unittest

import tensorflow as tf

from smartpig_inverse import build_vt_cnn


class InverseModelTest(unittest.TestCase):
    def test_untrained_forward_pass(self):
        model = build_vt_cnn()
        self.assertEqual(model.input_shape, (None, 5, 61, 1))
        self.assertEqual(model.output_shape, (None, 61, 65))
        self.assertEqual(model.count_params(), 3_570_038)

        output = model(tf.zeros((2, 5, 61, 1), dtype=tf.float32), training=False)
        self.assertEqual(tuple(output.shape), (2, 61, 65))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(output)).numpy()))
        self.assertTrue(bool(tf.reduce_all(output >= 0).numpy()))


if __name__ == "__main__":
    unittest.main()
