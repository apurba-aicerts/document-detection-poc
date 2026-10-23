import unittest

import numpy as np

from detect_doccornernet import decode_prediction, prepare_input


class DocCornerNetTests(unittest.TestCase):
    def test_rgb_normalization_and_shape(self):
        image = np.full((30, 60, 3), [0, 128, 255], dtype=np.uint8)
        actual = prepare_input(image)
        expected = (np.array([1, 128 / 255, 0]) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        self.assertEqual(actual.shape, (1, 224, 224, 3))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual[0, 0, 0], expected, atol=1e-6)

    def test_original_image_coordinates_and_presence(self):
        prediction = [0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8, 0.9]
        result = decode_prediction(prediction, (600, 800, 3))
        np.testing.assert_allclose(result["corners"], [[80, 120], [720, 120], [720, 480], [80, 480]])
        self.assertTrue(result["document_candidate_found"])
        prediction[-1] = 0.1
        self.assertFalse(decode_prediction(prediction, (600, 800, 3))["document_candidate_found"])

    def test_degenerate_corners_are_not_accepted(self):
        self.assertFalse(decode_prediction([0.5] * 8 + [0.99], (600, 800, 3))["document_candidate_found"])

    def test_coordinates_stay_inside_image(self):
        result = decode_prediction([0, 0, 1, 0, 1, 1, 0, 1, 0.9], (600, 800, 3))
        np.testing.assert_array_equal(result["corners"], [[0, 0], [799, 0], [799, 599], [0, 599]])

    def test_invalid_predictions(self):
        for prediction in ([0] * 8, [float("nan")] * 9, [2] * 9):
            with self.subTest(prediction=prediction):
                with self.assertRaises(ValueError):
                    decode_prediction(prediction, (600, 800, 3))


if __name__ == "__main__":
    unittest.main()