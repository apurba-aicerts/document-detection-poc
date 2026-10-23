import unittest

import cv2
import numpy as np

from detect_document import detect_document, order_corners, rectify_document, synthetic_cases


class DocumentDetectionTests(unittest.TestCase):
    def test_synthetic_conditions(self):
        for name, image, expected in synthetic_cases():
            with self.subTest(condition=name):
                candidate, _ = detect_document(image)
                if expected is None:
                    self.assertIsNone(candidate)
                    continue
                self.assertIsNotNone(candidate)
                actual = np.array(candidate["corners"], dtype=np.float32)
                expected_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                actual_mask = expected_mask.copy()
                cv2.fillConvexPoly(expected_mask, expected.astype(np.int32), 1)
                cv2.fillConvexPoly(actual_mask, actual.astype(np.int32), 1)
                intersection = np.count_nonzero(expected_mask & actual_mask)
                union = np.count_nonzero(expected_mask | actual_mask)
                self.assertGreater(intersection / union, 0.95)
                distances = np.linalg.norm(actual[:, None] - expected[None, :], axis=2)
                self.assertLess(float(distances.min(axis=1).max()), 8)
                crop = rectify_document(image, actual)
                self.assertGreater(min(crop.shape[:2]), 200)

    def test_downscaled_coordinates_use_original_image(self):
        _, image, expected = next(synthetic_cases())
        image = cv2.resize(image, None, fx=3, fy=3)
        candidate, _ = detect_document(image, max_side=800)
        self.assertIsNotNone(candidate)
        actual = np.array(candidate["corners"])
        distances = np.linalg.norm(actual[:, None] - (expected * 3)[None, :], axis=2)
        self.assertLess(float(distances.min(axis=1).max()), 24)

    def test_ordering_has_four_distinct_corners_at_45_degrees(self):
        corners = np.array([[200, 0], [400, 200], [200, 400], [0, 200]])
        ordered = order_corners(corners)
        self.assertEqual(len(np.unique(ordered, axis=0)), 4)
        self.assertGreater(cv2.contourArea(ordered, oriented=True), 0)

    def test_image_border_is_not_a_document(self):
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (799, 599), (255, 255, 255), 2)
        candidate, _ = detect_document(image)
        self.assertIsNone(candidate)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            detect_document(None)
        with self.assertRaises(ValueError):
            detect_document(np.zeros((100, 100, 3), np.uint8), min_area_ratio=0)


if __name__ == "__main__":
    unittest.main()