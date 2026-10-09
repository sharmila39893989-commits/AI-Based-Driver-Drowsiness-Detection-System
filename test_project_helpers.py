import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.src.model_utils import binary_probability, inspect_model, preprocess_image
from backend.src.training import validate_binary_dataset
from backend.split_drowsiness import split_dataset
from backend.src.face_detection import detect_face_region
from backend.src.face_utils import (
    detect_face,
    extract_eye_regions,
    extract_face,
    extract_mouth_region,
)


class DatasetValidationTests(unittest.TestCase):
    def test_validates_binary_train_and_test_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for split in ("train", "test"):
                for class_name in ("closed", "open"):
                    class_dir = root / split / class_name
                    class_dir.mkdir(parents=True)
                    (class_dir / "sample.png").write_bytes(b"image")

            classes = validate_binary_dataset(
                root / "train",
                root / "test",
                expected_classes=("closed", "open"),
            )

        self.assertEqual(classes, ["closed", "open"])

    def test_reports_empty_or_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for split in ("train", "test"):
                for class_name in ("no_yawn", "yawn"):
                    (root / split / class_name).mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "No supported image files"):
                validate_binary_dataset(
                    root / "train",
                    root / "test",
                    expected_classes=("no_yawn", "yawn"),
                )

            with self.assertRaisesRegex(FileNotFoundError, "Dataset split not found"):
                validate_binary_dataset(root / "missing", root / "test")


class ModelUtilityTests(unittest.TestCase):
    def test_reads_single_output_and_two_class_probabilities(self) -> None:
        self.assertEqual(binary_probability([[0.3]]), 0.3)
        self.assertEqual(binary_probability([[0.2, 0.8]]), 0.8)

    def test_rejects_invalid_prediction_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one probability"):
            binary_probability([])
        with self.assertRaisesRegex(ValueError, "exactly one sample"):
            binary_probability([[0.1], [0.9]])

    def test_converts_bgr_to_rgb_and_adds_batch_dimension(self) -> None:
        bgr_pixel = np.array([[[0, 0, 255]]], dtype=np.uint8)
        result = preprocess_image(
            bgr_pixel,
            target_size=(1, 1),
            color_order="bgr",
        )

        self.assertEqual(result.shape, (1, 1, 1, 3))
        np.testing.assert_allclose(result[0, 0, 0], [1.0, 0.0, 0.0])

    def test_inspect_model_handles_dynamic_batch_and_none_dims(self) -> None:
        class DummyModel:
            input_shape = (None, 224, 224, 3)
            output_shape = (None, 2)

        info = inspect_model(DummyModel())

        self.assertEqual(info["input_shape"], (224, 224, 3))
        self.assertEqual(info["output_shape"], (2,))
        self.assertEqual(info["num_outputs"], 2)


class FaceUtilityTests(unittest.TestCase):
    def test_face_helpers_handle_a_valid_frame_without_a_face(self) -> None:
        frame = np.zeros((96, 96, 3), dtype=np.uint8)

        self.assertEqual(detect_face(frame), (None, None))
        self.assertEqual(detect_face_region(frame), (None, None))
        face, box = extract_face(frame)
        self.assertIs(face, frame)
        self.assertIsNone(box)
        self.assertEqual(extract_eye_regions(frame), (None, None))
        self.assertIsNone(extract_mouth_region(frame))


class DatasetSplitTests(unittest.TestCase):
    def test_does_not_move_images_when_dataset_is_already_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory) / "drowsiness"
            for class_name in ("drowsy", "non_drowsy"):
                train_class = dataset / "train" / class_name
                train_class.mkdir(parents=True)
                (train_class / "train.jpg").write_bytes(b"train")
            test_class = dataset / "test" / "drowsy"
            test_class.mkdir(parents=True)
            (test_class / "test.jpg").write_bytes(b"test")

            with patch("backend.split_drowsiness.DATASET_DIR", dataset):
                changed = split_dataset()

            self.assertFalse(changed)
            self.assertTrue((dataset / "train" / "drowsy" / "train.jpg").exists())
            self.assertTrue((dataset / "test" / "drowsy" / "test.jpg").exists())


if __name__ == "__main__":
    unittest.main()
