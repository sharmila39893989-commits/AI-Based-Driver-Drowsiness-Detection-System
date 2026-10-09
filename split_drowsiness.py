from __future__ import annotations

import random
import shutil
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataset" / "drowsiness"
CLASSES = ("drowsy", "non_drowsy")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TEST_SIZE = 0.20
RANDOM_SEED = 42


def split_dataset() -> bool:
    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    if any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for class_name in CLASSES
        for path in (test_dir / class_name).glob("*")
    ):
        print("The drowsiness dataset already has a test split; no files were moved.")
        return False

    selected: dict[str, tuple[list[Path], int]] = {}
    for class_name in CLASSES:
        class_dir = train_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) < 2:
            raise ValueError(
                f"At least two images are required in {class_dir}; found {len(images)}."
            )

        random.Random(f"{RANDOM_SEED}:{class_name}").shuffle(images)
        test_count = max(1, int(len(images) * TEST_SIZE))
        selected[class_name] = (images[:test_count], len(images))

    for class_name, (images, total_count) in selected.items():
        destination_dir = test_dir / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            shutil.move(str(image), str(destination_dir / image.name))
        print(f"{class_name}: moved {len(images)} of {total_count} images to test.")

    return True


if __name__ == "__main__":
    try:
        split_dataset()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Dataset split failed: {error}") from error
