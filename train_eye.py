if __package__:
    from .src.training import run_training_cli
else:
    from src.training import run_training_cli


if __name__ == "__main__":
    run_training_cli(
        "eye",
        "eyes",
        "eye_model.h5",
        expected_classes=("Closed", "Open"),
    )
