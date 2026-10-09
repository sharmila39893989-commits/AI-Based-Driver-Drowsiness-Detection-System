if __package__:
    from .src.training import run_training_cli
else:
    from src.training import run_training_cli


if __name__ == "__main__":
    run_training_cli(
        "yawn",
        "yawn",
        "yawn_model.keras",
        expected_classes=("no_yawn", "yawn"),
    )

