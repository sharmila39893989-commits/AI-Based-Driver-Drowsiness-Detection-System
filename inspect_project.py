from __future__ import annotations

from backend.src.model_utils import PROJECT_ROOT, summarize_project, write_json_report


def inspect_project() -> dict:
    summary = summarize_project()
    print("Project root:", PROJECT_ROOT)
    print("Dataset summary:")
    for task_name, split_info in summary["dataset"].items():
        print(f"- {task_name}: {split_info}")
    print("Models:")
    for model_name, info in summary["models"].items():
        print(f"- {model_name}: {info}")
    return summary


if __name__ == "__main__":
    data = inspect_project()
    report_path = PROJECT_ROOT / "project_summary.json"
    write_json_report(report_path, data)
    print(f"Saved summary report to {report_path}")
