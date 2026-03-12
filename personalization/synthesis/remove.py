#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r") as handle:
        return json.load(handle)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=4)


def _ensure_list(data: Any, label: str) -> list:
    if not isinstance(data, list):
        raise ValueError(
            f"{label} must be a JSON array, got {type(data).__name__}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove a persona and all its associated data from combined JSON files."
    )
    parser.add_argument(
        "--output-dir",
        default="synthesis/output/email",
        help="Output directory containing personas.json/styles.json/tasks.json/dataset.json",
    )
    parser.add_argument(
        "--persona-id",
        type=int,
        required=True,
        help="Persona ID to remove",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    personas_path = output_dir / "personas.json"
    styles_path = output_dir / "styles.json"
    tasks_path = output_dir / "tasks.json"
    dataset_path = output_dir / "dataset.json"
    persona_dir = output_dir / f"persona_{args.persona_id}"

    personas = _ensure_list(_load_json(personas_path, []), "personas.json")
    styles = _ensure_list(_load_json(styles_path, []), "styles.json")
    tasks = _ensure_list(_load_json(tasks_path, []), "tasks.json")
    dataset = _ensure_list(_load_json(dataset_path, []), "dataset.json")

    # Find the persona to get its name for filtering tasks/dataset
    persona_to_remove = None
    for p in personas:
        if int(p.get("id", 0)) == args.persona_id:
            persona_to_remove = p.get("persona", {})
            break

    if persona_to_remove is None:
        print(
            f"Warning: Persona ID {args.persona_id} not found in {personas_path}")
        persona_name = None
    else:
        persona_name = persona_to_remove.get("name")
        print(f"Removing persona ID {args.persona_id}: {persona_name}")

    # Remove from personas.json (by ID)
    original_count = len(personas)
    personas = [p for p in personas if int(p.get("id", 0)) != args.persona_id]
    removed_personas = original_count - len(personas)
    print(f"Removed {removed_personas} persona(s) from {personas_path}")

    # Remove from styles.json (by ID)
    original_count = len(styles)
    styles = [s for s in styles if int(s.get("id", 0)) != args.persona_id]
    removed_styles = original_count - len(styles)
    print(f"Removed {removed_styles} style(s) from {styles_path}")

    # Remove from tasks.json (by persona name)
    original_count = len(tasks)
    if persona_name:
        tasks = [
            t for t in tasks
            if t.get("persona", {}).get("name") != persona_name
        ]
    removed_tasks = original_count - len(tasks)
    print(f"Removed {removed_tasks} task(s) from {tasks_path}")

    # Remove from dataset.json (by persona name)
    original_count = len(dataset)
    if persona_name:
        dataset = [
            d for d in dataset
            if d.get("persona", {}).get("name") != persona_name
        ]
    removed_dataset = original_count - len(dataset)
    print(f"Removed {removed_dataset} dataset entry(ies) from {dataset_path}")

    # Save updated JSON files
    _save_json(personas_path, personas)
    _save_json(styles_path, styles)
    _save_json(tasks_path, tasks)
    _save_json(dataset_path, dataset)

    # Remove persona folder
    if persona_dir.exists():
        shutil.rmtree(persona_dir)
        print(f"Removed persona folder: {persona_dir}")
    else:
        print(f"Warning: Persona folder not found: {persona_dir}")

    print("\nRemoval complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
