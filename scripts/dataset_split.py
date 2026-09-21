import argparse
import random
import shutil
from pathlib import Path


def split_dataset(base_path: str, test_size: int = 100, seed: int = 42):
    """
    Splits files in each subdirectory of base_path into train/ and test/ folders.

    Args:
        base_path: Path to the 'tasks' directory.
        test_size: Number of files to hold out for testing (default 100).
        seed: Random seed for reproducibility.
    """
    root = Path(base_path)

    if not root.exists():
        print(f"Error: Directory '{base_path}' does not exist.")
        return

    # Set seed for reproducibility (crucial for paper consistency)
    random.seed(seed)

    # Iterate over each task directory (e.g., tasks/knapsack, tasks/tsp)
    for task_dir in [d for d in root.iterdir() if d.is_dir()]:
        print(f"Processing {task_dir.name}...")

        # 1. Identify all valid task files (exclude existing train/test folders)
        all_files = [
            f
            for f in task_dir.iterdir()
            if f.is_file() and f.suffix == ".json" and not f.name.startswith(".")
        ]

        # Safety check: if we have fewer files than the requested test size
        if len(all_files) <= test_size:
            print(
                f"  [Warning] Not enough files in {task_dir.name} to create a train set."
            )
            print(f"  Total: {len(all_files)}, Required for Test: {test_size}")
            print("  Skipping...")
            continue

        # 2. Shuffle and Split
        random.shuffle(all_files)

        test_files = all_files[:test_size]
        train_files = all_files[test_size:]

        # 3. Create destinations
        train_dir = task_dir / "train"
        test_dir = task_dir / "test"

        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)

        # 4. Move files
        # Move Test Files
        for f in test_files:
            shutil.move(str(f), str(test_dir / f.name))

        # Move Train Files
        for f in train_files:
            shutil.move(str(f), str(train_dir / f.name))

        print(f"  -> Split complete: {len(train_files)} train, {len(test_files)} test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split task files into train/test directories."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="tasks",
        help="Root directory containing task folders",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=100,
        help="Number of files to reserve for test split",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )

    args = parser.parse_args()

    split_dataset(args.dir, args.test_count, args.seed)
