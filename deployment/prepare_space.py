"""Create a public Space bundle without credentials, private documents or indexes."""

import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def prepare(output):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Choose an output directory outside the source checkout.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "The output directory must be empty; nothing was overwritten."
        )
    output.mkdir(parents=True, exist_ok=True)
    for name in ["Dockerfile", "README.md"]:
        shutil.copy2(ROOT / "deployment/huggingface" / name, output / name)
    shutil.copy2(ROOT / "requirements.txt", output / "requirements.txt")
    shutil.copytree(
        ROOT / "app",
        output / "app",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".env", ".env.*"),
    )
    shutil.copytree(ROOT / "evaluations/documents", output / "evaluations/documents")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output))


if __name__ == "__main__":
    main()
