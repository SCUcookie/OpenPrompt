from __future__ import annotations

import argparse
from pathlib import Path


def resolve_repo_path(repo_root: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def replace_with_symlink(target: Path, source: Path, force: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Asset source does not exist: {source}")

    if target.is_symlink():
        target.unlink()
    elif target.exists():
        if not force:
            raise FileExistsError(
                f"Target path already exists and is not a symlink: {target}. "
                "Use --force after moving or deleting the existing local directory."
            )
        if target.is_dir():
            raise IsADirectoryError(
                f"Refusing to replace existing directory without manual cleanup: {target}"
            )
        target.unlink()

    target.symlink_to(source.resolve())
    print(f"linked {target} -> {source.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Link external OpenPrompt datasets and outputs into this repo.")
    parser.add_argument(
        "--dota-root",
        default=None,
        help="Directory containing DOTA local assets, including `train/`, `val/`, and versioned `labelTxt-*` folders.",
    )
    parser.add_argument(
        "--dotav2-root",
        default=None,
        help="Directory containing DOTA-v2 local assets, including `images/` and `labels/`.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Optional directory to expose as the repo-local `outputs/` path.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing symlinks or files.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dota_root = resolve_repo_path(repo_root, args.dota_root)
    dotav2_root = resolve_repo_path(repo_root, args.dotav2_root)
    outputs_dir = resolve_repo_path(repo_root, args.outputs_dir)

    if dota_root is None and dotav2_root is None and outputs_dir is None:
        raise ValueError("Provide at least one of --dota-root, --dotav2-root, or --outputs-dir.")

    if dota_root is not None:
        replace_with_symlink(repo_root / "DOTA", dota_root, args.force)

    if dotav2_root is not None:
        replace_with_symlink(repo_root / "DOTAv2", dotav2_root, args.force)
    if outputs_dir is not None:
        replace_with_symlink(repo_root / "outputs", outputs_dir, args.force)


if __name__ == "__main__":
    main()
