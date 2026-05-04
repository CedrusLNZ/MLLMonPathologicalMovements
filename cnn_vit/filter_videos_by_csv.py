#!/usr/bin/env python3
import argparse
import csv
import os
import sys


def load_mapping(csv_path):
    mapping = {}
    conflicts = {}
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        required = {"file_name", "90_file_name"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            old_name = (row.get("file_name") or "").strip()
            new_name = (row.get("90_file_name") or "").strip()
            if not old_name:
                continue
            if not new_name:
                print(f"warning: no 90_file_name for {old_name}", file=sys.stderr)
                continue
            if old_name in mapping and mapping[old_name] != new_name:
                conflicts.setdefault(old_name, set()).update([mapping[old_name], new_name])
                continue
            mapping[old_name] = new_name

    if conflicts:
        details = "; ".join(
            f"{name} -> {sorted(list(targets))}" for name, targets in conflicts.items()
        )
        raise ValueError(f"Conflicting mappings found: {details}")
    return mapping


def iter_files(root, extensions, all_files):
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if all_files:
                yield os.path.join(dirpath, filename)
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                yield os.path.join(dirpath, filename)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rename files listed in the CSV mapping and delete the rest, "
            "keeping the existing folder structure."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing the videos to process.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the CSV mapping file.",
    )
    parser.add_argument(
        "--extensions",
        default=".mp4",
        help="Comma-separated list of extensions to include (default: .mp4).",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Process all files under root, ignoring --extensions.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, the script runs in dry-run mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting if the destination file already exists.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    csv_path = os.path.abspath(args.csv)
    extensions = {
        ext.strip().lower()
        for ext in args.extensions.split(",")
        if ext.strip()
    }
    if not args.all_files and not extensions:
        raise ValueError("No extensions provided; use --all-files or set --extensions.")

    mapping = load_mapping(csv_path)

    actions = []
    seen_mapping = set()
    collisions = []

    for path in iter_files(root, extensions, args.all_files):
        basename = os.path.basename(path)
        if basename in mapping:
            seen_mapping.add(basename)
            new_name = mapping[basename]
            dest = os.path.join(os.path.dirname(path), new_name)
            if os.path.abspath(dest) == os.path.abspath(path):
                continue
            if os.path.exists(dest) and not args.overwrite:
                collisions.append((path, dest))
                continue
            actions.append(("rename", path, dest))
        else:
            actions.append(("delete", path, None))

    if collisions:
        print("error: destination files already exist:", file=sys.stderr)
        for src, dest in collisions:
            print(f"  {src} -> {dest}", file=sys.stderr)
        print("rerun with --overwrite if you want to replace them.", file=sys.stderr)
        sys.exit(1)

    for action, src, dest in actions:
        if action == "rename":
            print(f"RENAME {src} -> {dest}")
            if args.apply:
                if args.overwrite:
                    os.replace(src, dest)
                else:
                    os.rename(src, dest)
        elif action == "delete":
            print(f"DELETE {src}")
            if args.apply:
                os.remove(src)

    missing = sorted(set(mapping.keys()) - seen_mapping)
    if missing:
        print(f"warning: {len(missing)} mapping entries were not found under root.")


if __name__ == "__main__":
    main()
