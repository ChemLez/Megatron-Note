#!/usr/bin/env python3
"""Archive a note source into a topic directory and update the archive README."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
from pathlib import Path


def slug_to_title(value: str) -> str:
    cleaned = value.strip().replace("_", "-")
    parts = [part for part in cleaned.split("-") if part]
    return " ".join(part.capitalize() for part in parts) if parts else "Unsorted"


def conflict_safe_path(path: Path) -> Path:
    if not path.exists():
        return path

    if path.is_dir() or not path.suffix:
        base = path
        for index in range(2, 1000):
            candidate = base.with_name(f"{base.name}-v{index}")
            if not candidate.exists():
                return candidate
    else:
        for index in range(2, 1000):
            candidate = path.with_name(f"{path.stem}-v{index}{path.suffix}")
            if not candidate.exists():
                return candidate

    raise RuntimeError(f"Could not find a conflict-safe path for {path}")


def normalize_topic_dir(archive_root: Path, topic_dir: str) -> Path:
    raw = Path(topic_dir).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (archive_root / raw).resolve()


def relative_link(readme_path: Path, target: Path) -> str:
    try:
        return target.resolve().relative_to(readme_path.parent.resolve()).as_posix()
    except ValueError:
        return Path(__import__("os").path.relpath(target, readme_path.parent)).as_posix()


def move_into_directory(source: Path, target_dir: Path, dry_run: bool) -> Path:
    destination = conflict_safe_path(target_dir / source.name)
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    return destination


def place_visual(visual_path: Path | None, target_dir: Path, dry_run: bool) -> Path | None:
    if visual_path is None:
        return None

    source = visual_path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Visual summary file does not exist: {source}")

    if source.parent == target_dir.resolve():
        return source

    destination = conflict_safe_path(target_dir / source.name)
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    return destination


def build_entry(
    readme_path: Path,
    archived_path: Path,
    title: str,
    summary: str,
    reason: str,
    visual_path: Path | None,
    date: str,
) -> list[str]:
    entry = [
        f"- {date} [{title}]({relative_link(readme_path, archived_path)}) - {summary}",
        f"  - Reason: {reason}",
    ]
    if visual_path is not None:
        entry.append(f"  - Visual: [summary image]({relative_link(readme_path, visual_path)})")
    return entry


def find_topic_heading(lines: list[str], topic_title: str) -> int | None:
    wanted = f"### {topic_title}".strip().lower()
    for index, line in enumerate(lines):
        if line.strip().lower() == wanted:
            return index
    return None


def ensure_readme(
    archive_root: Path,
    topic_title: str,
    entry_lines: list[str],
    dry_run: bool,
) -> Path:
    readme_path = archive_root / "README.md"
    archive_title = archive_root.name or "Notes Archive"

    if readme_path.exists():
        lines = readme_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = [f"# {archive_title}", "", "## Topics", ""]

    if not any(line.strip() == "## Topics" for line in lines):
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(["## Topics", ""])

    heading_index = find_topic_heading(lines, topic_title)
    if heading_index is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([f"### {topic_title}", ""])
        heading_index = len(lines) - 2

    insert_at = heading_index + 1
    while insert_at < len(lines) and lines[insert_at].strip() == "":
        insert_at += 1

    new_block = entry_lines + [""]
    lines[insert_at:insert_at] = new_block

    if not dry_run:
        archive_root.mkdir(parents=True, exist_ok=True)
        readme_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return readme_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="File or folder to archive.")
    parser.add_argument("--archive-root", required=True, help="Root directory containing README.md.")
    parser.add_argument("--topic-dir", required=True, help="Topic directory name/path under archive root, or an absolute target directory.")
    parser.add_argument("--title", required=True, help="Title to display in README.")
    parser.add_argument("--summary", required=True, help="One-sentence summary for README.")
    parser.add_argument("--reason", required=True, help="Reason this topic directory was selected or created.")
    parser.add_argument("--visual-path", help="Optional generated visual summary image to move/link.")
    parser.add_argument("--date", default=_dt.date.today().isoformat(), help="Entry date, default: today.")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without moving files or writing README.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    source = Path(args.source).expanduser().resolve()
    archive_root = Path(args.archive_root).expanduser().resolve()
    topic_dir = normalize_topic_dir(archive_root, args.topic_dir)
    visual_source = Path(args.visual_path) if args.visual_path else None

    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if source == archive_root or archive_root in source.parents:
        raise ValueError("Source is already inside the archive root; choose a source outside the archive or a more specific action.")

    archived_path = move_into_directory(source, topic_dir, args.dry_run)
    final_visual_path = place_visual(visual_source, topic_dir, args.dry_run)
    topic_title = slug_to_title(topic_dir.name)
    readme_path = archive_root / "README.md"
    entry = build_entry(
        readme_path=readme_path,
        archived_path=archived_path,
        title=args.title,
        summary=args.summary,
        reason=args.reason,
        visual_path=final_visual_path,
        date=args.date,
    )
    readme_path = ensure_readme(archive_root, topic_title, entry, args.dry_run)

    result = {
        "dry_run": args.dry_run,
        "archive_root": str(archive_root),
        "topic_dir": str(topic_dir),
        "archived_path": str(archived_path),
        "readme_path": str(readme_path),
        "archived_link": relative_link(readme_path, archived_path),
        "visual_path": str(final_visual_path) if final_visual_path else None,
        "visual_link": relative_link(readme_path, final_visual_path) if final_visual_path else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"archive_note.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
