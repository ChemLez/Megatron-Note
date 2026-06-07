#!/usr/bin/env python3
"""Archive a note source into a topic directory and update the archive README."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import sys
from pathlib import Path

CHAPTER_DIGITS = "零一二三四五六七八九"


def slug_to_title(value: str) -> str:
    cleaned = value.strip().replace("_", "-")
    parts = [part for part in cleaned.split("-") if part]
    return " ".join(part.capitalize() for part in parts) if parts else "Unsorted"


def int_to_chinese(value: int) -> str:
    if value <= 0:
        raise ValueError("Chapter number must be positive.")
    if value < 10:
        return CHAPTER_DIGITS[value]
    if value == 10:
        return "十"
    if value < 20:
        return f"十{CHAPTER_DIGITS[value % 10]}"
    if value < 100:
        tens, ones = divmod(value, 10)
        return f"{CHAPTER_DIGITS[tens]}十" + (CHAPTER_DIGITS[ones] if ones else "")
    hundreds, remainder = divmod(value, 100)
    prefix = f"{CHAPTER_DIGITS[hundreds]}百"
    if remainder == 0:
        return prefix
    if remainder < 10:
        return f"{prefix}零{CHAPTER_DIGITS[remainder]}"
    return f"{prefix}{int_to_chinese(remainder)}"


def chinese_to_int(value: str) -> int | None:
    if value.isdigit():
        number = int(value)
        return number if number > 0 else None
    if value in CHAPTER_DIGITS:
        return CHAPTER_DIGITS.index(value)
    if value == "十":
        return 10
    if "百" in value:
        left, right = value.split("百", 1)
        hundreds = CHAPTER_DIGITS.index(left) if left in CHAPTER_DIGITS else 0
        if hundreds == 0:
            return None
        if not right:
            return hundreds * 100
        if right.startswith("零"):
            right = right[1:]
        remainder = chinese_to_int(right)
        return hundreds * 100 + remainder if remainder is not None else None
    if "十" in value:
        left, right = value.split("十", 1)
        tens = 1 if not left else CHAPTER_DIGITS.index(left) if left in CHAPTER_DIGITS else 0
        ones = 0 if not right else CHAPTER_DIGITS.index(right) if right in CHAPTER_DIGITS else -1
        if tens <= 0 or ones < 0:
            return None
        return tens * 10 + ones
    return None


def strip_chapter_prefix(title: str) -> str:
    match = re.match(r"^\s*第[0-9零一二三四五六七八九十百]+章[：:\-\s]*(.+?)\s*$", title)
    return match.group(1).strip() if match else title.strip()


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


def topic_section_lines(lines: list[str], topic_title: str) -> list[str]:
    heading_index = find_topic_heading(lines, topic_title)
    if heading_index is None:
        return []
    end = len(lines)
    for index in range(heading_index + 1, len(lines)):
        if lines[index].startswith("### "):
            end = index
            break
    return lines[heading_index + 1 : end]


def next_chapter_number(archive_root: Path, topic_title: str) -> int:
    readme_path = archive_root / "README.md"
    if not readme_path.exists():
        return 1

    lines = readme_path.read_text(encoding="utf-8").splitlines()
    max_chapter = 0
    for line in topic_section_lines(lines, topic_title):
        match = re.search(r"\[第([0-9零一二三四五六七八九十百]+)章[：:\-\s]*[^\]]*\]", line)
        if not match:
            continue
        number = chinese_to_int(match.group(1))
        if number is not None:
            max_chapter = max(max_chapter, number)
    return max_chapter + 1


def chapter_title(title: str, archive_root: Path, topic_title: str) -> str:
    core_title = strip_chapter_prefix(title)
    number = next_chapter_number(archive_root, topic_title)
    return f"第{int_to_chinese(number)}章：{core_title}"


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
    parser.add_argument("--title", required=True, help="Content-based title core to display in README; chapter prefix is added by default.")
    parser.add_argument("--summary", required=True, help="One-sentence summary for README.")
    parser.add_argument("--reason", required=True, help="Reason this topic directory was selected or created.")
    parser.add_argument("--visual-path", help="Optional generated visual summary image to move/link.")
    parser.add_argument("--date", default=_dt.date.today().isoformat(), help="Entry date, default: today.")
    parser.add_argument("--no-chapter-title", action="store_true", help="Use --title as-is instead of adding the next 第N章 prefix.")
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
    display_title = args.title.strip() if args.no_chapter_title else chapter_title(args.title, archive_root, topic_title)
    readme_path = archive_root / "README.md"
    entry = build_entry(
        readme_path=readme_path,
        archived_path=archived_path,
        title=display_title,
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
        "title": display_title,
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
