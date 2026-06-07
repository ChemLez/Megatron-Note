---
name: archive-notes
description: Archive files or folders into a notes library, choose an existing or new topic folder from the note content, update the archive README, and generate a default AI visual summary. Use when Codex is asked to archive, organize, file, move, classify, or summarize notes/documents, especially into Megatron-Note or a user-specified notes archive.
---

# Archive Notes

## Overview

Archive a file or folder into a notes library, update the archive README, and create a visual summary by default. If the user does not specify an archive destination, use `/Users/liz/code/ai_infra/Megatron-Note`; if the user specifies a root or target directory, use the user's destination instead.

## Workflow

1. Resolve the source path from the user's request. If there are multiple plausible source files and the user did not identify one, ask a concise clarification before moving anything.
2. Resolve the archive destination:
   - If the user names a final target directory, use it directly.
   - If the user names an archive root but not a target topic, classify into a topic directory under that root.
   - If the user does not name a destination, use `/Users/liz/code/ai_infra/Megatron-Note` as the archive root.
3. Inspect the source content before moving it:
   - For text/Markdown/code files, read enough content to identify topic, keywords, technical objects, purpose, and a short summary.
   - For folders, inspect filenames and a representative set of readable files; avoid loading large generated artifacts, caches, binary blobs, or vendored dependencies.
4. Choose the topic directory:
   - Scan existing first-level directories under the archive root and the archive root `README.md` if present.
   - Use an existing directory only when the content clearly matches its topic.
   - If confidence is low, create a new kebab-case topic directory and record the reason in the README entry.
5. Move the source by default. Do not copy unless the user explicitly asks for copying.
6. Create a content-based title core before archiving:
   - Summarize the note into a concise Chinese title that names the main technical topic or learning outcome.
   - Do not include a chapter prefix in the title you pass to the script; pass only the title core, such as `Megatron训练入口与初始化主链`.
   - Let the script add the next `第N章：` prefix by reading the target topic's existing README entries.
7. Run `scripts/archive_note.py` to perform deterministic filesystem and README updates. The model is responsible for classification, title core, summary, and archive reason; the script performs creation, moving, conflict-safe naming, chapter-title sequencing, and README maintenance.
8. Generate a visual summary by default unless the user explicitly says not to generate one. Use the `imagegen` skill / built-in image generation path for a project-bound PNG/JPG, then save it in the same archive directory as the note and add its path to README.
9. Report the archived path, README path, final chapter title, visual summary path if generated, and verification results. After all implementation or archive work is complete, ask whether the user wants to push the resulting changes to remote GitHub; do not push without explicit confirmation.

## README Format

Maintain the archive root `README.md` as a topic directory index. If it does not exist, create it with:

```markdown
# <Archive Root Name>

## Topics

### <Topic Name>

- <YYYY-MM-DD> [第N章：<Content-Based Title>](relative/path) - <one-sentence summary>
  - Reason: <why this topic directory was chosen or created>
  - Visual: [summary image](relative/path)
```

Rules:
- Keep topic headings in alphabetical order when practical.
- Add new entries under the chosen topic, newest first.
- Titles must continue the chapter sequence within the selected topic: `第一章：...`, `第二章：...`, `第三章：...`.
- If a README already contains chapter entries for that topic, use the next chapter number under that topic; new topics start at `第一章`.
- Omit `Visual:` only when the user opted out or generation failed.
- Keep summaries concise and specific to the archived note.

## Visual Summary Guidance

Create an AI-generated information graphic, not a decorative image. Prefer a clean educational diagram that captures the note's learning route, call chain, concept map, key conclusions, or operational flow.

Use a prompt shaped from the note summary, for example:

```text
Create a clean educational information graphic summarizing this technical note.
Subject: <topic>
Key points: <3-6 concise bullets>
Structure: concept map or learning-route diagram with readable labels.
Style: polished Codex/GPT-style technical explainer, light background, crisp typography, restrained colors.
Avoid: decorative-only imagery, fictional details, unreadable dense text, watermarks.
```

After generating the image, save it in the selected archive directory. Use `summary-visual.png` or a conflict-safe variant such as `summary-visual-v2.png`.

## Script Usage

Use the bundled script after classification decisions are made:

```bash
python /Users/liz/.codex/skills/archive-notes/scripts/archive_note.py \
  --source <source-path> \
  --archive-root <archive-root> \
  --topic-dir <topic-dir-name-or-path> \
  --title "<content-based-title-core>" \
  --summary "<one-sentence summary>" \
  --reason "<why this directory was selected or created>" \
  --visual-path <optional-generated-image-path>
```

The script adds the next `第N章：` prefix by default and prints JSON containing the final chapter title, archived path, topic directory, README path, and relative links. Use `--dry-run` to preview without moving files or writing README. Use `--no-chapter-title` only when the user explicitly asks not to use chapter sequencing.

## GitHub Push Flow

When all requested implementation, archive, and validation steps are complete:

1. Summarize the changed files and validation results.
2. Ask whether to push to remote GitHub.
3. If the user says yes, inspect git status, branch, and remotes first.
4. Commit and push only the relevant changes the user confirms.
5. If the user declines or does not clearly approve, do not push.

## Iteration

This skill is intentionally editable. Future updates may adjust the default archive root, classification strategy, README template, visual summary style, or push flow. After any skill modification, run the validator and at least one minimal script test when script behavior changes.
