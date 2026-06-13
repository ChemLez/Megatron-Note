---
name: archive-notes
description: Archive files or folders into a notes library, update existing archived notes, choose or maintain topic folders from note content, update the archive README, and generate or refresh a default AI visual summary. Use when Codex is asked to archive, organize, file, move, classify, update, revise, maintain, or summarize notes/documents, especially into Megatron-Note or a user-specified notes archive.
---

# Archive Notes

## Overview

Archive a file or folder into a notes library, update an existing archived note in place, update the archive README, and create or refresh a visual summary by default. If the user does not specify an archive destination, use `/Users/liz/code/ai_infra/Megatron-Note`; if the user specifies a root or target directory, use the user's destination instead.

## Action Selection

First decide which action the user is asking for:

- New archive: the user asks to archive, file, move, classify, organize, or import a source file/folder that is not yet in the archive.
- Existing document update: the user asks to update, revise, expand, correct, rename, refresh, add a summary/visual, or maintain a note/document that is already in the archive.
- README-only maintenance: the user asks to repair, reindex, rename an entry, update summaries/reasons/visual links, or otherwise maintain archive metadata without changing the article body.

If the request could mean either creating a new archive entry or updating an existing one, inspect the named paths first. If ambiguity remains and moving a file would be risky, ask a concise clarification before editing or moving anything.

## New Archive Workflow

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
8. Generate a visual summary by default unless the user explicitly says not to generate one. Use the `imagegen` skill / built-in image generation path for a project-bound PNG/JPG, then save it in the same archive directory as the note, add its path to README, and append the image link to the end of the archived article.
9. Report the archived path, README path, final chapter title, visual summary path, article path that received the visual link if any, and verification results. After all implementation or archive work is complete, ask whether the user wants to push the resulting changes to remote GitHub; do not push without explicit confirmation.

## Existing Document Update Workflow

Use this workflow when the target note/document already lives inside the archive root or when the user explicitly asks to update an existing archived document.

1. Check the Git working tree before editing:
   - If the archive root is inside a Git repository, run `git status --short` from that repository root before changing files.
   - If the target document, README, or related visual files already have changes, inspect the relevant `git diff` before editing.
   - Treat existing changes as user work. Do not overwrite, revert, or normalize them unless the user explicitly asks.
   - If existing changes directly conflict with the requested update and cannot be merged safely, ask a concise clarification before editing.
2. Resolve the target document, folder, or README entry:
   - If the user gives a direct file/folder path, use that target.
   - If the user names a chapter title, topic, keyword, or README entry, search the archive root README and topic directories to locate the best match.
   - If multiple plausible targets remain, ask a concise clarification before editing.
3. Inspect the existing content and metadata before editing:
   - Read the target article or a representative set of files for folders.
   - Read the archive root `README.md` and locate the matching topic section and entry when present.
   - Preserve the existing chapter title and topic directory unless the user asks to rename or reclassify it.
4. Update the document in place:
   - Do not move, copy, or create a new archive entry for an existing archived document unless the user explicitly asks.
   - Use normal editing tools for Markdown/text/code notes.
   - For binary office documents such as `.docx`, use the Documents skill when available and render/verify the result according to that skill's workflow.
   - Keep edits narrowly scoped to the requested update, but fix obvious stale cross-references caused by the edit.
5. Synchronize README metadata when the update changes user-facing meaning:
   - Update the README summary if the article's topic, conclusion, or scope changed.
   - Update the README link if a document or folder was explicitly renamed.
   - Update the README title only when the user asks for a rename or the existing title is clearly wrong; keep the original chapter number.
   - Preserve the original README entry date for existing archived documents. Treat that date as the archive/create date, not the last-updated date.
   - Do not add or change an update date unless the user explicitly asks for update-history metadata.
   - Add or refresh the `Visual:` line when a summary visual is generated or replaced.
6. Refresh the visual summary by default when the document's substantive content changes, unless the user explicitly says not to:
   - Generate a new information graphic based on the updated content.
   - For existing archived documents, follow the safe refresh process in Visual Summary Guidance: create a candidate image first, keep existing article and README links unchanged until the user confirms the new image, back up the old active image before switching, and remove the old active image only after confirmation when it is no longer referenced.
   - If the update is only a typo, formatting cleanup, link fix, or README-only change, do not regenerate the visual unless the user asks.
7. Verify the update:
   - Confirm the edited document exists and contains the intended changes.
   - Confirm README links resolve relative to the archive root README.
   - Confirm the article visual link resolves when a visual was added or refreshed.
8. Report the updated document path, README path if changed, whether pre-existing Git changes were present, visual summary path if changed, article path that received the visual link if any, and verification results. After all implementation or update work is complete, ask whether the user wants to push the resulting changes to remote GitHub; do not push without explicit confirmation.

## README-Only Maintenance Workflow

Use this workflow when the user asks to update archive metadata without changing article content.

1. Read the archive root `README.md` and relevant topic directories.
2. Make the requested metadata changes while preserving existing chapter numbers and links unless the user explicitly asks to rename or reorder them.
3. Keep topic headings alphabetical when practical and entries newest first within each topic.
4. Verify all changed relative links resolve.
5. Report the README path, changed entries, and verification results. Ask whether the user wants to push to remote GitHub; do not push without explicit confirmation.

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
- Treat each entry date as the archive/create date. For existing document updates, preserve this date unless the user explicitly asks to change it.
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

When a visual summary exists, link it at the end of the archived article:
- For a Markdown file, append a `## 总结图` section with `![总结图](relative/path/to/image)`.
- For an archived folder, append to the first available main Markdown file in this order: `README.md`, `readme.md`, `index.md`, then the first Markdown file found.
- If no Markdown article exists, keep the README visual link only and report that the article link was skipped.
- The script uses an `archive-notes-summary-visual` marker so a later run can replace the previous visual-link block instead of appending duplicates.

For new archive actions, the generated visual can be linked immediately by the archive script because no old active visual exists yet.

For existing archived document updates, use a confirmation-based refresh process:
- Resolve the current active visual before generating a replacement:
  - First read the article's `archive-notes-summary-visual` block and resolve its Markdown image path relative to the article.
  - If the article has no visual block, read the matching README entry's `Visual:` link and resolve it relative to the archive root README.
  - If the article visual and README visual disagree, treat the article visual as the active image and report that the README needs synchronization.
- Save the newly generated image as a candidate in the same topic directory without changing article or README links. Use `summary-visual-candidate.png` or a conflict-safe variant such as `summary-visual-candidate-v2.png`.
- Report the candidate image path and ask the user to confirm whether to adopt it. Until the user confirms, leave the article visual block, README `Visual:` line, and old active image unchanged.
- When the user confirms adoption:
  - Create `visual-backups/` in the same topic directory.
  - Copy the old active image into that directory before switching links. Name the backup `<old-stem>-backup-YYYYMMDD-HHMMSS<suffix>`.
  - Confirm that the backup exists before updating article or README links.
  - Update the article visual block to point to the adopted image, replacing the existing `archive-notes-summary-visual` block when present or appending the standard `## 总结图` section otherwise.
  - Update the matching README `Visual:` line to point to the adopted image.
  - Scan the archive root `README.md` and all Markdown files under the archive root for references to the old active image. Remove the old active image from its original location only when it is no longer referenced and the user did not ask to keep it.
  - Keep files in `visual-backups/`; never delete backup images automatically.
- If the user rejects the candidate or asks to regenerate:
  - Do not change article or README links.
  - Keep the old active image in place.
  - Keep the candidate image by default and report its path for comparison, unless the user explicitly asks to delete it.

## Script Usage

Use the bundled script after classification decisions are made for new archive actions only:

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

The script adds the next `第N章：` prefix by default and prints JSON containing the final chapter title, archived path, topic directory, README path, README relative links, and `article_visual_path` when it appended the visual link to a Markdown article. Use `--dry-run` to preview without moving files or writing README. Use `--no-chapter-title` only when the user explicitly asks not to use chapter sequencing.

Do not use `archive_note.py` to update an existing archived document, because it intentionally rejects sources already inside the archive root and performs move-based archiving. For existing document updates, edit the target files in place and update README metadata directly.

## GitHub Push Flow

When all requested implementation, archive, and validation steps are complete:

1. Summarize the changed files and validation results.
2. Ask whether to push to remote GitHub.
3. If the user says yes, inspect git status, branch, and remotes first.
4. Commit only the relevant changes the user confirms.
5. Add this trailer to every commit message:
   ```text
   Co-authored-by: Codex <noreply@openai.com>
   ```
6. Push only after the commit includes the Codex co-author trailer.
7. If the user declines or does not clearly approve, do not push.

## Iteration

This skill is intentionally editable. Future updates may adjust the default archive root, classification strategy, README template, visual summary style, or push flow. After any skill modification, run the validator and at least one minimal script test when script behavior changes.
