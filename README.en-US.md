# Mewgenics Simplified Chinese Localization Patch

A community-made Chinese localization patch for Edmund McMillen's cat-breeding roguelike game, **Mewgenics**.

> ⚠️ The game is currently in Early Access; translation content may change with version updates.

## Installation (Players)

### Method 1: One-Click Patch Tool (Recommended)

1. Download `mewpatch.exe` from [Releases](../../releases).
2. Place `mewpatch.exe` into the game directory (`Steam/steamapps/common/Mewgenics/`).
3. **Double-click and run** `mewpatch.exe`. Wait for it to complete automatically (Unpack → Translate → Font Replacement → Pack → Apply).
4. Launch the game and switch the language to Chinese in the settings.

The tool will automatically back up the original `resources.gpak` as `resources.gpak.bak`. To restore the game, simply rename the backup file back to `resources.gpak`.

> You can also use `mewpatch help` in the command line to view all available commands.

### Method 2: Direct Resource Replacement (Deprecated)

1. Download `resources.gpak` from [Releases](../../releases).
2. Back up the original `resources.gpak` in the game directory (rename it to `resources.gpak.bak`).
3. Place the downloaded `resources.gpak` into the game directory.
4. Launch the game and switch the language to Chinese in the settings.

To restore, rename the backup file back to `resources.gpak`.

## Participating in Translation (Contributors)

Translations are stored in `translation_progress.json` in the following format:

```json
{
  "filename::KEY": "Chinese Translation",
  "misc.csv::AREA_NAME_TUTORIAL": "The Path",
  ...
}
```

### How to Contribute

1. Fork this repository.
2. Edit `translation_progress.json` to correct any translations you find inaccurate.
3. Submit a Pull Request, explaining which entries were modified and why.

**Alternatively, submit an issue directly for discussion.**

### Translation Guidelines

- Refer to the `glossary.json` terminology list to maintain consistency.
- Preserve all markup tags; do not translate the content within tags:

| Tag | Meaning |
|------|------|
| `[m:happy]` | Character expression |
| `[s:1.5]` | Text scale |
| `[b]...[/b]` | Bold |
| `[i]...[/i]` | Italic |
| `[w:500]` | Wait (milliseconds) |
| `{catname}` | Dynamic variable (Cat name) |
| `{his}` `{he}` | Pronoun variables |
| `&nbsp;` | Non-breaking space |

- Translations should be natural and fluent, adhering to the reading habits of Chinese gamers.
- If the original text is only punctuation or does not require translation, leave it as is.

## Developer Guide

Clone this repository into the game directory (`Steam/steamapps/common/Mewgenics/`).

### Patch Tool (Go)

The patch tool is a Go program compiled into a single standalone binary, `mewpatch.exe`. Translation data (`translation_progress.json`), the Chinese font (`MaoKenZhuYuanTi-MaokenZhuyuanTi-2.ttf`), and a backup of the original game font (`unicodefont.swf.bak`, used as CJK fallback) are embedded into the binary via `go:embed`, so users do not need extra files.

#### Compilation

Requires [Go](https://go.dev/) 1.25+:

```bash
go mod tidy                        # Resolve dependencies
go build -o mewpatch.exe ./cmd/    # Compile
```

> ⚠️ **Compilation Prerequisite**: The repository does not include `unicodefont.swf.bak` (original game resource, large size and copyright reasons).
> Before compiling, manually copy the following from your game directory:
> 1. Use `mewpatch extract` to unpack the game's `resources.gpak`.
> 2. Copy `extracted/swfs/unicodefont.swf` to the root of this repository as `unicodefont.swf.bak`.

#### Commands

| Command | Description |
|------|------|
| `mewpatch patch` | One-click execution of all steps (Unpack → Translate → Font → Pack → Replace) |
| `mewpatch extract` | Unpack `resources.gpak` to `extracted/` |
| `mewpatch apply-translations` | Apply embedded translation data to CSV files |
| `mewpatch replace-font` | Replace the CJK font in `unicodefont.swf` |
| `mewpatch pack` | Repack into `resources_patched.gpak` |
| `mewpatch apply` | Replace `resources.gpak` with the patched file (automatic backup) |
| `mewpatch info` | View GPAK file information |
| `mewpatch help` | Show help |

> Running by double-clicking defaults to the `patch` command. Wait and press Enter to exit upon completion.

#### Source Structure

| File | Description |
|------|------|
| `embed.go` | Uses `go:embed` to include translation data and font files (package `mewpatch`) |
| `cmd/main.go` | CLI entry point and `patch` one-click workflow |
| `cmd/gpak.go` | GPAK format parsing, unpacking, and packing |
| `cmd/translate.go` | Reads embedded translations and applies them to the `zh` column of CSVs |
| `cmd/font.go` | TTF → SWF glyph conversion; replaces `unicodefont.swf` |

### Translation Workflow (Python)

Install [uv](https://docs.astral.sh/uv/), then follow this process.

### Complete Workflow

```bash
uv sync  # Initialize project and dependencies
uv run main.py extract        # 1. Unpack resource files
uv run main.py add-zh-column  # 2. Add 'zh' column to CSVs
uv run main.py translate      # 3. AI auto-translation (API Key required)
uv run main.py wrap           # 4. Auto-wrap text (by display width)
uv run main.py pack           # 5. Repack resources
uv run main.py apply          # 6. Apply patch
```

### Detailed Command Guide

#### `extract` — Unpack Resource Files

Unpacks all 18,524 files from `resources.gpak` into the `extracted/` directory (~4.7 GB). It also generates two metadata files for use by `pack`:

- `extracted/__gpak_header.bin` — GPAK file header (4 bytes)
- `extracted/__gpak_index.txt` — File list (preserving original order)

```bash
uv run main.py extract
```

#### `add-zh-column` — Add Chinese Column

Appends an empty `zh` column to the end of every row for all CSV files under `extracted/data/text/`. Skips if the `zh` column already exists. For `additions.csv`, it also automatically sets language metadata (`CURRENT_LANGUAGE_NAME` → `中文`).

```bash
uv run main.py add-zh-column
```

#### `translate` — AI Translation

Calls Tongyi Qianwen (qwen3-max) to translate English text to Chinese and writes it to the `zh` column of the CSVs. Translation progress is saved in `translation_progress.json`, allowing resumes after interruptions. It uses 20-thread concurrent API calls and refers to `glossary.json` to ensure consistent terminology.

An API Key must be set first:

```bash
export DASHSCOPE_API_KEY="your-api-key"        # Linux/macOS
set DASHSCOPE_API_KEY=your-api-key              # Windows CMD
$env:DASHSCOPE_API_KEY="your-api-key"           # PowerShell
```

```bash
uv run main.py translate                           # Translate all
uv run main.py translate --dry                     # Check progress only, no translation
uv run main.py translate --apply-only              # Apply existing translations only, no AI calls
uv run main.py translate --files events.csv        # Translate specific files only
uv run main.py translate --batch-size 100          # 100 entries per batch
```

| Parameter | Default | Description |
|------|--------|------|
| `--dry` | — | Only show progress stats, does not call API |
| `--apply-only` | — | Only write existing translations to CSV, does not call AI |
| `--files FILE [FILE ...]` | All 19 CSVs | Only translate specified CSV files |
| `--batch-size N` | 50 | Number of text entries per API call |

#### `wrap` — Auto-Wrap

Automatically inserts line breaks into translated text based on display width. Chinese characters count as 2, English/Numbers count as 1, and tags (`[...]`, `{...}`) count as 0. It prioritizes breaking after Chinese punctuation (periods, commas, etc.) to avoid splitting words.

```bash
uv run main.py wrap                        # Default wrap width: 40
uv run main.py wrap --max-width 50         # Specify maximum display width
uv run main.py wrap --dry                  # View entries that would be modified
uv run main.py wrap --files items.csv      # Process specific files only
```

| Parameter | Default | Description |
|------|--------|------|
| `--max-width N` | 40 | Max display width per line (CN=2, EN=1) |
| `--files FILE [FILE ...]` | All | Only process specified CSV files |
| `--dry` | — | Only show modified entries, does not write to file |

#### `pack` — Repack

Repacks the `extracted/` directory back into `resources_patched.gpak` following the original index order. Files that were not unpacked are read from the original `resources.gpak`.

```bash
uv run main.py pack
```

#### `apply` — Apply Patch

Replaces `resources.gpak` with `resources_patched.gpak`. On first execution, it automatically backs up the original file as `resources.gpak.bak`.

```bash
uv run main.py apply
```

#### `info` — View GPAK Info

Displays the number of files, size, file type distribution, and CSV file list of `resources.gpak`.

```bash
uv run main.py info
```

### Replacing the Chinese Font

The game includes `unicodefont.swf` as a CJK fallback font. If you find the default font unattractive, you can replace it:

```bash
uv run replace_unicode_font.py                    # Replace with default TTF
uv run replace_unicode_font.py --font other.ttf   # Replace with specified TTF
uv run replace_unicode_font.py --restore           # Restore original font
```

| Parameter | Default | Description |
|------|--------|------|
| `--font FILE` | `MaoKenZhuYuanTi-MaokenZhuyuanTi-2.ttf` | The TTF font file to embed |
| `--dry` | — | Only show conversion info, does not write to file |
| `--restore` | — | Restore original font from backup |

After replacing, you must `pack` and `apply` again for changes to take effect in-game.

## Target Localization Files

> **Starting from 2026/4 version**: The game merges all text into a single `data/text/combined.csv` (~6.4 MB, UTF-8 BOM).
> The file uses `// filename.csv` comment lines as section separators. `translation_progress.json` still uses `filename::KEY` indexing, and the tools will automatically identify sections.

The `combined.csv` contains 19 sections:

| Section | Content |
|---------|------|
| `events.csv` | Random event text (combat, exploration, plot events) |
| `npc_dialog.csv` | NPC dialogue |
| `abilities.csv` | Active skill names and effect descriptions |
| `items.csv` | Item names and descriptions |
| `passives.csv` | Passive skill names and effect descriptions |
| `units.csv` | Unit/character names and descriptions |
| `keyword_tooltips.csv` | Tooltip explanations for game keywords |
| `cutscene_text.csv` | Cutscene text |
| `furniture.csv` | Furniture names and descriptions |
| `misc.csv` | Miscellaneous text (UI, place names, system prompts, etc.) |
| `mutations.csv` | Cat mutation names and effect descriptions |
| `progression.csv` | Game progression (unlock notifications, achievements, etc.) |
| `enemy_abilities.csv` | Enemy skill names and descriptions |
| `additions.csv` | Additional text (including language metadata) |
| `weather.csv` | Weather names and descriptions |
| `teamnames.csv` | Team names |
| `additions2.csv` | Additional text 2 |
| `pronouns.csv` | Pronoun system (dynamic character gender replacements) |
| `additions3.csv` | Additional text 3 |

Run `uv run main.py translate --dry` to check the completion percentage of each section.

## Project Files

| File | Description |
|------|------|
| `embed.go` | Go embed for translation data and fonts |
| `cmd/main.go` | Go patch tool CLI entry point |
| `cmd/gpak.go` | GPAK format parsing, unpacking, and packing |
| `cmd/translate.go` | Applies embedded translations to CSV |
| `cmd/font.go` | TTF → SWF glyph conversion |
| `main.py` | Python CLI entry point (unpack/pack/apply patch) |
| `translate.py` | AI translation logic (batch translation, glossary extraction, progress management) |
| `ai.py` | AI API wrapper (Tongyi Qianwen qwen3-max) |
| `glossary.json` | Terminology list (English → Chinese mapping, enforced during translation) |
| `translation_progress.json` | Translation data (`filename::KEY` → Chinese translation, core file for community collaboration) |
| `replace_unicode_font.py` | Replaces CJK fallback font in `unicodefont.swf` |

## Dependencies

**Patch Tool (Compilation):**
- [Go](https://go.dev/) >= 1.25
- `golang.org/x/image` (TTF font parsing)

**Translation Workflow (Development):**
- Python >= 3.14
- [uv](https://docs.astral.sh/uv/)
- `openai` (Required for AI translation)
- `fonttools` (Required for font replacement)

## Other Notes

- If you need to run commands temporarily, just use `uv run python`.

## License

This project is a community localization tool and does not contain any original game resource files.
