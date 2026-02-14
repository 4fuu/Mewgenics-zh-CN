"""
Translation module for Mewgenics Chinese localization.

Reads CSV files, extracts English text, calls AI for translation,
and writes results back to the zh column.
"""

import csv
import os
import json
import time
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


TEXT_DIR = "extracted/data/text"
PROGRESS_FILE = "translation_progress.json"
GLOSSARY_FILE = "glossary.json"

CSV_FILES = [
    "misc.csv",
    "additions3.csv",
    "additions2.csv",
    "additions.csv",
    "pronouns.csv",
    "weather.csv",
    "teamnames.csv",
    "progression.csv",
    "keyword_tooltips.csv",
    "cutscene_text.csv",
    "furniture.csv",
    "mutations.csv",
    "enemy_abilities.csv",
    "units.csv",
    "passives.csv",
    "items.csv",
    "events.csv",
    "abilities.csv",
    "npc_dialog.csv",
]

FILE_CONTEXT = {
    "misc.csv": "杂项文本，包含UI界面、地名、系统提示等",
    "additions.csv": "追加文本",
    "additions2.csv": "追加文本2",
    "additions3.csv": "追加文本3",
    "pronouns.csv": "代词系统，用于动态替换角色性别代词",
    "weather.csv": "天气名称和描述",
    "teamnames.csv": "队伍名称",
    "progression.csv": "游戏进度相关文本，如解锁提示、成就等",
    "keyword_tooltips.csv": "游戏关键词的工具提示说明",
    "cutscene_text.csv": "过场动画文本",
    "furniture.csv": "家具名称和描述",
    "mutations.csv": "猫咪变异名称和效果描述",
    "enemy_abilities.csv": "敌人技能名称和描述",
    "units.csv": "单位/角色名称和描述",
    "passives.csv": "被动技能名称和效果描述",
    "items.csv": "物品名称和描述",
    "events.csv": "随机事件文本，包含战斗、探索、剧情事件等",
    "abilities.csv": "主动技能名称和效果描述",
    "npc_dialog.csv": "NPC对话文本",
}

ENTRY_SEP = "⟨SEP⟩"


def load_glossary() -> dict[str, str]:
    if os.path.exists(GLOSSARY_FILE):
        with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_glossary(glossary: dict[str, str]):
    with open(GLOSSARY_FILE, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)


def match_glossary(glossary: dict[str, str], entries: list[dict]) -> dict[str, str]:
    combined = "\n".join(e["en"] for e in entries).lower()
    matched = {}
    for en_term, zh_term in glossary.items():
        if en_term.lower() in combined:
            matched[en_term] = zh_term
    return matched


def build_prompt(entries: list[dict], glossary: dict[str, str]) -> str:
    csv_file = entries[0]["file"] if entries else ""
    context = FILE_CONTEXT.get(csv_file, "")

    matched = match_glossary(glossary, entries)

    lines = []
    lines.append("你是游戏《Mewgenics》的中文本地化翻译。")
    lines.append(
        "这是一款由Edmund McMillen制作的猫咪养成roguelike游戏，玩家收集、培育变异猫咪进行战斗。"
    )
    lines.append("")

    if matched:
        lines.append("【术语表】翻译时必须使用以下统一译名：")
        for en, zh in matched.items():
            lines.append(f"  {en} = {zh}")
        lines.append("")

    if context:
        lines.append(f"【当前文件】{csv_file} — {context}")
        lines.append("")

    lines.append("【翻译规则】")
    lines.append(
        "1. 保留所有标记标签，不翻译标签内容，包括但不限于：[m:happy] [s:1.5] [b]...[/b] [w:500] [i]...[/i] {catname} {his} {he} &nbsp; 等"
    )
    lines.append("2. 保留原文中的换行符")
    lines.append("3. 译文要自然流畅，符合中文游戏玩家的阅读习惯")
    lines.append("4. 如果原文只是一个标点或者无需翻译，请原样返回")
    lines.append(
        '5. 每条原文后可能附有"(备注: ...)"，这是开发者的内部注释，仅供你理解语境，严禁将备注内容写入译文'
    )
    lines.append(
        "6. 所有英文单词都必须翻译为中文，不要保留英文原文。唯一的例外是中文游戏语境下玩家习惯直接使用的英文词汇（如Boss、HP、MP、NPC、DPS等常见缩写），这类词可以保留英文"
    )
    lines.append("")
    lines.append(
        f"【输出格式】每条翻译之间用 {ENTRY_SEP} 分隔，严格按顺序输出，不要添加编号、KEY或任何额外内容。只输出译文。"
    )
    lines.append("")
    lines.append(
        f"以下共 {len(entries)} 条待翻译文本，每条格式为 [编号] KEY | 英文原文："
    )
    lines.append("")

    for i, entry in enumerate(entries):
        en_text = entry["en"].replace("\n", "\\n")
        line = f"[{i + 1}] {entry['key']} | {en_text}"
        if entry.get("notes"):
            line += f"  (备注: {entry['notes']})"
        lines.append(line)

    return "\n".join(lines)


def parse_response(response: str, expected_count: int) -> list[str]:
    parts = response.split(ENTRY_SEP)
    results = [p.strip() for p in parts]

    if len(results) == 1 and expected_count > 1:
        results = response.strip().split("\n")
        cleaned = []
        for line in results:
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\[\d+\]\s*", "", line)
            line = re.sub(r"^[A-Z_]+\s*\|\s*", "", line)
            cleaned.append(line)
        results = cleaned

    results = [r.replace("\\n", "\n") for r in results]

    while len(results) < expected_count:
        results.append("")
    return results[:expected_count]


def translate_batch(entries: list[dict]) -> list[str]:
    from ai import completion

    glossary = load_glossary()
    prompt = build_prompt(entries, glossary)

    messages = [{"role": "user", "content": prompt}]
    response = completion(messages)

    assert response, "response is None"

    return parse_response(response, len(entries))


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def read_csv(filepath: str) -> tuple[list[str], list[list[str]]]:
    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def write_csv(filepath: str, header: list[str], rows: list[list[str]]):
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def collect_entries(
    csv_file: str, done_keys: set
) -> tuple[list[str], list[list[str]], list[tuple[int, dict]]]:
    """Read a CSV and collect untranslated entries.

    Returns (header, rows, pending) where pending is a list of
    (row_index, entry_dict) for rows that need translation.
    """
    filepath = os.path.join(TEXT_DIR, csv_file)
    header, rows = read_csv(filepath)

    en_idx = header.index("en")
    notes_idx = header.index("notes") if "notes" in header else -1
    zh_idx = header.index("zh") if "zh" in header else -1

    if zh_idx == -1:
        header.append("zh")
        zh_idx = len(header) - 1
        for row in rows:
            row.append("")

    pending = []
    for i, row in enumerate(rows):
        while len(row) < len(header):
            row.append("")

        key = row[0]
        en_text = row[en_idx] if len(row) > en_idx else ""

        if not en_text.strip():
            continue
        if key.startswith("//"):
            continue

        full_key = f"{csv_file}::{key}"
        if full_key in done_keys:
            translated = done_keys[full_key] if isinstance(done_keys, dict) else ""
            if translated:
                row[zh_idx] = translated
            continue

        if row[zh_idx].strip():
            continue

        entry = {
            "key": key,
            "en": en_text,
            "notes": row[notes_idx] if notes_idx >= 0 and len(row) > notes_idx else "",
            "file": csv_file,
        }
        pending.append((i, entry))

    return header, rows, pending


def run_translate(
    batch_size: int = 50,
    files: list[str] | None = None,
    dry_run: bool = False,
    apply_only: bool = False,
):
    """Main translation loop.

    Args:
        batch_size: Number of entries per translate_batch() call.
        files: List of CSV filenames to translate. None = all files.
        dry_run: If True, only show stats without translating.
        apply_only: If True, only apply existing translations from progress
            file to CSVs without calling AI.
    """
    if not os.path.exists(TEXT_DIR):
        print(f"Error: {TEXT_DIR} not found. Run 'extract' or 'extract-text' first.")
        return

    progress = load_progress()
    target_files = files if files else CSV_FILES

    available = [f for f in target_files if os.path.exists(os.path.join(TEXT_DIR, f))]
    if not available:
        print("No CSV files found.")
        return

    total_pending = 0
    total_done = 0
    file_stats = []

    for csv_file in available:
        header, rows, pending = collect_entries(csv_file, progress)  # type: ignore
        zh_idx = header.index("zh") if "zh" in header else -1
        done_in_file = sum(
            1
            for row in rows
            if zh_idx >= 0
            and len(row) > zh_idx
            and row[zh_idx].strip()
            and not row[0].startswith("//")
            and row[0].strip()
        )
        en_idx = header.index("en")
        translatable = sum(
            1
            for row in rows
            if len(row) > en_idx and row[en_idx].strip() and not row[0].startswith("//")
        )
        if done_in_file > 0:
            write_csv(os.path.join(TEXT_DIR, csv_file), header, rows)
        total_pending += len(pending)
        total_done += done_in_file
        file_stats.append((csv_file, translatable, done_in_file, len(pending)))

    print(f"Translation status: {total_done} done, {total_pending} pending")
    print()
    for csv_file, translatable, done, pending in file_stats:
        bar_len = 20
        pct = done / translatable * 100 if translatable > 0 else 0
        filled = int(bar_len * done / translatable) if translatable > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  {csv_file:30s} {bar} {done:5d}/{translatable:<5d} ({pct:5.1f}%)")

    if dry_run or apply_only:
        if apply_only:
            print("\nApply-only mode: wrote existing translations to CSV files.")
        return

    if total_pending == 0:
        print("\nAll entries are translated.")
        return

    print(
        f"\nTranslating {total_pending} entries (batch size: {batch_size}, 4 threads)..."
    )
    print()

    translated_total = 0
    t_start = time.time()
    lock = Lock()
    has_error = False

    for csv_file in available:
        header, rows, pending = collect_entries(csv_file, progress)  # type: ignore
        if not pending:
            continue

        zh_idx = header.index("zh")
        print(f"[{csv_file}] {len(pending)} entries to translate")

        batches = []
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            batches.append((batch_start, batch))

        file_translated = 0

        def process_batch(batch_info):
            batch_start, batch = batch_info
            batch_entries = [entry for _, entry in batch]
            results = translate_batch(batch_entries)

            if len(results) != len(batch):
                with lock:
                    print(
                        f"  WARNING: translate_batch returned {len(results)} results for {len(batch)} entries"
                    )
                results = results[: len(batch)]
                results.extend([""] * (len(batch) - len(results)))

            return batch_start, batch, results

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(process_batch, b): b for b in batches}

            for future in as_completed(futures):
                if has_error:
                    break

                try:
                    batch_start, batch, results = future.result()
                except Exception as e:
                    with lock:
                        print(f"\n  ERROR in translate_batch(): {e}")
                        print("  Saving progress and stopping.")
                        save_progress(progress)
                        write_csv(os.path.join(TEXT_DIR, csv_file), header, rows)
                        has_error = True
                    break

                with lock:
                    for (row_idx, entry), zh_text in zip(batch, results):
                        if zh_text:
                            rows[row_idx][zh_idx] = zh_text
                            full_key = f"{csv_file}::{entry['key']}"
                            progress[full_key] = zh_text

                    count = sum(1 for r in results if r)
                    file_translated += count
                    translated_total += count
                    elapsed = time.time() - t_start
                    print(
                        f"  [{file_translated}/{len(pending)}] +{count} ({elapsed:.0f}s)"
                    )

        if has_error:
            return

        write_csv(os.path.join(TEXT_DIR, csv_file), header, rows)
        save_progress(progress)

    elapsed = time.time() - t_start
    print(f"\nDone. Translated {translated_total} entries in {elapsed:.1f}s")
    print(f"Progress saved to {PROGRESS_FILE}")


def char_width(c: str) -> int:
    eaw = unicodedata.east_asian_width(c)
    return 2 if eaw in ("W", "F") else 1


def display_width(text: str) -> int:
    width = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c in ("[", "{"):
            close = "]" if c == "[" else "}"
            j = text.find(close, i + 1)
            if j != -1:
                i = j + 1
                continue
        if c == "&":
            m = re.match(r"&[a-zA-Z]+;", text[i:])
            if m:
                i += len(m.group())
                continue
        width += char_width(c)
        i += 1
    return width


_TAG_RE = re.compile(r"\[[^\]]*\]|\{[^}]*\}|&[a-zA-Z]+;")
_BREAK_AFTER_PUNCT = set("。！？，、；：）》」』")


def wrap_text(text: str, max_width: int = 40) -> tuple[str, bool]:
    """Wrap *text* so each line fits within *max_width*.

    Returns ``(wrapped_text, overflow)`` where *overflow* is ``True`` when at
    least one line could not be broken because no punctuation break-point was
    found within the width limit.
    """
    lines = text.split("\n")
    result = []
    overflow = False
    for line in lines:
        if display_width(line) <= max_width:
            result.append(line)
            continue
        tokens = []
        last = 0
        for m in _TAG_RE.finditer(line):
            if m.start() > last:
                tokens.append(("text", line[last : m.start()]))
            tokens.append(("tag", m.group()))
            last = m.end()
        if last < len(line):
            tokens.append(("text", line[last:]))

        buf: list[str] = []
        w = 0
        punct_pos = -1
        punct_w = 0
        out_lines: list[str] = []

        def _flush_at_punct():
            nonlocal buf, w, punct_pos, punct_w
            out_lines.append("".join(buf[: punct_pos + 1]))
            buf = buf[punct_pos + 1 :]
            # recalculate width of remaining buffer
            w = 0
            for item in buf:
                if len(item) > 1:  # tag
                    pass
                else:
                    w += char_width(item)
            punct_pos = -1
            punct_w = 0

        for tok_type, tok_val in tokens:
            if tok_type == "tag":
                buf.append(tok_val)
                continue
            for c in tok_val:
                cw = char_width(c)
                if w + cw > max_width and w > 0:
                    if punct_pos >= 0:
                        _flush_at_punct()
                    else:
                        overflow = True
                buf.append(c)
                w += cw
                if c in _BREAK_AFTER_PUNCT:
                    punct_pos = len(buf) - 1
                    punct_w = w

        if buf:
            out_lines.append("".join(buf))
        result.append("\n".join(out_lines))
    return "\n".join(result), overflow


OVERFLOW_FILE = "wrap_overflow.json"


def run_wrap(
    max_width: int = 40,
    files: list[str] | None = None,
    dry_run: bool = False,
):
    """Auto-wrap long translated text lines.

    Args:
        max_width: Maximum display width per line.
        files: List of CSV filenames to process. None = all files.
        dry_run: If True, only show what would change.
    """
    progress = load_progress()
    if not progress:
        print("No translations found in progress file.")
        return

    # Import manually edited overflow entries back into progress
    if os.path.exists(OVERFLOW_FILE):
        with open(OVERFLOW_FILE, "r", encoding="utf-8") as f:
            overflow_edits: dict[str, str] = json.load(f)
        imported = 0
        for key, value in overflow_edits.items():
            if key in progress and progress[key] != value:
                progress[key] = value
                imported += 1
        if imported > 0:
            save_progress(progress)
            print(f"Imported {imported} entries from {OVERFLOW_FILE}")
        os.remove(OVERFLOW_FILE)
        print(f"Removed {OVERFLOW_FILE}")

    modified_count = 0
    overflow_entries: dict[str, str] = {}
    examples = []

    for key, value in list(progress.items()):
        if files:
            csv_file = key.split("::")[0]
            if csv_file not in files:
                continue
        wrapped, overflow = wrap_text(value, max_width)
        if overflow:
            overflow_entries[key] = value
        if wrapped != value:
            modified_count += 1
            if len(examples) < 5:
                examples.append((key, value, wrapped))
            if not dry_run:
                progress[key] = wrapped

    if dry_run:
        print(
            f"Dry run: {modified_count} entries would be modified (max_width={max_width})"
        )
        for key, old, new in examples:
            print(f"\n  [{key}]")
            print(f"    Before: {old!r}")
            print(f"    After:  {new!r}")
        if overflow_entries:
            print(
                f"\n{len(overflow_entries)} entries overflow (no punctuation break-point)."
            )
        return

    if overflow_entries:
        with open(OVERFLOW_FILE, "w", encoding="utf-8") as f:
            json.dump(overflow_entries, f, ensure_ascii=False, indent=2)
        print(f"{len(overflow_entries)} entries overflow — saved to {OVERFLOW_FILE}")
        print(
            "  Add punctuation to these entries in translation_progress.json, then re-run wrap."
        )

    if modified_count == 0 and not overflow_entries:
        print("No entries need wrapping.")
        return

    if modified_count > 0:
        save_progress(progress)
        print(f"Updated {modified_count} entries in {PROGRESS_FILE}")

    # Apply wrapped text to CSVs
    target_files = files if files else CSV_FILES
    available = [f for f in target_files if os.path.exists(os.path.join(TEXT_DIR, f))]

    for csv_file in available:
        filepath = os.path.join(TEXT_DIR, csv_file)
        header, rows = read_csv(filepath)
        if "zh" not in header:
            continue
        zh_idx = header.index("zh")
        changed = False
        for row in rows:
            while len(row) < len(header):
                row.append("")
            key = row[0]
            full_key = f"{csv_file}::{key}"
            if full_key in progress and row[zh_idx].strip():
                if row[zh_idx] != progress[full_key]:
                    row[zh_idx] = progress[full_key]
                    changed = True
        if changed:
            write_csv(filepath, header, rows)
            print(f"  Updated {csv_file}")

    print("Done.")


def run_auto_wrap(max_width: int = 40, batch_size: int = 30):
    """Use AI to automatically add line breaks to overflow entries.

    Reads wrap_overflow.json, sends entries that still need line breaks
    to AI in batches, and writes the results back.
    """
    from ai import completion

    if not os.path.exists(OVERFLOW_FILE):
        print(f"{OVERFLOW_FILE} not found. Run 'wrap' first.")
        return

    with open(OVERFLOW_FILE, "r", encoding="utf-8") as f:
        overflow: dict[str, str] = json.load(f)

    # Filter entries that still have at least one line exceeding max_width
    needs_wrap = {}
    for k, v in overflow.items():
        # Skip non-Chinese entries
        if not any("\u4e00" <= c <= "\u9fff" for c in v):
            continue
        needs_wrap[k] = v

    if not needs_wrap:
        print("No entries need auto-wrapping.")
        return

    print(f"Auto-wrapping {len(needs_wrap)} entries (max_width={max_width})...")

    keys = list(needs_wrap.keys())
    updated = 0
    skipped = 0

    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i : i + batch_size]
        batch_values = [needs_wrap[k] for k in batch_keys]

        # Build prompt: mark which lines overflow
        prompt_entries = []
        for k, v in zip(batch_keys, batch_values):
            entry_lines = []
            for line in v.split("\n"):
                w = display_width(line)
                if w > max_width:
                    entry_lines.append(f"{line}    ←此行宽度{w}，需要断行")
                else:
                    entry_lines.append(line)
            prompt_entries.append((k, "\n".join(entry_lines)))

        lines = []
        lines.append(
            f"你是一个游戏文本排版助手。以下文本中有些行的显示宽度超过了{max_width}个单位"
            f"（中文字符=2单位，英文/数字/标点=1单位），我已用←标记了超宽行。"
        )
        lines.append(
            f"你需要将这些超宽行拆分成多行，使每行宽度不超过{max_width}个单位。"
            "未标记的行不要改动。"
        )
        lines.append("")
        lines.append("规则：")
        lines.append("1. 只修改标记了←的超宽行，在语义自然的位置插入换行")
        lines.append("2. 没有标点可断的长句，直接在词语之间断行即可")
        lines.append(f"3. 断行后每行宽度必须≤{max_width}（中文字符=2，其他=1）")
        lines.append(
            "4. [img:xxx]、[b]...[/b]、{xxx} 等标记标签宽度为0，不要拆开"
        )
        lines.append("5. 不要修改文字内容，只添加换行")
        lines.append("6. 输出中不要包含←标记")
        lines.append("")
        lines.append(
            f"【输出格式】每条结果之间用一行 {ENTRY_SEP} 分隔（单独占一行），"
            "严格按顺序输出，只输出处理后的完整文本。"
        )
        lines.append("")
        lines.append(
            f"以下共 {len(prompt_entries)} 条文本（用 ---- 分隔每条）："
        )
        lines.append("")

        for j, (k, v) in enumerate(prompt_entries):
            lines.append(f"[{j + 1}] {k}")
            lines.append(v)
            if j < len(prompt_entries) - 1:
                lines.append("----")
            lines.append("")

        prompt = "\n".join(lines)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = completion(messages)
        except Exception as e:
            print(f"  ERROR at batch {i}: {e}")
            print("  Stopping.")
            return

        assert response, "response is None"

        # Parse by ENTRY_SEP on its own line
        parts = re.split(rf"\s*{re.escape(ENTRY_SEP)}\s*", response.strip())
        # Clean up: remove leading [N] or key prefixes
        results = []
        for p in parts:
            p = p.strip()
            p = re.sub(r"^\[\d+\]\s*", "", p)
            p = re.sub(r"^[A-Za-z_]+::[A-Za-z_]+\s*", "", p)
            results.append(p)

        while len(results) < len(batch_keys):
            results.append("")
        results = results[: len(batch_keys)]

        for k, new_value in zip(batch_keys, results):
            if not new_value:
                skipped += 1
                continue
            overflow[k] = new_value
            max_line = max(display_width(line) for line in new_value.split("\n"))
            if max_line <= max_width:
                updated += 1
            else:
                skipped += 1

        done = min(i + batch_size, len(keys))
        print(f"  [{done}/{len(keys)}] 已处理，更新 {updated} 条，跳过 {skipped} 条")

    # Save results back to overflow file for manual review
    with open(OVERFLOW_FILE, "w", encoding="utf-8") as f:
        json.dump(overflow, f, ensure_ascii=False, indent=2)

    print(f"✓ AI 自动换行完成：更新 {updated} 条，跳过 {skipped} 条。")
    print(f"  结果已写入 {OVERFLOW_FILE}，请检查后运行 wrap 命令应用。")


# --- Check translations for quality issues ---

# English words that are acceptable in Chinese translations
# (abbreviations, proper nouns, game terms, etc.)
ACCEPTABLE_ENGLISH = {
    # Common gaming/tech abbreviations
    "HP",
    "MP",
    "AP",
    "DPS",
    "SP",
    "EX",
    "XP",
    "DLC",
    "RPG",
    "AOE",
    "AoE",
    "NPC",
    "BGM",
    "SFX",
    "UI",
    "VIP",
    "AI",
    "MSAA",
    "VHS",
    "DVD",
    "UFO",
    "TNT",
    "USA",
    "AAA",
    "MC",
    "DJ",
    "TV",
    "PC",
    "CD",
    "VS",
    "vs",
    "OK",
    "ok",
    "DNA",
    "DIE",
    "OBEY",
    "STOP",
    "DUMB",
    "PvP",
    "PvE",
    # Roman numerals
    "II",
    "III",
    "IV",
    "VI",
    "VII",
    "VIII",
    "IX",
    "XI",
    "XII",
}

# Pattern to match trailing notes/remarks added by translators
NOTE_PATTERNS = [
    # （备注: ...）or (备注: ...)
    r"\s*[（(]\s*备注\s*[:：].*?[)）]\s*$",
    # （备注: ...  without closing bracket (end of string)
    r"\s*[（(]\s*备注\s*[:：].*$",
    # (备注: updated ✔️) style markers
    r"\s*[（(]\s*备注\s*[:：].*?✔.*?[)）]?\s*$",
]


def _strip_tags(text: str) -> str:
    """Remove markup tags and variables to isolate actual text."""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\{.*?\}", "", text)
    text = re.sub(r"&nbsp;", "", text)
    return text


def _find_mixed_english(text: str) -> list[str]:
    """Find English words (2+ letters) in Chinese text, excluding acceptable ones."""
    cleaned = _strip_tags(text)
    # Also strip note sections before checking
    for pattern in NOTE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)
    words = re.findall(r"[A-Za-z]{2,}", cleaned)
    return [w for w in words if w not in ACCEPTABLE_ENGLISH]


def _has_notes(text: str) -> re.Match | None:
    """Check if text contains translator notes/remarks."""
    for pattern in NOTE_PATTERNS:
        m = re.search(pattern, text, flags=re.DOTALL | re.MULTILINE)
        if m:
            return m
    return None


def _remove_notes(text: str) -> str:
    """Remove trailing translator notes from text."""
    for pattern in NOTE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)
    return text.rstrip()


def _apply_progress_to_csvs(progress: dict, files: list[str] | None = None):
    """Write progress values back to CSV files."""
    target_files = files if files else CSV_FILES
    available = [f for f in target_files if os.path.exists(os.path.join(TEXT_DIR, f))]
    for csv_file in available:
        filepath = os.path.join(TEXT_DIR, csv_file)
        header, rows = read_csv(filepath)
        if "zh" not in header:
            continue
        zh_idx = header.index("zh")
        changed = False
        for row in rows:
            while len(row) < len(header):
                row.append("")
            full_key = f"{csv_file}::{row[0]}"
            if full_key in progress and row[zh_idx].strip():
                if row[zh_idx] != progress[full_key]:
                    row[zh_idx] = progress[full_key]
                    changed = True
        if changed:
            write_csv(filepath, header, rows)
            print(f"  Updated {csv_file}")


def _build_fix_mixed_prompt(
    entries: list[tuple[str, str, list[str]]], glossary: dict[str, str]
) -> str:
    """Build a prompt to fix mixed Chinese-English translations."""
    lines = []
    lines.append("你是游戏《Mewgenics》的中文本地化校对员。")
    lines.append(
        "以下译文中残留了未翻译的英文单词，请将这些英文单词翻译为中文，修正译文。"
    )
    lines.append("")

    matched = {}
    combined = "\n".join(val for _, val, _ in entries).lower()
    for en_term, zh_term in glossary.items():
        if en_term.lower() in combined:
            matched[en_term] = zh_term

    if matched:
        lines.append("【术语表】翻译时必须使用以下统一译名：")
        for en, zh in matched.items():
            lines.append(f"  {en} = {zh}")
        lines.append("")

    lines.append("【规则】")
    lines.append("1. 只翻译残留的英文单词，不要改动译文的其余部分")
    lines.append(
        "2. 保留所有标记标签不变，包括 [m:happy] [s:1.5] [b]...[/b] {catname} {his} &nbsp; 等"
    )
    lines.append("3. 保留原文中的换行符")
    lines.append("4. 如果某个英文单词是专有名词或缩写，应保持原样不翻译")
    lines.append("")
    lines.append(
        f"【输出格式】每条修正后的译文之间用 {ENTRY_SEP} 分隔，严格按顺序输出，只输出修正后的完整译文。"
    )
    lines.append("")
    lines.append(f"以下共 {len(entries)} 条需要修正的译文：")
    lines.append("")

    for i, (key, value, eng_words) in enumerate(entries):
        val_display = value.replace("\n", "\\n")
        lines.append(f"[{i + 1}] {key}")
        lines.append(f"    当前译文：{val_display}")
        lines.append(f"    残留英文：{', '.join(eng_words)}")
        lines.append("")

    return "\n".join(lines)


def _fix_mixed_entries(
    mixed_entries: list[tuple[str, str, list[str]]],
    progress: dict,
    glossary: dict[str, str],
    files: list[str] | None = None,
    batch_size: int = 30,
):
    """Use AI to fix mixed Chinese-English translations."""
    from ai import completion

    total = len(mixed_entries)
    fixed_count = 0
    t_start = time.time()

    for batch_start in range(0, total, batch_size):
        batch = mixed_entries[batch_start : batch_start + batch_size]
        prompt = _build_fix_mixed_prompt(batch, glossary)

        messages = [{"role": "user", "content": prompt}]
        try:
            response = completion(messages)
        except Exception as e:
            print(f"  ERROR at batch {batch_start}: {e}")
            print("  Saving progress and stopping.")
            save_progress(progress)
            _apply_progress_to_csvs(progress, files)
            return

        assert response, "response is None"
        results = parse_response(response, len(batch))

        for (key, old_value, _), new_value in zip(batch, results):
            if new_value and new_value != old_value:
                progress[key] = new_value
                fixed_count += 1

        elapsed = time.time() - t_start
        done = min(batch_start + batch_size, total)
        print(f"  [{done}/{total}] 已修正 {fixed_count} 条 ({elapsed:.0f}s)")

    save_progress(progress)
    _apply_progress_to_csvs(progress, files)
    print(f"✓ AI 修正了 {fixed_count} 条混合中英文译文。")


def run_check(
    fix: bool = False, fix_mixed: bool = False, files: list[str] | None = None
):
    """Check translations for mixed Chinese-English and stray notes.

    Args:
        fix: If True, auto-remove trailing notes from translations.
        fix_mixed: If True, use AI to fix mixed Chinese-English translations.
        files: List of CSV filenames to check. None = all files.
    """
    progress = load_progress()
    if not progress:
        print("No translations found in progress file.")
        return

    glossary = load_glossary()
    # Glossary values (Chinese terms) are fine, but glossary keys mapped to
    # English proper nouns that appear in translations are also acceptable
    extra_acceptable = set()
    for en_term in glossary:
        # If the glossary keeps the English name as-is, it's acceptable
        if re.match(r"^[A-Za-z]", en_term):
            for word in en_term.split():
                if len(word) >= 2:
                    extra_acceptable.add(word)

    note_entries = []
    mixed_entries = []

    for key, value in sorted(progress.items()):
        if not value or not isinstance(value, str):
            continue
        if files:
            csv_file = key.split("::")[0]
            if csv_file not in files:
                continue
        # Must contain Chinese to be considered a translation
        if not re.search(r"[\u4e00-\u9fff]", value):
            continue

        # Check for notes
        if _has_notes(value):
            note_entries.append((key, value))

        # Check for mixed English (after stripping notes)
        eng_words = _find_mixed_english(value)
        eng_words = [w for w in eng_words if w not in extra_acceptable]
        if eng_words:
            mixed_entries.append((key, value, eng_words))

    # Report
    print(f"Checked {len(progress)} translations.\n")

    if note_entries:
        print(f"⚠ Translator notes found: {len(note_entries)}")
        for key, val in note_entries[:10]:
            short = val.replace("\n", " ")
            if len(short) > 80:
                short = short[:80] + "..."
            print(f"  {key}")
            print(f"    {short}")
        if len(note_entries) > 10:
            print(f"  ... and {len(note_entries) - 10} more")
        print()

    if mixed_entries:
        print(f"⚠ Mixed Chinese-English: {len(mixed_entries)}")
        for key, val, words in mixed_entries[:20]:
            short = val.replace("\n", " ")
            if len(short) > 80:
                short = short[:80] + "..."
            print(f"  {key}")
            print(f"    {short}")
            print(f"    残留英文: {words}")
        if len(mixed_entries) > 20:
            print(f"  ... and {len(mixed_entries) - 20} more")
        print()

    if not note_entries and not mixed_entries:
        print("✓ No issues found.")
        return

    # Fix mode: remove notes
    if fix and note_entries:
        fixed_count = 0
        for key, value in note_entries:
            cleaned = _remove_notes(value)
            if cleaned != value:
                progress[key] = cleaned
                fixed_count += 1
        save_progress(progress)
        print(f"✓ Removed notes from {fixed_count} entries.")
        _apply_progress_to_csvs(progress, files)
    elif fix:
        print("No notes to fix.")

    # Fix mixed Chinese-English via AI
    if fix_mixed and mixed_entries:
        print(
            f"\n🔧 Using AI to fix {len(mixed_entries)} mixed Chinese-English entries..."
        )
        _fix_mixed_entries(mixed_entries, progress, glossary, files)
    elif mixed_entries:
        print(
            f"\n💡 {len(mixed_entries)} entries have mixed English that may need manual review."
        )
        print("   Use --fix-mixed to auto-fix with AI.")
        print("   Add acceptable terms to glossary.json to suppress false positives.")
