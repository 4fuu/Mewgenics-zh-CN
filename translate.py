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
            if not row[zh_idx].strip():
                row[zh_idx] = done_keys[full_key] if isinstance(done_keys, dict) else ""
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
            print(f"\nApply-only mode: wrote existing translations to CSV files.")
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


def wrap_text(text: str, max_width: int = 40) -> str:
    lines = text.split("\n")
    result = []
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

        buf = []
        w = 0
        punct_pos = -1
        punct_w = 0
        out_lines = []

        for tok_type, tok_val in tokens:
            if tok_type == "tag":
                buf.append(tok_val)
                continue
            for c in tok_val:
                cw = char_width(c)
                if w + cw > max_width and w > 0:
                    if punct_pos >= 0:
                        out_lines.append("".join(buf[: punct_pos + 1]))
                        buf = buf[punct_pos + 1 :]
                        w = w - punct_w
                        punct_pos = -1
                        punct_w = 0
                    # else: no good break point, keep going (don't hard-break mid-word)
                buf.append(c)
                w += cw
                if c in _BREAK_AFTER_PUNCT:
                    punct_pos = len(buf) - 1
                    punct_w = w

        if buf:
            out_lines.append("".join(buf))
        result.append("\n".join(out_lines))
    return "\n".join(result)


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

    modified_count = 0
    examples = []

    for key, value in list(progress.items()):
        if files:
            csv_file = key.split("::")[0]
            if csv_file not in files:
                continue
        wrapped = wrap_text(value, max_width)
        if wrapped != value:
            modified_count += 1
            if len(examples) < 5:
                examples.append((key, value, wrapped))
            if not dry_run:
                progress[key] = wrapped

    if dry_run:
        print(f"Dry run: {modified_count} entries would be modified (max_width={max_width})")
        for key, old, new in examples:
            print(f"\n  [{key}]")
            print(f"    Before: {old!r}")
            print(f"    After:  {new!r}")
        return

    if modified_count == 0:
        print("No entries need wrapping.")
        return

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
