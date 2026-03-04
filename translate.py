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
    # 1. 术语定义类：建立核心词汇表
    "keyword_tooltips.csv",  # 游戏关键词定义，最先确立术语
    "misc.csv",  # UI、地名、系统文本，基础词汇
    "pronouns.csv",  # 代词系统（极小，结构性）
    # 2. 核心机制类：使用已确立的关键词
    "mutations.csv",  # 猫咪变异
    "weather.csv",  # 天气效果
    "furniture.csv",  # 家具
    "enemy_abilities.csv",  # 敌人技能
    "passives.csv",  # 被动技能
    "abilities.csv",  # 主动技能
    "items.csv",  # 物品
    # 3. 世界与角色
    "units.csv",  # 单位/角色
    "progression.csv",  # 游戏进度
    "teamnames.csv",  # 队伍名称
    # 4. 补充文本
    "additions.csv",
    "additions2.csv",
    "additions3.csv",
    # 5. 叙事文本（最长，受益于前面积累的全部术语）
    "cutscene_text.csv",  # 过场动画
    "events.csv",  # 随机事件
    "npc_dialog.csv",  # NPC 对话
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


def _extract_prefix(key: str) -> str:
    """Extract grouping prefix from a key to keep related entries together.

    Examples:
        ABILITY_CHAOSSHOT_NAME -> ABILITY_CHAOSSHOT
        ABILITY_CHAOSSHOT_DESC -> ABILITY_CHAOSSHOT
        ABILITY_CHAOSSHOT2_DESC -> ABILITY_CHAOSSHOT
    """
    suffixes = [
        "_NAME",
        "_DESC",
        "_FLAVOR",
        "_TOOLTIP",
        "_EFFECT",
        "_HEADER",
        "_TITLE",
        "_BODY",
        "_TEXT",
        "_SHORT",
        "_LONG",
        "_LABEL",
        "_BTN",
        "_BUTTON",
        "_MSG",
        "_MESSAGE",
        "_INFO",
        "_HINT",
    ]
    result = key
    for suffix in suffixes:
        if result.endswith(suffix):
            result = result[: -len(suffix)]
            break
    # Strip trailing digits to group variants (CHAOSSHOT2 -> CHAOSSHOT)
    # But don't strip if the last segment is entirely digits (e.g. ITEM_123)
    parts = result.rsplit("_", 1)
    if len(parts) == 2 and parts[1] and not parts[1].isdigit():
        stripped = re.sub(r"\d+$", "", parts[1])
        if stripped:
            result = parts[0] + "_" + stripped
    return result


def _build_prefix_batches(
    pending: list[tuple[int, dict]], batch_size: int
) -> list[list[tuple[int, dict]]]:
    """Build batches that keep entries with the same prefix together."""
    from collections import OrderedDict

    groups: OrderedDict[str, list[tuple[int, dict]]] = OrderedDict()
    for item in pending:
        _, entry = item
        prefix = _extract_prefix(entry["key"])
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(item)

    batches: list[list[tuple[int, dict]]] = []
    current_batch: list[tuple[int, dict]] = []

    for _prefix, group in groups.items():
        if current_batch and len(current_batch) + len(group) > batch_size:
            batches.append(current_batch)
            current_batch = []
        if len(group) > batch_size:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            for i in range(0, len(group), batch_size):
                batches.append(group[i : i + batch_size])
        else:
            current_batch.extend(group)

    if current_batch:
        batches.append(current_batch)

    return batches


def _find_reference_translations(entries: list[dict], progress: dict) -> list[dict]:
    """Find already-translated entries with the same prefix as reference."""
    csv_file = entries[0]["file"] if entries else ""
    prefixes = set(_extract_prefix(e["key"]) for e in entries)
    entry_keys = set(f"{e['file']}::{e['key']}" for e in entries)

    references = []
    for full_key, zh_text in progress.items():
        if not zh_text or full_key in entry_keys:
            continue
        parts = full_key.split("::", 1)
        if len(parts) != 2:
            continue
        file_name, key = parts
        if file_name != csv_file:
            continue
        if _extract_prefix(key) in prefixes:
            references.append({"key": key, "zh": zh_text})

    return references[:20]


def build_prompt(
    entries: list[dict],
    glossary: dict[str, str],
    references: list[dict] | None = None,
) -> str:
    csv_file = entries[0]["file"] if entries else ""
    context = FILE_CONTEXT.get(csv_file, "")
    matched = match_glossary(glossary, entries)

    sections = []

    # Role & game context
    sections.append(
        "你是游戏《Mewgenics》的中文本地化翻译。\n"
        "这是一款由Edmund McMillen制作的猫咪养成roguelike游戏，玩家收集、培育变异猫咪进行战斗。\n"
        "游戏简介：根据你的策略，培育猫咪，组建终极喵喵大军，派他们踏上颇具深度和难度的回合制冒险之旅。"
        "抽选能力，获得物品，改变影响数代的遗传特性，享受这款类Rogue策略游戏，"
        "感受《以撒的结合》与《终结将至》作者的创意和设计。\n"
        "===="
    )

    # Glossary
    if matched:
        glossary_lines = ["【术语表】翻译时参考使用以下译名（不强制）："]
        for en, zh in matched.items():
            glossary_lines.append(f"  {en} = {zh}")
        sections.append("\n".join(glossary_lines))

    # Reference translations
    if references:
        ref_lines = ["【参考译文】以下是同系列已翻译条目，请参考以精确理解："]
        for ref in references:
            zh_short = ref["zh"].replace("\n", "\\n")
            ref_lines.append(f"  {ref['key']} → {zh_short}")
        sections.append("\n".join(ref_lines))

    # Current file context
    if context:
        sections.append(f"【当前文件】{csv_file} — {context}")

    # Tag reference — consolidates tag explanations and preservation rules
    sections.append(
        "【标记标签】以下标签必须原样保留，不翻译标签内容，位置可随中文语序调整：\n"
        "  [m:表情] — 角色表情，如 [m:happy]、[m:angry]\n"
        "  [s:数字] — 文字缩放，如 [s:1.5]\n"
        "  [b]...[/b] — 粗体  |  [i]...[/i] — 斜体\n"
        "  [w:数字] — 停顿（毫秒），如 [w:500]\n"
        "  {变量名} — 动态变量，如 {catname}、{his}、{he}\n"
        "  &nbsp; — 不换行空格"
    )

    # Translation rules — grouped by concern
    sections.append(
        "【翻译规则】\n"
        "\n"
        "▸ 格式\n"
        "  - 保留所有标记标签，翻译前后标签数量必须一致\n"
        "  - 无需保留原文换行符，按中文语序重新组织文本和标签位置\n"
        '  - 原文后的"(备注: ...)"是开发者注释，仅供理解语境，严禁写入译文\n'
        "\n"
        "▸ 语言\n"
        "  - 所有英文必须译为中文；例外：中文玩家惯用的缩写（Boss、HP、MP、NPC、DPS等）可保留\n"
        "  - 复合条件句须拆分重组，用'当…时，若…则…'等结构衔接，避免条件堆叠\n"
        '    ✓ "当敌人下次结束移动时，若其处于你的基础攻击范围内，则对其发动攻击。"\n'
        '    ✗ "下次当敌人结束移动并位于你的基础攻击范围内时，攻击它。"\n'
        "\n"
        "▸ 风格\n"
        "  - 完整翻译，译文要易懂、自然流畅，不得直接机翻，可联系上下文补全缺失含义\n"
        "  - 描述文本应精炼简洁：省略可推断的主语（'该技能''你的猫咪'），直接以动词开头\n"
        '    ✓ "对所有敌人造成5点伤害"  ✗ "该技能对所有敌人造成5点伤害"\n'
        '    ✓ "获得+2攻击力，持续3回合"  ✗ "你的猫咪获得+2攻击力，持续3回合"'
    )

    # Output format
    sections.append(
        f"【输出格式】每条翻译之间用 {ENTRY_SEP} 分隔，严格按顺序输出，"
        "不要添加编号、KEY或任何额外内容。只输出译文。"
    )

    # Input entries
    entry_lines = [
        f"以下共 {len(entries)} 条待翻译文本，每条格式为 [编号] KEY | 英文原文：",
        "",
    ]
    for i, entry in enumerate(entries):
        en_text = entry["en"].replace("\n", "\\n")
        line = f"[{i + 1}] {entry['key']} | {en_text}"
        if entry.get("notes"):
            line += f"  (备注: {entry['notes']})"
        entry_lines.append(line)
    sections.append("\n".join(entry_lines))

    return "\n\n".join(sections)


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


def translate_batch(
    entries: list[dict],
    progress: dict | None = None,
    glossary: dict | None = None,
) -> list[str]:
    from ai import completion

    if glossary is None:
        glossary = load_glossary()
    references = _find_reference_translations(entries, progress) if progress else None
    prompt = build_prompt(entries, glossary, references)

    messages = [{"role": "user", "content": prompt}]
    response = completion(messages, 3000)

    assert response, "response is None"

    return parse_response(response, len(entries))


def _extract_glossary_from_batch(
    entries: list[dict],
    translations: list[str],
    glossary: dict[str, str],
) -> dict[str, str]:
    """Call AI to extract notable terms from a translated batch."""
    from ai import completion1

    pairs = []
    for entry, zh in zip(entries, translations):
        if zh:
            pairs.append(f"  {entry['key']}: {entry['en']} → {zh}")

    if not pairs or len(pairs) < 5:
        return {}

    existing = "\n".join(f"  {en} = {zh}" for en, zh in glossary.items())

    prompt_lines = [
        "你是游戏《Mewgenics》的中文本地化术语管理员。",
        "请从以下翻译中提取【跨条目复用的通用术语】，确保后续翻译保持一致。",
        "",
        "【什么是术语】术语是翻译时容易产生歧义、需要统一译法的词汇，例如：",
        "  - 有多种译法的游戏概念：Rune=符文, Cleave=劈砍, Brace=防御姿态",
        "  - 游戏自创/特殊含义的词：creep=诡异痕迹, Bloodzerked=嗜血狂怒",
        "  - 专有人名/地名/阵营名等需要固定的译名",
        "",
        "【什么不是术语】以下内容不要提取：",
        "  - 含义明确的常用词、词组等（如 damage=伤害, attack=攻击, cat=猫咪, speed=速度，increased by=提升）",
        "  - 单个条目的完整翻译（如某个具体技能名、物品名、关键词名的整体翻译）",
        "  - 只在一个条目中出现一次的专有名称，或者常用词的特殊含义（water=水域）",
        "  - CSV的key名（如 KEYWORD_xxx_NAME, ABILITY_xxx_DESC 等）",
        "  - 已有术语表中已存在的词条",
        "",
        "【已有术语表】",
        existing if existing else "  （暂无）",
        "",
        "【本批翻译】",
        *pairs,
        "",
        "【输出格式】仅输出新术语，每行一个，格式为: English = 中文",
        '如果没有值得添加的新术语，输出"无"。',
    ]

    messages = [{"role": "user", "content": "\n".join(prompt_lines)}]
    try:
        response = completion1(messages)
    except Exception:
        return {}

    if not response or response.strip() == "无":
        return {}

    existing_lower = {k.lower() for k in glossary}
    new_terms = {}
    for line in response.strip().split("\n"):
        line = line.strip().lstrip("- ").strip()
        if "=" in line:
            parts = line.split("=", 1)
            en = parts[0].strip()
            zh = parts[1].strip()
            if en and zh and en.lower() not in existing_lower:
                # Reject CSV key names (e.g. KEYWORD_TRANSFORM_NAME)
                if re.match(r"^[A-Z][A-Z0-9_]{3,}$", en):
                    continue
                new_terms[en] = zh

    return new_terms


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
    wave_size: int = 10,
):
    """Main translation loop.

    Args:
        batch_size: Number of entries per translate_batch() call.
        files: List of CSV filenames to translate. None = all files.
        dry_run: If True, only show stats without translating.
        apply_only: If True, only apply existing translations from progress
            file to CSVs without calling AI.
        wave_size: Number of batches to run concurrently per wave.
            After each wave completes, glossary is updated before next wave.
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
        f"\nTranslating {total_pending} entries (batch size: {batch_size}, wave size: {wave_size})..."
    )
    print()

    translated_total = 0
    t_start = time.time()
    lock = Lock()
    has_error = False
    glossary = load_glossary()

    for csv_file in available:
        header, rows, pending = collect_entries(csv_file, progress)  # type: ignore
        if not pending:
            continue

        zh_idx = header.index("zh")
        print(f"[{csv_file}] {len(pending)} entries to translate")

        prefix_batches = _build_prefix_batches(pending, batch_size)
        batches = list(enumerate(prefix_batches))

        file_new_terms: dict[str, str] = {}
        file_translated = 0

        # Process batches in waves: each wave runs wave_size batches
        # concurrently, then updates glossary snapshot before next wave
        for wave_start in range(0, len(batches), wave_size):
            wave = batches[wave_start : wave_start + wave_size]

            # Fresh snapshot each wave so new terms are visible
            glossary_snapshot = {**glossary, **file_new_terms}
            progress_snapshot = dict(progress)

            def process_batch(
                batch_info, _glossary=glossary_snapshot, _progress=progress_snapshot
            ):
                batch_idx, batch = batch_info
                batch_entries = [entry for _, entry in batch]
                results = translate_batch(batch_entries, _progress, _glossary)

                if len(results) != len(batch):
                    with lock:
                        print(
                            f"  WARNING: translate_batch returned {len(results)} results for {len(batch)} entries"
                        )
                    results = results[: len(batch)]
                    results.extend([""] * (len(batch) - len(results)))

                return batch_idx, batch, results

            with ThreadPoolExecutor(max_workers=wave_size) as executor:
                futures = {executor.submit(process_batch, b): b for b in wave}

                for future in as_completed(futures):
                    if has_error:
                        break

                    try:
                        batch_idx, batch, results = future.result()
                    except Exception as e:
                        with lock:
                            print(f"\n  ERROR in translate_batch(): {e}")
                            print("  Saving progress and stopping.")
                            save_progress(progress)
                            write_csv(os.path.join(TEXT_DIR, csv_file), header, rows)
                            has_error = True
                        break

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

                    # Extract glossary terms (main thread only, sequential)
                    batch_entries = [entry for _, entry in batch]
                    latest_glossary = {**glossary, **file_new_terms}
                    new_terms = _extract_glossary_from_batch(
                        batch_entries, results, latest_glossary
                    )
                    if new_terms:
                        # Case-insensitive dedup: skip if key already exists
                        existing_lower = {k.lower() for k in glossary} | {
                            k.lower() for k in file_new_terms
                        }
                        new_terms = {
                            k: v
                            for k, v in new_terms.items()
                            if k.lower() not in existing_lower
                        }
                    if new_terms:
                        file_new_terms.update(new_terms)
                        terms_str = ", ".join(f"{k}={v}" for k, v in new_terms.items())
                        print(f"  📚 新术语: {terms_str}")

            if has_error:
                break

        # Merge accumulated glossary terms after all batches of this file
        if file_new_terms:
            # Case-insensitive dedup before merging into main glossary
            glossary_lower = {k.lower(): k for k in glossary}
            for en, zh in file_new_terms.items():
                existing_key = glossary_lower.get(en.lower())
                if existing_key is None:
                    glossary[en] = zh
                    glossary_lower[en.lower()] = en
            save_glossary(glossary)
            print(f"  📚 {csv_file}: 共新增 {len(file_new_terms)} 条术语")

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
    npc_width: int = 40,
    events_width: int = 40,
    abilities_width: int = 40,
):
    """Auto-wrap long translated text lines.

    Args:
        max_width: Maximum display width per line.
        files: List of CSV filenames to process. None = all files.
        dry_run: If True, only show what would change.
        npc_width: Width for npc_dialog.csv (independent of max_width).
        events_width: Width for events.csv (independent of max_width).
        abilities_width: Width for abilities.csv (independent of max_width).
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
        csv_file = key.split("::")[0]
        if files:
            if csv_file not in files:
                continue
        if csv_file == "npc_dialog.csv":
            entry_width = npc_width
        elif csv_file == "events.csv":
            entry_width = events_width
        elif csv_file == "abilities.csv":
            entry_width = abilities_width
        else:
            entry_width = max_width
        wrapped, overflow = wrap_text(value, entry_width)
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
        lines.append("4. [img:xxx]、[b]...[/b]、{xxx} 等标记标签宽度为0，不要拆开")
        lines.append("5. 不要修改文字内容，只添加换行")
        lines.append("6. 输出中不要包含←标记")
        lines.append("")
        lines.append(
            f"【输出格式】每条结果之间用一行 {ENTRY_SEP} 分隔（单独占一行），"
            "严格按顺序输出，只输出处理后的完整文本。"
        )
        lines.append("")
        lines.append(f"以下共 {len(prompt_entries)} 条文本（用 ---- 分隔每条）：")
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
