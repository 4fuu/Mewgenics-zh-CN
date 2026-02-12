"""
Mewgenics Chinese Localization Patch Tool

Workflow:
  1. Extract text from resources.gpak
  2. Add Chinese translations to CSV files
  3. Repack into a patched resources.gpak

Usage:
  uv run main.py extract          - Extract all files from resources.gpak
  uv run main.py add-zh-column    - Add empty 'zh' column to all CSV files
  uv run main.py translate              - Translate all CSV files
  uv run main.py translate --dry        - Show translation progress without translating
  uv run main.py translate --apply-only - Apply existing translations only, no AI
  uv run main.py pack             - Repack extracted/ into resources_patched.gpak
  uv run main.py apply            - Replace resources.gpak with patched version (backs up original)
  uv run main.py info             - Show archive info
"""

import struct
import os
import sys
import shutil
import argparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

GPAK_PATH = "resources.gpak"
GPAK_BACKUP = "resources.gpak.bak"
GPAK_PATCHED = "resources_patched.gpak"
EXTRACT_DIR = "extracted"
TEXT_DIR = "extracted/data/text"


def parse_index(filepath):
    entries = []
    with open(filepath, "rb") as f:
        header = f.read(4)
        while True:
            pos = f.tell()
            raw = f.read(2)
            if len(raw) < 2:
                break
            name_len = struct.unpack("<H", raw)[0]
            if name_len == 0 or name_len > 500:
                f.seek(pos)
                break
            name_bytes = f.read(name_len)
            if len(name_bytes) < name_len:
                f.seek(pos)
                break
            try:
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError:
                f.seek(pos)
                break
            if not all(c.isprintable() for c in name):
                f.seek(pos)
                break
            size = struct.unpack("<I", f.read(4))[0]
            entries.append((name, size))
        data_start = f.tell()

    result = []
    current = data_start
    for name, size in entries:
        result.append((name, size, current))
        current += size
    return header, result, data_start


def cmd_info():
    header, entries, data_start = parse_index(GPAK_PATH)
    header_val = struct.unpack("<I", header)[0]
    file_size = os.path.getsize(GPAK_PATH)
    print(f"File: {GPAK_PATH}")
    print(f"Size: {file_size} bytes ({file_size / 1024 / 1024:.1f} MB)")
    print(f"Header (entry count): {header_val}")
    print(f"Entries: {len(entries)}")
    print(f"Data starts at: 0x{data_start:X}")

    ext_counts = {}
    ext_sizes = {}
    for name, size, _ in entries:
        ext = name.rsplit(".", 1)[-1] if "." in name else "(none)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        ext_sizes[ext] = ext_sizes.get(ext, 0) + size
    print("\nFile types:")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        total_mb = ext_sizes[ext] / 1024 / 1024
        print(f"  .{ext}: {count} files ({total_mb:.1f} MB)")

    csv_files = [(n, s) for n, s, _ in entries if n.endswith(".csv")]
    if csv_files:
        print("\nCSV text files (localization targets):")
        for name, size in csv_files:
            print(f"  {name:<50} {size:>10} bytes")


def cmd_extract():
    header, entries, data_start = parse_index(GPAK_PATH)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    header_path = os.path.join(EXTRACT_DIR, "__gpak_header.bin")
    with open(header_path, "wb") as hf:
        hf.write(header)

    index_path = os.path.join(EXTRACT_DIR, "__gpak_index.txt")
    with open(index_path, "w", encoding="utf-8") as idx:
        for name, size, _ in entries:
            idx.write(f"{name}\n")

    print(f"Extracting {len(entries)} files...")

    with open(GPAK_PATH, "rb") as f:
        for i, (name, size, offset) in enumerate(entries):
            out_path = os.path.join(EXTRACT_DIR, name.replace("/", os.sep))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            f.seek(offset)
            with open(out_path, "wb") as out:
                remaining = size
                while remaining > 0:
                    chunk = min(remaining, 8 * 1024 * 1024)
                    data = f.read(chunk)
                    if not data:
                        break
                    out.write(data)
                    remaining -= len(data)
            if (i + 1) % 200 == 0 or i == len(entries) - 1:
                print(f"  [{i + 1}/{len(entries)}] {name}")

    print(f"Done. Extracted to {EXTRACT_DIR}/")


ZH_LANGUAGE_META = {
    "CURRENT_LANGUAGE_NAME": "中文",
    "CURRENT_LANGUAGE_SHIPPABLE": "yes",
}


def cmd_add_zh_column():
    if not os.path.exists(TEXT_DIR):
        print(f"Error: {TEXT_DIR} not found. Run 'extract' first.")
        return

    csv_files = [f for f in os.listdir(TEXT_DIR) if f.endswith(".csv")]
    print(f"Adding 'zh' column to {len(csv_files)} CSV files...")

    for csv_file in sorted(csv_files):
        filepath = os.path.join(TEXT_DIR, csv_file)
        with open(filepath, "r", encoding="utf-8-sig") as f:
            content = f.read()

        lines = content.split("\n")
        if not lines:
            continue

        header_line = lines[0]
        if ",zh" in header_line.lower():
            print(f"  {csv_file}: 'zh' column already exists, skipping")
        else:
            new_lines = []
            for line in lines:
                if line.strip() == "":
                    new_lines.append(line)
                else:
                    new_lines.append(line + ",")
            new_lines[0] = header_line + ",zh"
            lines = new_lines

            with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
                f.write("\n".join(lines))

            print(f"  {csv_file}: added 'zh' column")

        if csv_file == "additions.csv":
            _apply_zh_language_meta(filepath, lines)

    print("Done. You can now edit the CSV files to add Chinese translations.")
    print(f"CSV files are in: {TEXT_DIR}/")


def _apply_zh_language_meta(filepath, lines):
    import csv as _csv
    import io

    content = "\n".join(lines)
    reader = _csv.reader(io.StringIO(content))
    header = next(reader)
    rows = list(reader)

    if "zh" not in header:
        return

    zh_idx = header.index("zh")
    changed = False

    for row in rows:
        while len(row) < len(header):
            row.append("")
        key = row[0]
        if key in ZH_LANGUAGE_META and not row[zh_idx].strip():
            row[zh_idx] = ZH_LANGUAGE_META[key]
            print(f"  additions.csv: set {key} zh = {ZH_LANGUAGE_META[key]}")
            changed = True

    if changed:
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = _csv.writer(f, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)


def cmd_pack():
    if not os.path.exists(EXTRACT_DIR):
        print(f"Error: {EXTRACT_DIR}/ not found. Run 'extract' first.")
        return

    header_path = os.path.join(EXTRACT_DIR, "__gpak_header.bin")
    index_path = os.path.join(EXTRACT_DIR, "__gpak_index.txt")

    if not os.path.exists(header_path) or not os.path.exists(index_path):
        print("Error: __gpak_header.bin or __gpak_index.txt not found.")
        return

    with open(header_path, "rb") as hf:
        header = hf.read(4)

    with open(index_path, "r", encoding="utf-8") as idx:
        file_list = [line.strip() for line in idx if line.strip()]

    print(f"Packing {len(file_list)} files into {GPAK_PATCHED}...")

    entries = []
    for name in file_list:
        local_path = os.path.join(EXTRACT_DIR, name.replace("/", os.sep))
        if not os.path.exists(local_path):
            print(f"  WARNING: {name} not found, will read from original GPAK")
            entries.append((name, None))
        else:
            entries.append((name, local_path))

    _, orig_entries, _ = parse_index(GPAK_PATH)
    orig_map = {n: (s, o) for n, s, o in orig_entries}

    index_data = bytearray()
    file_infos = []
    for name, local_path in entries:
        if local_path is not None:
            size = os.path.getsize(local_path)
            file_infos.append((name, size, local_path, None))
        else:
            orig_size, orig_offset = orig_map[name]
            file_infos.append((name, orig_size, None, orig_offset))
        name_bytes = name.encode("utf-8")
        index_data += struct.pack("<H", len(name_bytes))
        index_data += name_bytes
        index_data += struct.pack("<I", file_infos[-1][1])

    with open(GPAK_PATCHED, "wb") as out:
        out.write(header)
        out.write(index_data)

        orig_f = open(GPAK_PATH, "rb")
        try:
            for i, (name, size, local_path, orig_offset) in enumerate(file_infos):
                if local_path is not None:
                    with open(local_path, "rb") as src:
                        remaining = size
                        while remaining > 0:
                            chunk = min(remaining, 8 * 1024 * 1024)
                            data = src.read(chunk)
                            if not data:
                                break
                            out.write(data)
                            remaining -= len(data)
                else:
                    orig_f.seek(orig_offset)
                    remaining = size
                    while remaining > 0:
                        chunk = min(remaining, 8 * 1024 * 1024)
                        data = orig_f.read(chunk)
                        if not data:
                            break
                        out.write(data)
                        remaining -= len(data)

                if (i + 1) % 500 == 0 or i == len(file_infos) - 1:
                    print(f"  [{i + 1}/{len(file_infos)}] {name}")
        finally:
            orig_f.close()

    patched_size = os.path.getsize(GPAK_PATCHED)
    print(f"Done. Output: {GPAK_PATCHED} ({patched_size / 1024 / 1024:.1f} MB)")


def cmd_apply():
    if not os.path.exists(GPAK_PATCHED):
        print(f"Error: {GPAK_PATCHED} not found. Run 'pack' first.")
        return

    if not os.path.exists(GPAK_BACKUP):
        print(f"Backing up {GPAK_PATH} -> {GPAK_BACKUP}")
        shutil.copy2(GPAK_PATH, GPAK_BACKUP)
    else:
        print(f"Backup already exists: {GPAK_BACKUP}")

    print(f"Replacing {GPAK_PATH} with {GPAK_PATCHED}")
    shutil.move(GPAK_PATCHED, GPAK_PATH)
    print("Done. Patch applied.")


def main():
    parser = argparse.ArgumentParser(
        description="Mewgenics Chinese Localization Patch Tool"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Show GPAK archive info")
    sub.add_parser("extract", help="Extract all files from GPAK")
    sub.add_parser("add-zh-column", help="Add empty 'zh' column to CSV files")
    sub.add_parser("pack", help="Repack into patched GPAK")
    sub.add_parser("apply", help="Apply patch (replace resources.gpak)")
    tr_parser = sub.add_parser("translate", help="Translate CSV files to Chinese")
    tr_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Entries per translation batch (default: 50)",
    )
    tr_parser.add_argument(
        "--files", nargs="+", help="Only translate specific CSV files"
    )
    tr_parser.add_argument(
        "--dry", action="store_true", help="Show progress without translating"
    )
    tr_parser.add_argument(
        "--apply-only",
        action="store_true",
        help="Only apply existing translations from progress file, no AI",
    )
    args = parser.parse_args()

    if args.command == "info":
        cmd_info()
    elif args.command == "extract":
        cmd_extract()
    elif args.command == "add-zh-column":
        cmd_add_zh_column()
    elif args.command == "pack":
        cmd_pack()
    elif args.command == "translate":
        from translate import run_translate

        run_translate(
            batch_size=args.batch_size,
            files=args.files,
            dry_run=args.dry,
            apply_only=args.apply_only,
        )
    elif args.command == "apply":
        cmd_apply()
    else:
        parser.print_help()
        print("\nTypical workflow:")
        print("  1. uv run main.py extract        # Extract all files")
        print("  2. uv run main.py add-zh-column  # Add Chinese column to CSVs")
        print("  3. uv run main.py translate      # Auto-translate to Chinese")
        print("  4. uv run main.py pack           # Create patched GPAK")
        print("  5. uv run main.py apply          # Apply patch to game")


if __name__ == "__main__":
    main()
