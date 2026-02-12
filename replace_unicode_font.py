"""
Replace the font in unicodefont.swf with a TTF font file.

Reads the TTF, converts glyph outlines to SWF DefineFont3 shape records,
and rebuilds unicodefont.swf with the new font while preserving the SWF
structure and font identity (ID + name) so the game continues to use it.

Usage:
  uv run replace_unicode_font.py                    # replace font
  uv run replace_unicode_font.py --restore          # restore from backup
  uv run replace_unicode_font.py --dry              # show what would happen
  uv run replace_unicode_font.py --font other.ttf   # use a different TTF
"""

import struct
import sys
import os
import zlib
import shutil
import argparse
import io

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fontTools.ttLib import TTFont
from fontTools.pens.pointPen import SegmentToPointPen
from fontTools.pens.basePen import BasePen

UNICODE_SWF = os.path.join("extracted", "swfs", "unicodefont.swf")
UNICODE_SWF_BAK = "unicodefont.swf.bak"
DEFAULT_TTF = "MaoKenZhuYuanTi-MaokenZhuyuanTi-2.ttf"

SWF_EM = 1024 * 20  # 20480 twips


class BitWriter:
    def __init__(self):
        self.bits = []

    def write_ub(self, nbits, value):
        for i in range(nbits - 1, -1, -1):
            self.bits.append((value >> i) & 1)

    def write_sb(self, nbits, value):
        if value < 0:
            value = (1 << nbits) + value
        self.write_ub(nbits, value)

    def align(self):
        while len(self.bits) % 8 != 0:
            self.bits.append(0)

    def to_bytes(self):
        self.align()
        result = bytearray()
        for i in range(0, len(self.bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(self.bits):
                    byte = (byte << 1) | self.bits[i + j]
                else:
                    byte = byte << 1
            result.append(byte)
        return bytes(result)


def bits_needed_signed(value):
    if value == 0:
        return 1
    if value > 0:
        return value.bit_length() + 1
    return ((-value - 1).bit_length() if value != -1 else 0) + 2


def bits_needed_unsigned(value):
    if value == 0:
        return 0
    return value.bit_length()


def encode_shape(contours):
    bw = BitWriter()
    bw.write_ub(4, 1)  # NumFillBits = 1
    bw.write_ub(4, 0)  # NumLineBits = 0

    for contour in contours:
        if len(contour) == 0:
            continue

        segments = contour
        start_x, start_y = segments[0][0]
        cur_x, cur_y = start_x, start_y

        move_bits = max(bits_needed_signed(start_x), bits_needed_signed(start_y))
        if move_bits < 1:
            move_bits = 1

        # StyleChangeRecord with MoveTo and FillStyle1
        bw.write_ub(1, 0)  # TypeFlag = 0 (non-edge)
        bw.write_ub(1, 0)  # StateNewStyles
        bw.write_ub(1, 0)  # StateLineStyle
        bw.write_ub(1, 1)  # StateFillStyle1
        bw.write_ub(1, 0)  # StateFillStyle0
        bw.write_ub(1, 1)  # StateMoveTo
        bw.write_ub(5, move_bits)
        bw.write_sb(move_bits, start_x)
        bw.write_sb(move_bits, start_y)
        bw.write_ub(1, 1)  # FillStyle1 = 1

        for seg in segments:
            seg_type = seg[-1] if isinstance(seg[-1], str) else None
            if seg_type == "line":
                _, x, y = seg[0], seg[1][0], seg[1][1]
                dx = x - cur_x
                dy = y - cur_y
                if dx == 0 and dy == 0:
                    continue
                nbits = max(bits_needed_signed(dx), bits_needed_signed(dy))
                if nbits < 2:
                    nbits = 2
                bw.write_ub(1, 1)  # TypeFlag = 1 (edge)
                bw.write_ub(1, 1)  # StraightFlag
                bw.write_ub(4, nbits - 2)
                if dx == 0:
                    bw.write_ub(1, 0)  # GeneralLine = 0
                    bw.write_ub(1, 1)  # VertLine = 1
                    bw.write_sb(nbits, dy)
                elif dy == 0:
                    bw.write_ub(1, 0)  # GeneralLine = 0
                    bw.write_ub(1, 0)  # VertLine = 0
                    bw.write_sb(nbits, dx)
                else:
                    bw.write_ub(1, 1)  # GeneralLine = 1
                    bw.write_sb(nbits, dx)
                    bw.write_sb(nbits, dy)
                cur_x, cur_y = x, y

            elif seg_type == "curve":
                _, ctrl, anchor = seg[0], seg[1], seg[2]
                cx, cy = ctrl
                ax, ay = anchor
                cdx = cx - cur_x
                cdy = cy - cur_y
                adx = ax - cx
                ady = ay - cy
                nbits = max(
                    bits_needed_signed(cdx),
                    bits_needed_signed(cdy),
                    bits_needed_signed(adx),
                    bits_needed_signed(ady),
                )
                if nbits < 2:
                    nbits = 2
                bw.write_ub(1, 1)  # TypeFlag = 1 (edge)
                bw.write_ub(1, 0)  # StraightFlag = 0 (curve)
                bw.write_ub(4, nbits - 2)
                bw.write_sb(nbits, cdx)
                bw.write_sb(nbits, cdy)
                bw.write_sb(nbits, adx)
                bw.write_sb(nbits, ady)
                cur_x, cur_y = ax, ay

    # EndShapeRecord
    bw.write_ub(6, 0)
    return bw.to_bytes()


def ttf_contours_to_swf(glyph, glyf_table, scale):
    if glyph.isComposite():
        glyph.recalcBounds(glyf_table)
        coords, end_pts, flags = glyph.getCoordinates(glyf_table)
    else:
        if glyph.numberOfContours <= 0:
            return []
        coords = glyph.coordinates
        end_pts = glyph.endPtsOfContours
        flags = glyph.flags

    if not end_pts:
        return []

    contours = []
    start = 0
    for end in end_pts:
        points = []
        for i in range(start, end + 1):
            x = round(coords[i][0] * scale)
            y = round(-coords[i][1] * scale)  # flip Y
            on_curve = bool(flags[i] & 1)
            points.append((x, y, on_curve))
        start = end + 1

        expanded = []
        n = len(points)
        for i in range(n):
            p = points[i]
            p_next = points[(i + 1) % n]
            expanded.append(p)
            if not p[2] and not p_next[2]:
                mid_x = (p[0] + p_next[0]) // 2
                mid_y = (p[1] + p_next[1]) // 2
                expanded.append((mid_x, mid_y, True))

        if not expanded:
            continue

        if not expanded[0][2]:
            last_on = None
            for ep in expanded:
                if ep[2]:
                    last_on = ep
                    break
            if last_on is None:
                mid_x = (expanded[0][0] + expanded[-1][0]) // 2
                mid_y = (expanded[0][1] + expanded[-1][1]) // 2
                expanded.insert(0, (mid_x, mid_y, True))
            else:
                while not expanded[0][2]:
                    expanded.append(expanded.pop(0))

        segments = []
        first_point = (expanded[0][0], expanded[0][1])
        i = 0
        while i < len(expanded):
            cur = expanded[i]
            if i + 1 < len(expanded) and not expanded[i + 1][2]:
                off = expanded[i + 1]
                if i + 2 < len(expanded):
                    on = expanded[i + 2]
                    segments.append((
                        (cur[0], cur[1]),
                        (off[0], off[1]),
                        (on[0], on[1]),
                        "curve",
                    ))
                    i += 2
                else:
                    segments.append((
                        (cur[0], cur[1]),
                        (off[0], off[1]),
                        first_point,
                        "curve",
                    ))
                    i += 2
            elif i + 1 < len(expanded):
                nxt = expanded[i + 1]
                segments.append(((cur[0], cur[1]), (nxt[0], nxt[1]), "line"))
                i += 1
            else:
                if (cur[0], cur[1]) != first_point:
                    segments.append(((cur[0], cur[1]), first_point, "line"))
                i += 1

        if segments:
            last_end = (
                segments[-1][2] if segments[-1][-1] == "curve" else segments[-1][1]
            )
            if last_end != first_point:
                segments.append((last_end, first_point, "line"))

        contours.append(segments)

    return contours


def build_define_font3(
    font_id, font_name_bytes, codes, shapes_data, advances, ascent, descent
):
    num_glyphs = len(codes)

    flags = 0x80 | 0x04  # HasLayout | WideCodes
    total_shape = sum(len(s) for s in shapes_data)
    if total_shape + (num_glyphs + 1) * 4 > 65535:
        flags |= 0x08  # WideOffsets
    wide_offsets = bool(flags & 0x08)

    result = bytearray()
    result += struct.pack("<H", font_id)
    result += bytes([flags])
    result += bytes([0])  # language code
    result += bytes([len(font_name_bytes)])
    result += font_name_bytes
    result += struct.pack("<H", num_glyphs)

    if wide_offsets:
        offset_table_size = (num_glyphs + 1) * 4
    else:
        offset_table_size = (num_glyphs + 1) * 2

    running = offset_table_size
    offsets = []
    for s in shapes_data:
        offsets.append(running)
        running += len(s)
    offsets.append(running)  # code_table_offset

    if wide_offsets:
        for o in offsets:
            result += struct.pack("<I", o)
    else:
        for o in offsets:
            result += struct.pack("<H", o)

    for s in shapes_data:
        result += s

    for c in codes:
        result += struct.pack("<H", c)

    # Layout
    result += struct.pack("<H", ascent)
    result += struct.pack("<H", descent)
    result += struct.pack("<h", 0)  # leading
    for a in advances:
        result += struct.pack("<h", a)
    for _ in range(num_glyphs):
        result += bytes([0x00])  # empty RECT bounds
    result += struct.pack("<H", 0)  # kerning count

    return bytes(result)


def build_tag(tag_type, data):
    data_len = len(data)
    if data_len < 0x3F:
        tc = (tag_type << 6) | data_len
        return struct.pack("<H", tc) + data
    else:
        tc = (tag_type << 6) | 0x3F
        return struct.pack("<H", tc) + struct.pack("<I", data_len) + data


def build_define_font_name(font_id, full_name, copyright_str=""):
    data = struct.pack("<H", font_id)
    data += full_name.encode("utf-8") + b"\x00"
    data += copyright_str.encode("utf-8") + b"\x00"
    return data


def build_font_align_zones(font_id, num_glyphs):
    data = struct.pack("<H", font_id)
    csm_hint = 0  # thin
    data += bytes([csm_hint])
    for _ in range(num_glyphs):
        data += bytes([2])  # NumZoneData = 2 (always for ZoneRecord)
        data += struct.pack("<H", 0)  # ZoneData0 AlignmentCoord
        data += struct.pack("<H", 0)  # ZoneData0 Range
        data += struct.pack("<H", 0)  # ZoneData1 AlignmentCoord
        data += struct.pack("<H", 0)  # ZoneData1 Range
        data += bytes([0x03])  # ZoneMaskY | ZoneMaskX
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Replace unicodefont.swf font with a TTF"
    )
    parser.add_argument("--font", default=DEFAULT_TTF, help="TTF font file path")
    parser.add_argument("--restore", action="store_true", help="Restore from backup")
    parser.add_argument("--dry", action="store_true", help="Dry run")
    args = parser.parse_args()

    if args.restore:
        if os.path.exists(UNICODE_SWF_BAK):
            shutil.copy2(UNICODE_SWF_BAK, UNICODE_SWF)
            print(f"Restored {UNICODE_SWF} from backup.")
        else:
            print(f"No backup found at {UNICODE_SWF_BAK}")
        return

    if not os.path.exists(UNICODE_SWF):
        print(f"Error: {UNICODE_SWF} not found. Run 'extract' first.")
        return
    if not os.path.exists(args.font):
        print(f"Error: {args.font} not found.")
        return

    print(f"Reading TTF: {args.font} ...")
    ttf = TTFont(args.font)
    upm = ttf["head"].unitsPerEm
    scale = SWF_EM / upm
    print(f"  unitsPerEm: {upm}, scale factor: {scale:.4f}")

    hhea = ttf["hhea"]
    ascent_swf = min(abs(round(hhea.ascent * scale)), 65535)
    descent_swf = min(abs(round(hhea.descent * scale)), 65535)
    print(f"  ascent: {ascent_swf}, descent: {descent_swf}")

    cmap = ttf.getBestCmap()
    glyf_table = ttf["glyf"]
    hmtx = ttf["hmtx"]

    print(f"  cmap entries: {len(cmap)}")

    print("Reading original unicodefont.swf ...")
    with open(UNICODE_SWF, "rb") as f:
        orig_sig = f.read(3)
        orig_ver = struct.unpack("<B", f.read(1))[0]
        orig_flen = struct.unpack("<I", f.read(4))[0]
        orig_rest = f.read()

    if orig_sig == b"CWS":
        orig_body = zlib.decompress(orig_rest)
    elif orig_sig == b"FWS":
        orig_body = orig_rest
    else:
        print(f"Unknown SWF signature: {orig_sig}")
        return

    nbits = orig_body[0] >> 3
    rect_size = (5 + nbits * 4 + 7) // 8
    header_size = rect_size + 4  # RECT + frame_rate(2) + frame_count(2)

    off = header_size
    tags = []
    while off < len(orig_body) - 1:
        tc = struct.unpack_from("<H", orig_body, off)[0]
        tag_type = tc >> 6
        tag_length = tc & 0x3F
        hdr_len = 2
        if tag_length == 0x3F:
            tag_length = struct.unpack_from("<I", orig_body, off + 2)[0]
            hdr_len = 6
        tags.append((tag_type, off, hdr_len, tag_length))
        off += hdr_len + tag_length
        if tag_type == 0:
            break

    orig_font_id = None
    orig_font_name = None
    for tag_type, o, hl, tl in tags:
        if tag_type == 75:
            data = orig_body[o + hl : o + hl + tl]
            orig_font_id = struct.unpack_from("<H", data, 0)[0]
            nlen = data[4]
            orig_font_name = data[5 : 5 + nlen]
            orig_font_name_str = orig_font_name.decode("utf-8", "replace").rstrip(
                "\x00"
            )
            print(f'  Original font: ID={orig_font_id}, name="{orig_font_name_str}"')
            break

    if orig_font_id is None:
        print("Error: No DefineFont3 found in unicodefont.swf")
        return

    print("Converting glyphs ...")
    codes = []
    shapes_data = []
    advances = []
    failed = 0

    sorted_codepoints = sorted(cp for cp in cmap.keys() if cp <= 0xFFFF)
    for cp in sorted_codepoints:
        glyph_name = cmap[cp]
        try:
            glyph = glyf_table[glyph_name]
        except KeyError:
            failed += 1
            continue

        adv_width = hmtx[glyph_name][0]
        adv_swf = round(adv_width * scale)

        if glyph.numberOfContours == 0:
            bw = BitWriter()
            bw.write_ub(4, 1)  # NumFillBits
            bw.write_ub(4, 0)  # NumLineBits
            bw.write_ub(6, 0)  # EndShapeRecord
            shape_bytes = bw.to_bytes()
        elif glyph.numberOfContours < 0:
            contours = ttf_contours_to_swf(glyph, glyf_table, scale)
            if not contours:
                bw = BitWriter()
                bw.write_ub(4, 1)
                bw.write_ub(4, 0)
                bw.write_ub(6, 0)
                shape_bytes = bw.to_bytes()
            else:
                shape_bytes = encode_shape(contours)
        else:
            contours = ttf_contours_to_swf(glyph, glyf_table, scale)
            if not contours:
                bw = BitWriter()
                bw.write_ub(4, 1)
                bw.write_ub(4, 0)
                bw.write_ub(6, 0)
                shape_bytes = bw.to_bytes()
            else:
                shape_bytes = encode_shape(contours)

        codes.append(cp)
        shapes_data.append(shape_bytes)
        advances.append(adv_swf)

    print(f"  Converted {len(codes)} glyphs ({failed} failed)")

    font3_data = build_define_font3(
        orig_font_id,
        orig_font_name,
        codes,
        shapes_data,
        advances,
        ascent_swf,
        descent_swf,
    )
    print(f"  DefineFont3 tag: {len(font3_data)} bytes")

    align_zones_data = build_font_align_zones(orig_font_id, len(codes))
    font_name_data = build_define_font_name(
        orig_font_id, orig_font_name.decode("utf-8", "replace").rstrip("\x00")
    )

    if args.dry:
        print(f"\n[DRY RUN] Would replace font in {UNICODE_SWF}")
        print(f"  New glyph count: {len(codes)}")
        print(f"  DefineFont3 size: {len(font3_data)} bytes")
        return

    if not os.path.exists(UNICODE_SWF_BAK):
        shutil.copy2(UNICODE_SWF, UNICODE_SWF_BAK)
        print(f"\nBackup: {UNICODE_SWF_BAK}")

    print("Building new SWF ...")
    new_body = bytearray(orig_body[:header_size])

    for tag_type, o, hl, tl in tags:
        if tag_type == 75:
            new_body += build_tag(75, font3_data)
        elif tag_type == 73:
            new_body += build_tag(73, align_zones_data)
        elif tag_type == 88:
            new_body += build_tag(88, font_name_data)
        else:
            new_body += orig_body[o : o + hl + tl]

    file_length = len(new_body) + 8
    with open(UNICODE_SWF, "wb") as f:
        f.write(b"FWS")
        f.write(struct.pack("<B", orig_ver))
        f.write(struct.pack("<I", file_length))
        f.write(new_body)

    new_size = os.path.getsize(UNICODE_SWF)
    orig_size = os.path.getsize(UNICODE_SWF_BAK)
    print(f"\nDone. Replaced font in {UNICODE_SWF}")
    print(f"  {orig_size / 1024 / 1024:.1f} MB -> {new_size / 1024 / 1024:.1f} MB")
    print(f"  Glyphs: {len(codes)}")


if __name__ == "__main__":
    main()
