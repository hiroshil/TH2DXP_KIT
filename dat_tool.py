#!/usr/bin/env python3
"""
data_dat_cli.py

Single CLI for this game's data.dat-style archive:
  - extract: unpack archive, write raw files, and create editable sidecars for known assets
  - rebuild: rebuild archive from extracted folder using metadata, without accidentally packing sidecars
  - info: print archive table
  - self-test: validate TPL and 8192-byte BIN roundtrips on local samples/synthetic data

Important safety model:
  * Rebuild packs only entries listed in data_archive_meta.json, in original order.
  * Sidecar PNG/JSON/payload files are never packed as extra archive entries.
  * TPL rebuild is template-based and preserves headers/records/offsets/gaps/file size.
  * 8192-byte n/*.bin assets are not PNG internally; they are 256 glyphs, each 16x16 1bpp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import struct
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

# -----------------------------------------------------------------------------
# Archive format
# -----------------------------------------------------------------------------

ARCHIVE_ENTRY_SIZE = 24
DEFAULT_CHUNK_SIZE = 32768
META_NAME = "data_archive_meta.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_c_string_at(f, off: int) -> str:
    f.seek(off)
    out = bytearray()
    while True:
        b = f.read(1)
        if not b:
            raise ValueError(f"unterminated name string at 0x{off:X}")
        if b == b"\x00":
            break
        out += b
    return out.decode("latin1")


@dataclass
class ArchiveEntry:
    index: int
    name: str
    name_offset: int
    compressed_size: int
    unknown0: int
    uncompressed_size: int
    unknown1: int
    file_offset: int
    chunk_count: int = 0
    flags: Optional[List[int]] = None
    raw_sha256: Optional[str] = None
    asset_kind: str = "raw"


def parse_archive_table(archive_path: Path) -> Tuple[Dict, List[ArchiveEntry]]:
    with archive_path.open("rb") as f:
        data = f.read(16)
        if len(data) < 16:
            raise ValueError("archive too small for header")
        file_count, unk_header0, unk_header1, entries_offset = struct.unpack(">IIII", data)
        if file_count > 200000:
            raise ValueError(f"unreasonable file count: {file_count}")
        if entries_offset < 16:
            raise ValueError(f"invalid entries_offset: 0x{entries_offset:X}")
        f.seek(0)
        pre_entries = f.read(entries_offset)
        entries: List[ArchiveEntry] = []
        for i in range(file_count):
            f.seek(entries_offset + i * ARCHIVE_ENTRY_SIZE)
            raw = f.read(ARCHIVE_ENTRY_SIZE)
            if len(raw) != ARCHIVE_ENTRY_SIZE:
                raise ValueError(f"truncated entry {i}")
            name_offset, comp_size, u0, uncomp_size, u1, file_offset = struct.unpack(">IIIIII", raw)
            name = read_c_string_at(f, name_offset)
            entries.append(ArchiveEntry(i, name, name_offset, comp_size, u0, uncomp_size, u1, file_offset))
        meta = {
            "tool": "data_dat_cli.py",
            "archive": str(archive_path),
            "archive_size": archive_path.stat().st_size,
            "file_count": file_count,
            "header_unknown0": unk_header0,
            "header_unknown1": unk_header1,
            "entries_offset": entries_offset,
            "pre_entries_hex": pre_entries.hex(),
            "entry_size": ARCHIVE_ENTRY_SIZE,
        }
        return meta, entries


def decompress_archive_entry(blob: bytes, entry: ArchiveEntry) -> Tuple[bytes, List[int]]:
    if entry.compressed_size == 0:
        out = blob[entry.file_offset:entry.file_offset + entry.uncompressed_size]
        if len(out) != entry.uncompressed_size:
            raise ValueError(f"{entry.name}: raw entry exceeds archive size")
        return out, []

    pos = entry.file_offset
    end = entry.file_offset + entry.compressed_size
    if end > len(blob):
        raise ValueError(f"{entry.name}: compressed range exceeds archive size")
    out = bytearray()
    flags: List[int] = []
    while pos < end:
        if pos + 4 > end:
            raise ValueError(f"{entry.name}: truncated chunk header")
        prefix = struct.unpack_from(">I", blob, pos)[0]
        pos += 4
        flag = (prefix >> 24) & 0xFF
        chunk_size = prefix & 0x00FFFFFF
        flags.append(flag)
        if pos + chunk_size > end:
            raise ValueError(f"{entry.name}: chunk exceeds compressed range")
        chunk = blob[pos:pos + chunk_size]
        pos += chunk_size
        if flag in (0x00, 0x80):
            out += lzma.decompress(chunk, format=lzma.FORMAT_ALONE)
        elif flag == 0x40:
            out += chunk
        else:
            raise ValueError(f"{entry.name}: unsupported chunk flag 0x{flag:02X}")
        pad = (4 - ((4 + chunk_size) % 4)) % 4
        if pad:
            pos += pad
        if pos > end:
            raise ValueError(f"{entry.name}: padding exceeds compressed range")
    if len(out) != entry.uncompressed_size:
        raise ValueError(
            f"{entry.name}: decompressed size 0x{len(out):X} != table 0x{entry.uncompressed_size:X}"
        )
    return bytes(out), flags


def compress_archive_data(data: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bytes:
    if not data:
        return b""
    out = bytearray()
    for off in range(0, len(data), chunk_size):
        chunk = data[off:off + chunk_size]
        comp = lzma.compress(chunk, format=lzma.FORMAT_ALONE)
        flag = 0x00 if off + chunk_size >= len(data) else 0x80
        out += struct.pack(">I", (flag << 24) | (len(comp) & 0x00FFFFFF))
        out += comp
        out += b"\x00" * ((4 - ((4 + len(comp)) % 4)) % 4)
    return bytes(out)


# -----------------------------------------------------------------------------
# TPL general support, integrated from tpl_image_tool safety model
# -----------------------------------------------------------------------------

TPL_REC_SIZE = 0x14
ENGINE_FONT_ORDER = ["R", "G", "B", "A"]
RAW_CHANNELS = ["A", "R", "G", "B"]
FMT_A4R4G4B4 = {0x83, 0xA3}
FMT_ARGB8888 = {0x85, 0xA5, 0xBE}
FMT_L8 = {0x81, 0xA1}
FMT_G8B8 = {0x8B, 0xAB}
FMT_DXT1 = {0x86, 0xA6}


@dataclass
class TplRecord:
    index: int
    record_offset: int
    raw0: int
    raw1: int
    width: int
    height: int
    pitch: int
    fmt: int
    flags: int
    data_offset: int
    wrap_s: int
    wrap_t: int
    filter_mode: int
    mip_flag: int
    layout: str
    payload_size: int
    span_size: int

    @property
    def end_offset(self) -> int:
        return self.data_offset + self.payload_size

    @property
    def gap_size(self) -> int:
        return max(0, self.span_size - self.payload_size)

    @property
    def class_name(self) -> str:
        return classify_tpl_fmt(self.fmt)


def classify_tpl_fmt(fmt: int) -> str:
    if fmt in FMT_A4R4G4B4:
        return "A4R4G4B4_16bpp"
    if fmt in FMT_ARGB8888:
        return "A8R8G8B8_32bpp"
    if fmt in FMT_L8:
        return "L8_8bpp"
    if fmt in FMT_G8B8:
        return "G8B8_16bpp_LA"
    if fmt in FMT_DXT1:
        return "DXT1"
    return f"unknown_0x{fmt:02X}"


def tpl_row_pitch(fmt: int, width: int, pitch: int) -> int:
    if fmt in FMT_ARGB8888:
        return pitch if pitch else width * 4
    if fmt in FMT_A4R4G4B4 or fmt in FMT_G8B8:
        return pitch if pitch else width * 2
    if fmt in FMT_L8:
        return pitch if pitch else width
    if fmt in FMT_DXT1:
        return pitch if pitch else ((width + 3) // 4) * 8
    return pitch


def tpl_expected_payload_size(fmt: int, width: int, height: int, pitch: int) -> Optional[int]:
    if width <= 0 or height <= 0:
        return None
    row = tpl_row_pitch(fmt, width, pitch)
    if row <= 0:
        return None
    if fmt in FMT_DXT1:
        min_row = ((width + 3) // 4) * 8
        if row < min_row:
            return None
        return row * ((height + 3) // 4)
    if fmt in FMT_ARGB8888:
        if row < width * 4:
            return None
    elif fmt in FMT_A4R4G4B4 or fmt in FMT_G8B8:
        if row < width * 2:
            return None
    elif fmt in FMT_L8:
        if row < width:
            return None
    else:
        return None
    return row * height


def parse_tpl_records(buf: bytes) -> List[TplRecord]:
    if len(buf) < 4:
        raise ValueError("TPL too small for count")
    count = struct.unpack_from(">I", buf, 0)[0]
    if count <= 0 or count > 4096:
        raise ValueError(f"invalid/unreasonable TPL record count: {count}")
    header_min = 4 + count * TPL_REC_SIZE
    if len(buf) < header_min:
        raise ValueError(f"TPL too small for {count} records")
    raw_records = []
    for i in range(count):
        off = 4 + i * TPL_REC_SIZE
        raw0, raw1, pitch = struct.unpack_from(">HHH", buf, off)
        fmt = buf[off + 6]
        flags = buf[off + 7]
        data_offset = struct.unpack_from(">I", buf, off + 8)[0]
        wrap_s, wrap_t, filter_mode, mip_flag = struct.unpack_from(">HHHH", buf, off + 12)
        raw_records.append((off, raw0, raw1, pitch, fmt, flags, data_offset, wrap_s, wrap_t, filter_mode, mip_flag))
    data_offsets = [r[6] for r in raw_records]
    records: List[TplRecord] = []
    for i, r in enumerate(raw_records):
        off, raw0, raw1, pitch, fmt, flags, data_offset, wrap_s, wrap_t, filter_mode, mip_flag = r
        if data_offset < header_min or data_offset > len(buf):
            raise ValueError(f"record {i}: invalid data_offset 0x{data_offset:X}")
        next_off = data_offsets[i + 1] if i + 1 < count else len(buf)
        if next_off < data_offset:
            raise ValueError(f"record {i}: non-monotonic data offsets")
        span = next_off - data_offset
        candidates = []
        for layout, width, height in (("height_width", raw1, raw0), ("width_height", raw0, raw1)):
            size = tpl_expected_payload_size(fmt, width, height, pitch)
            if size is None or data_offset + size > len(buf) or size > span:
                continue
            row = tpl_row_pitch(fmt, width, pitch)
            score = 0
            if size == span:
                score += 1000
            elif span - size in (0x10, 0x20, 0x40, 0x60, 0x80, 0x100, 0x800, 0x1000):
                score += 700
            else:
                score += max(0, 500 - (span - size) // 0x10)
            if layout == "height_width":
                score += 50
            if fmt in FMT_DXT1 and row == ((width + 3) // 4) * 8:
                score += 100
            if fmt in FMT_ARGB8888 and row == width * 4:
                score += 100
            if (fmt in FMT_A4R4G4B4 or fmt in FMT_G8B8) and row == width * 2:
                score += 100
            if fmt in FMT_L8 and row == width:
                score += 100
            candidates.append((score, layout, width, height, size))
        if not candidates:
            raise ValueError(
                f"record {i}: cannot infer payload for fmt=0x{fmt:02X}, raw=(0x{raw0:X},0x{raw1:X}), "
                f"pitch=0x{pitch:X}, span=0x{span:X}"
            )
        candidates.sort(reverse=True)
        _, layout, width, height, size = candidates[0]
        records.append(TplRecord(i, off, raw0, raw1, width, height, pitch, fmt, flags, data_offset,
                                 wrap_s, wrap_t, filter_mode, mip_flag, layout, size, span))
    return records


def tpl_payload(buf: bytes, rec: TplRecord) -> bytes:
    return buf[rec.data_offset:rec.data_offset + rec.payload_size]


def require_pillow() -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for PNG sidecars. Install with: pip install pillow")


def rgb565_to_rgba(c: int) -> Tuple[int, int, int, int]:
    r = ((c >> 11) & 0x1F) * 255 // 31
    g = ((c >> 5) & 0x3F) * 255 // 63
    b = (c & 0x1F) * 255 // 31
    return r, g, b, 255


def decode_tpl_a4(buf: bytes, rec: TplRecord):
    require_pillow()
    row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch)
    out = bytearray(rec.width * rec.height * 4)
    for y in range(rec.height):
        src = rec.data_offset + y * row
        for x in range(rec.width):
            o = src + x * 2
            px = (buf[o] << 8) | buf[o + 1]
            a = ((px >> 12) & 0xF) * 17
            r = ((px >> 8) & 0xF) * 17
            g = ((px >> 4) & 0xF) * 17
            b = (px & 0xF) * 17
            d = (y * rec.width + x) * 4
            out[d:d + 4] = bytes((r, g, b, a))
    return Image.frombytes("RGBA", (rec.width, rec.height), bytes(out))


def extract_tpl_a4_channels(buf: bytes, rec: TplRecord) -> Dict[str, object]:
    require_pillow()
    row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch)
    planes = {ch: bytearray(rec.width * rec.height) for ch in RAW_CHANNELS}
    for y in range(rec.height):
        src = rec.data_offset + y * row
        ri = y * rec.width
        for x in range(rec.width):
            o = src + x * 2
            px = (buf[o] << 8) | buf[o + 1]
            planes["A"][ri + x] = ((px >> 12) & 0xF) * 17
            planes["R"][ri + x] = ((px >> 8) & 0xF) * 17
            planes["G"][ri + x] = ((px >> 4) & 0xF) * 17
            planes["B"][ri + x] = (px & 0xF) * 17
    return {ch: Image.frombytes("L", (rec.width, rec.height), bytes(data)) for ch, data in planes.items()}


def decode_tpl_argb(buf: bytes, rec: TplRecord):
    require_pillow()
    row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch)
    out = bytearray(rec.width * rec.height * 4)
    for y in range(rec.height):
        src = rec.data_offset + y * row
        for x in range(rec.width):
            o = src + x * 4
            a, r, g, b = buf[o], buf[o + 1], buf[o + 2], buf[o + 3]
            d = (y * rec.width + x) * 4
            out[d:d + 4] = bytes((r, g, b, a))
    return Image.frombytes("RGBA", (rec.width, rec.height), bytes(out))


def decode_tpl_l8(buf: bytes, rec: TplRecord):
    require_pillow()
    row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch)
    out = bytearray(rec.width * rec.height)
    for y in range(rec.height):
        src = rec.data_offset + y * row
        out[y * rec.width:(y + 1) * rec.width] = buf[src:src + rec.width]
    return Image.frombytes("L", (rec.width, rec.height), bytes(out))


def decode_tpl_g8b8(buf: bytes, rec: TplRecord):
    require_pillow()
    row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch)
    out = bytearray(rec.width * rec.height * 2)
    for y in range(rec.height):
        src = rec.data_offset + y * row
        for x in range(rec.width):
            o = src + x * 2
            d = (y * rec.width + x) * 2
            out[d] = buf[o]
            out[d + 1] = buf[o + 1]
    return Image.frombytes("LA", (rec.width, rec.height), bytes(out))


def decode_dxt1_payload(payload: bytes, width: int, height: int, pitch: int):
    require_pillow()
    row_pitch = pitch if pitch else ((width + 3) // 4) * 8
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    img = Image.new("RGBA", (width, height))
    px = img.load()
    for by in range(blocks_y):
        row = by * row_pitch
        for bx in range(blocks_x):
            off = row + bx * 8
            if off + 8 > len(payload):
                continue
            c0, c1, bits = struct.unpack_from("<HHI", payload, off)
            colors = [rgb565_to_rgba(c0), rgb565_to_rgba(c1)]
            if c0 > c1:
                colors.append(tuple((2 * colors[0][i] + colors[1][i]) // 3 for i in range(3)) + (255,))
                colors.append(tuple((colors[0][i] + 2 * colors[1][i]) // 3 for i in range(3)) + (255,))
            else:
                colors.append(tuple((colors[0][i] + colors[1][i]) // 2 for i in range(3)) + (255,))
                colors.append((0, 0, 0, 0))
            for py in range(4):
                for px_i in range(4):
                    x = bx * 4 + px_i
                    y = by * 4 + py
                    if x < width and y < height:
                        idx = (bits >> (2 * (py * 4 + px_i))) & 3
                        px[x, y] = colors[idx]
    return img


def decode_tpl_record(buf: bytes, rec: TplRecord):
    if rec.fmt in FMT_A4R4G4B4:
        return decode_tpl_a4(buf, rec)
    if rec.fmt in FMT_ARGB8888:
        return decode_tpl_argb(buf, rec)
    if rec.fmt in FMT_L8:
        return decode_tpl_l8(buf, rec)
    if rec.fmt in FMT_G8B8:
        return decode_tpl_g8b8(buf, rec)
    if rec.fmt in FMT_DXT1:
        return decode_dxt1_payload(tpl_payload(buf, rec), rec.width, rec.height, rec.pitch)
    raise ValueError(f"record {rec.index}: unsupported TPL fmt 0x{rec.fmt:02X}")


def extract_tpl_asset(tpl_path: Path, dump_raw_payload: bool = True) -> Path:
    require_pillow()
    buf = tpl_path.read_bytes()
    records = parse_tpl_records(buf)
    out_dir = tpl_path.parent
    stem = tpl_path.stem
    meta = {
        "tool": "data_dat_cli.py/tpl",
        "template_file": tpl_path.name,
        "file_size": len(buf),
        "count": len(records),
        "record_size": TPL_REC_SIZE,
        "safe_rebuild": "template-based; preserve header/records/offsets/gaps; replace exact payload ranges only",
        "records": [],
    }
    for rec in records:
        base = f"{stem}_rec{rec.index:02d}_fmt{rec.fmt:02X}"
        rec_meta = asdict(rec)
        rec_meta.update({
            "format_class": rec.class_name,
            "payload_bin": None,
            "png": None,
            "channel_pngs": {},
            "note": "",
        })
        if dump_raw_payload:
            payload_name = f"{base}.payload.bin"
            (out_dir / payload_name).write_bytes(tpl_payload(buf, rec))
            rec_meta["payload_bin"] = payload_name
        img = decode_tpl_record(buf, rec)
        if rec.fmt in FMT_A4R4G4B4:
            preview = f"{base}_preview_rgba.png"
            img.save(out_dir / preview)
            rec_meta["png"] = preview
            for ch, ch_img in extract_tpl_a4_channels(buf, rec).items():
                ch_name = f"{base}_ch{ch}.png"
                ch_img.save(out_dir / ch_name)
                rec_meta["channel_pngs"][ch] = ch_name
            rec_meta["font_engine_order"] = ENGINE_FONT_ORDER
            rec_meta["note"] = "A4R4G4B4: rebuild uses chA/chR/chG/chB. Font engine pages are R,G,B,A per texture."
        else:
            png_name = f"{base}.png"
            img.save(out_dir / png_name)
            rec_meta["png"] = png_name
            if rec.fmt in FMT_DXT1:
                rec_meta["note"] = "DXT1: default rebuild preserves payload.bin; PNG re-encode is intentionally not done here."
        meta["records"].append(rec_meta)
    meta_path = out_dir / f"{stem}.tplmeta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def q4(v: int) -> int:
    return (int(v) * 15 + 127) // 255


def load_luma_4bit(path: Path, size: Tuple[int, int]) -> bytes:
    require_pillow()
    img = Image.open(path)
    if img.size != size:
        raise ValueError(f"{path}: size {img.size} != expected {size}")
    return bytes(q4(v) for v in img.convert("L").tobytes())


def pack_tpl_a4(template_payload: bytes, rec: TplRecord, paths: Dict[str, Path]) -> bytes:
    size = (rec.width, rec.height)
    row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch)
    planes = {ch: load_luma_4bit(paths[ch], size) for ch in RAW_CHANNELS}
    out = bytearray(template_payload)
    for y in range(rec.height):
        dst = y * row
        src_row = y * rec.width
        for x in range(rec.width):
            a = planes["A"][src_row + x]
            r = planes["R"][src_row + x]
            g = planes["G"][src_row + x]
            b = planes["B"][src_row + x]
            px = (a << 12) | (r << 8) | (g << 4) | b
            o = dst + x * 2
            out[o] = (px >> 8) & 0xFF
            out[o + 1] = px & 0xFF
    return bytes(out)


def pack_tpl_argb(template_payload: bytes, rec: TplRecord, png: Path) -> bytes:
    require_pillow()
    img = Image.open(png).convert("RGBA")
    if img.size != (rec.width, rec.height):
        raise ValueError(f"{png}: size {img.size} != expected {(rec.width, rec.height)}")
    src = img.tobytes()
    row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch)
    out = bytearray(template_payload)
    for y in range(rec.height):
        dst_row = y * row
        src_row = y * rec.width * 4
        for x in range(rec.width):
            s = src_row + x * 4
            r, g, b, a = src[s], src[s + 1], src[s + 2], src[s + 3]
            d = dst_row + x * 4
            out[d:d + 4] = bytes((a, r, g, b))
    return bytes(out)


def pack_tpl_l8(template_payload: bytes, rec: TplRecord, png: Path) -> bytes:
    require_pillow()
    img = Image.open(png).convert("L")
    if img.size != (rec.width, rec.height):
        raise ValueError(f"{png}: size {img.size} != expected {(rec.width, rec.height)}")
    src = img.tobytes()
    row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch)
    out = bytearray(template_payload)
    for y in range(rec.height):
        out[y * row:y * row + rec.width] = src[y * rec.width:(y + 1) * rec.width]
    return bytes(out)


def pack_tpl_g8b8(template_payload: bytes, rec: TplRecord, png: Path) -> bytes:
    require_pillow()
    img = Image.open(png)
    if img.size != (rec.width, rec.height):
        raise ValueError(f"{png}: size {img.size} != expected {(rec.width, rec.height)}")
    if img.mode == "LA":
        la = img.tobytes()
    else:
        rgba = img.convert("RGBA")
        r, g, b, a = rgba.split()
        l = Image.merge("RGB", (r, g, b)).convert("L")
        la = Image.merge("LA", (l, a)).tobytes()
    row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch)
    out = bytearray(template_payload)
    for y in range(rec.height):
        out[y * row:y * row + rec.width * 2] = la[y * rec.width * 2:(y + 1) * rec.width * 2]
    return bytes(out)


def rebuild_tpl_asset(template_tpl: Path, meta_path: Path) -> bytes:
    buf = bytearray(template_tpl.read_bytes())
    records = parse_tpl_records(buf)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rec_meta = {int(r["index"]): r for r in meta.get("records", [])}
    if len(records) != int(meta.get("count", len(records))):
        raise ValueError(f"{template_tpl}: record count differs from TPL metadata")
    base_dir = meta_path.parent
    for rec in records:
        m = rec_meta.get(rec.index)
        if m is None:
            raise ValueError(f"{template_tpl}: metadata missing TPL record {rec.index}")
        if int(m["fmt"]) != rec.fmt or int(m["width"]) != rec.width or int(m["height"]) != rec.height:
            raise ValueError(f"{template_tpl}: record {rec.index} differs from metadata")
        old = tpl_payload(buf, rec)
        if rec.fmt in FMT_A4R4G4B4:
            ch_meta = m.get("channel_pngs", {})
            paths = {ch: base_dir / ch_meta[ch] for ch in RAW_CHANNELS}
            new = pack_tpl_a4(old, rec, paths)
        elif rec.fmt in FMT_ARGB8888:
            new = pack_tpl_argb(old, rec, base_dir / m["png"])
        elif rec.fmt in FMT_L8:
            new = pack_tpl_l8(old, rec, base_dir / m["png"])
        elif rec.fmt in FMT_G8B8:
            new = pack_tpl_g8b8(old, rec, base_dir / m["png"])
        elif rec.fmt in FMT_DXT1:
            # Safe archive rebuilder never re-encodes DXT1 from PNG; use original payload.bin.
            payload_bin = m.get("payload_bin")
            if not payload_bin:
                raise ValueError(f"{template_tpl}: DXT1 record {rec.index} needs payload.bin")
            new = (base_dir / payload_bin).read_bytes()
        else:
            raise ValueError(f"{template_tpl}: unsupported TPL fmt 0x{rec.fmt:02X}")
        if len(new) != rec.payload_size:
            raise ValueError(f"{template_tpl}: record {rec.index} new payload size mismatch")
        buf[rec.data_offset:rec.data_offset + rec.payload_size] = new
    return bytes(buf)


# Drop-in-ish function name retained for callers.
def process_tpl_to_png(tpl_data: bytes, name: str, output_dir: str) -> bool:
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / Path(name).name
        tmp.write_bytes(tpl_data)
        extract_tpl_asset(tmp)
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# 8192-byte n/*.bin font/tile assets
# -----------------------------------------------------------------------------

NBIN_CHUNK_SIZE = 8192
NBIN_GLYPHS_PER_CHUNK = 16
NBIN_CELL = 32
NBIN_COLS = 4
NBIN_ROWS = 4
NBIN_BYTES_PER_GLYPH = 0x200
NBIN_BYTES_PER_ROW = NBIN_CELL // 2  # 4bpp, two pixels per byte.

# Backward-compatible aliases retained because the rest of the CLI and older
# callers use the previous bin8192 function names. The previous interpretation
# (256 glyphs, 16x16, 1bpp) was incorrect for this game's n/*.bin font chunks.
BIN8192_SIZE = NBIN_CHUNK_SIZE
BIN8192_GLYPHS = NBIN_GLYPHS_PER_CHUNK
BIN8192_CELL = NBIN_CELL
BIN8192_COLS = NBIN_COLS
BIN8192_ROWS = NBIN_ROWS


def nbin_chunk_base_code(rel_path: str) -> Optional[int]:
    """Return the 0xXXXX base code encoded in n/XXXX.bin, if present."""
    rel = rel_path.replace('\\', '/')
    parts = rel.split('/')
    if len(parts) != 2 or parts[0] != 'n' or not parts[1].lower().endswith('.bin'):
        return None
    stem = parts[1][:-4]
    if len(stem) != 4:
        return None
    try:
        value = int(stem, 16)
    except ValueError:
        return None
    return value if (value & 0xF) == 0 else None


def looks_like_n_bin(rel_path: str, data: bytes) -> bool:
    # Engine-side evidence and sample filenames show n/XXXX.bin is a 16-glyph
    # chunk, where XXXX is aligned to a 0x10 code block. Size is exactly 0x2000.
    return nbin_chunk_base_code(rel_path) is not None and len(data) == NBIN_CHUNK_SIZE


def _u4_to_u8(v: int) -> int:
    return (v & 0xF) * 17


def _u8_to_u4(v: int) -> int:
    return max(0, min(15, (int(v) * 15 + 127) // 255))


def bin8192_to_image(data: bytes):
    """Decode one n/XXXX.bin chunk to an editable 4x4 glyph grid PNG.

    Actual engine layout: 16 glyphs per file, each glyph is 32x32 pixels at
    4bpp. One glyph is 0x200 bytes; one chunk is 0x2000 bytes. Bytes are packed
    high-nibble first, then low-nibble.
    """
    require_pillow()
    if len(data) != NBIN_CHUNK_SIZE:
        raise ValueError(f"expected {NBIN_CHUNK_SIZE} bytes, got {len(data)}")
    img = Image.new("L", (NBIN_COLS * NBIN_CELL, NBIN_ROWS * NBIN_CELL), 0)
    px = img.load()
    for glyph in range(NBIN_GLYPHS_PER_CHUNK):
        gx = (glyph % NBIN_COLS) * NBIN_CELL
        gy = (glyph // NBIN_COLS) * NBIN_CELL
        base = glyph * NBIN_BYTES_PER_GLYPH
        for y in range(NBIN_CELL):
            row = base + y * NBIN_BYTES_PER_ROW
            for xb in range(NBIN_BYTES_PER_ROW):
                b = data[row + xb]
                x = xb * 2
                px[gx + x, gy + y] = _u4_to_u8(b >> 4)
                px[gx + x + 1, gy + y] = _u4_to_u8(b)
    return img


def image_to_bin8192(png_path: Path) -> bytes:
    require_pillow()
    img = Image.open(png_path).convert("L")
    expected = (NBIN_COLS * NBIN_CELL, NBIN_ROWS * NBIN_CELL)
    if img.size != expected:
        raise ValueError(f"{png_path}: size {img.size} != expected {expected}")
    px = img.load()
    out = bytearray(NBIN_CHUNK_SIZE)
    for glyph in range(NBIN_GLYPHS_PER_CHUNK):
        gx = (glyph % NBIN_COLS) * NBIN_CELL
        gy = (glyph // NBIN_COLS) * NBIN_CELL
        base = glyph * NBIN_BYTES_PER_GLYPH
        for y in range(NBIN_CELL):
            row = base + y * NBIN_BYTES_PER_ROW
            for xb in range(NBIN_BYTES_PER_ROW):
                x = xb * 2
                hi = _u8_to_u4(px[gx + x, gy + y])
                lo = _u8_to_u4(px[gx + x + 1, gy + y])
                out[row + xb] = (hi << 4) | lo
    return bytes(out)


def extract_bin8192(data: bytes, raw_path: Path) -> Path:
    img = bin8192_to_image(data)
    png_path = raw_path.with_suffix(raw_path.suffix + ".png")
    img.save(png_path)
    base_code = nbin_chunk_base_code(str(raw_path).replace('\\', '/'))
    meta = {
        "tool": "data_dat_cli.py/nbin4bpp",
        "template_file": raw_path.name,
        "format": "n/XXXX.bin font chunk: 16 glyphs, 32x32, 4bpp grayscale/alpha, 0x200 bytes per glyph, high nibble first",
        "chunk_size": NBIN_CHUNK_SIZE,
        "glyphs_per_chunk": NBIN_GLYPHS_PER_CHUNK,
        "cell": [NBIN_CELL, NBIN_CELL],
        "png_grid": [NBIN_COLS, NBIN_ROWS],
        "base_code_hex": f"0x{base_code:04X}" if base_code is not None else None,
        "glyph_code_hex": [f"0x{(base_code or 0) + i:04X}" for i in range(NBIN_GLYPHS_PER_CHUNK)] if base_code is not None else None,
        "png": png_path.name,
        "raw_sha256": sha256_bytes(data),
    }
    meta_path = raw_path.with_suffix(raw_path.suffix + ".bin8192.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return png_path


def process_bin_to_png(bin_data: bytes, name: str, output_dir: str) -> bool:
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / Path(name).name
        raw_path.write_bytes(bin_data)
        extract_bin8192(bin_data, raw_path)
        return True
    except Exception:
        return False



# -----------------------------------------------------------------------------
# Hash-aware sidecar extraction and template-based selective rebuild
# -----------------------------------------------------------------------------

IMAGE_ASSET_KINDS = {"tpl", "bin8192"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_to(root: Path, path: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def is_image_asset_kind(kind: str) -> bool:
    return kind in IMAGE_ASSET_KINDS


def extract_tpl_asset_from_bytes(tpl_data: bytes, out_path: Path, dump_raw_payload: bool = True, write_template: bool = True) -> Path:
    """Extract editable TPL sidecars near out_path.

    out_path is the logical archive path where the raw .tpl would live. In
    --image-only mode write_template=False, so only PNG/meta sidecars are emitted.
    """
    require_pillow()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if write_template:
        out_path.write_bytes(tpl_data)
    records = parse_tpl_records(tpl_data)
    out_dir = out_path.parent
    stem = out_path.stem
    meta = {
        "tool": "data_dat_cli.py/tpl",
        "template_file": out_path.name,
        "logical_archive_path": str(out_path).replace("\\", "/"),
        "file_size": len(tpl_data),
        "count": len(records),
        "record_size": TPL_REC_SIZE,
        "safe_rebuild": "template-based; preserve header/records/offsets/gaps; replace exact payload ranges only",
        "records": [],
    }
    for rec in records:
        base = f"{stem}_rec{rec.index:02d}_fmt{rec.fmt:02X}"
        rec_meta = asdict(rec)
        rec_meta.update({
            "format_class": rec.class_name,
            "payload_bin": None,
            "png": None,
            "channel_pngs": {},
            "note": "",
            "reimport_from_png": rec.fmt not in FMT_DXT1,
        })
        if dump_raw_payload:
            payload_name = f"{base}.payload.bin"
            (out_dir / payload_name).write_bytes(tpl_payload(tpl_data, rec))
            rec_meta["payload_bin"] = payload_name
        img = decode_tpl_record(tpl_data, rec)
        if rec.fmt in FMT_A4R4G4B4:
            preview = f"{base}_preview_rgba.png"
            img.save(out_dir / preview)
            rec_meta["png"] = preview
            for ch, ch_img in extract_tpl_a4_channels(tpl_data, rec).items():
                ch_name = f"{base}_ch{ch}.png"
                ch_img.save(out_dir / ch_name)
                rec_meta["channel_pngs"][ch] = ch_name
            rec_meta["font_engine_order"] = ENGINE_FONT_ORDER
            rec_meta["note"] = "A4R4G4B4: rebuild uses chA/chR/chG/chB. Font engine pages are R,G,B,A per texture."
        else:
            png_name = f"{base}.png"
            img.save(out_dir / png_name)
            rec_meta["png"] = png_name
            if rec.fmt in FMT_DXT1:
                rec_meta["note"] = "DXT1: PNG is preview-only in this tool; default rebuild preserves payload.bin/template payload."
        meta["records"].append(rec_meta)
    meta_path = out_dir / f"{stem}.tplmeta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


# Override earlier helper with a path-compatible wrapper.
def extract_tpl_asset(tpl_path: Path, dump_raw_payload: bool = True) -> Path:
    return extract_tpl_asset_from_bytes(tpl_path.read_bytes(), tpl_path, dump_raw_payload=dump_raw_payload, write_template=True)


def rebuild_tpl_asset_from_bytes(template_data: bytes, meta_path: Path) -> bytes:
    buf = bytearray(template_data)
    records = parse_tpl_records(buf)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rec_meta = {int(r["index"]): r for r in meta.get("records", [])}
    if len(records) != int(meta.get("count", len(records))):
        raise ValueError(f"{meta_path}: record count differs from TPL metadata")
    base_dir = meta_path.parent
    for rec in records:
        m = rec_meta.get(rec.index)
        if m is None:
            raise ValueError(f"{meta_path}: metadata missing TPL record {rec.index}")
        if int(m["fmt"]) != rec.fmt or int(m["width"]) != rec.width or int(m["height"]) != rec.height:
            raise ValueError(f"{meta_path}: record {rec.index} differs from template")
        old = tpl_payload(buf, rec)
        if rec.fmt in FMT_A4R4G4B4:
            ch_meta = m.get("channel_pngs", {})
            paths = {ch: base_dir / ch_meta[ch] for ch in RAW_CHANNELS}
            new = pack_tpl_a4(old, rec, paths)
        elif rec.fmt in FMT_ARGB8888:
            new = pack_tpl_argb(old, rec, base_dir / m["png"])
        elif rec.fmt in FMT_L8:
            new = pack_tpl_l8(old, rec, base_dir / m["png"])
        elif rec.fmt in FMT_G8B8:
            new = pack_tpl_g8b8(old, rec, base_dir / m["png"])
        elif rec.fmt in FMT_DXT1:
            # No lossy PNG re-encode in data.dat CLI. Use payload.bin if present;
            # otherwise leave template payload unchanged.
            payload_bin = m.get("payload_bin")
            if payload_bin and (base_dir / payload_bin).exists():
                new = (base_dir / payload_bin).read_bytes()
            else:
                new = old
        else:
            raise ValueError(f"{meta_path}: unsupported TPL fmt 0x{rec.fmt:02X}")
        if len(new) != rec.payload_size:
            raise ValueError(f"{meta_path}: record {rec.index} new payload size mismatch")
        buf[rec.data_offset:rec.data_offset + rec.payload_size] = new
    return bytes(buf)


# Override earlier helper with a path-compatible wrapper.
def rebuild_tpl_asset(template_tpl: Path, meta_path: Path) -> bytes:
    return rebuild_tpl_asset_from_bytes(template_tpl.read_bytes(), meta_path)


def extract_bin8192_from_bytes(data: bytes, raw_path: Path, write_raw: bool = True) -> Path:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if write_raw:
        raw_path.write_bytes(data)
    img = bin8192_to_image(data)
    png_path = raw_path.with_suffix(raw_path.suffix + ".png")
    img.save(png_path)
    base_code = nbin_chunk_base_code(rel_to(raw_path.parents[1], raw_path) if len(raw_path.parents) > 1 else raw_path.name)
    meta = {
        "tool": "data_dat_cli.py/nbin4bpp",
        "template_file": raw_path.name,
        "format": "n/XXXX.bin font chunk: 16 glyphs, 32x32, 4bpp grayscale/alpha, 0x200 bytes per glyph, high nibble first",
        "chunk_size": NBIN_CHUNK_SIZE,
        "glyphs_per_chunk": NBIN_GLYPHS_PER_CHUNK,
        "cell": [NBIN_CELL, NBIN_CELL],
        "png_grid": [NBIN_COLS, NBIN_ROWS],
        "base_code_hex": f"0x{base_code:04X}" if base_code is not None else None,
        "glyph_code_hex": [f"0x{(base_code or 0) + i:04X}" for i in range(NBIN_GLYPHS_PER_CHUNK)] if base_code is not None else None,
        "png": png_path.name,
        "raw_sha256": sha256_bytes(data),
    }
    meta_path = raw_path.with_suffix(raw_path.suffix + ".bin8192.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return png_path


# Override earlier helper with a compatible signature.
def extract_bin8192(data: bytes, raw_path: Path) -> Path:
    return extract_bin8192_from_bytes(data, raw_path, write_raw=True)


def detect_asset_kind(rel_path: str, data: bytes) -> str:
    if rel_path.endswith(".tpl"):
        try:
            parse_tpl_records(data)
            return "tpl"
        except Exception:
            return "raw"
    if looks_like_n_bin(rel_path, data):
        return "bin8192"
    return "raw"


def safe_out_path(root: Path, rel_path: str) -> Path:
    if ".." in Path(rel_path).parts or Path(rel_path).is_absolute():
        raise ValueError(f"unsafe archive path: {rel_path}")
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def watched_files_for_entry(output_dir: Path, entry_record: Dict, include_raw: bool) -> List[Dict]:
    watched: List[Dict] = []
    raw_path = output_dir / entry_record["path"]
    kind = entry_record.get("asset_kind", "raw")
    sidecars = entry_record.get("sidecars", {})
    if include_raw and raw_path.exists():
        watched.append({"path": entry_record["path"], "role": "raw", "sha256": sha256_file(raw_path)})
    if kind == "tpl" and sidecars.get("tpl_meta"):
        tpl_meta = json.loads((output_dir / sidecars["tpl_meta"]).read_text(encoding="utf-8"))
        base = (output_dir / sidecars["tpl_meta"]).parent
        for rec in tpl_meta.get("records", []):
            if rec.get("reimport_from_png", True) is False:
                # DXT1 preview PNG is intentionally not watched because this CLI
                # does not encode DXT1 back from PNG.
                continue
            if rec.get("channel_pngs"):
                for ch, fname in sorted(rec["channel_pngs"].items()):
                    p = base / fname
                    if p.exists():
                        watched.append({"path": rel_to(output_dir, p), "role": f"tpl_channel_{ch}", "sha256": sha256_file(p)})
            elif rec.get("png"):
                p = base / rec["png"]
                if p.exists():
                    watched.append({"path": rel_to(output_dir, p), "role": "tpl_png", "sha256": sha256_file(p)})
            payload_bin = rec.get("payload_bin")
            if payload_bin:
                p = base / payload_bin
                if p.exists():
                    watched.append({"path": rel_to(output_dir, p), "role": "tpl_payload_bin", "sha256": sha256_file(p)})
    elif kind == "bin8192" and sidecars.get("png"):
        p = output_dir / sidecars["png"]
        if p.exists():
            watched.append({"path": rel_to(output_dir, p), "role": "bin8192_png", "sha256": sha256_file(p)})
    return watched


def extract_archive(archive_path: Path, output_dir: Path, decode_assets: bool = True,
                    use_hash: bool = False, image_only: bool = False) -> None:
    if image_only:
        decode_assets = True
    output_dir.mkdir(parents=True, exist_ok=True)
    blob = archive_path.read_bytes()
    meta, entries = parse_archive_table(archive_path)
    meta_entries = []
    extracted_count = 0
    skipped_non_images = 0
    for ent in entries:
        data, flags = decompress_archive_entry(blob, ent)
        kind = detect_asset_kind(ent.name, data) if decode_assets else "raw"
        sidecars: Dict[str, str] = {}
        raw_path = safe_out_path(output_dir, ent.name)

        write_raw = not image_only
        if image_only and not is_image_asset_kind(kind):
            skipped_non_images += 1
        else:
            if write_raw:
                raw_path.write_bytes(data)
                extracted_count += 1
            elif is_image_asset_kind(kind):
                # Ensure parent exists even though raw file is intentionally absent.
                raw_path.parent.mkdir(parents=True, exist_ok=True)

            if decode_assets and is_image_asset_kind(kind):
                try:
                    if kind == "tpl":
                        meta_path = extract_tpl_asset_from_bytes(
                            data,
                            raw_path,
                            dump_raw_payload=not image_only,
                            write_template=write_raw,
                        )
                        sidecars["tpl_meta"] = rel_to(output_dir, meta_path)
                    elif kind == "bin8192":
                        png_path = extract_bin8192_from_bytes(data, raw_path, write_raw=write_raw)
                        sidecars["png"] = rel_to(output_dir, png_path)
                        sidecars["bin_meta"] = rel_to(output_dir, raw_path.with_suffix(raw_path.suffix + ".bin8192.json"))
                    extracted_count += 1
                except Exception as e:
                    print(f"warning: failed to decode sidecar for {ent.name}: {e}", file=sys.stderr)
                    kind = "raw"
        ent.raw_sha256 = sha256_bytes(data)
        ent.asset_kind = kind
        ent.flags = flags
        ent.chunk_count = len(flags)
        rec = asdict(ent)
        rec["path"] = ent.name
        rec["sidecars"] = sidecars
        rec["extracted_raw"] = write_raw and raw_path.exists()
        rec["image_only_skipped"] = image_only and not is_image_asset_kind(kind)
        rec["hash_files"] = []
        meta_entries.append(rec)
        if image_only and not is_image_asset_kind(kind):
            print(f"skipped [{ent.index+1}/{len(entries)}] {ent.name} (non-image)")
        else:
            print(f"extracted [{ent.index+1}/{len(entries)}] {ent.name} ({kind})")

    meta["entries"] = meta_entries
    meta["decode_assets"] = decode_assets
    meta["image_only"] = image_only
    meta["use_hash"] = use_hash
    meta["extract_policy"] = "image_only" if image_only else "full"
    meta["template_archive"] = str(archive_path)
    meta["hash_semantics"] = (
        "Hashes are recorded for files emitted by extract that are relevant for reimport. "
        "Selective rebuild with --use-hash copies the template archive and appends/repoints only entries whose watched files changed."
        if use_hash else "disabled"
    )

    if use_hash:
        for rec in meta_entries:
            rec["hash_files"] = watched_files_for_entry(output_dir, rec, include_raw=not image_only)
    (output_dir / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"metadata: {output_dir / META_NAME}")
    if image_only:
        print(f"image-only: emitted image assets/sidecars; skipped {skipped_non_images} non-image entries")


def get_template_archive_path(meta: Dict, explicit_template: Optional[Path]) -> Path:
    if explicit_template is not None:
        return explicit_template
    template = meta.get("template_archive") or meta.get("archive")
    if not template:
        raise ValueError("metadata has no template archive path; pass --template")
    path = Path(template)
    if not path.exists():
        raise ValueError(f"template archive not found: {path}; pass --template")
    return path


def entry_original_data_from_template(template_blob: bytes, template_entries: List[ArchiveEntry], entry_meta: Dict) -> bytes:
    idx = int(entry_meta["index"])
    if idx < 0 or idx >= len(template_entries):
        raise ValueError(f"entry index {idx} not present in template archive")
    ent = template_entries[idx]
    if ent.name != entry_meta["path"]:
        raise ValueError(f"template entry name mismatch at {idx}: {ent.name} != {entry_meta['path']}")
    data, _ = decompress_archive_entry(template_blob, ent)
    return data


def current_changed_roles(root: Path, entry_meta: Dict) -> Tuple[bool, List[str]]:
    changed_roles: List[str] = []
    for item in entry_meta.get("hash_files", []):
        p = root / item["path"]
        if not p.exists():
            raise ValueError(f"watched file missing: {p}")
        now = sha256_file(p)
        if now != item.get("sha256"):
            changed_roles.append(item.get("role", "unknown"))
    return bool(changed_roles), changed_roles


def materialize_entry_data(root: Path, entry_meta: Dict, raw_only: bool = False,
                           template_data: Optional[bytes] = None,
                           prefer_raw: bool = False) -> bytes:
    rel_path = entry_meta["path"]
    raw_path = root / rel_path
    kind = entry_meta.get("asset_kind", "raw")
    sidecars = entry_meta.get("sidecars", {})
    if raw_only:
        if raw_path.exists():
            return raw_path.read_bytes()
        if template_data is not None:
            return template_data
        raise ValueError(f"raw file not extracted and no template data available: {rel_path}")
    if prefer_raw and raw_path.exists():
        return raw_path.read_bytes()
    if kind == "tpl" and sidecars.get("tpl_meta"):
        meta_path = root / sidecars["tpl_meta"]
        if meta_path.exists():
            if raw_path.exists():
                return rebuild_tpl_asset(raw_path, meta_path)
            if template_data is not None:
                return rebuild_tpl_asset_from_bytes(template_data, meta_path)
            raise ValueError(f"TPL template not available for {rel_path}")
    if kind == "bin8192" and sidecars.get("png"):
        png_path = root / sidecars["png"]
        if png_path.exists():
            return image_to_bin8192(png_path)
    if raw_path.exists():
        return raw_path.read_bytes()
    if template_data is not None:
        return template_data
    raise ValueError(f"cannot materialize entry: {rel_path}")


def rebuild_archive_full(extracted_dir: Path, output_dat: Path, raw_only: bool = False,
                         chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    meta_path = extracted_dir / META_NAME
    if not meta_path.exists():
        raise ValueError(f"missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    entries = meta.get("entries", [])
    if not entries:
        raise ValueError("metadata has no entries")
    entries_offset = int(meta.get("entries_offset", 16))
    pre_entries = bytes.fromhex(meta.get("pre_entries_hex", ""))
    if len(pre_entries) != entries_offset:
        pre_entries = struct.pack(">IIII", len(entries), meta.get("header_unknown0", 0), meta.get("header_unknown1", 0), entries_offset)
        if len(pre_entries) < entries_offset:
            pre_entries += b"\x00" * (entries_offset - len(pre_entries))
    pre_entries = bytearray(pre_entries)
    struct.pack_into(">I", pre_entries, 0, len(entries))
    if entries_offset >= 16:
        struct.pack_into(">I", pre_entries, 12, entries_offset)

    names = bytearray()
    name_offsets: List[int] = []
    names_offset_start = entries_offset + len(entries) * ARCHIVE_ENTRY_SIZE
    for ent in entries:
        name_offsets.append(names_offset_start + len(names))
        names += ent["path"].encode("latin1") + b"\x00"
    while len(names) % 4:
        names += b"\x00"

    data_start = names_offset_start + len(names)
    output_dat.parent.mkdir(parents=True, exist_ok=True)
    out = bytearray(pre_entries)
    if len(out) < entries_offset:
        out += b"\x00" * (entries_offset - len(out))
    out += b"\x00" * (len(entries) * ARCHIVE_ENTRY_SIZE)
    out += names
    if len(out) < data_start:
        out += b"\x00" * (data_start - len(out))

    # If this was image-only, full rebuild still needs template data for entries
    # that were not extracted as raw files.
    template_blob = None
    template_entries = None
    if meta.get("image_only"):
        template_path = get_template_archive_path(meta, None)
        template_blob = template_path.read_bytes()
        _, template_entries = parse_archive_table(template_path)

    new_entries = []
    for idx, ent in enumerate(entries):
        template_data = None
        if template_blob is not None and template_entries is not None:
            template_data = entry_original_data_from_template(template_blob, template_entries, ent)
        data = materialize_entry_data(extracted_dir, ent, raw_only=raw_only, template_data=template_data)
        file_offset = len(out)
        comp = b"" if len(data) == 0 else compress_archive_data(data, chunk_size=chunk_size)
        out += comp
        new_entries.append((name_offsets[idx], len(comp), int(ent.get("unknown0", 0)), len(data), int(ent.get("unknown1", 0)), file_offset))
        print(f"packed [{idx+1}/{len(entries)}] {ent['path']} size=0x{len(data):X} comp=0x{len(comp):X}")

    for idx, rec in enumerate(new_entries):
        off = entries_offset + idx * ARCHIVE_ENTRY_SIZE
        struct.pack_into(">IIIIII", out, off, *rec)
    output_dat.write_bytes(bytes(out))
    print(f"rebuilt archive: {output_dat}")


def selective_reimport_archive(extracted_dir: Path, output_dat: Path, template_archive: Optional[Path] = None,
                               raw_only: bool = False, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    meta_path = extracted_dir / META_NAME
    if not meta_path.exists():
        raise ValueError(f"missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("use_hash"):
        raise ValueError("metadata was not extracted with --use-hash; cannot selective rebuild")
    entries = meta.get("entries", [])
    template_path = get_template_archive_path(meta, template_archive)
    template_blob = bytearray(template_path.read_bytes())
    _, template_entries = parse_archive_table(template_path)
    if len(template_entries) != len(entries):
        raise ValueError("template archive entry count differs from metadata")
    out = bytearray(template_blob)
    changed = 0
    skipped = 0
    for ent_meta in entries:
        is_changed, roles = current_changed_roles(extracted_dir, ent_meta)
        if not is_changed:
            skipped += 1
            continue
        idx = int(ent_meta["index"])
        template_ent = template_entries[idx]
        if template_ent.name != ent_meta["path"]:
            raise ValueError(f"template entry mismatch at {idx}")
        template_data, _ = decompress_archive_entry(template_blob, template_ent)
        prefer_raw = any(role == "raw" for role in roles) and not any(role.startswith("tpl_") or role == "bin8192_png" for role in roles)
        new_data = materialize_entry_data(
            extracted_dir,
            ent_meta,
            raw_only=raw_only,
            template_data=template_data,
            prefer_raw=prefer_raw,
        )
        comp = b"" if len(new_data) == 0 else compress_archive_data(new_data, chunk_size=chunk_size)
        file_offset = len(out)
        out += comp
        entry_off = int(meta["entries_offset"]) + idx * ARCHIVE_ENTRY_SIZE
        struct.pack_into(">IIIIII", out, entry_off,
                         template_ent.name_offset,
                         len(comp),
                         int(ent_meta.get("unknown0", template_ent.unknown0)),
                         len(new_data),
                         int(ent_meta.get("unknown1", template_ent.unknown1)),
                         file_offset)
        changed += 1
        print(f"reimported [{idx+1}/{len(entries)}] {ent_meta['path']} roles={','.join(roles)} size=0x{len(new_data):X} comp=0x{len(comp):X}")
    output_dat.parent.mkdir(parents=True, exist_ok=True)
    output_dat.write_bytes(bytes(out))
    print(f"selective rebuild: changed={changed}, skipped={skipped}, output={output_dat}")


def rebuild_archive(extracted_dir: Path, output_dat: Path, raw_only: bool = False,
                    chunk_size: int = DEFAULT_CHUNK_SIZE, use_hash: bool = False,
                    template_archive: Optional[Path] = None) -> None:
    meta_path = extracted_dir / META_NAME
    if not meta_path.exists():
        raise ValueError(f"missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if use_hash or meta.get("image_only"):
        # image-only extraction is intended to be paired with template-based
        # rebuild. If --use-hash is omitted for image-only, reimport all watched
        # image entries by falling back to full rebuild with template data.
        if use_hash:
            selective_reimport_archive(extracted_dir, output_dat, template_archive=template_archive, raw_only=raw_only, chunk_size=chunk_size)
        else:
            rebuild_archive_full(extracted_dir, output_dat, raw_only=raw_only, chunk_size=chunk_size)
    else:
        rebuild_archive_full(extracted_dir, output_dat, raw_only=raw_only, chunk_size=chunk_size)


def print_archive_info(archive_path: Path) -> None:
    meta, entries = parse_archive_table(archive_path)
    print(f"file: {archive_path}")
    print(f"size: 0x{meta['archive_size']:X}")
    print(f"count: {meta['file_count']} entries_offset=0x{meta['entries_offset']:X}")
    for ent in entries:
        print(
            f"[{ent.index:04d}] {ent.name} name=0x{ent.name_offset:X} "
            f"comp=0x{ent.compressed_size:X} uncomp=0x{ent.uncompressed_size:X} data=0x{ent.file_offset:X} "
            f"unk=({ent.unknown0},{ent.unknown1})"
        )


# -----------------------------------------------------------------------------
# Self-tests
# -----------------------------------------------------------------------------


def self_test() -> None:
    print("self-test: bin8192 roundtrip")
    sample = bytes(((i * 37 + 13) & 0xFF) for i in range(BIN8192_SIZE))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "n" / "8000.bin"
        p.parent.mkdir()
        extract_bin8192_from_bytes(sample, p, write_raw=True)
        rebuilt = image_to_bin8192(p.with_suffix(p.suffix + ".png"))
        assert rebuilt == sample, "bin8192 PNG roundtrip failed"
    print("  OK")

    print("self-test: TPL roundtrip on local samples")
    samples = [p for p in [Path("/mnt/data/font1.tpl"), Path("/mnt/data/thumb.tpl"), Path("/mnt/data/v199011.tpl")] if p.exists()]
    for tpl in samples:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / tpl.name
            shutil.copy2(tpl, work)
            meta_path = extract_tpl_asset(work)
            rebuilt = rebuild_tpl_asset(work, meta_path)
            assert rebuilt == work.read_bytes(), f"TPL roundtrip failed: {tpl}"
            print(f"  OK {tpl.name}")

    print("self-test: archive image-only/hash selective synthetic")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "root"
        (root / "n").mkdir(parents=True)
        (root / "n" / "8000.bin").write_bytes(sample)
        (root / "plain.txt").write_bytes(b"hello")
        entries = []
        for idx, rel in enumerate(["n/8000.bin", "plain.txt"]):
            raw = (root / rel).read_bytes()
            kind = detect_asset_kind(rel, raw)
            sidecars = {}
            if kind == "bin8192":
                png = extract_bin8192_from_bytes(raw, root / rel, write_raw=True)
                sidecars["png"] = str(png.relative_to(root)).replace("\\", "/")
            entries.append({"index": idx, "path": rel, "unknown0": 0, "unknown1": 0, "asset_kind": kind, "sidecars": sidecars})
        meta = {"entries_offset": 16, "pre_entries_hex": struct.pack(">IIII", 2, 0, 0, 16).hex(), "entries": entries}
        (root / META_NAME).write_text(json.dumps(meta), encoding="utf-8")
        dat = td / "test.dat"
        rebuild_archive_full(root, dat)
        out = td / "imgonly"
        extract_archive(dat, out, use_hash=True, image_only=True)
        # No changes -> selective output should be byte-identical to template.
        out_dat = td / "sel.dat"
        selective_reimport_archive(out, out_dat, template_archive=dat)
        assert out_dat.read_bytes() == dat.read_bytes(), "selective unchanged archive should match template"
        # Modify one pixel in PNG and verify logical raw data changes on extract.
        png = out / "n" / "8000.bin.png"
        img = Image.open(png).convert("L")
        pix = img.load()
        pix[0, 0] = 0 if pix[0, 0] > 128 else 255
        img.save(png)
        out_dat2 = td / "sel2.dat"
        selective_reimport_archive(out, out_dat2, template_archive=dat)
        out2 = td / "out2"
        extract_archive(out_dat2, out2, decode_assets=False)
        assert (out2 / "plain.txt").read_bytes() == b"hello"
        assert (out2 / "n" / "8000.bin").read_bytes() != sample
    print("  OK")
    print("self-test OK")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Safe extractor/rebuilder for this game's data.dat archive")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info")
    p.add_argument("archive", type=Path)

    p = sub.add_parser("extract")
    p.add_argument("archive", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--no-decode-assets", action="store_true", help="only write raw decompressed files, no PNG/TPL sidecars")
    p.add_argument("--image-only", action="store_true", help="only emit editable image assets/sidecars; non-image raw files are not written")
    p.add_argument("--use-hash", action="store_true", help="store SHA-256 hashes for emitted reimportable files")

    p = sub.add_parser("rebuild")
    p.add_argument("extracted_dir", type=Path)
    p.add_argument("output_dat", type=Path)
    p.add_argument("--raw-only", action="store_true", help="ignore editable TPL/BIN sidecars and pack raw files only")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--use-hash", action="store_true", help="copy template archive and only reimport watched files whose hash changed")
    p.add_argument("--template", type=Path, help="original data.dat to use as selective/template source; overrides metadata path")

    sub.add_parser("self-test")

    args = ap.parse_args()
    try:
        if args.cmd == "info":
            print_archive_info(args.archive)
        elif args.cmd == "extract":
            extract_archive(args.archive, args.output_dir, decode_assets=not args.no_decode_assets,
                            use_hash=args.use_hash, image_only=args.image_only)
        elif args.cmd == "rebuild":
            rebuild_archive(args.extracted_dir, args.output_dat, raw_only=args.raw_only,
                            chunk_size=args.chunk_size, use_hash=args.use_hash,
                            template_archive=args.template)
        elif args.cmd == "self-test":
            self_test()
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
