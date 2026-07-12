#!/usr/bin/env python3
"""
tpl_image_tool.py

General extractor/rebuilder for this game's record-based .tpl texture files.

This is NOT the font-only pages tool. It supports the texture formats observed in
this batch:

  0xA3 / 0x83  A4R4G4B4, big-endian 16-bit pixels
               For font atlases, editable pages are exported as R/G/B/A channel PNGs.
  0x85 / 0xA5 / 0xBE  A8R8G8B8-like 32bpp payload, stored as bytes A,R,G,B.
  0x81 / 0xA1  8bpp single-channel payload.
  0x8B / 0xAB  G8B8 two-channel payload, exported as PNG LA (L, alpha).
  0x84 / 0xA4  R5G6B5 / RGB565, big-endian 16-bit pixels.
  0x86 / 0xA6  DXT1 block-compressed payload.

Safe rebuild model:
  - Use the original .tpl as template.
  - Preserve count, records, offsets, flags, padding, gaps, and file size.
  - Replace only the exact payload byte range of each record.
  - DXT1 is preserved byte-identical from extracted .bin by default. Re-encoding
    from PNG is available with --encode-dxt1, but it is lossy.

Requires Pillow:
  pip install pillow
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise SystemExit("Pillow is required. Install with: pip install pillow") from e

REC_SIZE = 0x14
ENGINE_FONT_ORDER = ["R", "G", "B", "A"]
RAW_CHANNELS = ["A", "R", "G", "B"]
A4_SHIFT = {"A": 12, "R": 8, "G": 4, "B": 0}

FMT_A4R4G4B4 = {0x83, 0xA3}
FMT_ARGB8888 = {0x85, 0xA5, 0xBE}
FMT_L8 = {0x81, 0xA1}
FMT_G8B8 = {0x8B, 0xAB}
FMT_RGB565 = {0x84, 0xA4}
FMT_DXT1 = {0x86, 0xA6}


# ---------------------------------------------------------------------------
# Storage layout helpers
# ---------------------------------------------------------------------------
# Some game TPLs (for example extra.tpl and ci.tpl) store ordinary uncompressed
# texels in a PS3/GCM-style Morton tiled order instead of linear rows.  The
# record format byte is still the same (0x81 L8, 0x85 ARGB8888, etc.); in the
# observed files flags bit 0 marks this swizzled storage.  Decode functions
# expose normal row-major PNGs, and pack functions swizzle PNG pixels back into
# the original payload layout so template-based rebuild remains byte-stable.

def _is_pow2(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def is_swizzled_storage(rec: Record) -> bool:
    return (
        (rec.flags & 0x01) != 0
        and rec.fmt not in FMT_DXT1
        and _is_pow2(rec.width)
        and _is_pow2(rec.height)
    )


def storage_layout_name(rec: Record) -> str:
    return "morton_xy_swizzled" if is_swizzled_storage(rec) else "linear"


def morton_index_xy(x: int, y: int, width: int, height: int) -> int:
    """Morton order used by this game's swizzled rectangular pow2 textures.

    Bits are interleaved as x bit, then y bit, skipping dimensions once their
    bit range is exhausted.  This handles both square and rectangular pow2
    textures such as 1024x128 and 16x256.
    """
    off = 0
    shift = 0
    bit = 1
    while bit < width or bit < height:
        if bit < width:
            if x & bit:
                off |= 1 << shift
            shift += 1
        if bit < height:
            if y & bit:
                off |= 1 << shift
            shift += 1
        bit <<= 1
    return off


def texel_offset_in_file(rec: Record, x: int, y: int, bpp: int) -> int:
    if is_swizzled_storage(rec):
        return rec.data_offset + morton_index_xy(x, y, rec.width, rec.height) * bpp
    row_pitch = row_pitch_for(rec.fmt, rec.width, rec.pitch)
    return rec.data_offset + y * row_pitch + x * bpp


def texel_offset_in_payload(rec: Record, x: int, y: int, bpp: int) -> int:
    if is_swizzled_storage(rec):
        return morton_index_xy(x, y, rec.width, rec.height) * bpp
    row_pitch = row_pitch_for(rec.fmt, rec.width, rec.pitch)
    return y * row_pitch + x * bpp


@dataclass
class Record:
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
        return classify_fmt(self.fmt)


def classify_fmt(fmt: int) -> str:
    if fmt in FMT_A4R4G4B4:
        return "A4R4G4B4_16bpp"
    if fmt in FMT_ARGB8888:
        return "A8R8G8B8_32bpp"
    if fmt in FMT_L8:
        return "L8_8bpp"
    if fmt in FMT_G8B8:
        return "G8B8_16bpp_LA"
    if fmt in FMT_RGB565:
        return "RGB565_16bpp"
    if fmt in FMT_DXT1:
        return "DXT1"
    return f"unknown_0x{fmt:02X}"


def row_pitch_for(fmt: int, width: int, pitch: int) -> int:
    if fmt in FMT_ARGB8888:
        return pitch if pitch else width * 4
    if fmt in FMT_A4R4G4B4 or fmt in FMT_G8B8 or fmt in FMT_RGB565:
        return pitch if pitch else width * 2
    if fmt in FMT_L8:
        return pitch if pitch else width
    if fmt in FMT_DXT1:
        return pitch if pitch else ((width + 3) // 4) * 8
    # Unknown: assume explicit pitch is meaningful; otherwise cannot infer.
    return pitch


def expected_payload_size(fmt: int, width: int, height: int, pitch: int) -> Optional[int]:
    if width <= 0 or height <= 0:
        return None
    row_pitch = row_pitch_for(fmt, width, pitch)
    if row_pitch <= 0:
        return None
    if fmt in FMT_DXT1:
        min_row = ((width + 3) // 4) * 8
        if row_pitch < min_row:
            return None
        return row_pitch * ((height + 3) // 4)
    if fmt in FMT_ARGB8888:
        if row_pitch < width * 4:
            return None
    elif fmt in FMT_A4R4G4B4 or fmt in FMT_G8B8 or fmt in FMT_RGB565:
        if row_pitch < width * 2:
            return None
    elif fmt in FMT_L8:
        if row_pitch < width:
            return None
    else:
        return None
    return row_pitch * height


def parse_records(buf: bytes) -> List[Record]:
    if len(buf) < 4:
        raise ValueError("file too small for TPL count")
    count = struct.unpack_from(">I", buf, 0)[0]
    if count <= 0 or count > 4096:
        raise ValueError(f"invalid/unreasonable record count: {count}")
    header_min = 4 + count * REC_SIZE
    if len(buf) < header_min:
        raise ValueError(f"file too small for {count} records")

    raw_records = []
    for i in range(count):
        off = 4 + i * REC_SIZE
        raw0, raw1, pitch = struct.unpack_from(">HHH", buf, off)
        fmt = buf[off + 6]
        flags = buf[off + 7]
        data_offset = struct.unpack_from(">I", buf, off + 8)[0]
        wrap_s, wrap_t, filter_mode, mip_flag = struct.unpack_from(">HHHH", buf, off + 12)
        raw_records.append((off, raw0, raw1, pitch, fmt, flags, data_offset,
                            wrap_s, wrap_t, filter_mode, mip_flag))

    data_offsets = [r[6] for r in raw_records]
    records: List[Record] = []
    for i, r in enumerate(raw_records):
        off, raw0, raw1, pitch, fmt, flags, data_offset, wrap_s, wrap_t, filter_mode, mip_flag = r
        if data_offset < header_min or data_offset > len(buf):
            raise ValueError(f"record {i}: invalid data_offset 0x{data_offset:X}")
        next_off = data_offsets[i + 1] if i + 1 < count else len(buf)
        if next_off < data_offset:
            raise ValueError(f"record {i}: non-monotonic data offsets")
        span_size = next_off - data_offset

        candidates = []
        for layout, width, height in (
            ("height_width", raw1, raw0),
            ("width_height", raw0, raw1),
        ):
            size = expected_payload_size(fmt, width, height, pitch)
            if size is None:
                continue
            if data_offset + size > len(buf):
                continue
            if size > span_size:
                continue
            row_pitch = row_pitch_for(fmt, width, pitch)
            score = 0
            # Prefer exact span match, but allow alignment/gap.
            if size == span_size:
                score += 1000
            elif span_size - size in (0x10, 0x20, 0x40, 0x60, 0x80, 0x100, 0x800):
                score += 700
            else:
                score += max(0, 500 - (span_size - size) // 0x10)
            # Prefer the observed game convention: first field is height, second is width.
            if layout == "height_width":
                score += 50
            # Prefer pitch matching a natural row pitch.
            if fmt in FMT_DXT1 and row_pitch == ((width + 3) // 4) * 8:
                score += 100
            if fmt in FMT_ARGB8888 and row_pitch == width * 4:
                score += 100
            if (fmt in FMT_A4R4G4B4 or fmt in FMT_G8B8 or fmt in FMT_RGB565) and row_pitch == width * 2:
                score += 100
            if fmt in FMT_L8 and row_pitch == width:
                score += 100
            candidates.append((score, layout, width, height, size))

        if not candidates:
            raise ValueError(
                f"record {i}: cannot infer payload size/dimensions for fmt=0x{fmt:02X}, "
                f"raw_dims=(0x{raw0:X},0x{raw1:X}), pitch=0x{pitch:X}, span=0x{span_size:X}"
            )
        candidates.sort(reverse=True)
        _, layout, width, height, payload_size = candidates[0]
        records.append(Record(
            index=i, record_offset=off, raw0=raw0, raw1=raw1, width=width, height=height,
            pitch=pitch, fmt=fmt, flags=flags, data_offset=data_offset,
            wrap_s=wrap_s, wrap_t=wrap_t, filter_mode=filter_mode, mip_flag=mip_flag,
            layout=layout, payload_size=payload_size, span_size=span_size,
        ))
    return records


def record_payload(buf: bytes, rec: Record) -> bytes:
    return buf[rec.data_offset:rec.data_offset + rec.payload_size]


def iter_u16be_row(buf: bytes, rec: Record, y: int) -> Iterable[int]:
    row_pitch = row_pitch_for(rec.fmt, rec.width, rec.pitch)
    base = rec.data_offset + y * row_pitch
    for x in range(rec.width):
        o = base + x * 2
        yield (buf[o] << 8) | buf[o + 1]


def decode_a4r4g4b4(buf: bytes, rec: Record) -> Image.Image:
    out = bytearray(rec.width * rec.height * 4)
    for y in range(rec.height):
        for x in range(rec.width):
            o = texel_offset_in_file(rec, x, y, 2)
            px = (buf[o] << 8) | buf[o + 1]
            a = ((px >> 12) & 0xF) * 17
            r = ((px >> 8) & 0xF) * 17
            g = ((px >> 4) & 0xF) * 17
            b = (px & 0xF) * 17
            d = (y * rec.width + x) * 4
            out[d:d + 4] = bytes((r, g, b, a))
    return Image.frombytes("RGBA", (rec.width, rec.height), bytes(out))

def extract_a4_channels(buf: bytes, rec: Record) -> Dict[str, Image.Image]:
    planes: Dict[str, bytearray] = {ch: bytearray(rec.width * rec.height) for ch in RAW_CHANNELS}
    for y in range(rec.height):
        row_i = y * rec.width
        for x in range(rec.width):
            o = texel_offset_in_file(rec, x, y, 2)
            px = (buf[o] << 8) | buf[o + 1]
            planes["A"][row_i + x] = ((px >> 12) & 0xF) * 17
            planes["R"][row_i + x] = ((px >> 8) & 0xF) * 17
            planes["G"][row_i + x] = ((px >> 4) & 0xF) * 17
            planes["B"][row_i + x] = (px & 0xF) * 17
    return {ch: Image.frombytes("L", (rec.width, rec.height), bytes(data)) for ch, data in planes.items()}

def decode_argb8888(buf: bytes, rec: Record) -> Image.Image:
    out = bytearray(rec.width * rec.height * 4)
    for y in range(rec.height):
        for x in range(rec.width):
            o = texel_offset_in_file(rec, x, y, 4)
            a, r, g, b = buf[o], buf[o + 1], buf[o + 2], buf[o + 3]
            d = (y * rec.width + x) * 4
            out[d:d + 4] = bytes((r, g, b, a))
    return Image.frombytes("RGBA", (rec.width, rec.height), bytes(out))

def decode_l8(buf: bytes, rec: Record) -> Image.Image:
    out = bytearray(rec.width * rec.height)
    for y in range(rec.height):
        for x in range(rec.width):
            out[y * rec.width + x] = buf[texel_offset_in_file(rec, x, y, 1)]
    return Image.frombytes("L", (rec.width, rec.height), bytes(out))

def decode_g8b8(buf: bytes, rec: Record) -> Image.Image:
    # CELL_GCM_TEXTURE_G8B8 family. For editing, expose as LA: first byte=L, second byte=alpha.
    out = bytearray(rec.width * rec.height * 2)
    for y in range(rec.height):
        for x in range(rec.width):
            o = texel_offset_in_file(rec, x, y, 2)
            d = (y * rec.width + x) * 2
            out[d] = buf[o]
            out[d + 1] = buf[o + 1]
    return Image.frombytes("LA", (rec.width, rec.height), bytes(out))

def rgb565_to_rgba(c: int) -> Tuple[int, int, int, int]:
    r = ((c >> 11) & 0x1F) * 255 // 31
    g = ((c >> 5) & 0x3F) * 255 // 63
    b = (c & 0x1F) * 255 // 31
    return r, g, b, 255


def decode_dxt1_payload(payload: bytes, width: int, height: int, pitch: int) -> Image.Image:
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


def decode_rgb565(buf: bytes, rec: Record) -> Image.Image:
    out = bytearray(rec.width * rec.height * 4)
    for y in range(rec.height):
        for x in range(rec.width):
            o = texel_offset_in_file(rec, x, y, 2)
            px = (buf[o] << 8) | buf[o + 1]
            r = ((px >> 11) & 0x1F) * 255 // 31
            g = ((px >> 5) & 0x3F) * 255 // 63
            b = (px & 0x1F) * 255 // 31
            d = (y * rec.width + x) * 4
            out[d:d + 4] = bytes((r, g, b, 255))
    return Image.frombytes("RGBA", (rec.width, rec.height), bytes(out))


def decode_record(buf: bytes, rec: Record) -> Image.Image:
    if rec.fmt in FMT_A4R4G4B4:
        return decode_a4r4g4b4(buf, rec)
    if rec.fmt in FMT_ARGB8888:
        return decode_argb8888(buf, rec)
    if rec.fmt in FMT_L8:
        return decode_l8(buf, rec)
    if rec.fmt in FMT_G8B8:
        return decode_g8b8(buf, rec)
    if rec.fmt in FMT_RGB565:
        return decode_rgb565(buf, rec)
    if rec.fmt in FMT_DXT1:
        return decode_dxt1_payload(record_payload(buf, rec), rec.width, rec.height, rec.pitch)
    raise ValueError(f"record {rec.index}: unsupported format 0x{rec.fmt:02X}")


def q4(v: int) -> int:
    return (int(v) * 15 + 127) // 255


def load_luma_4bit(path: Path, size: Tuple[int, int]) -> bytes:
    img = Image.open(path)
    if img.size != size:
        raise ValueError(f"{path.name}: size {img.size} != expected {size}")
    lum = img.convert("L").tobytes()
    return bytes(q4(v) for v in lum)


def pack_a4r4g4b4(template_payload: bytes, rec: Record, png_paths: Dict[str, Path]) -> bytes:
    size = (rec.width, rec.height)
    planes = {ch: load_luma_4bit(png_paths[ch], size) for ch in RAW_CHANNELS}
    out = bytearray(template_payload)
    for y in range(rec.height):
        src_row = y * rec.width
        for x in range(rec.width):
            a = planes["A"][src_row + x]
            r = planes["R"][src_row + x]
            g = planes["G"][src_row + x]
            b = planes["B"][src_row + x]
            px = (a << 12) | (r << 8) | (g << 4) | b
            o = texel_offset_in_payload(rec, x, y, 2)
            out[o] = (px >> 8) & 0xFF
            out[o + 1] = px & 0xFF
    return bytes(out)

def pack_argb8888(template_payload: bytes, rec: Record, png_path: Path) -> bytes:
    img = Image.open(png_path).convert("RGBA")
    if img.size != (rec.width, rec.height):
        raise ValueError(f"{png_path.name}: size {img.size} != expected {(rec.width, rec.height)}")
    src = img.tobytes()
    out = bytearray(template_payload)
    for y in range(rec.height):
        for x in range(rec.width):
            s = (y * rec.width + x) * 4
            r, g, b, a = src[s], src[s + 1], src[s + 2], src[s + 3]
            d = texel_offset_in_payload(rec, x, y, 4)
            out[d:d + 4] = bytes((a, r, g, b))
    return bytes(out)

def pack_l8(template_payload: bytes, rec: Record, png_path: Path) -> bytes:
    img = Image.open(png_path).convert("L")
    if img.size != (rec.width, rec.height):
        raise ValueError(f"{png_path.name}: size {img.size} != expected {(rec.width, rec.height)}")
    src = img.tobytes()
    out = bytearray(template_payload)
    for y in range(rec.height):
        for x in range(rec.width):
            out[texel_offset_in_payload(rec, x, y, 1)] = src[y * rec.width + x]
    return bytes(out)

def pack_g8b8(template_payload: bytes, rec: Record, png_path: Path) -> bytes:
    img = Image.open(png_path)
    if img.size != (rec.width, rec.height):
        raise ValueError(f"{png_path.name}: size {img.size} != expected {(rec.width, rec.height)}")
    if img.mode == "LA":
        la = img.tobytes()
    else:
        rgba = img.convert("RGBA")
        r, g, b, a = rgba.split()
        l = Image.merge("RGB", (r, g, b)).convert("L")
        la = Image.merge("LA", (l, a)).tobytes()
    out = bytearray(template_payload)
    for y in range(rec.height):
        for x in range(rec.width):
            s = (y * rec.width + x) * 2
            d = texel_offset_in_payload(rec, x, y, 2)
            out[d:d + 2] = la[s:s + 2]
    return bytes(out)

def pack_rgb565(template_payload: bytes, rec: Record, png_path: Path) -> bytes:
    img = Image.open(png_path).convert("RGBA")
    if img.size != (rec.width, rec.height):
        raise ValueError(f"{png_path.name}: size {img.size} != expected {(rec.width, rec.height)}")
    src = img.tobytes()
    out = bytearray(template_payload)
    for y in range(rec.height):
        for x in range(rec.width):
            s = (y * rec.width + x) * 4
            r, g, b = src[s], src[s + 1], src[s + 2]
            px = ((r * 31 + 127) // 255 << 11) | ((g * 63 + 127) // 255 << 5) | ((b * 31 + 127) // 255)
            d = texel_offset_in_payload(rec, x, y, 2)
            out[d] = (px >> 8) & 0xFF
            out[d + 1] = px & 0xFF
    return bytes(out)


def rgba_to_565(r: int, g: int, b: int) -> int:
    return ((r * 31 + 127) // 255 << 11) | ((g * 63 + 127) // 255 << 5) | ((b * 31 + 127) // 255)


def unpack_565(c: int) -> Tuple[int, int, int]:
    r, g, b, _ = rgb565_to_rgba(c)
    return r, g, b


def color_distance_sq(c: Tuple[int, int, int], p: Tuple[int, int, int]) -> int:
    return (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 + (c[2] - p[2]) ** 2


def encode_dxt1_image(img: Image.Image, width: int, height: int, pitch: int) -> bytes:
    rgba_img = img.convert("RGBA")
    if rgba_img.size != (width, height):
        raise ValueError(f"DXT1 PNG size {rgba_img.size} != expected {(width, height)}")
    pix = rgba_img.load()
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    row_pitch = pitch if pitch else blocks_x * 8
    if row_pitch < blocks_x * 8:
        raise ValueError("DXT1 pitch is smaller than block row size")
    out = bytearray(row_pitch * blocks_y)

    for by in range(blocks_y):
        for bx in range(blocks_x):
            samples = []
            has_alpha = False
            for py in range(4):
                for px in range(4):
                    x = min(width - 1, bx * 4 + px)
                    y = min(height - 1, by * 4 + py)
                    r, g, b, a = pix[x, y]
                    if a < 128:
                        has_alpha = True
                    samples.append((r, g, b, a))

            opaque = [(r, g, b) for r, g, b, a in samples if a >= 128]
            if not opaque:
                c0 = 0
                c1 = 0
                bits = 0xFFFFFFFF  # all transparent index 3 in c0 <= c1 mode
            else:
                # Choose endpoints by luminance extrema. This is simple but valid.
                min_col = min(opaque, key=lambda c: c[0] * 299 + c[1] * 587 + c[2] * 114)
                max_col = max(opaque, key=lambda c: c[0] * 299 + c[1] * 587 + c[2] * 114)
                c_min = rgba_to_565(*min_col)
                c_max = rgba_to_565(*max_col)
                if has_alpha:
                    # 3-color + transparent mode requires color0 <= color1.
                    c0, c1 = sorted((c_min, c_max))
                    p0 = unpack_565(c0)
                    p1 = unpack_565(c1)
                    palette = [p0, p1, tuple((p0[i] + p1[i]) // 2 for i in range(3)), (0, 0, 0)]
                else:
                    # 4-color mode requires color0 > color1.
                    c0, c1 = c_max, c_min
                    if c0 <= c1:
                        c0, c1 = c1, c0
                    p0 = unpack_565(c0)
                    p1 = unpack_565(c1)
                    palette = [
                        p0,
                        p1,
                        tuple((2 * p0[i] + p1[i]) // 3 for i in range(3)),
                        tuple((p0[i] + 2 * p1[i]) // 3 for i in range(3)),
                    ]
                bits = 0
                for i, (r, g, b, a) in enumerate(samples):
                    if has_alpha and a < 128:
                        idx = 3
                    else:
                        idx = min(range(3 if has_alpha else 4), key=lambda k: color_distance_sq(palette[k], (r, g, b)))
                    bits |= idx << (2 * i)
            dst = by * row_pitch + bx * 8
            struct.pack_into("<HHI", out, dst, c0, c1, bits)
    return bytes(out)


def extract_tpl(tpl_path: Path, out_dir: Path, name: Optional[str] = None, dump_raw: bool = True) -> bool:
    buf = tpl_path.read_bytes()
    records = parse_records(buf)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name or tpl_path.stem

    meta = {
        "tool": "tpl_image_tool.py",
        "source": str(tpl_path),
        "file_name": tpl_path.name,
        "file_size": len(buf),
        "count": len(records),
        "record_size": REC_SIZE,
        "safe_rebuild": "template-based; replace exact payload bytes only",
        "records": [],
    }

    for rec in records:
        base = f"{stem}_rec{rec.index:02d}_fmt{rec.fmt:02X}"
        payload_name = f"{base}.payload.bin"
        if dump_raw:
            (out_dir / payload_name).write_bytes(record_payload(buf, rec))

        rec_meta = asdict(rec)
        rec_meta.update({
            "format_class": rec.class_name,
            "storage_layout": storage_layout_name(rec),
            "payload_bin": payload_name if dump_raw else None,
            "png": None,
            "channel_pngs": {},
            "note": "",
        })

        img = decode_record(buf, rec)
        if rec.fmt in FMT_A4R4G4B4:
            # Save a RGBA preview plus editable raw channel pages.
            preview_name = f"{base}_preview_rgba.png"
            img.save(out_dir / preview_name)
            rec_meta["png"] = preview_name
            channels = extract_a4_channels(buf, rec)
            for ch in RAW_CHANNELS:
                ch_name = f"{base}_ch{ch}.png"
                channels[ch].save(out_dir / ch_name)
                rec_meta["channel_pngs"][ch] = ch_name
            rec_meta["font_engine_order"] = ENGINE_FONT_ORDER
            rec_meta["note"] = "A4R4G4B4: rebuild uses chA/chR/chG/chB PNGs; font engine page order is R,G,B,A."
        else:
            png_name = f"{base}.png"
            img.save(out_dir / png_name)
            rec_meta["png"] = png_name
            if rec.fmt in FMT_DXT1:
                rec_meta["note"] = "DXT1: default rebuild preserves payload.bin byte-identical; use --encode-dxt1 to rebuild from PNG lossy."
        meta["records"].append(rec_meta)

    meta_path = out_dir / f"{stem}.tplmeta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return True


def rebuild_tpl(template_tpl: Path, extracted_dir: Path, out_tpl: Path, meta_path: Optional[Path] = None,
                encode_dxt1: bool = False) -> None:
    buf = bytearray(template_tpl.read_bytes())
    records = parse_records(buf)
    if meta_path is None:
        metas = sorted(extracted_dir.glob("*.tplmeta.json"))
        if len(metas) != 1:
            raise ValueError("cannot auto-select metadata; pass --meta explicitly")
        meta_path = metas[0]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_records = {int(r["index"]): r for r in meta.get("records", [])}
    if len(records) != int(meta.get("count", len(records))):
        raise ValueError("template record count differs from metadata")

    for rec in records:
        m = meta_records.get(rec.index)
        if m is None:
            raise ValueError(f"metadata missing record {rec.index}")
        if int(m["fmt"]) != rec.fmt or int(m["width"]) != rec.width or int(m["height"]) != rec.height:
            raise ValueError(f"record {rec.index}: template differs from metadata")
        old_payload = record_payload(buf, rec)

        if rec.fmt in FMT_A4R4G4B4:
            ch_meta = m.get("channel_pngs", {})
            ch_paths = {ch: extracted_dir / ch_meta[ch] for ch in RAW_CHANNELS}
            for ch, p in ch_paths.items():
                if not p.exists():
                    raise ValueError(f"record {rec.index}: missing channel {ch}: {p}")
            new_payload = pack_a4r4g4b4(old_payload, rec, ch_paths)
        elif rec.fmt in FMT_ARGB8888:
            new_payload = pack_argb8888(old_payload, rec, extracted_dir / m["png"])
        elif rec.fmt in FMT_L8:
            new_payload = pack_l8(old_payload, rec, extracted_dir / m["png"])
        elif rec.fmt in FMT_G8B8:
            new_payload = pack_g8b8(old_payload, rec, extracted_dir / m["png"])
        elif rec.fmt in FMT_RGB565:
            new_payload = pack_rgb565(old_payload, rec, extracted_dir / m["png"])
        elif rec.fmt in FMT_DXT1:
            if encode_dxt1:
                img = Image.open(extracted_dir / m["png"])
                new_payload = encode_dxt1_image(img, rec.width, rec.height, rec.pitch)
            else:
                payload_bin = m.get("payload_bin")
                if not payload_bin:
                    raise ValueError(f"record {rec.index}: DXT1 needs payload.bin or --encode-dxt1")
                new_payload = (extracted_dir / payload_bin).read_bytes()
        else:
            raise ValueError(f"record {rec.index}: unsupported format 0x{rec.fmt:02X}")

        if len(new_payload) != rec.payload_size:
            raise ValueError(
                f"record {rec.index}: new payload size 0x{len(new_payload):X} != expected 0x{rec.payload_size:X}"
            )
        buf[rec.data_offset:rec.data_offset + rec.payload_size] = new_payload

    out_tpl.write_bytes(buf)



def rebuild_tpl_from_meta(template_tpl: Path, meta_path: Path, out_tpl: Path, encode_dxt1: bool = False) -> None:
    """Rebuild using a metadata JSON path; image/payload files are resolved relative to it."""
    rebuild_tpl(template_tpl, meta_path.parent, out_tpl, meta_path=meta_path, encode_dxt1=encode_dxt1)


def default_meta_path_for(tpl_path: Path, name: Optional[str] = None, out_dir: Optional[Path] = None) -> Path:
    """Metadata path used by the no-folder CLI."""
    stem = name or tpl_path.stem
    base_dir = out_dir if out_dir is not None else tpl_path.parent
    return base_dir / f"{stem}.tplmeta.json"


def print_info(tpl_path: Path) -> None:
    buf = tpl_path.read_bytes()
    records = parse_records(buf)
    print(f"file: {tpl_path}")
    print(f"size: 0x{len(buf):X} ({len(buf)} bytes)")
    print(f"records: {len(records)}")
    for rec in records:
        print(
            f"[{rec.index}] raw_dims=(0x{rec.raw0:04X},0x{rec.raw1:04X}) "
            f"layout={rec.layout} resolved={rec.width}x{rec.height} "
            f"pitch=0x{rec.pitch:X} fmt=0x{rec.fmt:02X} flags=0x{rec.flags:02X} "
            f"storage={storage_layout_name(rec)} class={rec.class_name} data=0x{rec.data_offset:X}..0x{rec.end_offset:X} "
            f"span=0x{rec.span_size:X} gap=0x{rec.gap_size:X}"
        )

def roundtrip_check(tpl_path: Path, encode_dxt1: bool = False) -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "extract"
        rebuilt = Path(td) / "rebuilt.tpl"
        extract_tpl(tpl_path, out_dir)
        meta = next(out_dir.glob("*.tplmeta.json"))
        rebuild_tpl(tpl_path, out_dir, rebuilt, meta_path=meta, encode_dxt1=encode_dxt1)
        a = tpl_path.read_bytes()
        b = rebuilt.read_bytes()
        if a != b:
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    raise SystemExit(
                        f"roundtrip differs at 0x{i:X}: original=0x{x:02X}, rebuilt=0x{y:02X}. "
                        f"For DXT1 this is expected only if --encode-dxt1 was used."
                    )
            raise SystemExit(f"roundtrip size mismatch: {len(a)} vs {len(b)}")
        print("roundtrip OK: rebuilt file is byte-identical")


# Drop-in compatible wrapper for the user's old function signature.
def process_tpl_to_png(tpl_data, name, output_dir):
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_tpl = out_dir / f"{name}.tpl.input.tmp"
        tmp_tpl.write_bytes(tpl_data)
        try:
            return extract_tpl(tmp_tpl, out_dir, name=name)
        finally:
            try:
                tmp_tpl.unlink()
            except OSError:
                pass
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract/rebuild this game's record-based .tpl textures safely; no font pages folder required")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info")
    p.add_argument("tpl", type=Path)

    p = sub.add_parser("extract")
    p.add_argument("tpl", type=Path)
    p.add_argument("--out-dir", "-o", type=Path, help="optional output directory; default is next to the .tpl")
    p.add_argument("--name", help="override output filename stem")
    p.add_argument("--no-raw", action="store_true", help="do not dump exact payload .bin files")

    p = sub.add_parser("rebuild")
    p.add_argument("template_tpl", type=Path)
    p.add_argument("meta_json", type=Path, help="metadata JSON produced by extract; PNG/bin files are resolved relative to it")
    p.add_argument("out_tpl", type=Path)
    p.add_argument("--encode-dxt1", action="store_true", help="rebuild DXT1 from PNG using a simple lossy encoder")

    p = sub.add_parser("roundtrip-check")
    p.add_argument("tpl", type=Path)
    p.add_argument("--encode-dxt1", action="store_true", help="allow lossy DXT1 encode during test; result will usually not be byte-identical")

    args = ap.parse_args()
    try:
        if args.cmd == "info":
            print_info(args.tpl)
        elif args.cmd == "extract":
            out_dir = args.out_dir if args.out_dir is not None else args.tpl.parent
            extract_tpl(args.tpl, out_dir, name=args.name, dump_raw=not args.no_raw)
            meta_path = default_meta_path_for(args.tpl, name=args.name, out_dir=out_dir)
            print(f"extracted to {out_dir}")
            print(f"metadata: {meta_path}")
        elif args.cmd == "rebuild":
            rebuild_tpl_from_meta(args.template_tpl, args.meta_json, args.out_tpl, encode_dxt1=args.encode_dxt1)
            print(f"rebuilt {args.out_tpl}")
        elif args.cmd == "roundtrip-check":
            roundtrip_check(args.tpl, encode_dxt1=args.encode_dxt1)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
