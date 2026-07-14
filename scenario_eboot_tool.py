"""
scenario_eboot_tool_structured_singlefile_v16.py

Single-file logical-message extractor/rebuilder for ToHeart2 DX PLUS scenario blocks
embedded in EBOOT.ELF.

Default workflow:
  python3 scenario_eboot_tool_structured_singlefile_v16.py extract EBOOT.ELF out_json --use-hash
  # edit translated_text fields in script_*.json
  python3 scenario_eboot_tool_structured_singlefile_v16.py rebuild EBOOT.ELF out_json EBOOT_MODDED.ELF --use-hash

With font.tbl byte<->text mapping:
  python3 scenario_eboot_tool_structured_singlefile_v16.py extract EBOOT.ELF out_json --use-hash --font-tbl font.tbl
  python3 scenario_eboot_tool_structured_singlefile_v16.py rebuild EBOOT.ELF out_json EBOOT_MODDED.ELF --use-hash --font-tbl font.tbl

Important model:
  * Extraction is logical-message JSON by default. There is no --clean-logical flag.
  * A logical message may consist of multiple TEXT_OPCODE records plus a terminal
    control such as \\p0001 or \\k\\n.
  * Rebuild requires pylzma and preserves LZMA-ALONE uncompressed-size metadata.
  * Strict length is default; pass --allow-growth only when intentionally testing
    relocated/grown logical messages.
  * No auto-padding and no auto-wrap are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import re
import struct
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TOOL_NAME = "scenario_eboot_cli"
TOOL_VERSION = 0.19
DEFAULT_TABLE_START = 0x000EE8B4
DEFAULT_MAX_ENTRIES = 2000
DEFAULT_INPLACE_OFFSET_BASE = 0x0029F000
DEFAULT_INPLACE_VADDR_BASE = 0x002AF000
DEFAULT_INPLACE_MAX_SIZE = 3734108
META_FILENAME = "scenario_meta.json"
TEXT_OPCODE = b"\x02\x0B\x00\x0A"
DEFAULT_ENCODING = "cp932"


@dataclass
class ProgramHeader:
    vaddr: int
    memsz: int
    offset: int


@dataclass
class ScenarioEntry:
    entry_index: int
    filename: str
    json_filename: str
    name_ptr: int
    data_ptr: int
    file_offset: int
    uncomp_size: int
    comp_size_table: int
    comp_size_actual: int
    raw_chunk_size: int
    sha256_bin: str
    text_count: int = 0
    json_sha256: Optional[str] = None


def parse_int(value: str) -> int:
    return int(value, 0)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_u32be(buf: bytes, off: int) -> int:
    return struct.unpack_from(">I", buf, off)[0]


def _strip_tbl_comment(line: str) -> str:
    """Strip common non-data comments while preserving mapped spaces."""
    # A leading comment line is ignored. Inline comments are only recognized when
    # preceded by whitespace so values like "#" can still be mapped.
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#") or stripped.startswith("//") or stripped.startswith(";"):
        return ""
    for marker in (" //", " #", " ;"):
        pos = line.find(marker)
        if pos >= 0:
            return line[:pos].rstrip("\r\n")
    return line.rstrip("\r\n")


def _parse_hex_bytes(token: str) -> Optional[bytes]:
    t = token.strip()
    if not t:
        return None
    if t.lower().startswith("0x"):
        t = t[2:]
    t = t.replace(" ", "").replace("_", "")
    if len(t) % 2 != 0 or not t:
        return None
    try:
        return bytes.fromhex(t)
    except ValueError:
        return None


class FontTable:
    """
    Bidirectional byte<->text mapping loaded from a .tbl file.

    Supported line styles are intentionally permissive:
      8140=あ
      8140<TAB>あ
      0x8140 = あ
      あ=8140

    Byte decoding is longest-match over table keys, then falls back to the
    configured Python encoding for ordinary game text bytes. Encoding is greedy
    over mapped text tokens, then falls back to the configured Python encoding.
    """

    def __init__(self, byte_to_text: Dict[bytes, str], source: Optional[Path] = None):
        if not byte_to_text:
            raise ValueError("font table contains no mappings")
        self.byte_to_text: Dict[bytes, str] = dict(byte_to_text)
        self.text_to_bytes: Dict[str, bytes] = {}
        duplicates: List[str] = []
        for raw, text in self.byte_to_text.items():
            if text in self.text_to_bytes and self.text_to_bytes[text] != raw:
                duplicates.append(text)
                continue
            self.text_to_bytes[text] = raw
        self.max_key_len = max(len(k) for k in self.byte_to_text)
        self.text_tokens = sorted(self.text_to_bytes.keys(), key=len, reverse=True)
        self.source = source
        self.duplicate_text_count = len(duplicates)

    @classmethod
    def load(cls, path: Path) -> "FontTable":
        mapping: Dict[bytes, str] = {}
        text = path.read_text(encoding="utf-8-sig")
        for lineno, original_line in enumerate(text.splitlines(), 1):
            line = _strip_tbl_comment(original_line)
            if not line.strip():
                continue

            parts: Optional[Tuple[str, str]] = None
            if "=" in line:
                left, right = line.split("=", 1)
                parts = (left, right)
            else:
                split = line.split(None, 1)
                if len(split) == 2:
                    parts = (split[0], split[1])

            if not parts:
                raise ValueError(f"{path}:{lineno}: cannot parse mapping line: {original_line!r}")
            left, right = parts
            left_bytes = _parse_hex_bytes(left)
            right_bytes = _parse_hex_bytes(right)

            if left_bytes is not None and right_bytes is None:
                raw = left_bytes
                value = right
            elif right_bytes is not None and left_bytes is None:
                raw = right_bytes
                value = left
            elif left_bytes is not None and right_bytes is not None:
                # Ambiguous but common tbl convention is HEX=TEXT. Treat RHS hex
                # as literal text only when both sides are hex-like.
                raw = left_bytes
                value = right
            else:
                raise ValueError(f"{path}:{lineno}: no hex byte sequence found: {original_line!r}")

            if value == "<space>":
                value = " "
            elif value == "<tab>":
                value = "\t"
            elif value == "<empty>":
                value = ""
            if value == "":
                raise ValueError(f"{path}:{lineno}: empty text token is not allowed")
            if raw in mapping and mapping[raw] != value:
                raise ValueError(
                    f"{path}:{lineno}: duplicate byte key {raw.hex().upper()} maps to both "
                    f"{mapping[raw]!r} and {value!r}"
                )
            mapping[raw] = value
        return cls(mapping, source=path)

    def decode_bytes(self, data: bytes, encoding: str) -> str:
        out: List[str] = []
        i = 0
        while i < len(data):
            matched = False
            limit = min(self.max_key_len, len(data) - i)
            for size in range(limit, 0, -1):
                chunk = data[i : i + size]
                value = self.byte_to_text.get(chunk)
                if value is not None:
                    out.append(value)
                    i += size
                    matched = True
                    break
            if matched:
                continue
            # Fallback to a decodable byte unit. Prefer one byte for ASCII/control,
            # then try two-byte code units for Shift-JIS-like text.
            for size in (1, 2):
                if i + size <= len(data):
                    try:
                        out.append(data[i : i + size].decode(encoding))
                        i += size
                        matched = True
                        break
                    except Exception:
                        pass
            if not matched:
                out.append("�")
                i += 1
        return "".join(out)

    def encode_text(self, text: str, encoding: str, errors: str = "ignore") -> bytes:
        out = bytearray()
        i = 0
        while i < len(text):
            matched = False
            for token in self.text_tokens:
                if token and text.startswith(token, i):
                    out.extend(self.text_to_bytes[token])
                    i += len(token)
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            try:
                out.extend(ch.encode(encoding))
            except Exception:
                if errors == "strict":
                    raise
                out.extend(ch.encode(encoding, errors=errors))
            i += 1
        return bytes(out)

    def meta(self) -> Dict[str, Any]:
        return {
            "source": str(self.source) if self.source else None,
            "mapping_count": len(self.byte_to_text),
            "duplicate_text_count": self.duplicate_text_count,
            "max_byte_key_len": self.max_key_len,
        }


def load_font_table_arg(path_arg: Optional[str]) -> Optional[FontTable]:
    if not path_arg:
        return None
    path = Path(path_arg)
    if not path.exists():
        raise SystemExit(f"font.tbl not found: {path}")
    return FontTable.load(path)


def decode_text_bytes(raw: bytes, encoding: str, font_table: Optional[FontTable]) -> str:
    if font_table is not None:
        return font_table.decode_bytes(raw, encoding)
    return raw.decode(encoding)


def _is_hex_digit_char(ch: str) -> bool:
    return ("0" <= ch <= "9") or ("a" <= ch <= "f") or ("A" <= ch <= "F")


def _split_preserving_engine_escapes(text: str) -> List[Tuple[bool, str]]:
    """Split text into (is_engine_escape, token) chunks.

    Scenario text uses printable backslash control escapes inside text records,
    for example:
      \\p0001  page/line state marker
      \\k      wait/next indicator control
      \\n      line break control

    These bytes must remain literal ASCII bytes even when font.tbl maps ASCII
    letters/digits/punctuation to custom glyph codes.  Without this, a table
    that maps digits will turn \\p0001 into mixed control/custom bytes and the
    engine no longer recognizes the page/next-line command.
    """
    out: List[Tuple[bool, str]] = []
    i = 0
    while i < len(text):
        if text[i] != "\\":
            j = text.find("\\", i)
            if j < 0:
                j = len(text)
            out.append((False, text[i:j]))
            i = j
            continue

        # Preserve \p followed by four hex digits.  Corpus contains both
        # decimal-looking forms (\p0001) and hex-looking forms (\p000a).
        if (
            i + 6 <= len(text)
            and text[i + 1:i + 2] == "p"
            and all(_is_hex_digit_char(c) for c in text[i + 2:i + 6])
        ):
            out.append((True, text[i:i + 6]))
            i += 6
            continue

        # Preserve all one-letter backslash controls seen in corpus, and be
        # conservative for future controls by preserving backslash+next char.
        if i + 1 < len(text):
            out.append((True, text[i:i + 2]))
            i += 2
        else:
            # Rare corpus case: trailing literal backslash. Preserve it.
            out.append((True, "\\"))
            i += 1
    return out


def encode_text_bytes(text: str, encoding: str, font_table: Optional[FontTable]) -> bytes:
    if font_table is None:
        return encode_sjis_lossy(text, encoding=encoding)

    out = bytearray()
    for is_escape, chunk in _split_preserving_engine_escapes(text):
        if not chunk:
            continue
        if is_escape:
            # Engine control escapes are byte-level ASCII protocol, not glyph
            # text.  Keep them out of font.tbl mapping.
            out.extend(chunk.encode("ascii", errors="strict"))
        else:
            out.extend(font_table.encode_text(chunk, encoding, errors="ignore"))
    return bytes(out)


def read_elf_program_headers(eboot_data: bytes) -> List[ProgramHeader]:
    if len(eboot_data) < 0x40:
        raise ValueError("EBOOT is too small to contain an ELF64 header")
    if eboot_data[:4] != b"\x7FELF":
        raise ValueError("Input is not an ELF file")
    if eboot_data[4] != 2:
        raise ValueError("Expected ELF64")
    if eboot_data[5] != 2:
        raise ValueError("Expected big-endian ELF")

    e_phoff = struct.unpack_from(">Q", eboot_data, 0x20)[0]
    e_phentsize = struct.unpack_from(">H", eboot_data, 0x36)[0]
    e_phnum = struct.unpack_from(">H", eboot_data, 0x38)[0]

    headers: List[ProgramHeader] = []
    for i in range(e_phnum):
        ph_start = e_phoff + i * e_phentsize
        if ph_start + e_phentsize > len(eboot_data):
            break
        p_type = struct.unpack_from(">I", eboot_data, ph_start)[0]
        if p_type == 1:  # PT_LOAD
            p_offset = struct.unpack_from(">Q", eboot_data, ph_start + 8)[0]
            p_vaddr = struct.unpack_from(">Q", eboot_data, ph_start + 16)[0]
            p_memsz = struct.unpack_from(">Q", eboot_data, ph_start + 40)[0]
            headers.append(ProgramHeader(vaddr=p_vaddr, memsz=p_memsz, offset=p_offset))
    if not headers:
        raise ValueError("No PT_LOAD program headers found")
    return headers


def vaddr_to_fileoff(vaddr: int, ph_headers: Iterable[ProgramHeader]) -> int:
    for ph in ph_headers:
        if ph.vaddr <= vaddr < ph.vaddr + ph.memsz:
            return ph.offset + (vaddr - ph.vaddr)
    return vaddr


def lzma_decompress_chunk(eboot_data: bytes, file_off: int) -> Tuple[bytes, int, int]:
    if file_off < 0 or file_off + 4 > len(eboot_data):
        raise ValueError(f"Invalid LZMA chunk offset 0x{file_off:X}")
    comp_size = read_u32be(eboot_data, file_off)
    start = file_off + 4
    end = start + comp_size
    if end > len(eboot_data):
        raise ValueError(
            f"LZMA chunk 0x{file_off:X} size 0x{comp_size:X} exceeds EBOOT size"
        )
    lzma_alone = eboot_data[start:end]
    if len(lzma_alone) < 13:
        raise ValueError(f"LZMA chunk 0x{file_off:X} is too small")
    if lzma_alone[0] != 0x5D:
        raise ValueError(
            f"LZMA chunk 0x{file_off:X} has unexpected properties byte 0x{lzma_alone[0]:02X}"
        )
    data = lzma.decompress(lzma_alone, format=lzma.FORMAT_ALONE)
    return data, comp_size, 4 + comp_size


def compress_lzma_alone(data: bytes, compressor: str = "pylzma") -> bytes:
    """Return a complete LZMA-ALONE stream.

    V9 production default requires pylzma because the game appears stricter than
    Python's stdlib decoder. The stdlib path remains available only for explicit
    diagnostics via --compressor python-lzma.
    """
    if compressor not in {"auto", "pylzma", "python-lzma"}:
        raise ValueError(f"Unknown compressor: {compressor}")

    if compressor in {"auto", "pylzma"}:
        try:
            import pylzma  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "pylzma is required for rebuild. Install/use the uploaded wheel, e.g.\n"
                "  python3 -m pip install --target ./pylzma_pkg pylzma-*.whl\n"
                "  PYTHONPATH=./pylzma_pkg python3 scenario_eboot_tool_fonttbl_v9_pylzma_required.py ..."
            ) from exc
        raw = pylzma.compress(data, dictionary=13, fastBytes=273, eos=0)
        if len(raw) < 5 or raw[0] != 0x5D:
            raise ValueError("pylzma returned an invalid LZMA stream")
        return raw[:5] + struct.pack("<Q", len(data)) + raw[5:]

    # Diagnostic fallback only. Do not use for production unless proven safe on hardware.
    filters = [
        {
            "id": lzma.FILTER_LZMA1,
            "dict_size": 1 << 13,
            "lc": 3,
            "lp": 0,
            "pb": 2,
        }
    ]
    out = bytearray(lzma.compress(data, format=lzma.FORMAT_ALONE, filters=filters))
    if len(out) < 13 or out[0] != 0x5D:
        raise ValueError("invalid LZMA-ALONE stream generated by stdlib lzma")
    struct.pack_into("<Q", out, 5, len(data))
    return bytes(out)

def parse_scenario_texts(data: bytes, encoding: str = DEFAULT_ENCODING, font_table: Optional[FontTable] = None) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    i = 16
    while i < len(data) - 6:
        if data[i : i + 4] == TEXT_OPCODE:
            length = struct.unpack_from("<H", data, i + 4)[0]
            text_start = i + 6
            text_end = text_start + length
            if text_end <= len(data):
                raw = data[text_start:text_end]
                try:
                    s = decode_text_bytes(raw, encoding, font_table)
                except Exception:
                    i += 1
                    continue
                lines.append(
                    {
                        "offset": f"0x{i:X}",
                        "length": length,
                        "original_text": s,
                        "translated_text": s,
                    }
                )
                i = text_end
            else:
                i += 1
        else:
            i += 1
    return lines


def wrap_text(text: str, max_len: int = 30) -> List[str]:
    if not text:
        return ["　"]

    if "\\n" in text or "\\k" in text:
        parts = text.split("\\n")
        return [p + "\\n" if i < len(parts) - 1 else p for i, p in enumerate(parts) if p or i < len(parts) - 1]

    words = text.split(" ")
    pages: List[str] = []
    current_page: List[str] = []
    current_line: List[str] = []
    current_len = 0

    for word in words:
        extra = len(word) + (1 if current_len > 0 else 0)
        if current_len + extra > max_len:
            if not current_line:
                current_line = [word]
                current_len = len(word)
            else:
                current_page.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)
                if len(current_page) == 2:
                    pages.append("\\n".join(current_page) + "\\k\\n")
                    current_page = []
        else:
            current_line.append(word)
            current_len += extra

    if current_line:
        current_page.append(" ".join(current_line))
    if current_page:
        pages.append("\\n".join(current_page))

    wrapped = "".join(pages)
    parts = wrapped.split("\\n")
    out: List[str] = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            out.append(p + "\\n")
        elif p:
            out.append(p)
    return out or ["　"]


def translated_to_segments(value: Any, wrap: bool, max_len: int) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value) if value is not None else ""
    if text == "":
        return ["　"]
    return wrap_text(text, max_len=max_len) if wrap else [text]


def encode_sjis_lossy(text: str, encoding: str = DEFAULT_ENCODING) -> bytes:
    try:
        return text.encode(encoding)
    except Exception:
        return text.encode(encoding, errors="ignore")



def _scenario_code_start(data: bytes) -> int:
    if len(data) < 0x10:
        raise ValueError("Scenario data is too small for a VE header")
    label_count = struct.unpack_from("<I", data, 8)[0]
    return 0x10 + label_count * 0x24


def _scan_exact_le32_label_offset_hits(
    data: bytes,
    *,
    text_spans: List[Tuple[int, int]],
    label_ref_spans: Optional[List[Tuple[int, int]]] = None,
) -> List[Dict[str, Any]]:
    """Find exact embedded u32le label offsets outside known text/label-name operands.

    This is a conservative production guard for --allow-growth. The corpus probe
    found zero such hits across all 1824 scripts. If a future script has one, the
    rebuilder refuses growth instead of silently missing a relocation.
    """
    label_count = struct.unpack_from("<I", data, 8)[0] if len(data) >= 0x10 else 0
    code_start = _scenario_code_start(data)
    code_size = struct.unpack_from("<I", data, 4)[0]
    labels: Dict[int, str] = {}
    for i in range(label_count):
        entry_off = 0x10 + i * 0x24
        if entry_off + 0x24 > len(data):
            break
        name = data[entry_off:entry_off + 0x20].split(b"\0", 1)[0].decode("ascii", "replace")
        ptr = struct.unpack_from("<I", data, entry_off + 0x20)[0]
        if 0 <= ptr < code_size:
            labels[ptr] = name
    if not labels:
        return []

    excluded = bytearray(len(data))
    for a, b in text_spans:
        a = max(0, a); b = min(len(data), b)
        excluded[a:b] = b"\x01" * max(0, b - a)
    if label_ref_spans:
        for a, b in label_ref_spans:
            a = max(0, a); b = min(len(data), b)
            excluded[a:b] = b"\x01" * max(0, b - a)

    hits: List[Dict[str, Any]] = []
    for off in range(code_start, len(data) - 3):
        if any(excluded[off:off + 4]):
            continue
        v = struct.unpack_from("<I", data, off)[0]
        if v in labels:
            hits.append({
                "file_offset": f"0x{off:X}",
                "code_relative_offset": f"0x{off - code_start:X}",
                "value": f"0x{v:X}",
                "label": labels[v],
                "context_hex": data[max(0, off - 8):off + 12].hex(),
            })
    return hits


def _find_label_name_operand_spans(data: bytes, text_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    code_start = _scenario_code_start(data)
    text_mask = bytearray(len(data))
    for a, b in text_spans:
        a = max(0, a); b = min(len(data), b)
        text_mask[a:b] = b"\x01" * max(0, b - a)
    spans: List[Tuple[int, int]] = []
    i = code_start
    while i < len(data) - 2:
        if text_mask[i]:
            while i < len(data) and text_mask[i]:
                i += 1
            continue
        if data[i] == 0x05:
            ln = data[i + 1]
            end = i + 2 + ln
            if 1 <= ln <= 0x20 and end <= len(data):
                raw = data[i + 2:end]
                if raw.startswith(b"@") and all(0x20 <= c <= 0x7E for c in raw):
                    spans.append((i, end))
                    i = end
                    continue
        i += 1
    return spans


def rebuild_scenario_from_json(
    orig_data: bytes,
    lines: List[Dict[str, Any]],
    *,
    encoding: str = DEFAULT_ENCODING,
    wrap: bool = False,
    max_len: int = 30,
    font_table: Optional[FontTable] = None,
    allow_growth: bool = False,
    verify_relocations: bool = True,
) -> bytes:
    """Rebuild one decompressed VE scenario.

    Production relocation model verified against the whole extracted corpus:
      * label table offsets are relative to code_start, not file start;
      * label-name operands (05 len "@...") resolve through that label table;
      * no direct u32le label offsets were found outside text records and
        label-name operands in the 1824-script corpus;
      * labels that point into text records point to record+1, the 0x0B type byte.

    Default is strict: a translated record may not become larger unless
    --allow-growth is passed. Shrink/same-size edits still use the same relocation
    path because later code positions can change.
    """
    if len(orig_data) < 0x10:
        raise ValueError("Invalid scenario: smaller than 0x10-byte header")
    label_count = struct.unpack_from("<I", orig_data, 8)[0]
    code_size = struct.unpack_from("<I", orig_data, 4)[0]
    code_start = 0x10 + label_count * 0x24
    if len(orig_data) != code_start + code_size:
        raise ValueError(
            f"Invalid scenario size formula: file=0x{len(orig_data):X}, "
            f"code_start=0x{code_start:X}, code_size=0x{code_size:X}"
        )

    sorted_lines = sorted(lines, key=lambda item: int(str(item["offset"]), 16))
    new_payload = bytearray()
    last_idx = 16
    relocation_records: List[Dict[str, int]] = []
    original_text_spans: List[Tuple[int, int]] = []

    for line in sorted_lines:
        offset = int(str(line["offset"]), 16)
        orig_len = int(line["length"])
        old_total = 6 + orig_len
        old_end = offset + old_total
        if offset < last_idx or old_end > len(orig_data):
            raise ValueError(f"Invalid or overlapping text record at 0x{offset:X}")
        if orig_data[offset : offset + 4] != TEXT_OPCODE:
            raise ValueError(f"Text opcode mismatch at 0x{offset:X}")
        if offset < code_start:
            raise ValueError(f"Text record 0x{offset:X} is before code_start 0x{code_start:X}")

        new_payload.extend(orig_data[last_idx:offset])

        translated = line.get("translated_text", line.get("original_text", ""))
        segments = translated_to_segments(translated, wrap=wrap, max_len=max_len)

        encoded_segments: List[bytes] = []
        new_total = 0
        for segment in segments:
            encoded = encode_text_bytes(segment, encoding, font_table)
            if len(encoded) > 0xFFFF:
                raise ValueError(f"Encoded text at 0x{offset:X} exceeds 65535 bytes")
            encoded_segments.append(encoded)
            new_total += 6 + len(encoded)

        if (not allow_growth) and new_total > old_total:
            raise ValueError(
                f"Text growth is disabled at 0x{offset:X}: old_total={old_total}, "
                f"new_total={new_total}. Pass --allow-growth after verifying this build."
            )

        for encoded in encoded_segments:
            new_payload.extend(TEXT_OPCODE)
            new_payload.extend(struct.pack("<H", len(encoded)))
            new_payload.extend(encoded)

        delta = new_total - old_total
        relocation_records.append({
            "record_abs": offset,
            "record_rel": offset - code_start,
            "old_end_abs": old_end,
            "old_end_rel": old_end - code_start,
            "old_total": old_total,
            "new_total": new_total,
            "delta": delta,
        })
        original_text_spans.append((offset, old_end))
        last_idx = old_end

    new_payload.extend(orig_data[last_idx:])

    new_header = bytearray(orig_data[:16])
    total_delta = sum(r["delta"] for r in relocation_records)
    struct.pack_into("<I", new_header, 4, code_size + total_delta)

    # Update code-relative label offsets. Labels inside the edited text record are
    # valid only for the observed record+1 anchor (the 0x0B text type byte).
    if label_count:
        for i in range(label_count):
            entry_abs = 0x10 + i * 0x24
            ptr_field_abs = entry_abs + 0x20
            ptr_payload_off = ptr_field_abs - 16
            if ptr_payload_off + 4 > len(new_payload):
                raise ValueError(f"Label entry {i} is outside rebuilt payload")
            p = struct.unpack_from("<I", new_payload, ptr_payload_off)[0]
            if not (0 <= p < code_size):
                raise ValueError(f"Label entry {i} has invalid code-relative offset 0x{p:X}")

            shift = 0
            for r in relocation_records:
                rec_rel = r["record_rel"]
                end_rel = r["old_end_rel"]
                if p >= end_rel:
                    shift += r["delta"]
                elif rec_rel <= p < end_rel:
                    # Corpus proof: all labels inside text records target record+1.
                    # This remains the same semantic anchor after length/payload edits.
                    if p != rec_rel + 1:
                        name = new_payload[entry_abs - 16:entry_abs - 16 + 0x20].split(b"\0", 1)[0].decode("ascii", "replace")
                        raise ValueError(
                            f"Label {name!r} points inside edited text record at non-anchor "
                            f"offset 0x{p:X}; refusing unsafe relocation"
                        )
            if shift:
                struct.pack_into("<I", new_payload, ptr_payload_off, p + shift)

    rebuilt = bytes(new_header + new_payload)

    if verify_relocations:
        label_ref_spans = _find_label_name_operand_spans(orig_data, original_text_spans)
        hits = _scan_exact_le32_label_offset_hits(
            orig_data,
            text_spans=original_text_spans,
            label_ref_spans=label_ref_spans,
        )
        if hits:
            raise ValueError(
                "Possible direct u32 label-offset relocations found outside known operands; "
                "refusing production rebuild. First hits: " + json.dumps(hits[:5], ensure_ascii=False)
            )

        new_label_count = struct.unpack_from("<I", rebuilt, 8)[0]
        new_code_size = struct.unpack_from("<I", rebuilt, 4)[0]
        new_code_start = 0x10 + new_label_count * 0x24
        if len(rebuilt) != new_code_start + new_code_size:
            raise ValueError(
                f"Rebuilt scenario size formula failed: file=0x{len(rebuilt):X}, "
                f"code_start=0x{new_code_start:X}, code_size=0x{new_code_size:X}"
            )
        for i in range(new_label_count):
            ptr = struct.unpack_from("<I", rebuilt, 0x10 + i * 0x24 + 0x20)[0]
            if not (0 <= ptr < new_code_size):
                raise ValueError(f"Rebuilt label {i} has invalid code-relative offset 0x{ptr:X}")

    return rebuilt

def scan_entries(
    eboot_data: bytes,
    *,
    table_start: int,
    max_entries: int,
    encoding: str,
    font_table: Optional[FontTable] = None,
) -> List[Tuple[ScenarioEntry, bytes]]:
    ph_headers = read_elf_program_headers(eboot_data)
    entries: List[Tuple[ScenarioEntry, bytes]] = []

    for count in range(max_entries):
        entry_off = table_start + count * 16
        if entry_off + 16 > len(eboot_data):
            break
        name_ptr, data_ptr, uncomp_size, comp_size_table = struct.unpack_from(">IIII", eboot_data, entry_off)
        if data_ptr == 0 or name_ptr == 0:
            break

        file_off = vaddr_to_fileoff(data_ptr, ph_headers)
        try:
            dec, comp_size_actual, raw_chunk_size = lzma_decompress_chunk(eboot_data, file_off)
        except Exception:
            continue

        filename = f"script_{count:04d}.bin"
        json_filename = f"script_{count:04d}.json"
        text_count = len(parse_scenario_texts(dec, encoding=encoding, font_table=font_table))
        entry = ScenarioEntry(
            entry_index=count,
            filename=filename,
            json_filename=json_filename,
            name_ptr=name_ptr,
            data_ptr=data_ptr,
            file_offset=file_off,
            uncomp_size=uncomp_size,
            comp_size_table=comp_size_table,
            comp_size_actual=comp_size_actual,
            raw_chunk_size=raw_chunk_size,
            sha256_bin=sha256_bytes(dec),
            text_count=text_count,
        )
        entries.append((entry, dec))

    return entries


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_meta(
    *,
    eboot_path: Path,
    eboot_data: bytes,
    extraction_mode: str,
    table_start: int,
    max_entries: int,
    entries: List[ScenarioEntry],
    use_hash: bool,
    encoding: str,
    font_table: Optional[FontTable] = None,
) -> Dict[str, Any]:
    return {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "source_eboot": str(eboot_path),
        "source_eboot_size": len(eboot_data),
        "source_eboot_sha256": sha256_bytes(eboot_data),
        "table_start": table_start,
        "max_entries": max_entries,
        "encoding": encoding,
        "font_tbl": font_table.meta() if font_table is not None else None,
        "extraction_mode": extraction_mode,
        "hash_algorithm": "sha256" if use_hash else None,
        "entries": [asdict(e) for e in entries],
    }


def command_extract(args: argparse.Namespace) -> None:
    if args.use_hash and not args.clean:
        raise SystemExit("extract --use-hash is valid only together with --clean")

    font_table = load_font_table_arg(args.font_tbl)

    eboot_path = Path(args.eboot)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eboot_data = eboot_path.read_bytes()

    scanned = scan_entries(
        eboot_data,
        table_start=args.table_start,
        max_entries=args.max_entries,
        encoding=args.encoding,
        font_table=font_table,
    )

    extracted_entries: List[ScenarioEntry] = []
    total_text = 0
    written_files = 0

    for entry, dec in scanned:
        if args.clean:
            lines = parse_scenario_texts(dec, encoding=args.encoding, font_table=font_table)
            entry.text_count = len(lines)
            total_text += entry.text_count
            if lines:
                json_path = out_dir / entry.json_filename
                write_json(json_path, lines)
                written_files += 1
                if args.use_hash:
                    entry.json_sha256 = sha256_file(json_path)
        else:
            bin_path = out_dir / entry.filename
            bin_path.write_bytes(dec)
            written_files += 1
        extracted_entries.append(entry)

    mode = "clean_json" if args.clean else "raw_bin"
    meta = build_meta(
        eboot_path=eboot_path,
        eboot_data=eboot_data,
        extraction_mode=mode,
        table_start=args.table_start,
        max_entries=args.max_entries,
        entries=extracted_entries,
        use_hash=args.use_hash,
        encoding=args.encoding,
        font_table=font_table,
    )
    write_json(out_dir / META_FILENAME, meta)

    if args.clean:
        print(
            f"Extracted {written_files} JSON files from {len(extracted_entries)} LZMA entries "
            f"({total_text} text records) to {out_dir}"
        )
    else:
        print(f"Extracted {written_files} binary scenario files to {out_dir}")
    print(f"Metadata saved to {out_dir / META_FILENAME}")


def normalize_entry(raw: Dict[str, Any]) -> ScenarioEntry:
    fields = {
        "entry_index": int(raw["entry_index"]),
        "filename": str(raw.get("filename", f"script_{int(raw['entry_index']):04d}.bin")),
        "json_filename": str(raw.get("json_filename", f"script_{int(raw['entry_index']):04d}.json")),
        "name_ptr": int(raw["name_ptr"]),
        "data_ptr": int(raw["data_ptr"]),
        "file_offset": int(raw["file_offset"]),
        "uncomp_size": int(raw.get("uncomp_size", 0)),
        "comp_size_table": int(raw.get("comp_size_table", 0)),
        "comp_size_actual": int(raw.get("comp_size_actual", 0)),
        "raw_chunk_size": int(raw.get("raw_chunk_size", 0)),
        "sha256_bin": str(raw.get("sha256_bin", "")),
        "text_count": int(raw.get("text_count", 0)),
        "json_sha256": raw.get("json_sha256"),
    }
    return ScenarioEntry(**fields)


def detect_input_kind(meta: Dict[str, Any], input_dir: Path) -> str:
    mode = str(meta.get("extraction_mode", "")).lower()
    if mode in {"clean_json", "json", "clean"}:
        return "json"
    if mode in {"raw_bin", "bin", "binary"}:
        return "bin"

    json_count = len(list(input_dir.glob("script_*.json")))
    bin_count = len(list(input_dir.glob("script_*.bin")))
    if json_count and json_count >= bin_count:
        return "json"
    if bin_count:
        return "bin"
    raise ValueError("Cannot detect input kind: no script_*.json or script_*.bin files found")


def locate_meta(input_dir: Path, meta_arg: Optional[str]) -> Path:
    if meta_arg:
        path = Path(meta_arg)
        if path.is_dir():
            return path / META_FILENAME
        return path
    return input_dir / META_FILENAME


def get_original_entry_data(
    eboot_data: bytes,
    ph_headers: List[ProgramHeader],
    entry: ScenarioEntry,
) -> bytes:
    file_off = vaddr_to_fileoff(entry.data_ptr, ph_headers) if entry.data_ptr else entry.file_offset
    data, _, _ = lzma_decompress_chunk(eboot_data, file_off)
    return data


def repack_entries_into_eboot(
    *,
    eboot_data: bytearray,
    table_start: int,
    entries_to_import: List[Tuple[ScenarioEntry, bytes]],
    inplace_offset_base: int,
    inplace_vaddr_base: int,
    inplace_max_size: int,
    compressor: str,
) -> Tuple[int, int]:
    current_offset = inplace_offset_base
    success = 0

    for entry, uncomp_data in entries_to_import:
        lzma_alone = compress_lzma_alone(uncomp_data, compressor=compressor)
        new_comp_size = len(lzma_alone)
        raw_chunk_size = 4 + new_comp_size
        padding = (4 - (raw_chunk_size % 4)) % 4
        total_chunk_size = raw_chunk_size + padding

        used_after = (current_offset - inplace_offset_base) + total_chunk_size
        if inplace_max_size and used_after > inplace_max_size:
            raise RuntimeError(
                f"Repack exceeds reserved area: need {used_after} bytes, limit is {inplace_max_size}"
            )
        if current_offset + total_chunk_size > len(eboot_data):
            raise RuntimeError(
                f"Repack write exceeds EBOOT size at 0x{current_offset:X}; "
                "refusing to extend executable"
            )

        new_vaddr = inplace_vaddr_base + (current_offset - inplace_offset_base)
        eboot_data[current_offset : current_offset + 4] = struct.pack(">I", new_comp_size)
        current_offset += 4
        eboot_data[current_offset : current_offset + new_comp_size] = lzma_alone
        current_offset += new_comp_size
        if padding:
            eboot_data[current_offset : current_offset + padding] = b"\x00" * padding
            current_offset += padding

        entry_start = table_start + entry.entry_index * 16
        if entry_start + 16 > len(eboot_data):
            raise RuntimeError(f"Index entry {entry.entry_index} is outside EBOOT")
        eboot_data[entry_start + 4 : entry_start + 8] = struct.pack(">I", new_vaddr)
        eboot_data[entry_start + 8 : entry_start + 12] = struct.pack(">I", len(uncomp_data))
        eboot_data[entry_start + 12 : entry_start + 16] = struct.pack(">I", total_chunk_size)
        success += 1

    return success, current_offset - inplace_offset_base


def command_rebuild(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    meta_path = locate_meta(input_dir, args.meta)
    if not meta_path.exists():
        raise SystemExit(f"Metadata not found: {meta_path}")

    meta = load_json(meta_path)
    if meta.get("tool") not in {TOOL_NAME, None}:
        print(f"warning: metadata tool is {meta.get('tool')!r}, expected {TOOL_NAME!r}")

    entries = [normalize_entry(e) for e in meta.get("entries", [])]
    if not entries:
        raise SystemExit("Metadata contains no entries")

    input_kind = detect_input_kind(meta, input_dir)
    if args.use_hash and input_kind != "json":
        raise SystemExit("rebuild --use-hash is valid only for clean JSON extractions")
    if args.use_hash and not meta.get("hash_algorithm"):
        raise SystemExit("Metadata has no JSON hashes. Re-run extract with --clean --use-hash.")

    eboot_path = Path(args.eboot)
    eboot_data = bytearray(eboot_path.read_bytes())
    ph_headers = read_elf_program_headers(bytes(eboot_data))
    table_start = int(meta.get("table_start", args.table_start or DEFAULT_TABLE_START))
    encoding = str(meta.get("encoding", args.encoding))
    meta_font_tbl = meta.get("font_tbl") or {}
    font_tbl_arg = args.font_tbl or meta_font_tbl.get("source")
    font_table = load_font_table_arg(font_tbl_arg) if font_tbl_arg else None

    entries_to_import: List[Tuple[ScenarioEntry, bytes]] = []
    skipped_unchanged = 0
    skipped_missing = 0

    if input_kind == "bin":
        for entry in entries:
            bin_path = input_dir / entry.filename
            if not bin_path.exists():
                skipped_missing += 1
                continue
            entries_to_import.append((entry, bin_path.read_bytes()))
    else:
        for entry in entries:
            json_path = input_dir / entry.json_filename
            if not json_path.exists():
                skipped_missing += 1
                continue
            if args.use_hash:
                old_hash = entry.json_sha256
                new_hash = sha256_file(json_path)
                if old_hash and new_hash == old_hash:
                    skipped_unchanged += 1
                    continue

            lines = load_json(json_path)
            if not isinstance(lines, list):
                raise ValueError(f"{json_path} must contain a JSON list")
            orig_data = get_original_entry_data(bytes(eboot_data), ph_headers, entry)
            rebuilt = rebuild_scenario_from_json(
                orig_data,
                lines,
                encoding=encoding,
                wrap=args.wrap,
                max_len=args.wrap_len,
                font_table=font_table,
                allow_growth=args.allow_growth,
                verify_relocations=not args.no_verify_relocations,
            )
            entries_to_import.append((entry, rebuilt))

    if not entries_to_import:
        print("No entries selected for import; output EBOOT will be a copy of input EBOOT.")

    success, used = repack_entries_into_eboot(
        eboot_data=eboot_data,
        table_start=table_start,
        entries_to_import=entries_to_import,
        inplace_offset_base=args.inplace_offset_base,
        inplace_vaddr_base=args.inplace_vaddr_base,
        inplace_max_size=args.inplace_max_size,
        compressor=args.compressor,
    )

    out_path = Path(args.output_eboot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(eboot_data)

    print(f"Input kind: {input_kind}")
    print(f"Imported {success} entries into {out_path}")
    if args.use_hash:
        print(f"Skipped unchanged JSON files: {skipped_unchanged}")
    if skipped_missing:
        print(f"Skipped missing input files: {skipped_missing}")
    print(f"Reserved area used: {used} bytes")


@dataclass
class TextRecordInfo:
    offset_int: int
    offset: str
    length: int
    raw: bytes
    text: str
    total_size: int
    end_int: int


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text_records_structured(data: bytes, encoding: str, font_table: Optional[FontTable]) -> List[TextRecordInfo]:
    out: List[TextRecordInfo] = []
    i = 16
    while i < len(data) - 6:
        if data[i:i + 4] == TEXT_OPCODE:
            length = struct.unpack_from("<H", data, i + 4)[0]
            start = i + 6
            end = start + length
            if end <= len(data):
                raw = data[start:end]
                try:
                    text = decode_text_bytes(raw, encoding, font_table)
                except Exception:
                    i += 1
                    continue
                out.append(TextRecordInfo(
                    offset_int=i,
                    offset=f"0x{i:X}",
                    length=length,
                    raw=raw,
                    text=text,
                    total_size=6 + length,
                    end_int=end,
                ))
                i = end
                continue
        i += 1
    return out


_P_RECORD_RE = re.compile(r"^\\p[0-9A-Fa-f]{4}$")
_P_SUFFIX_RE = re.compile(r"\\p[0-9A-Fa-f]{4}$")


def split_text_control_suffix(text: str) -> Tuple[str, str, str]:
    """Return (visible_text, control_suffix, control_kind)."""
    if text.endswith("\\k\\n"):
        return text[:-4], "\\k\\n", "k_n_suffix"
    if text.endswith("\\k"):
        return text[:-2], "\\k", "k_suffix"
    if text.endswith("\\n"):
        return text[:-2], "\\n", "n_suffix"
    m = _P_SUFFIX_RE.search(text)
    if m:
        return text[:m.start()], text[m.start():], "p_suffix"
    return text, "", "none"


def classify_text_record(text: str) -> Tuple[str, str, str, str]:
    """Return (kind, visible, control, control_kind)."""
    if _P_RECORD_RE.fullmatch(text):
        return "control_text", "", text, "p_record"
    visible, control, control_kind = split_text_control_suffix(text)
    return "text", visible, control, control_kind


def _hex(data: bytes) -> str:
    return data.hex().upper()


# ---------------------------------------------------------------------------
# Scenario VM opcode-gap decoder
# ---------------------------------------------------------------------------

_BYTECODE_DECODER_NAME = "scenario_vm_gap_decoder"
_BYTECODE_DECODER_VERSION = 2

_SINGLE_BYTE_OP_NAMES: Dict[int, str] = {
    0x00: "zero_or_padding",
    0x01: "op_01",
    0x03: "op_03",
    0x04: "op_04",
    0x06: "op_06",
    0x17: "op_17",
    0x18: "op_18",
    0x19: "op_19",
    0x1A: "op_1A",
    0x22: "op_22",
    0x2A: "op_2A",
}

_CMD8_HINTS: Dict[int, str] = {
    # Names are intentionally conservative.  The bytecode handler table is not
    # symbolized in the stripped EBOOT, so these are corpus-facing labels rather
    # than hard semantic claims.
    0x19: "vm_cmd_19",
    0x20: "message/page-flow marker",
    0x27: "vm_cmd_27",
    0x30: "vm_cmd_30",
    0x46: "vm_cmd_46",
    0x47: "vm_cmd_47",
    0x49: "vm_cmd_49",
    0x4D: "vm_cmd_4D",
    0x4E: "vm_cmd_4E",
    0x51: "vm_cmd_51",
    0x52: "vm_cmd_52",
    0x6A: "asset/layer command family",
    0x6B: "asset/layer command family",
}


def _json_hex(raw: bytes) -> str:
    return raw.hex().upper()


def _is_printable_ascii_bytes(raw: bytes) -> bool:
    return all(0x20 <= b <= 0x7E for b in raw)


def _decode_bytecode_string(raw: bytes, encoding: str, font_table: Optional[FontTable]) -> str:
    if not raw:
        return ""
    try:
        if font_table is not None:
            return font_table.decode_bytes(raw, encoding)
        # ASCII asset names and labels are common inside opcode gaps; try ASCII
        # first so binary-ish text fallback does not hide malformed data.
        if _is_printable_ascii_bytes(raw):
            return raw.decode("ascii")
        return raw.decode(encoding)
    except Exception:
        try:
            return raw.decode(encoding, errors="replace")
        except Exception:
            return raw.decode("latin-1", errors="replace")


def _literal_values(raw: bytes) -> Dict[str, Any]:
    values: Dict[str, Any] = {"hex": _json_hex(raw), "length": len(raw)}
    if len(raw) == 1:
        values["u8"] = raw[0]
        values["i8"] = struct.unpack("<b", raw)[0]
    elif len(raw) == 2:
        values["u16le"] = struct.unpack("<H", raw)[0]
        values["i16le"] = struct.unpack("<h", raw)[0]
    elif len(raw) == 4:
        values["u32le"] = struct.unpack("<I", raw)[0]
        values["i32le"] = struct.unpack("<i", raw)[0]
        values["f32le"] = struct.unpack("<f", raw)[0]
    elif len(raw) == 8:
        values["u64le"] = struct.unpack("<Q", raw)[0]
        values["i64le"] = struct.unpack("<q", raw)[0]
        values["f64le"] = struct.unpack("<d", raw)[0]
    return values


def decode_scenario_opcode_gap(
    raw: bytes,
    *,
    base_offset: Optional[int] = None,
    encoding: str = DEFAULT_ENCODING,
    font_table: Optional[FontTable] = None,
) -> Dict[str, Any]:
    """Decode a structured ``opcode_gap`` into scenario-VM tokens.

    Important distinction: these bytes are not PowerPC instructions.  They are
    scenario VM bytecode embedded between TEXT records.  Capstone is useful for
    auditing the stripped EBOOT handler, but gap bytes must be decoded with this
    VM grammar:
      02                         statement/end marker
      08 xx                      command family / command id
      09 len payload             typed numeric literal, commonly len=4 f32le
      0A len16 payload           string literal, commonly ASCII asset names
      05 len payload             label/name operand, often @label
      07 cstring\0               zero-terminated short symbol
      02 0B 00 0A len16 payload  embedded text record, even if text decoding fails

    The decoder is deliberately lossless: every emitted token includes the raw
    bytes.  Rebuild still uses the original ``bytes`` field; decoded output is
    for editing diagnostics and reverse-engineering only.
    """
    tokens: List[Dict[str, Any]] = []
    asm_lines: List[str] = []
    pseudo_lines: List[str] = []
    warnings: List[str] = []
    i = 0

    def token_offset(pos: int) -> str:
        if base_offset is None:
            return f"+0x{pos:X}"
        return f"0x{base_offset + pos:X}"

    def add_token(kind: str, start: int, end: int, asm: str, pseudo: str, **extra: Any) -> None:
        item: Dict[str, Any] = {
            "offset": token_offset(start),
            "rel_offset": f"+0x{start:X}",
            "kind": kind,
            "length": end - start,
            "raw": _json_hex(raw[start:end]),
            "asm": asm,
            "pseudo_c": pseudo,
        }
        item.update(extra)
        tokens.append(item)
        asm_lines.append(f"{token_offset(start)}: {asm}")
        pseudo_lines.append(pseudo)

    while i < len(raw):
        start = i

        # Existing extractor uses TEXT_OPCODE = 02 0B 00 0A.  A few corpus gaps
        # contain such records with non-standard glyph bytes that fail Shift-JIS
        # decoding, so make the bytecode decoder identify them explicitly instead
        # of letting them appear as raw unknown bytes.
        if raw[i:i + 4] == TEXT_OPCODE and i + 6 <= len(raw):
            length = struct.unpack_from("<H", raw, i + 4)[0]
            end = i + 6 + length
            if end <= len(raw):
                payload = raw[i + 6:end]
                text = _decode_bytecode_string(payload, encoding, font_table)
                add_token(
                    "text_record",
                    start,
                    end,
                    f"text len=0x{length:X} {text!r}",
                    f"emit_text({text!r});",
                    text=text,
                    payload_hex=_json_hex(payload),
                )
                i = end
                continue
            warnings.append(f"{token_offset(start)}: truncated TEXT record length 0x{length:X}")

        b = raw[i]

        if b == 0x02:
            add_token("statement_end", start, start + 1, "end", "vm_end_statement();")
            i += 1
            continue

        if b == 0x08 and i + 1 < len(raw):
            cmd = raw[i + 1]
            hint = _CMD8_HINTS.get(cmd)
            suffix = f" ; {hint}" if hint else ""
            add_token(
                "command_08",
                start,
                start + 2,
                f"cmd8 0x{cmd:02X}{suffix}",
                f"vm_cmd8(0x{cmd:02X});",
                command_id=f"0x{cmd:02X}",
                hint=hint,
            )
            i += 2
            continue

        if b == 0x09 and i + 1 < len(raw):
            n = raw[i + 1]
            end = i + 2 + n
            if n in {1, 2, 4, 8, 16} and end <= len(raw):
                payload = raw[i + 2:end]
                values = _literal_values(payload)
                if n == 4:
                    asm = f"literal len=4 f32={values['f32le']:.8g} i32={values['i32le']}"
                    pseudo = f"vm_push_literal_f32({values['f32le']:.8g});"
                else:
                    asm = f"literal len={n} 0x{_json_hex(payload)}"
                    pseudo = f"vm_push_literal_bytes({values['hex']!r});"
                add_token(
                    "literal_09",
                    start,
                    end,
                    asm,
                    pseudo,
                    values=values,
                )
                i = end
                continue
            if n in {1, 2, 4, 8, 16}:
                warnings.append(f"{token_offset(start)}: truncated literal_09 length {n}")

        if b == 0x0A and i + 2 < len(raw):
            n = struct.unpack_from("<H", raw, i + 1)[0]
            end = i + 3 + n
            if n <= 0x400 and end <= len(raw):
                payload = raw[i + 3:end]
                text = _decode_bytecode_string(payload, encoding, font_table)
                add_token(
                    "string_0A",
                    start,
                    end,
                    f"string len=0x{n:X} {text!r}",
                    f"vm_push_string({text!r});",
                    text=text,
                    payload_hex=_json_hex(payload),
                )
                i = end
                continue
            if n <= 0x400:
                warnings.append(f"{token_offset(start)}: truncated string_0A length {n}")

        if b == 0x05 and i + 1 < len(raw):
            n = raw[i + 1]
            end = i + 2 + n
            if n <= 0x40 and end <= len(raw):
                payload = raw[i + 2:end]
                text = _decode_bytecode_string(payload, encoding, font_table)
                add_token(
                    "label_ref_05",
                    start,
                    end,
                    f"label len=0x{n:X} {text!r}",
                    f"vm_label_ref({text!r});",
                    text=text,
                    payload_hex=_json_hex(payload),
                )
                i = end
                continue
            if n <= 0x40:
                warnings.append(f"{token_offset(start)}: truncated label_ref_05 length {n}")

        if b == 0x07:
            nul = raw.find(b"\0", i + 1)
            if nul >= 0:
                payload = raw[i + 1:nul]
                # Limit accidental runaway cstrings; genuine label-like operands
                # in this corpus are short printable ASCII identifiers.
                if len(payload) <= 0x40 and (not payload or _is_printable_ascii_bytes(payload)):
                    text = _decode_bytecode_string(payload, encoding, font_table)
                    add_token(
                        "cstring_07",
                        start,
                        nul + 1,
                        f"cstring7 {text!r}",
                        f"vm_cstring7({text!r});",
                        text=text,
                        payload_hex=_json_hex(payload),
                    )
                    i = nul + 1
                    continue

        name = _SINGLE_BYTE_OP_NAMES.get(b, f"op_{b:02X}")
        add_token(
            "single_byte_op",
            start,
            start + 1,
            name,
            f"vm_op(0x{b:02X});",
            opcode=f"0x{b:02X}",
            name=name,
        )
        i += 1

    return {
        "decoder": _BYTECODE_DECODER_NAME,
        "version": _BYTECODE_DECODER_VERSION,
        "byte_length": len(raw),
        "token_count": len(tokens),
        "asm": asm_lines,
        "pseudo_c": pseudo_lines,
        "tokens": tokens,
        "warnings": warnings,
    }


def _with_offset(item: Dict[str, Any], offset: str, include_offsets: bool) -> Dict[str, Any]:
    if include_offsets:
        item["offset"] = offset
    return item


def make_structured_units(
    data: bytes,
    records: List[TextRecordInfo],
    *,
    include_offsets: bool = False,
    encoding: str = DEFAULT_ENCODING,
    font_table: Optional[FontTable] = None,
    decode_gaps: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build editable units from the script bytecode.

    Default JSON is editor-facing and hides offsets because rebuild can infer
    original positions from unit/item order.  Use include_offsets=True only for
    diagnostics.

    Important policy:
      * TEXT_OPCODE records that are adjacent, or only separated by empty gaps,
        can be grouped together.
      * Any non-empty gap between text records is bytecode, not padding. It is
        emitted as opcode_gap and terminates the current unit after the gap.
      * Terminal controls (\\p#### as a standalone record, or suffixes like
        \\k\\n) also terminate the current unit. If a non-empty gap immediately
        follows a terminal control, it is included in that same unit as an
        opcode_gap so editors can see and preserve it.
    """
    units: List[Dict[str, Any]] = []
    cur_items: List[Dict[str, Any]] = []
    unit_index = 0

    def flush(end_offset: int, reason: str) -> None:
        nonlocal unit_index, cur_items
        if not cur_items:
            return
        unit: Dict[str, Any] = {
            "end_reason": reason,
            "items": cur_items,
        }
        if include_offsets:
            # Debug-only metadata. Editor-facing JSON intentionally omits
            # stable IDs/indexes because unit order is the source of truth and
            # rebuild recomputes indexes/offsets.
            unit["unit_index"] = unit_index
            start = cur_items[0].get("offset", f"0x{end_offset:X}")
            unit["start_offset"] = start
            unit["end_offset"] = f"0x{end_offset:X}"
        units.append(unit)
        unit_index += 1
        cur_items = []

    for idx, rec in enumerate(records):
        kind, visible, control, control_kind = classify_text_record(rec.text)
        if kind == "control_text":
            base_item: Dict[str, Any] = {
                "kind": "control_text",
                "length": rec.length,
                "control": control,
                "editable": False,
            }
            if include_offsets:
                base_item["control_kind"] = control_kind
            item = _with_offset(base_item, rec.offset, include_offsets)
        else:
            base_item = {
                "kind": "text",
                "length": rec.length,
                "original_text": visible,
                "translated_text": visible,
                "control_suffix": control,
                "editable": True,
            }
            if include_offsets:
                base_item["control_kind"] = control_kind
            item = _with_offset(base_item, rec.offset, include_offsets)
        cur_items.append(item)

        next_off = records[idx + 1].offset_int if idx + 1 < len(records) else len(data)
        gap = data[rec.end_int:next_off]
        has_gap = bool(gap)
        if has_gap:
            gap_item: Dict[str, Any] = {
                "kind": "opcode_gap",
                "length": len(gap),
                "bytes": _hex(gap),
                "editable": False,
            }
            if decode_gaps:
                gap_item["decoded"] = decode_scenario_opcode_gap(
                    gap,
                    base_offset=rec.end_int,
                    encoding=encoding,
                    font_table=font_table,
                )
            cur_items.append(_with_offset(gap_item, f"0x{rec.end_int:X}", include_offsets))

        terminal = kind == "control_text" or control_kind != "none"
        if terminal:
            flush(next_off, "terminal_control_plus_gap" if has_gap else "terminal_control")
        elif has_gap:
            flush(next_off, "opcode_gap")

    flush(len(data), "eof")
    return units


def _get_item_offset(item: Dict[str, Any]) -> Optional[int]:
    value = item.get("offset")
    if value in {None, "", "new", "NEW"}:
        return None
    return int(str(value), 16)


def _encode_control_ascii(control: str, encoding: str) -> bytes:
    # Engine control strings must stay as raw script ASCII / Shift-JIS bytes and
    # must not be remapped through font.tbl.
    return encode_text_bytes(control, encoding, None)


def _encode_structured_text_item(item: Dict[str, Any], encoding: str, font_table: Optional[FontTable]) -> bytes:
    text = str(item.get("translated_text", item.get("original_text", "")))
    control_suffix = str(item.get("control_suffix", ""))
    payload = encode_text_bytes(text, encoding, font_table)
    if control_suffix:
        payload += _encode_control_ascii(control_suffix, encoding)
    return payload


def _encode_structured_control_item(item: Dict[str, Any], encoding: str) -> bytes:
    return _encode_control_ascii(str(item.get("control", "")), encoding)


def _make_text_record(payload: bytes, where: str) -> bytes:
    if len(payload) > 0xFFFF:
        raise ValueError(f"TEXT payload at {where} exceeds 65535 bytes")
    return TEXT_OPCODE + struct.pack("<H", len(payload)) + payload


def _sorted_json_items(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("structured JSON must be a list of unit objects")
        items = unit.get("items")
        if not isinstance(items, list):
            raise ValueError("structured unit has no item list")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("structured unit contains a non-object item")
            cloned = dict(item)
            out.append(cloned)
    return out


def _item_is_insert(item: Dict[str, Any]) -> bool:
    return bool(item.get("insert") or item.get("new") or item.get("_insert"))


def _unit_is_insert(unit: Dict[str, Any]) -> bool:
    return bool(unit.get("insert") or unit.get("new") or unit.get("_insert"))


def _item_signature(item: Dict[str, Any]) -> Tuple[Any, ...]:
    kind = str(item.get("kind", ""))
    if kind == "text":
        return (
            kind,
            int(item.get("length", -1)),
            str(item.get("original_text", "")),
            str(item.get("control_suffix", "")),
        )
    if kind == "control_text":
        return (kind, int(item.get("length", -1)), str(item.get("control", "")))
    if kind == "opcode_gap":
        return (kind, int(item.get("length", -1)), str(item.get("bytes", "")).upper())
    return (kind,)


def _unit_signature(unit: Dict[str, Any]) -> Tuple[Tuple[Any, ...], ...]:
    items = unit.get("items")
    if not isinstance(items, list):
        return tuple()
    return tuple(_item_signature(item) for item in items if isinstance(item, dict) and not _item_is_insert(item))


def _unit_matches_template(unit: Dict[str, Any], template_unit: Dict[str, Any]) -> bool:
    """Return True when an editor unit still corresponds to the next original unit.

    Offsets and unit indexes are intentionally ignored. The stable matching
    fields are non-editable provenance fields such as original_text, control
    strings, gap bytes, item kinds, and original lengths. translated_text is
    ignored so normal edits still match.
    """
    return _unit_signature(unit) == _unit_signature(template_unit)


def _as_inserted_unit(unit: Dict[str, Any]) -> Dict[str, Any]:
    """Clone a unit and mark every child item as an inserted stream item."""
    out_unit = dict(unit)
    out_unit["insert"] = True
    out_unit.pop("unit_index", None)
    out_unit.pop("start_offset", None)
    out_unit.pop("end_offset", None)
    out_items: List[Dict[str, Any]] = []
    for item_index, item in enumerate(unit.get("items", [])):
        if not isinstance(item, dict):
            raise ValueError(f"inserted unit item {item_index} is not an object")
        cloned = dict(item)
        cloned["insert"] = True
        cloned.pop("offset", None)
        out_items.append(cloned)
    out_unit["items"] = out_items
    return out_unit


def _hydrate_offsets_from_template(
    orig_data: bytes,
    units: List[Dict[str, Any]],
    *,
    encoding: str,
    font_table: Optional[FontTable],
) -> List[Dict[str, Any]]:
    """
    Editor JSON hides unit indexes and offsets by default.  Rebuild recovers
    original offsets by aligning non-insert units/items with a freshly parsed
    template from orig_data, using order as the source of truth.

    To add a completely new unit/page, add a unit object with ``"insert": true``.
    All text/control/opcode items inside that new unit are treated as inserted
    stream bytes and require --allow-growth later.  Inserting a new unit at the
    very beginning of a script is rejected for now because the safe insertion
    coordinate before the first original text unit is not yet proven.

    To add an item inside an existing unit, add ``"insert": true`` to that item.
    """
    records = read_text_records_structured(orig_data, encoding, font_table)
    template_units = make_structured_units(
        orig_data, records, include_offsets=True,
        encoding=encoding, font_table=font_table, decode_gaps=False,
    )
    hydrated: List[Dict[str, Any]] = []

    tmpl_unit_pos = 0
    seen_original_unit = False

    for json_unit_pos, unit in enumerate(units):
        if not isinstance(unit, dict):
            raise ValueError("structured JSON must be a list of unit objects")
        src_items = unit.get("items")
        if not isinstance(src_items, list):
            raise ValueError(f"unit #{json_unit_pos} has no item list")

        explicit_insert = _unit_is_insert(unit)
        auto_insert = False
        if not explicit_insert:
            if tmpl_unit_pos >= len(template_units):
                auto_insert = True
            else:
                # If the current JSON unit no longer matches the next original
                # unit by non-editable provenance fields, treat it as a newly
                # inserted unit. This lets users clone an existing page/unit and
                # paste it into the list without manually tagging every child
                # item with insert=true. The following JSON unit will still align
                # to the same original template unit.
                auto_insert = not _unit_matches_template(unit, template_units[tmpl_unit_pos])

        if explicit_insert or auto_insert:
            if not seen_original_unit:
                raise ValueError(
                    "Inserted unit before the first original unit is not supported yet; "
                    "insert after an existing unit so the byte-stream insertion point is unambiguous."
                )
            hydrated.append(_as_inserted_unit(unit))
            continue

        tmpl_items = template_units[tmpl_unit_pos].get("items", [])
        tmpl_unit_pos += 1
        seen_original_unit = True

        out_unit = dict(unit)
        out_items: List[Dict[str, Any]] = []
        tmpl_pos = 0
        for item_index, item in enumerate(src_items):
            if not isinstance(item, dict):
                raise ValueError(f"unit #{json_unit_pos} item {item_index} is not an object")
            cloned = dict(item)
            if _item_is_insert(cloned):
                cloned.pop("offset", None)
                out_items.append(cloned)
                continue

            # Existing debug JSON may already contain offsets. Keep them, while
            # advancing the template cursor so following hidden-offset items align.
            if cloned.get("offset") not in {None, "", "new", "NEW"}:
                if tmpl_pos < len(tmpl_items):
                    tmpl_pos += 1
                out_items.append(cloned)
                continue

            if tmpl_pos >= len(tmpl_items):
                raise ValueError(
                    f"unit #{json_unit_pos} has extra non-insert item #{item_index}; "
                    "mark newly added items with insert=true"
                )
            tmpl = tmpl_items[tmpl_pos]
            tmpl_pos += 1
            if cloned.get("kind") != tmpl.get("kind"):
                raise ValueError(
                    f"unit #{json_unit_pos} item #{item_index} kind mismatch: "
                    f"JSON={cloned.get('kind')!r}, original={tmpl.get('kind')!r}. "
                    "Do not reorder existing items across text/control/opcode boundaries."
                )
            if "offset" not in tmpl:
                raise ValueError(f"internal error: template unit #{json_unit_pos} item #{item_index} has no offset")
            cloned["offset"] = tmpl["offset"]
            out_items.append(cloned)

        out_unit["items"] = out_items
        hydrated.append(out_unit)

    # It is valid for a JSON file to omit trailing untouched units only if it is
    # an old/debug partial file; normal extraction writes all units.  Rebuild will
    # preserve omitted original data via the final copy of orig_data[last:].
    return hydrated


def rebuild_scenario_from_structured_json(
    orig_data: bytes,
    units: List[Dict[str, Any]],
    *,
    encoding: str,
    font_table: Optional[FontTable],
    allow_growth: bool,
    verify_relocations: bool = True,
) -> bytes:
    """
    Rebuild a script using structured JSON items.

    Existing text/control_text items replace their original TEXT_OPCODE record.
    New text/control_text items may be inserted by adding an item without
    "offset"; this requires --allow-growth. Existing opcode_gap items validate
    and preserve non-text bytecode. New opcode_gap items without offset can be
    inserted only with --allow-growth and are considered advanced/unsafe.
    """
    if len(orig_data) < 0x10:
        raise ValueError("Invalid scenario: smaller than header")
    label_count = struct.unpack_from("<I", orig_data, 8)[0]
    code_size = struct.unpack_from("<I", orig_data, 4)[0]
    code_start = 0x10 + label_count * 0x24
    if len(orig_data) != code_start + code_size:
        raise ValueError("Invalid scenario size formula")

    units = _hydrate_offsets_from_template(orig_data, units, encoding=encoding, font_table=font_table)

    text_records = read_text_records_structured(orig_data, encoding, font_table)
    text_by_off: Dict[int, TextRecordInfo] = {r.offset_int: r for r in text_records}
    consumed_text_offsets: set[int] = set()
    items = _sorted_json_items(units)

    new_tail = bytearray()
    last = 16
    reloc_events: List[Dict[str, int]] = []

    def copy_until(old_abs: int) -> None:
        nonlocal last
        if old_abs < last:
            raise ValueError(f"structured JSON item order overlaps original bytecode: 0x{old_abs:X} < 0x{last:X}")
        new_tail.extend(orig_data[last:old_abs])
        last = old_abs

    def add_shift_event(old_start: int, old_end: int, delta: int) -> None:
        if delta:
            reloc_events.append({"old_start": old_start, "old_end": old_end, "delta": delta})

    for item in items:
        kind = str(item.get("kind", "text"))
        off = _get_item_offset(item)
        if kind in {"text", "control_text"}:
            if kind == "text":
                payload = _encode_structured_text_item(item, encoding, font_table)
            else:
                payload = _encode_structured_control_item(item, encoding)
            new_record = _make_text_record(payload, str(item.get("offset", "<new>")))
            if off is None:
                if not allow_growth:
                    raise ValueError("Inserted text/control_text item requires --allow-growth")
                # Insert at current original coordinate without consuming original bytes.
                insert_at = last
                new_tail.extend(new_record)
                add_shift_event(insert_at, insert_at, len(new_record))
                continue
            rec = text_by_off.get(off)
            if rec is None:
                raise ValueError(f"JSON references non-text offset 0x{off:X}")
            if off in consumed_text_offsets:
                raise ValueError(f"Duplicate text/control_text item for offset 0x{off:X}")
            consumed_text_offsets.add(off)
            copy_until(off)
            old_total = rec.total_size
            new_total = len(new_record)
            if (not allow_growth) and new_total > old_total:
                raise ValueError(
                    f"Text growth is disabled at 0x{off:X}: old_total={old_total}, new_total={new_total}. "
                    "Pass --allow-growth after verification."
                )
            new_tail.extend(new_record)
            last = rec.end_int
            add_shift_event(off, rec.end_int, new_total - old_total)
            continue

        if kind == "opcode_gap":
            raw = bytes.fromhex(str(item.get("bytes", "")))
            if off is None:
                if not allow_growth:
                    raise ValueError("Inserted opcode_gap item requires --allow-growth")
                insert_at = last
                new_tail.extend(raw)
                add_shift_event(insert_at, insert_at, len(raw))
                continue
            length = int(item.get("length", len(raw)))
            if len(raw) != length:
                raise ValueError(f"opcode_gap at 0x{off:X}: length field does not match bytes")
            copy_until(off)
            old_gap = orig_data[off:off + length]
            if old_gap != raw:
                raise ValueError(
                    f"opcode_gap mismatch at 0x{off:X}: JSON={raw.hex().upper()} original={old_gap.hex().upper()}"
                )
            new_tail.extend(old_gap)
            last = off + length
            continue

        raise ValueError(f"Unsupported structured item kind: {kind!r}")

    # Copy the remainder of the original script. This also preserves text records
    # not present in JSON, but extraction should normally include all text records.
    new_tail.extend(orig_data[last:])

    new_header = bytearray(orig_data[:16])
    total_delta = sum(e["delta"] for e in reloc_events)
    struct.pack_into("<I", new_header, 4, code_size + total_delta)
    new_data = bytearray(new_header + new_tail)

    # Update code-relative label table offsets. Labels observed inside text
    # records may anchor to TEXT_OPCODE+1; that anchor is kept stable relative to
    # the record start. Other inside-record anchors are rejected.
    for i in range(label_count):
        ptr_abs = 0x10 + i * 0x24 + 0x20
        p = struct.unpack_from("<I", new_data, ptr_abs)[0]
        old_file = code_start + p
        shift = 0
        for ev in reloc_events:
            old_start = ev["old_start"]
            old_end = ev["old_end"]
            delta = ev["delta"]
            if old_start == old_end:
                if old_file >= old_start:
                    shift += delta
                continue
            if old_file >= old_end:
                shift += delta
            elif old_start <= old_file < old_end:
                if old_file != old_start + 1:
                    raise ValueError(
                        f"Label {i} points inside edited text record at 0x{old_file:X}; "
                        "only record+1 anchor is currently supported"
                    )
        if shift:
            new_file = old_file + shift
            struct.pack_into("<I", new_data, ptr_abs, new_file - code_start)

    if verify_relocations:
        new_code_size = struct.unpack_from("<I", new_data, 4)[0]
        if len(new_data) != code_start + new_code_size:
            raise ValueError("Rebuilt scenario size formula mismatch")
    return bytes(new_data)


def build_meta(eboot_path: Path, eboot_data: bytes, entries: List[ScenarioEntry], encoding: str, font_table: Optional[FontTable], use_hash: bool) -> Dict[str, Any]:
    return {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "source_eboot": str(eboot_path),
        "source_eboot_size": len(eboot_data),
        "source_eboot_sha256": sha256_bytes(eboot_data),
        "table_start": DEFAULT_TABLE_START,
        "max_entries": DEFAULT_MAX_ENTRIES,
        "encoding": encoding,
        "extraction_mode": "structured_json",
        "hash_algorithm": "sha256" if use_hash else None,
        "font_tbl": font_table.meta() if font_table is not None else None,
        "notes": [
            "v19 extracts structured item lists; offsets are hidden by default and opcode_gap is raw-only by default.",
            "opcode_gap.bytes is the default/rebuild source of truth; pass --decode-gaps to add lossless VM-token asm/pseudo-C diagnostics.",
            "Edit only text.translated_text by default.",
            "control_text and opcode_gap are preserved as explicit non-editable bytecode items.",
            "Offsets are omitted by default; rebuild aligns existing items by unit/item order. Use extract --debug-offsets for diagnostics.",
            "New inserted text/control_text/opcode_gap items must be marked insert=true and require --allow-growth.",
        ],
        "entries": [asdict(e) for e in entries],
    }


def command_extract(args: argparse.Namespace) -> None:
    font_table = load_font_table_arg(args.font_tbl)
    eboot_path = Path(args.eboot)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eboot_data = eboot_path.read_bytes()
    scanned = scan_entries(
        eboot_data,
        table_start=args.table_start,
        max_entries=args.max_entries,
        encoding=args.encoding,
        font_table=font_table,
    )

    entries: List[ScenarioEntry] = []
    written = 0
    total_units = 0
    total_items = 0
    total_records = 0
    for entry, dec in scanned:
        records = read_text_records_structured(dec, args.encoding, font_table)
        units = make_structured_units(
            dec, records, include_offsets=args.debug_offsets,
            encoding=args.encoding, font_table=font_table, decode_gaps=args.decode_gaps,
        )
        entry.text_count = len(units)
        total_units += len(units)
        total_items += sum(len(u.get("items", [])) for u in units)
        total_records += len(records)
        if units:
            path = out_dir / entry.json_filename
            write_json(path, units)
            written += 1
            if args.use_hash:
                entry.json_sha256 = sha256_file(path)
        entries.append(entry)

    meta = build_meta(eboot_path, eboot_data, entries, args.encoding, font_table, args.use_hash)
    write_json(out_dir / META_FILENAME, meta)
    print(
        f"Extracted {written} structured JSON files; "
        f"units={total_units}, items={total_items}, underlying text_records={total_records}"
    )
    print(f"Metadata saved to {out_dir / META_FILENAME}")


def command_rebuild(args: argparse.Namespace) -> None:
    font_table = load_font_table_arg(args.font_tbl)
    input_dir = Path(args.input_dir)
    meta_path = Path(args.meta) if args.meta else input_dir / META_FILENAME
    meta = load_json(meta_path)
    entries = [normalize_entry(e) for e in meta.get("entries", [])]
    if not entries:
        raise SystemExit("Metadata contains no entries")
    if meta.get("extraction_mode") not in {"structured_json", "clean_logical_json"}:
        raise SystemExit("This v14 structured tool rebuilds only structured_json extractions")
    if args.use_hash and not meta.get("hash_algorithm"):
        raise SystemExit("Metadata has no JSON hashes. Re-run extract with --use-hash.")

    eboot_path = Path(args.eboot)
    eboot_data = bytearray(eboot_path.read_bytes())
    ph = read_elf_program_headers(bytes(eboot_data))
    table_start = int(meta.get("table_start", args.table_start))
    encoding = str(meta.get("encoding", args.encoding))

    entries_to_import: List[Tuple[ScenarioEntry, bytes]] = []
    skipped_unchanged = 0
    skipped_missing = 0
    for entry in entries:
        path = input_dir / entry.json_filename
        if not path.exists():
            skipped_missing += 1
            continue
        if args.use_hash:
            old_hash = entry.json_sha256
            new_hash = sha256_file(path)
            if old_hash and new_hash == old_hash:
                skipped_unchanged += 1
                continue
        units = load_json(path)
        if not isinstance(units, list):
            raise ValueError(f"{path} must contain a JSON list")
        orig_data = get_original_entry_data(bytes(eboot_data), ph, entry)
        rebuilt = rebuild_scenario_from_structured_json(
            orig_data, units, encoding=encoding, font_table=font_table,
            allow_growth=args.allow_growth, verify_relocations=not args.no_verify_relocations,
        )
        entries_to_import.append((entry, rebuilt))

    if not entries_to_import:
        print("No changed entries selected for import; output EBOOT will be a copy.")
    success, used = repack_entries_into_eboot(
        eboot_data=eboot_data,
        table_start=table_start,
        entries_to_import=entries_to_import,
        inplace_offset_base=args.inplace_offset_base,
        inplace_vaddr_base=args.inplace_vaddr_base,
        inplace_max_size=args.inplace_max_size,
        compressor="pylzma",
    )
    out_path = Path(args.output_eboot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(eboot_data)
    print(f"Imported {success} structured scenario entries into {out_path}")
    if args.use_hash:
        print(f"Skipped unchanged JSON files: {skipped_unchanged}")
    if skipped_missing:
        print(f"Skipped missing input files: {skipped_missing}")
    print(f"Reserved area used: {used} bytes")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Structured scenario extractor/rebuilder for EBOOT embedded scripts. "
                    "JSON exposes TEXT records, control_text, and opcode_gap items directly; offsets are hidden unless --debug-offsets is used."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract")
    p.add_argument("eboot")
    p.add_argument("output_dir")
    p.add_argument("--use-hash", action="store_true")
    p.add_argument("--font-tbl")
    p.add_argument("--table-start", type=parse_int, default=DEFAULT_TABLE_START)
    p.add_argument("--max-entries", type=int, default=DEFAULT_MAX_ENTRIES)
    p.add_argument("--encoding", default=DEFAULT_ENCODING)
    p.add_argument("--debug-offsets", action="store_true", help="include original byte offsets in extracted JSON for diagnostics")
    p.add_argument("--decode-gaps", action="store_true", help="add decoded asm/pseudo-C/tokens diagnostics to opcode_gap items; default is raw-only")
    p.add_argument("--raw-gaps-only", dest="raw_gaps_only", action="store_true", help="accepted for compatibility; raw-only is already the default")
    p.set_defaults(func=command_extract)

    p = sub.add_parser("rebuild")
    p.add_argument("eboot")
    p.add_argument("input_dir")
    p.add_argument("output_eboot")
    p.add_argument("--meta")
    p.add_argument("--use-hash", action="store_true")
    p.add_argument("--allow-growth", action="store_true")
    p.add_argument("--no-verify-relocations", action="store_true")
    p.add_argument("--font-tbl")
    p.add_argument("--table-start", type=parse_int, default=DEFAULT_TABLE_START)
    p.add_argument("--inplace-offset-base", type=parse_int, default=DEFAULT_INPLACE_OFFSET_BASE)
    p.add_argument("--inplace-vaddr-base", type=parse_int, default=DEFAULT_INPLACE_VADDR_BASE)
    p.add_argument("--inplace-max-size", type=parse_int, default=DEFAULT_INPLACE_MAX_SIZE)
    p.add_argument("--encoding", default=DEFAULT_ENCODING)
    p.set_defaults(func=command_rebuild)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
