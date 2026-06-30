# scenario_eboot_tool Rust port

Rust port of the uploaded Python structured scenario extractor/rebuilder.

## Build

```bash
cargo build --release
```

The binary will be at:

```bash
target/release/scenario_eboot_tool
```

## Extract

Serial/default mode:

```bash
scenario_eboot_tool extract EBOOT.ELF out_json --use-hash --font-tbl font.tbl
```

Rayon mode with 8 worker threads:

```bash
scenario_eboot_tool extract EBOOT.ELF out_json --use-hash --font-tbl font.tbl -j 8
# equivalent:
scenario_eboot_tool extract EBOOT.ELF out_json --use-hash --font-tbl font.tbl --thread 8
```

## Rebuild

Serial/default mode:

```bash
scenario_eboot_tool rebuild EBOOT.ELF out_json EBOOT_MODDED.ELF --use-hash --font-tbl font.tbl
```

Rayon mode with 8 worker threads:

```bash
scenario_eboot_tool rebuild EBOOT.ELF out_json EBOOT_MODDED.ELF --use-hash --font-tbl font.tbl -j 8
# equivalent:
scenario_eboot_tool rebuild EBOOT.ELF out_json EBOOT_MODDED.ELF --use-hash --font-tbl font.tbl --thread 8
```

## Notes

- `-j/--thread` is optional. If omitted, the tool runs serially.
- `-j 0` is rejected.
- LZMA uses `lzma-sdk-rs` and manually wraps/unwraps LZMA-ALONE streams: 5-byte props + 8-byte little-endian unpacked size + raw LZMA payload.
- Rebuild parallelizes per-entry JSON parsing, scenario rebuild, and LZMA compression, then writes packed chunks back sequentially to preserve deterministic table layout.
- Extract parallelizes LZMA chunk decoding and structured JSON construction.
- Text decoding now follows CP932 / Windows-31J semantics because the engine uses CP932, not strict JIS Shift-JIS. The default `--encoding` is `cp932`, and legacy labels `shift_jis`, `sjis`, `ms932`, and `windows-31j` are accepted as aliases.
- CP932 single-byte handling is explicit: ASCII `0x00..0x7F` is decoded literally so `0x5C` stays `\` for engine controls; half-width kana `0xA1..0xDF` round-trips; CP932 PUA single bytes `0x80`, `0xA0`, `0xFD`, `0xFE`, and `0xFF` are preserved as `U+0080`, `U+F8F0`, `U+F8F1`, `U+F8F2`, and `U+F8F3`.
- CP932 double-byte handling accepts vendor/EUDC ranges through lead bytes `0xF0..0xFC` and builds the rebuild encoder map from the same decoder, so PUA/EUDC characters such as `U+E000..` encode back to their CP932 byte pairs where supported.
- Unmappable characters are still ignored during rebuild, matching the Python tool's lossy encode behavior.

### CP932 / PUA glyph notes

The engine uses CP932/Windows-31J, including vendor/EUDC/private-use glyph slots. Bytes such as `F0 40`, `F0 41`, and single-byte `FD` may appear in extracted JSON as Unicode Private Use Area characters like `\uE000`, `\uE001`, or `\uF8F1`. These characters may render as boxes, icons, a magnifier, a check mark, or nothing in a normal text editor depending on the editor font. That display does not mean the bytes are wrong.

For rebuild, these PUA characters are safe as long as the JSON editor preserves the UTF-8 codepoints. The encoder maps them back to their original CP932 bytes where CP932 defines the reverse mapping, for example `U+E001 -> F0 41` and `U+F8F1 -> FD`. Avoid editors or formatter steps that normalize, delete, or replace PUA characters with `?` or `\uFFFD`.
