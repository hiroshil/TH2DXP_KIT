# dat_tool Rust port

Rust port of the original Python `data_dat_cli.py` archive tool.

## Main CLI changes

- `extract` defaults to raw files only. It will not decode `.tpl` or `n/*.bin` to PNG sidecars unless `--clean` is passed.
- `--clean` enables editable PNG/JSON sidecars for supported image-like assets.
- `--image-only` emits only editable image sidecars and skips non-image raw files. It implies `--clean`.
- `--only-path <PATH>` filters extraction to archive paths under the given path prefix. `\` and `/` are normalized, so `--only-path ev` matches `ev\file1.tpl` and `ev/file1.tpl`.
- `--only-path` is rejected together with `--image-only`.
- Archive I/O is streaming per entry/chunk; the tool no longer reads the whole `data.dat` into RAM.
- `-j/--thread <N>` enables parallel extract/rebuild jobs. Default is `1`.

## Build

```bash
cargo build --release
```

## Examples

```bash
# Raw extraction only
dat_tool extract data.dat out

# Extract only ev/ paths as raw files
dat_tool extract data.dat out --only-path ev

# Decode supported image sidecars
dat_tool extract data.dat out --clean

# Parallel extract
dat_tool extract data.dat out --clean -j 8

# Rebuild
dat_tool rebuild out rebuilt.dat

# Selective/template rebuild from hash metadata
dat_tool extract data.dat out --clean --use-hash
dat_tool rebuild out rebuilt.dat --use-hash --template data.dat -j 8
```

## `n/*.bin` font chunks

The `n/XXXX.bin` assets are treated as engine font chunks, not as independent
256-glyph 1bpp images. Each valid file is exactly `0x2000` bytes and the file
stem must be a 0x10-aligned hex code such as `8020.bin`.

Layout used by `--clean` extraction and rebuild:

- 16 glyph slots per file: `XXXX+0` through `XXXX+F`.
- Each glyph is 32x32 pixels.
- Each glyph is `0x200` bytes.
- Pixels are 4bpp grayscale/alpha, two pixels per byte.
- High nibble is the left pixel, low nibble is the right pixel.
- PNG sidecar is a 4x4 grid, 128x128 pixels total.

The sidecar metadata filename remains `*.bin8192.json` for compatibility with
older extracted folders, but the encoder/decoder is now the corrected 4bpp
format.
