use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use image::ColorType;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tempfile::TempDir;

const ARCHIVE_ENTRY_SIZE: usize = 24;
const DEFAULT_CHUNK_SIZE: usize = 32_768;
const META_NAME: &str = "data_archive_meta.json";
const TPL_REC_SIZE: usize = 0x14;
const BIN8192_SIZE: usize = 8192;
const BIN8192_GLYPHS: usize = 16;
const BIN8192_CELL: usize = 32;
const BIN8192_COLS: usize = 4;
const BIN8192_ROWS: usize = 4;
const BIN8192_BYTES_PER_GLYPH: usize = 0x200;
const BIN8192_BYTES_PER_ROW: usize = BIN8192_CELL / 2;
const RAW_CHANNELS: [&str; 4] = ["A", "R", "G", "B"];
const ENGINE_FONT_ORDER: [&str; 4] = ["R", "G", "B", "A"];

fn is_a4r4g4b4(fmt: u8) -> bool { matches!(fmt, 0x83 | 0xA3) }
fn is_argb8888(fmt: u8) -> bool { matches!(fmt, 0x85 | 0xA5 | 0xBE) }
fn is_l8(fmt: u8) -> bool { matches!(fmt, 0x81 | 0xA1) }
fn is_g8b8(fmt: u8) -> bool { matches!(fmt, 0x8B | 0xAB) }
fn is_dxt1(fmt: u8) -> bool { matches!(fmt, 0x86 | 0xA6) }

#[derive(Parser, Debug)]
#[command(name = "dat_tool")]
#[command(about = "Safe extractor/rebuilder for this game's data.dat archive", long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Info {
        archive: PathBuf,
    },
    Extract {
        archive: PathBuf,
        output_dir: PathBuf,
        #[arg(long, help = "Decode supported image assets to editable PNG/JSON sidecars. Default is raw files only.")]
        clean: bool,
        #[arg(long, help = "Only process .tpl entries. With --clean: emit TPL PNG/JSON sidecars only. Without --clean: write raw .tpl files only.")]
        image_only: bool,
        #[arg(long = "only-path", value_name = "PATH", conflicts_with = "image_only", help = "Only extract entries whose archive path is under PATH. Both / and \\ are treated as separators.")]
        only_path: Option<String>,
        #[arg(long, help = "Store SHA-256 hashes for emitted reimportable files.")]
        use_hash: bool,
        #[arg(short = 'j', long = "thread", default_value_t = 1, value_name = "N", help = "Worker threads for extract. Default: 1.")]
        threads: usize,
    },
    Rebuild {
        extracted_dir: PathBuf,
        output_dat: PathBuf,
        #[arg(long, default_value_t = DEFAULT_CHUNK_SIZE)]
        chunk_size: usize,
        #[arg(long, help = "Copy template archive and only reimport watched files whose hashes changed.")]
        use_hash: bool,
        #[arg(long, help = "Original data.dat to use as selective/template source; overrides metadata path.")]
        template: Option<PathBuf>,
        #[arg(short = 'j', long = "thread", default_value_t = 1, value_name = "N", help = "Worker threads for rebuild compression. Default: 1.")]
        threads: usize,
    },
    SelfTest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ArchiveEntry {
    index: usize,
    name: String,
    name_offset: u32,
    compressed_size: u32,
    unknown0: u32,
    uncompressed_size: u32,
    unknown1: u32,
    file_offset: u32,
    #[serde(default)]
    chunk_count: usize,
    #[serde(default)]
    flags: Vec<u8>,
    #[serde(default)]
    raw_sha256: Option<String>,
    #[serde(default = "default_raw_kind")]
    asset_kind: String,
}

fn default_raw_kind() -> String { "raw".to_string() }

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
struct Sidecars {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tpl_meta: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    png: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bin_meta: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
struct HashFile {
    path: String,
    role: String,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EntryMeta {
    index: usize,
    name: String,
    name_offset: u32,
    compressed_size: u32,
    unknown0: u32,
    uncompressed_size: u32,
    unknown1: u32,
    file_offset: u32,
    chunk_count: usize,
    flags: Vec<u8>,
    raw_sha256: Option<String>,
    asset_kind: String,
    path: String,
    disk_path: String,
    sidecars: Sidecars,
    extracted_raw: bool,
    image_only_skipped: bool,
    #[serde(default)]
    hash_files: Vec<HashFile>,
}

impl EntryMeta {
    fn from_entry(ent: &ArchiveEntry, disk_path: String, sidecars: Sidecars, extracted_raw: bool, skipped: bool) -> Self {
        Self {
            index: ent.index,
            name: ent.name.clone(),
            name_offset: ent.name_offset,
            compressed_size: ent.compressed_size,
            unknown0: ent.unknown0,
            uncompressed_size: ent.uncompressed_size,
            unknown1: ent.unknown1,
            file_offset: ent.file_offset,
            chunk_count: ent.chunk_count,
            flags: ent.flags.clone(),
            raw_sha256: ent.raw_sha256.clone(),
            asset_kind: ent.asset_kind.clone(),
            path: ent.name.clone(),
            disk_path,
            sidecars,
            extracted_raw,
            image_only_skipped: skipped,
            hash_files: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ArchiveMeta {
    tool: String,
    archive: String,
    archive_size: u64,
    file_count: usize,
    selected_count: usize,
    header_unknown0: u32,
    header_unknown1: u32,
    entries_offset: u32,
    pre_entries_hex: String,
    entry_size: usize,
    entries: Vec<EntryMeta>,
    decode_assets: bool,
    clean: bool,
    image_only: bool,
    use_hash: bool,
    extract_policy: String,
    template_archive: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path_filter: Option<String>,
    hash_semantics: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TplRecord {
    index: usize,
    record_offset: usize,
    raw0: u16,
    raw1: u16,
    width: usize,
    height: usize,
    pitch: usize,
    fmt: u8,
    flags: u8,
    data_offset: usize,
    wrap_s: u16,
    wrap_t: u16,
    filter_mode: u16,
    mip_flag: u16,
    layout: String,
    payload_size: usize,
    span_size: usize,
}

impl TplRecord {
    fn class_name(&self) -> String { classify_tpl_fmt(self.fmt) }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TplRecordMeta {
    #[serde(flatten)]
    rec: TplRecord,
    format_class: String,
    payload_bin: Option<String>,
    png: Option<String>,
    channel_pngs: BTreeMap<String, String>,
    note: String,
    reimport_from_png: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    font_engine_order: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TplMeta {
    tool: String,
    template_file: String,
    logical_archive_path: String,
    file_size: usize,
    count: usize,
    record_size: usize,
    safe_rebuild: String,
    records: Vec<TplRecordMeta>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Info { archive } => print_archive_info(&archive),
        Command::Extract { archive, output_dir, clean, image_only, only_path, use_hash, threads } => {
            if threads == 0 { bail!("--thread must be >= 1"); }
            if image_only && !clean {
                eprintln!("note: --image-only without --clean extracts raw .tpl files only");
            }
            extract_archive(&archive, &output_dir, clean, use_hash, image_only, only_path.as_deref(), threads)
        }
        Command::Rebuild { extracted_dir, output_dat, chunk_size, use_hash, template, threads } => {
            if threads == 0 { bail!("--thread must be >= 1"); }
            if chunk_size == 0 { bail!("--chunk-size must be >= 1"); }
            rebuild_archive(&extracted_dir, &output_dat, false, chunk_size, use_hash, template.as_deref(), threads)
        }
        Command::SelfTest => self_test(),
    }
}

fn read_u16_be(buf: &[u8], off: usize) -> Result<u16> {
    let bytes = buf.get(off..off + 2).ok_or_else(|| anyhow!("short read u16 at 0x{off:X}"))?;
    Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
}

fn read_u32_be(buf: &[u8], off: usize) -> Result<u32> {
    let bytes = buf.get(off..off + 4).ok_or_else(|| anyhow!("short read u32 at 0x{off:X}"))?;
    Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn write_u32_be_at(buf: &mut [u8], off: usize, v: u32) -> Result<()> {
    let dst = buf.get_mut(off..off + 4).ok_or_else(|| anyhow!("short write u32 at 0x{off:X}"))?;
    dst.copy_from_slice(&v.to_be_bytes());
    Ok(())
}

fn write_entry_record<W: Write>(mut w: W, rec: (u32, u32, u32, u32, u32, u32)) -> Result<()> {
    for v in [rec.0, rec.1, rec.2, rec.3, rec.4, rec.5] {
        w.write_all(&v.to_be_bytes())?;
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

fn hex_decode(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 { bail!("hex string has odd length"); }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = (bytes[i] as char).to_digit(16).ok_or_else(|| anyhow!("invalid hex"))?;
        let lo = (bytes[i + 1] as char).to_digit(16).ok_or_else(|| anyhow!("invalid hex"))?;
        out.push(((hi << 4) | lo) as u8);
    }
    Ok(out)
}

fn latin1_to_string(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
}

fn string_to_latin1(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len());
    for ch in s.chars() {
        let v = ch as u32;
        if v > 0xFF { bail!("archive path contains non-latin1 character: {s}"); }
        out.push(v as u8);
    }
    Ok(out)
}

fn normalize_archive_path(s: &str) -> String {
    let mut out = s.replace('\\', "/");
    while out.contains("//") { out = out.replace("//", "/"); }
    out.trim_matches('/').to_string()
}

fn path_matches_filter(name: &str, filter: Option<&str>) -> bool {
    let Some(filter) = filter else { return true; };
    let p = normalize_archive_path(filter);
    if p.is_empty() { return true; }
    let n = normalize_archive_path(name);
    n == p || n.starts_with(&(p + "/"))
}

fn safe_out_path(root: &Path, archive_path: &str) -> Result<PathBuf> {
    let rel = normalize_archive_path(archive_path);
    let path = Path::new(&rel);
    if path.is_absolute() || path.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        bail!("unsafe archive path: {archive_path}");
    }
    // Do not create directories here. In --image-only mode many non-image
    // entries are intentionally skipped; creating parents during validation
    // produced large trees of empty folders before any asset was emitted.
    Ok(root.join(path))
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create directory {}", parent.display()))?;
    }
    Ok(())
}

fn rel_to(root: &Path, path: &Path) -> Result<String> {
    Ok(path.strip_prefix(root)?.to_string_lossy().replace('\\', "/"))
}

fn file_name_lossy(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "asset".to_string())
}


fn sha256_bytes(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex_encode(&h.finalize())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut h = Sha256::new();
    let mut f = File::open(path).with_context(|| format!("open for sha256: {}", path.display()))?;
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        h.update(&buf[..n]);
    }
    Ok(hex_encode(&h.finalize()))
}

fn parse_archive_table(archive_path: &Path) -> Result<(ArchiveMeta, Vec<ArchiveEntry>)> {
    let mut f = File::open(archive_path).with_context(|| format!("open archive: {}", archive_path.display()))?;
    let mut hdr = [0u8; 16];
    f.read_exact(&mut hdr).context("archive too small for header")?;
    let file_count = read_u32_be(&hdr, 0)? as usize;
    let unk_header0 = read_u32_be(&hdr, 4)?;
    let unk_header1 = read_u32_be(&hdr, 8)?;
    let entries_offset = read_u32_be(&hdr, 12)?;
    if file_count > 200_000 { bail!("unreasonable file count: {file_count}"); }
    if entries_offset < 16 { bail!("invalid entries_offset: 0x{entries_offset:X}"); }

    f.seek(SeekFrom::Start(0))?;
    let mut pre_entries = vec![0u8; entries_offset as usize];
    f.read_exact(&mut pre_entries)?;

    let mut entries = Vec::with_capacity(file_count);
    let mut rec_buf = [0u8; ARCHIVE_ENTRY_SIZE];
    for i in 0..file_count {
        f.seek(SeekFrom::Start(entries_offset as u64 + (i * ARCHIVE_ENTRY_SIZE) as u64))?;
        f.read_exact(&mut rec_buf).with_context(|| format!("truncated entry {i}"))?;
        let name_offset = read_u32_be(&rec_buf, 0)?;
        let compressed_size = read_u32_be(&rec_buf, 4)?;
        let unknown0 = read_u32_be(&rec_buf, 8)?;
        let uncompressed_size = read_u32_be(&rec_buf, 12)?;
        let unknown1 = read_u32_be(&rec_buf, 16)?;
        let file_offset = read_u32_be(&rec_buf, 20)?;
        let name = read_c_string_at(&mut f, name_offset as u64)?;
        entries.push(ArchiveEntry {
            index: i,
            name,
            name_offset,
            compressed_size,
            unknown0,
            uncompressed_size,
            unknown1,
            file_offset,
            chunk_count: 0,
            flags: Vec::new(),
            raw_sha256: None,
            asset_kind: "raw".into(),
        });
    }

    let meta = ArchiveMeta {
        tool: "dat_tool Rust".to_string(),
        archive: archive_path.to_string_lossy().into_owned(),
        archive_size: archive_path.metadata()?.len(),
        file_count,
        selected_count: file_count,
        header_unknown0: unk_header0,
        header_unknown1: unk_header1,
        entries_offset,
        pre_entries_hex: hex_encode(&pre_entries),
        entry_size: ARCHIVE_ENTRY_SIZE,
        entries: Vec::new(),
        decode_assets: false,
        clean: false,
        image_only: false,
        use_hash: false,
        extract_policy: "raw".to_string(),
        template_archive: archive_path.to_string_lossy().into_owned(),
        path_filter: None,
        hash_semantics: "disabled".to_string(),
    };
    Ok((meta, entries))
}

fn read_c_string_at(f: &mut File, off: u64) -> Result<String> {
    f.seek(SeekFrom::Start(off))?;
    let mut out = Vec::new();
    let mut b = [0u8; 1];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 { bail!("unterminated name string at 0x{off:X}"); }
        if b[0] == 0 { break; }
        out.push(b[0]);
    }
    Ok(latin1_to_string(&out))
}

struct HashingWriter<W> {
    inner: W,
    hasher: Sha256,
    written: u64,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self { Self { inner, hasher: Sha256::new(), written: 0 } }
    fn finish(self) -> (W, u64, String) { (self.inner, self.written, hex_encode(&self.hasher.finalize())) }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> { self.inner.flush() }
}

fn copy_exact_with_hash<R: Read, W: Write>(mut r: R, mut w: W, size: u64) -> Result<(u64, String)> {
    let mut h = Sha256::new();
    let mut left = size;
    let mut buf = vec![0u8; 1024 * 1024];
    let mut total = 0u64;
    while left > 0 {
        let take = left.min(buf.len() as u64) as usize;
        let n = r.read(&mut buf[..take])?;
        if n == 0 { bail!("unexpected EOF while copying raw entry"); }
        h.update(&buf[..n]);
        w.write_all(&buf[..n])?;
        total += n as u64;
        left -= n as u64;
    }
    Ok((total, hex_encode(&h.finalize())))
}

fn decompress_entry_to_writer<W: Write>(archive_path: &Path, entry: &ArchiveEntry, mut writer: W) -> Result<(Vec<u8>, u64, String)> {
    let mut f = File::open(archive_path)?;
    f.seek(SeekFrom::Start(entry.file_offset as u64))?;
    if entry.compressed_size == 0 {
        let (written, sha) = copy_exact_with_hash(&mut f, &mut writer, entry.uncompressed_size as u64)?;
        if written != entry.uncompressed_size as u64 { bail!("{}: raw size mismatch", entry.name); }
        return Ok((Vec::new(), written, sha));
    }

    let mut pos = entry.file_offset as u64;
    let end = entry.file_offset as u64 + entry.compressed_size as u64;
    let mut flags = Vec::new();
    let hashing = HashingWriter::new(writer);
    let mut hw = hashing;
    while pos < end {
        if pos + 4 > end { bail!("{}: truncated chunk header", entry.name); }
        let mut prefix_buf = [0u8; 4];
        f.read_exact(&mut prefix_buf)?;
        pos += 4;
        let prefix = u32::from_be_bytes(prefix_buf);
        let flag = ((prefix >> 24) & 0xFF) as u8;
        let chunk_size = (prefix & 0x00FF_FFFF) as usize;
        flags.push(flag);
        if pos + chunk_size as u64 > end { bail!("{}: chunk exceeds compressed range", entry.name); }
        let mut chunk = vec![0u8; chunk_size];
        f.read_exact(&mut chunk)?;
        pos += chunk_size as u64;
        match flag {
            0x00 | 0x80 => {
                let mut out = Vec::new();
                lzma_rs::lzma_decompress(&mut Cursor::new(chunk), &mut out)
                    .map_err(|e| anyhow!("{}: lzma decompress failed: {e:?}", entry.name))?;
                hw.write_all(&out)?;
            }
            0x40 => hw.write_all(&chunk)?,
            _ => bail!("{}: unsupported chunk flag 0x{flag:02X}", entry.name),
        }
        let pad = (4 - ((4 + chunk_size) % 4)) % 4;
        if pad != 0 {
            f.seek(SeekFrom::Current(pad as i64))?;
            pos += pad as u64;
        }
        if pos > end { bail!("{}: padding exceeds compressed range", entry.name); }
    }
    let (_inner, written, sha) = hw.finish();
    if written != entry.uncompressed_size as u64 {
        bail!("{}: decompressed size 0x{:X} != table 0x{:X}", entry.name, written, entry.uncompressed_size);
    }
    Ok((flags, written, sha))
}

fn decompress_entry_to_vec(archive_path: &Path, entry: &ArchiveEntry) -> Result<(Vec<u8>, Vec<u8>, String)> {
    let mut data = Vec::with_capacity(entry.uncompressed_size as usize);
    let (flags, _, sha) = decompress_entry_to_writer(archive_path, entry, &mut data)?;
    Ok((data, flags, sha))
}

fn copy_original_compressed_entry(archive_path: &Path, entry: &ArchiveEntry, out: &mut File) -> Result<u64> {
    let mut f = File::open(archive_path)?;
    f.seek(SeekFrom::Start(entry.file_offset as u64))?;
    let bytes = if entry.compressed_size == 0 { entry.uncompressed_size } else { entry.compressed_size } as u64;
    let mut limited = f.take(bytes);
    io::copy(&mut limited, out).with_context(|| format!("copy original compressed entry: {}", entry.name))
}

fn classify_tpl_fmt(fmt: u8) -> String {
    if is_a4r4g4b4(fmt) { "A4R4G4B4_16bpp".into() }
    else if is_argb8888(fmt) { "A8R8G8B8_32bpp".into() }
    else if is_l8(fmt) { "L8_8bpp".into() }
    else if is_g8b8(fmt) { "G8B8_16bpp_LA".into() }
    else if is_dxt1(fmt) { "DXT1".into() }
    else { format!("unknown_0x{fmt:02X}") }
}

fn tpl_row_pitch(fmt: u8, width: usize, pitch: usize) -> usize {
    if is_argb8888(fmt) { if pitch != 0 { pitch } else { width * 4 } }
    else if is_a4r4g4b4(fmt) || is_g8b8(fmt) { if pitch != 0 { pitch } else { width * 2 } }
    else if is_l8(fmt) { if pitch != 0 { pitch } else { width } }
    else if is_dxt1(fmt) { if pitch != 0 { pitch } else { ((width + 3) / 4) * 8 } }
    else { pitch }
}

fn tpl_expected_payload_size(fmt: u8, width: usize, height: usize, pitch: usize) -> Option<usize> {
    if width == 0 || height == 0 { return None; }
    let row = tpl_row_pitch(fmt, width, pitch);
    if row == 0 { return None; }
    if is_dxt1(fmt) {
        let min_row = ((width + 3) / 4) * 8;
        if row < min_row { return None; }
        return Some(row * ((height + 3) / 4));
    }
    if is_argb8888(fmt) && row < width * 4 { return None; }
    if (is_a4r4g4b4(fmt) || is_g8b8(fmt)) && row < width * 2 { return None; }
    if is_l8(fmt) && row < width { return None; }
    if !(is_argb8888(fmt) || is_a4r4g4b4(fmt) || is_g8b8(fmt) || is_l8(fmt)) { return None; }
    Some(row * height)
}

fn parse_tpl_records(buf: &[u8]) -> Result<Vec<TplRecord>> {
    if buf.len() < 4 { bail!("TPL too small for count"); }
    let count = read_u32_be(buf, 0)? as usize;
    if count == 0 || count > 4096 { bail!("invalid/unreasonable TPL record count: {count}"); }
    let header_min = 4 + count * TPL_REC_SIZE;
    if buf.len() < header_min { bail!("TPL too small for {count} records"); }

    #[derive(Clone)]
    struct RawRec { off: usize, raw0: u16, raw1: u16, pitch: u16, fmt: u8, flags: u8, data_offset: usize, wrap_s: u16, wrap_t: u16, filter_mode: u16, mip_flag: u16 }
    let mut raw_records = Vec::with_capacity(count);
    for i in 0..count {
        let off = 4 + i * TPL_REC_SIZE;
        raw_records.push(RawRec {
            off,
            raw0: read_u16_be(buf, off)?,
            raw1: read_u16_be(buf, off + 2)?,
            pitch: read_u16_be(buf, off + 4)?,
            fmt: buf[off + 6],
            flags: buf[off + 7],
            data_offset: read_u32_be(buf, off + 8)? as usize,
            wrap_s: read_u16_be(buf, off + 12)?,
            wrap_t: read_u16_be(buf, off + 14)?,
            filter_mode: read_u16_be(buf, off + 16)?,
            mip_flag: read_u16_be(buf, off + 18)?,
        });
    }
    let data_offsets: Vec<usize> = raw_records.iter().map(|r| r.data_offset).collect();
    let mut records = Vec::with_capacity(count);
    for (i, r) in raw_records.iter().enumerate() {
        if r.data_offset < header_min || r.data_offset > buf.len() { bail!("record {i}: invalid data_offset 0x{:X}", r.data_offset); }
        let next_off = if i + 1 < count { data_offsets[i + 1] } else { buf.len() };
        if next_off < r.data_offset { bail!("record {i}: non-monotonic data offsets"); }
        let span = next_off - r.data_offset;
        let mut candidates: Vec<(isize, &str, usize, usize, usize)> = Vec::new();
        for (layout, width, height) in [("height_width", r.raw1 as usize, r.raw0 as usize), ("width_height", r.raw0 as usize, r.raw1 as usize)] {
            let Some(size) = tpl_expected_payload_size(r.fmt, width, height, r.pitch as usize) else { continue; };
            if r.data_offset + size > buf.len() || size > span { continue; }
            let row = tpl_row_pitch(r.fmt, width, r.pitch as usize);
            let mut score: isize = 0;
            if size == span { score += 1000; }
            else if matches!(span - size, 0x10 | 0x20 | 0x40 | 0x60 | 0x80 | 0x100 | 0x800 | 0x1000) { score += 700; }
            else { score += std::cmp::max(0isize, 500isize - ((span - size) / 0x10) as isize); }
            if layout == "height_width" { score += 50; }
            if is_dxt1(r.fmt) && row == ((width + 3) / 4) * 8 { score += 100; }
            if is_argb8888(r.fmt) && row == width * 4 { score += 100; }
            if (is_a4r4g4b4(r.fmt) || is_g8b8(r.fmt)) && row == width * 2 { score += 100; }
            if is_l8(r.fmt) && row == width { score += 100; }
            candidates.push((score, layout, width, height, size));
        }
        if candidates.is_empty() {
            bail!("record {i}: cannot infer payload for fmt=0x{:02X}, raw=(0x{:X},0x{:X}), pitch=0x{:X}, span=0x{:X}", r.fmt, r.raw0, r.raw1, r.pitch, span);
        }
        candidates.sort_by(|a, b| b.0.cmp(&a.0));
        let (_, layout, width, height, size) = candidates[0];
        records.push(TplRecord {
            index: i,
            record_offset: r.off,
            raw0: r.raw0,
            raw1: r.raw1,
            width,
            height,
            pitch: r.pitch as usize,
            fmt: r.fmt,
            flags: r.flags,
            data_offset: r.data_offset,
            wrap_s: r.wrap_s,
            wrap_t: r.wrap_t,
            filter_mode: r.filter_mode,
            mip_flag: r.mip_flag,
            layout: layout.to_string(),
            payload_size: size,
            span_size: span,
        });
    }
    Ok(records)
}

fn tpl_payload<'a>(buf: &'a [u8], rec: &TplRecord) -> &'a [u8] {
    &buf[rec.data_offset..rec.data_offset + rec.payload_size]
}

fn decode_tpl_record(buf: &[u8], rec: &TplRecord) -> Result<(Vec<u8>, ColorType)> {
    if is_a4r4g4b4(rec.fmt) { decode_tpl_a4(buf, rec) }
    else if is_argb8888(rec.fmt) { decode_tpl_argb(buf, rec) }
    else if is_l8(rec.fmt) { decode_tpl_l8(buf, rec) }
    else if is_g8b8(rec.fmt) { decode_tpl_g8b8(buf, rec) }
    else if is_dxt1(rec.fmt) { Ok((decode_dxt1_payload(tpl_payload(buf, rec), rec.width, rec.height, rec.pitch)?, ColorType::Rgba8)) }
    else { bail!("record {}: unsupported TPL fmt 0x{:02X}", rec.index, rec.fmt) }
}

fn decode_tpl_a4(buf: &[u8], rec: &TplRecord) -> Result<(Vec<u8>, ColorType)> {
    let row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch);
    let mut out = vec![0u8; rec.width * rec.height * 4];
    for y in 0..rec.height {
        let src = rec.data_offset + y * row;
        for x in 0..rec.width {
            let o = src + x * 2;
            let px = ((buf[o] as u16) << 8) | buf[o + 1] as u16;
            let a = (((px >> 12) & 0xF) * 17) as u8;
            let r = (((px >> 8) & 0xF) * 17) as u8;
            let g = (((px >> 4) & 0xF) * 17) as u8;
            let b = ((px & 0xF) * 17) as u8;
            let d = (y * rec.width + x) * 4;
            out[d..d + 4].copy_from_slice(&[r, g, b, a]);
        }
    }
    Ok((out, ColorType::Rgba8))
}

fn extract_tpl_a4_channels(buf: &[u8], rec: &TplRecord) -> Result<BTreeMap<String, Vec<u8>>> {
    let row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch);
    let mut planes: BTreeMap<String, Vec<u8>> = RAW_CHANNELS.iter().map(|ch| ((*ch).to_string(), vec![0u8; rec.width * rec.height])).collect();
    for y in 0..rec.height {
        let src = rec.data_offset + y * row;
        let ri = y * rec.width;
        for x in 0..rec.width {
            let o = src + x * 2;
            let px = ((buf[o] as u16) << 8) | buf[o + 1] as u16;
            planes.get_mut("A").unwrap()[ri + x] = (((px >> 12) & 0xF) * 17) as u8;
            planes.get_mut("R").unwrap()[ri + x] = (((px >> 8) & 0xF) * 17) as u8;
            planes.get_mut("G").unwrap()[ri + x] = (((px >> 4) & 0xF) * 17) as u8;
            planes.get_mut("B").unwrap()[ri + x] = ((px & 0xF) * 17) as u8;
        }
    }
    Ok(planes)
}

fn decode_tpl_argb(buf: &[u8], rec: &TplRecord) -> Result<(Vec<u8>, ColorType)> {
    let row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch);
    let mut out = vec![0u8; rec.width * rec.height * 4];
    for y in 0..rec.height {
        let src = rec.data_offset + y * row;
        for x in 0..rec.width {
            let o = src + x * 4;
            let (a, r, g, b) = (buf[o], buf[o + 1], buf[o + 2], buf[o + 3]);
            let d = (y * rec.width + x) * 4;
            out[d..d + 4].copy_from_slice(&[r, g, b, a]);
        }
    }
    Ok((out, ColorType::Rgba8))
}

fn decode_tpl_l8(buf: &[u8], rec: &TplRecord) -> Result<(Vec<u8>, ColorType)> {
    let row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch);
    let mut out = vec![0u8; rec.width * rec.height];
    for y in 0..rec.height {
        let src = rec.data_offset + y * row;
        let dst = y * rec.width;
        out[dst..dst + rec.width].copy_from_slice(&buf[src..src + rec.width]);
    }
    Ok((out, ColorType::L8))
}

fn decode_tpl_g8b8(buf: &[u8], rec: &TplRecord) -> Result<(Vec<u8>, ColorType)> {
    let row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch);
    let mut out = vec![0u8; rec.width * rec.height * 2];
    for y in 0..rec.height {
        let src = rec.data_offset + y * row;
        for x in 0..rec.width {
            let o = src + x * 2;
            let d = (y * rec.width + x) * 2;
            out[d] = buf[o];
            out[d + 1] = buf[o + 1];
        }
    }
    Ok((out, ColorType::La8))
}

fn rgb565_to_rgba(c: u16) -> [u8; 4] {
    let r = (((c >> 11) & 0x1F) as u32 * 255 / 31) as u8;
    let g = (((c >> 5) & 0x3F) as u32 * 255 / 63) as u8;
    let b = ((c & 0x1F) as u32 * 255 / 31) as u8;
    [r, g, b, 255]
}

fn decode_dxt1_payload(payload: &[u8], width: usize, height: usize, pitch: usize) -> Result<Vec<u8>> {
    let row_pitch = if pitch != 0 { pitch } else { ((width + 3) / 4) * 8 };
    let blocks_x = (width + 3) / 4;
    let blocks_y = (height + 3) / 4;
    let mut out = vec![0u8; width * height * 4];
    for by in 0..blocks_y {
        let row = by * row_pitch;
        for bx in 0..blocks_x {
            let off = row + bx * 8;
            if off + 8 > payload.len() { continue; }
            let c0 = u16::from_le_bytes([payload[off], payload[off + 1]]);
            let c1 = u16::from_le_bytes([payload[off + 2], payload[off + 3]]);
            let bits = u32::from_le_bytes([payload[off + 4], payload[off + 5], payload[off + 6], payload[off + 7]]);
            let mut colors = [[0u8; 4]; 4];
            colors[0] = rgb565_to_rgba(c0);
            colors[1] = rgb565_to_rgba(c1);
            if c0 > c1 {
                for i in 0..3 {
                    colors[2][i] = ((2 * colors[0][i] as u16 + colors[1][i] as u16) / 3) as u8;
                    colors[3][i] = ((colors[0][i] as u16 + 2 * colors[1][i] as u16) / 3) as u8;
                }
                colors[2][3] = 255;
                colors[3][3] = 255;
            } else {
                for i in 0..3 { colors[2][i] = ((colors[0][i] as u16 + colors[1][i] as u16) / 2) as u8; }
                colors[2][3] = 255;
                colors[3] = [0, 0, 0, 0];
            }
            for py in 0..4 {
                for px in 0..4 {
                    let x = bx * 4 + px;
                    let y = by * 4 + py;
                    if x < width && y < height {
                        let idx = ((bits >> (2 * (py * 4 + px))) & 3) as usize;
                        let d = (y * width + x) * 4;
                        out[d..d + 4].copy_from_slice(&colors[idx]);
                    }
                }
            }
        }
    }
    Ok(out)
}

fn save_png(path: &Path, data: &[u8], width: usize, height: usize, color: ColorType) -> Result<()> {
    image::save_buffer(path, data, width as u32, height as u32, color)
        .with_context(|| format!("save png: {}", path.display()))
}

fn extract_tpl_asset_from_bytes(tpl_data: &[u8], out_path: &Path, dump_raw_payload: bool, write_template: bool) -> Result<PathBuf> {
    if let Some(parent) = out_path.parent() { fs::create_dir_all(parent)?; }
    if write_template { fs::write(out_path, tpl_data)?; }
    let records = parse_tpl_records(tpl_data)?;
    let out_dir = out_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = out_path.file_stem().and_then(|s| s.to_str()).unwrap_or("asset");
    let mut meta = TplMeta {
        tool: "dat_tool Rust/tpl".to_string(),
        template_file: file_name_lossy(out_path),
        logical_archive_path: out_path.to_string_lossy().replace('\\', "/"),
        file_size: tpl_data.len(),
        count: records.len(),
        record_size: TPL_REC_SIZE,
        safe_rebuild: "template-based; preserve header/records/offsets/gaps; replace exact payload ranges only".to_string(),
        records: Vec::new(),
    };
    for rec in &records {
        let base = format!("{stem}_rec{:02}_fmt{:02X}", rec.index, rec.fmt);
        let mut rec_meta = TplRecordMeta {
            rec: rec.clone(),
            format_class: rec.class_name(),
            payload_bin: None,
            png: None,
            channel_pngs: BTreeMap::new(),
            note: String::new(),
            reimport_from_png: !is_dxt1(rec.fmt),
            font_engine_order: None,
        };
        if dump_raw_payload {
            let payload_name = format!("{base}.payload.bin");
            fs::write(out_dir.join(&payload_name), tpl_payload(tpl_data, rec))?;
            rec_meta.payload_bin = Some(payload_name);
        }
        let (img, color) = decode_tpl_record(tpl_data, rec)?;
        if is_a4r4g4b4(rec.fmt) {
            let preview = format!("{base}_preview_rgba.png");
            save_png(&out_dir.join(&preview), &img, rec.width, rec.height, color)?;
            rec_meta.png = Some(preview);
            for (ch, plane) in extract_tpl_a4_channels(tpl_data, rec)? {
                let ch_name = format!("{base}_ch{ch}.png");
                save_png(&out_dir.join(&ch_name), &plane, rec.width, rec.height, ColorType::L8)?;
                rec_meta.channel_pngs.insert(ch, ch_name);
            }
            rec_meta.font_engine_order = Some(ENGINE_FONT_ORDER.iter().map(|s| s.to_string()).collect());
            rec_meta.note = "A4R4G4B4: rebuild uses chA/chR/chG/chB. Font engine pages are R,G,B,A per texture.".to_string();
        } else {
            let png_name = format!("{base}.png");
            save_png(&out_dir.join(&png_name), &img, rec.width, rec.height, color)?;
            rec_meta.png = Some(png_name);
            if is_dxt1(rec.fmt) {
                rec_meta.note = "DXT1: PNG is preview-only; rebuild preserves payload.bin/template payload.".to_string();
            }
        }
        meta.records.push(rec_meta);
    }
    let meta_path = out_dir.join(format!("{stem}.tplmeta.json"));
    fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)?;
    Ok(meta_path)
}

fn q4(v: u8) -> u8 { ((v as u16 * 15 + 127) / 255) as u8 }

fn load_luma_4bit(path: &Path, size: (usize, usize)) -> Result<Vec<u8>> {
    let img = image::open(path).with_context(|| format!("open png: {}", path.display()))?.to_luma8();
    if img.dimensions() != (size.0 as u32, size.1 as u32) {
        bail!("{}: size {:?} != expected {:?}", path.display(), img.dimensions(), size);
    }
    Ok(img.as_raw().iter().map(|&v| q4(v)).collect())
}

fn pack_tpl_a4(template_payload: &[u8], rec: &TplRecord, paths: &BTreeMap<String, PathBuf>) -> Result<Vec<u8>> {
    let size = (rec.width, rec.height);
    let row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch);
    let mut planes: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for ch in RAW_CHANNELS {
        let p = paths.get(ch).ok_or_else(|| anyhow!("missing A4 channel {ch}"))?;
        planes.insert(ch.to_string(), load_luma_4bit(p, size)?);
    }
    let mut out = template_payload.to_vec();
    for y in 0..rec.height {
        let dst = y * row;
        let src_row = y * rec.width;
        for x in 0..rec.width {
            let a = planes["A"][src_row + x] as u16;
            let r = planes["R"][src_row + x] as u16;
            let g = planes["G"][src_row + x] as u16;
            let b = planes["B"][src_row + x] as u16;
            let px = (a << 12) | (r << 8) | (g << 4) | b;
            let o = dst + x * 2;
            out[o] = (px >> 8) as u8;
            out[o + 1] = px as u8;
        }
    }
    Ok(out)
}

fn pack_tpl_argb(template_payload: &[u8], rec: &TplRecord, png: &Path) -> Result<Vec<u8>> {
    let img = image::open(png).with_context(|| format!("open png: {}", png.display()))?.to_rgba8();
    if img.dimensions() != (rec.width as u32, rec.height as u32) { bail!("{}: size mismatch", png.display()); }
    let src = img.as_raw();
    let row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch);
    let mut out = template_payload.to_vec();
    for y in 0..rec.height {
        let dst_row = y * row;
        let src_row = y * rec.width * 4;
        for x in 0..rec.width {
            let s = src_row + x * 4;
            let (r, g, b, a) = (src[s], src[s + 1], src[s + 2], src[s + 3]);
            let d = dst_row + x * 4;
            out[d..d + 4].copy_from_slice(&[a, r, g, b]);
        }
    }
    Ok(out)
}

fn pack_tpl_l8(template_payload: &[u8], rec: &TplRecord, png: &Path) -> Result<Vec<u8>> {
    let img = image::open(png).with_context(|| format!("open png: {}", png.display()))?.to_luma8();
    if img.dimensions() != (rec.width as u32, rec.height as u32) { bail!("{}: size mismatch", png.display()); }
    let src = img.as_raw();
    let row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch);
    let mut out = template_payload.to_vec();
    for y in 0..rec.height {
        out[y * row..y * row + rec.width].copy_from_slice(&src[y * rec.width..(y + 1) * rec.width]);
    }
    Ok(out)
}

fn pack_tpl_g8b8(template_payload: &[u8], rec: &TplRecord, png: &Path) -> Result<Vec<u8>> {
    let img = image::open(png).with_context(|| format!("open png: {}", png.display()))?.to_rgba8();
    if img.dimensions() != (rec.width as u32, rec.height as u32) { bail!("{}: size mismatch", png.display()); }
    let src = img.as_raw();
    let row = tpl_row_pitch(rec.fmt, rec.width, rec.pitch);
    let mut out = template_payload.to_vec();
    for y in 0..rec.height {
        for x in 0..rec.width {
            let s = (y * rec.width + x) * 4;
            let r = src[s] as u32;
            let g = src[s + 1] as u32;
            let b = src[s + 2] as u32;
            let a = src[s + 3];
            let l = ((299 * r + 587 * g + 114 * b + 500) / 1000) as u8;
            let d = y * row + x * 2;
            out[d] = l;
            out[d + 1] = a;
        }
    }
    Ok(out)
}

fn rebuild_tpl_asset_from_bytes(template_data: &[u8], meta_path: &Path) -> Result<Vec<u8>> {
    let mut buf = template_data.to_vec();
    let records = parse_tpl_records(&buf)?;
    let meta: TplMeta = serde_json::from_slice(&fs::read(meta_path)?)?;
    if records.len() != meta.count { bail!("{}: record count differs from TPL metadata", meta_path.display()); }
    let base_dir = meta_path.parent().unwrap_or_else(|| Path::new("."));
    let rec_meta: BTreeMap<usize, TplRecordMeta> = meta.records.into_iter().map(|r| (r.rec.index, r)).collect();
    for rec in &records {
        let m = rec_meta.get(&rec.index).ok_or_else(|| anyhow!("{}: metadata missing TPL record {}", meta_path.display(), rec.index))?;
        if m.rec.fmt != rec.fmt || m.rec.width != rec.width || m.rec.height != rec.height {
            bail!("{}: record {} differs from template", meta_path.display(), rec.index);
        }
        let old = tpl_payload(&buf, rec).to_vec();
        let new = if is_a4r4g4b4(rec.fmt) {
            let paths: BTreeMap<String, PathBuf> = m.channel_pngs.iter().map(|(ch, f)| (ch.clone(), base_dir.join(f))).collect();
            pack_tpl_a4(&old, rec, &paths)?
        } else if is_argb8888(rec.fmt) {
            pack_tpl_argb(&old, rec, &base_dir.join(m.png.as_ref().ok_or_else(|| anyhow!("missing TPL png"))?))?
        } else if is_l8(rec.fmt) {
            pack_tpl_l8(&old, rec, &base_dir.join(m.png.as_ref().ok_or_else(|| anyhow!("missing TPL png"))?))?
        } else if is_g8b8(rec.fmt) {
            pack_tpl_g8b8(&old, rec, &base_dir.join(m.png.as_ref().ok_or_else(|| anyhow!("missing TPL png"))?))?
        } else if is_dxt1(rec.fmt) {
            if let Some(payload_bin) = &m.payload_bin {
                let p = base_dir.join(payload_bin);
                if p.exists() { fs::read(p)? } else { old }
            } else { old }
        } else {
            bail!("{}: unsupported TPL fmt 0x{:02X}", meta_path.display(), rec.fmt);
        };
        if new.len() != rec.payload_size { bail!("{}: record {} new payload size mismatch", meta_path.display(), rec.index); }
        buf[rec.data_offset..rec.data_offset + rec.payload_size].copy_from_slice(&new);
    }
    Ok(buf)
}

fn nbin_chunk_base_code(rel_path: &str) -> Option<u32> {
    let norm = normalize_archive_path(rel_path);
    let parts: Vec<&str> = norm.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 || !parts[parts.len() - 2].eq_ignore_ascii_case("n") {
        return None;
    }
    let file_name = parts[parts.len() - 1];
    if !file_name.to_ascii_lowercase().ends_with(".bin") {
        return None;
    }
    let stem = &file_name[..file_name.len() - 4];
    if stem.len() != 4 {
        return None;
    }
    let value = u32::from_str_radix(stem, 16).ok()?;
    if (value & 0xF) == 0 { Some(value) } else { None }
}

fn looks_like_n_bin(rel_path: &str, data_len: usize) -> bool {
    // Engine-side evidence shows n/XXXX.bin is a 0x2000-byte font chunk:
    // 16 glyphs per file, 32x32 pixels per glyph, 4bpp, high nibble first.
    nbin_chunk_base_code(rel_path).is_some() && data_len == BIN8192_SIZE
}

fn u4_to_u8(v: u8) -> u8 {
    (v & 0x0F) * 17
}

fn u8_to_u4(v: u8) -> u8 {
    (((v as u16) * 15 + 127) / 255).min(15) as u8
}

fn bin8192_to_image(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() != BIN8192_SIZE { bail!("expected 8192 bytes, got {}", data.len()); }
    let w = BIN8192_COLS * BIN8192_CELL;
    let h = BIN8192_ROWS * BIN8192_CELL;
    let mut img = vec![0u8; w * h];
    for glyph in 0..BIN8192_GLYPHS {
        let gx = (glyph % BIN8192_COLS) * BIN8192_CELL;
        let gy = (glyph / BIN8192_COLS) * BIN8192_CELL;
        let base = glyph * BIN8192_BYTES_PER_GLYPH;
        for y in 0..BIN8192_CELL {
            let row = base + y * BIN8192_BYTES_PER_ROW;
            for xb in 0..BIN8192_BYTES_PER_ROW {
                let b = data[row + xb];
                let x = xb * 2;
                img[(gy + y) * w + gx + x] = u4_to_u8(b >> 4);
                img[(gy + y) * w + gx + x + 1] = u4_to_u8(b);
            }
        }
    }
    Ok(img)
}

fn image_to_bin8192(png_path: &Path) -> Result<Vec<u8>> {
    let img = image::open(png_path).with_context(|| format!("open png: {}", png_path.display()))?.to_luma8();
    let expected = ((BIN8192_COLS * BIN8192_CELL) as u32, (BIN8192_ROWS * BIN8192_CELL) as u32);
    if img.dimensions() != expected { bail!("{}: size {:?} != expected {:?}", png_path.display(), img.dimensions(), expected); }
    let px = img.as_raw();
    let mut out = vec![0u8; BIN8192_SIZE];
    let w = expected.0 as usize;
    for glyph in 0..BIN8192_GLYPHS {
        let gx = (glyph % BIN8192_COLS) * BIN8192_CELL;
        let gy = (glyph / BIN8192_COLS) * BIN8192_CELL;
        let base = glyph * BIN8192_BYTES_PER_GLYPH;
        for y in 0..BIN8192_CELL {
            let row = base + y * BIN8192_BYTES_PER_ROW;
            for xb in 0..BIN8192_BYTES_PER_ROW {
                let x = xb * 2;
                let hi = u8_to_u4(px[(gy + y) * w + gx + x]);
                let lo = u8_to_u4(px[(gy + y) * w + gx + x + 1]);
                out[row + xb] = (hi << 4) | lo;
            }
        }
    }
    Ok(out)
}

fn extract_bin8192_from_bytes(data: &[u8], raw_path: &Path, write_raw: bool) -> Result<PathBuf> {
    if let Some(parent) = raw_path.parent() { fs::create_dir_all(parent)?; }
    if write_raw { fs::write(raw_path, data)?; }
    let img = bin8192_to_image(data)?;
    let png_path = raw_path.with_file_name(format!("{}.png", file_name_lossy(raw_path)));
    save_png(&png_path, &img, BIN8192_COLS * BIN8192_CELL, BIN8192_ROWS * BIN8192_CELL, ColorType::L8)?;
    let base_code = nbin_chunk_base_code(&raw_path.to_string_lossy());
    let glyph_code_hex = base_code.map(|base| {
        (0..BIN8192_GLYPHS)
            .map(|i| format!("0x{:04X}", base + i as u32))
            .collect::<Vec<String>>()
    });
    let meta = serde_json::json!({
        "tool": "dat_tool Rust/nbin4bpp",
        "template_file": file_name_lossy(raw_path),
        "format": "n/XXXX.bin font chunk: 16 glyphs, 32x32, 4bpp grayscale/alpha, 0x200 bytes per glyph, high nibble first",
        "chunk_size": BIN8192_SIZE,
        "glyphs_per_chunk": BIN8192_GLYPHS,
        "cell": [BIN8192_CELL, BIN8192_CELL],
        "png_grid": [BIN8192_COLS, BIN8192_ROWS],
        "base_code_hex": base_code.map(|v| format!("0x{:04X}", v)),
        "glyph_code_hex": glyph_code_hex,
        "png": file_name_lossy(&png_path),
        "raw_sha256": sha256_bytes(data),
    });
    let meta_path = raw_path.with_file_name(format!("{}.bin8192.json", file_name_lossy(raw_path)));
    fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)?;
    Ok(png_path)
}

fn detect_asset_kind(rel_path: &str, data: &[u8]) -> String {
    if normalize_archive_path(rel_path).ends_with(".tpl") {
        if parse_tpl_records(data).is_ok() { return "tpl".to_string(); }
    }
    if looks_like_n_bin(rel_path, data.len()) { return "bin8192".to_string(); }
    "raw".to_string()
}

fn is_image_asset_kind(kind: &str) -> bool { matches!(kind, "tpl" | "bin8192") }

fn is_tpl_entry_name(name: &str) -> bool {
    normalize_archive_path(name).ends_with(".tpl")
}

fn should_try_image_entry(ent: &ArchiveEntry) -> bool {
    is_tpl_entry_name(&ent.name) || looks_like_n_bin(&ent.name, ent.uncompressed_size as usize)
}

fn extract_one_entry(archive_path: &Path, output_dir: &Path, ent: &ArchiveEntry, decode_assets: bool, image_only: bool) -> Result<(EntryMeta, String)> {
    let disk_path = normalize_archive_path(&ent.name);
    let raw_path = safe_out_path(output_dir, &ent.name)?;
    let mut entry = ent.clone();
    let mut sidecars = Sidecars::default();
    let mut extracted_raw = false;
    let mut skipped = false;

    if image_only && !is_tpl_entry_name(&ent.name) {
        skipped = true;
        entry.asset_kind = "raw".to_string();
        let rec = EntryMeta::from_entry(&entry, disk_path, sidecars, false, true);
        return Ok((rec, format!("skipped [{}] {} (non-tpl)", ent.index + 1, ent.name)));
    }

    if decode_assets && should_try_image_entry(ent) {
        let (data, flags, sha) = decompress_entry_to_vec(archive_path, ent)?;
        let kind = detect_asset_kind(&ent.name, &data);
        entry.flags = flags;
        entry.chunk_count = entry.flags.len();
        entry.raw_sha256 = Some(sha);
        entry.asset_kind = kind.clone();
        if image_only && !is_image_asset_kind(&kind) {
            skipped = true;
            let rec = EntryMeta::from_entry(&entry, disk_path, sidecars, false, true);
            return Ok((rec, format!("skipped [{}] {} (non-image)", ent.index + 1, ent.name)));
        }
        if kind == "tpl" {
            let meta_path = extract_tpl_asset_from_bytes(&data, &raw_path, !image_only, !image_only)
                .with_context(|| format!("decode TPL sidecar for {}", ent.name))?;
            sidecars.tpl_meta = Some(rel_to(output_dir, &meta_path)?);
            extracted_raw = !image_only;
        } else if kind == "bin8192" {
            let png_path = extract_bin8192_from_bytes(&data, &raw_path, !image_only)
                .with_context(|| format!("decode BIN8192 sidecar for {}", ent.name))?;
            sidecars.png = Some(rel_to(output_dir, &png_path)?);
            let meta_path = raw_path.with_file_name(format!("{}.bin8192.json", file_name_lossy(&raw_path)));
            sidecars.bin_meta = Some(rel_to(output_dir, &meta_path)?);
            extracted_raw = !image_only;
        } else if !image_only {
            ensure_parent_dir(&raw_path)?;
            fs::write(&raw_path, &data)?;
            extracted_raw = true;
        }
        let rec = EntryMeta::from_entry(&entry, disk_path, sidecars, extracted_raw, skipped);
        return Ok((rec, format!("extracted [{}] {} ({})", ent.index + 1, ent.name, entry.asset_kind)));
    }

    if image_only {
        // --image-only without --clean still filters to TPL entries, but writes
        // their raw bytes instead of decoding editable PNG/JSON sidecars.
        ensure_parent_dir(&raw_path)?;
        let f = File::create(&raw_path).with_context(|| format!("create {}", raw_path.display()))?;
        let (flags, _, sha) = decompress_entry_to_writer(archive_path, ent, f)?;
        entry.flags = flags;
        entry.chunk_count = entry.flags.len();
        entry.raw_sha256 = Some(sha);
        entry.asset_kind = "tpl".to_string();
        extracted_raw = true;
        let rec = EntryMeta::from_entry(&entry, disk_path, sidecars, extracted_raw, skipped);
        return Ok((rec, format!("extracted [{}] {} (tpl raw)", ent.index + 1, ent.name)));
    }

    ensure_parent_dir(&raw_path)?;
    let f = File::create(&raw_path).with_context(|| format!("create {}", raw_path.display()))?;
    let (flags, _, sha) = decompress_entry_to_writer(archive_path, ent, f)?;
    entry.flags = flags;
    entry.chunk_count = entry.flags.len();
    entry.raw_sha256 = Some(sha);
    entry.asset_kind = "raw".to_string();
    extracted_raw = true;
    let rec = EntryMeta::from_entry(&entry, disk_path, sidecars, extracted_raw, skipped);
    Ok((rec, format!("extracted [{}] {} (raw)", ent.index + 1, ent.name)))
}

fn watched_files_for_entry(output_dir: &Path, entry: &EntryMeta, include_raw: bool) -> Result<Vec<HashFile>> {
    let mut watched = Vec::new();
    let raw_path = output_dir.join(&entry.disk_path);
    if include_raw && raw_path.exists() {
        watched.push(HashFile { path: entry.disk_path.clone(), role: "raw".to_string(), sha256: sha256_file(&raw_path)? });
    }
    if entry.asset_kind == "tpl" {
        if let Some(tpl_meta_rel) = &entry.sidecars.tpl_meta {
            let tpl_meta_path = output_dir.join(tpl_meta_rel);
            if tpl_meta_path.exists() {
                let tpl_meta: TplMeta = serde_json::from_slice(&fs::read(&tpl_meta_path)?)?;
                let base = tpl_meta_path.parent().unwrap_or(output_dir);
                for rec in tpl_meta.records {
                    if !rec.reimport_from_png { continue; }
                    if !rec.channel_pngs.is_empty() {
                        for (ch, fname) in rec.channel_pngs {
                            let p = base.join(fname);
                            if p.exists() {
                                watched.push(HashFile { path: rel_to(output_dir, &p)?, role: format!("tpl_channel_{ch}"), sha256: sha256_file(&p)? });
                            }
                        }
                    } else if let Some(png) = rec.png {
                        let p = base.join(png);
                        if p.exists() {
                            watched.push(HashFile { path: rel_to(output_dir, &p)?, role: "tpl_png".to_string(), sha256: sha256_file(&p)? });
                        }
                    }
                    if let Some(payload_bin) = rec.payload_bin {
                        let p = base.join(payload_bin);
                        if p.exists() {
                            watched.push(HashFile { path: rel_to(output_dir, &p)?, role: "tpl_payload_bin".to_string(), sha256: sha256_file(&p)? });
                        }
                    }
                }
            }
        }
    } else if entry.asset_kind == "bin8192" {
        if let Some(png_rel) = &entry.sidecars.png {
            let p = output_dir.join(png_rel);
            if p.exists() {
                watched.push(HashFile { path: png_rel.clone(), role: "bin8192_png".to_string(), sha256: sha256_file(&p)? });
            }
        }
    }
    Ok(watched)
}


#[derive(Debug)]
struct ProgressBarText {
    label: String,
    total: usize,
    current: AtomicUsize,
    last_draw: Mutex<Instant>,
}

impl ProgressBarText {
    fn new(label: impl Into<String>, total: usize) -> Arc<Self> {
        let pb = Arc::new(Self {
            label: label.into(),
            total,
            current: AtomicUsize::new(0),
            last_draw: Mutex::new(Instant::now() - Duration::from_secs(1)),
        });
        pb.draw(0, false);
        pb
    }

    fn inc(&self, delta: usize) {
        let current = self.current.fetch_add(delta, Ordering::SeqCst).saturating_add(delta).min(self.total);
        let mut last = match self.last_draw.lock() {
            Ok(v) => v,
            Err(_) => return,
        };
        if current < self.total && last.elapsed() < Duration::from_millis(100) {
            return;
        }
        *last = Instant::now();
        drop(last);
        self.draw(current, false);
    }

    fn finish(&self) {
        self.current.store(self.total, Ordering::SeqCst);
        self.draw(self.total, true);
        eprintln!();
    }

    fn draw(&self, current: usize, done: bool) {
        let width = 32usize;
        let total = self.total.max(1);
        let filled = (current.saturating_mul(width) / total).min(width);
        let empty = width.saturating_sub(filled);
        let pct = (current.saturating_mul(100) / total).min(100);
        let status = if done { "done" } else { "work" };
        eprint!(
            "\r{}: [{}{}] {}/{} {:>3}% {}",
            self.label,
            "=".repeat(filled),
            " ".repeat(empty),
            current.min(self.total),
            self.total,
            pct,
            status,
        );
        let _ = io::stderr().flush();
    }
}

fn with_thread_pool<T, F>(threads: usize, f: F) -> Result<T>
where
    T: Send,
    F: FnOnce() -> Result<T> + Send,
{
    // Windows worker threads often have small default stacks. PNG/TPL decode and
    // lzma-rs can use enough stack to overflow Rayon defaults when -j is used,
    // so run even the single-worker path inside a pool with an explicit stack.
    let worker_stack = std::env::var("DAT_TOOL_WORKER_STACK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64 * 1024 * 1024);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .stack_size(worker_stack)
        .build()?;
    pool.install(f)
}

fn extract_archive(archive_path: &Path, output_dir: &Path, decode_assets: bool, use_hash: bool, image_only: bool, only_path: Option<&str>, threads: usize) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let (mut meta, entries_all) = parse_archive_table(archive_path)?;
    let entries: Vec<ArchiveEntry> = entries_all.into_iter().filter(|e| path_matches_filter(&e.name, only_path)).collect();

    let extract_pb = ProgressBarText::new("extract", entries.len());
    let extract_pb_for_work = Arc::clone(&extract_pb);
    let results: Vec<Result<(EntryMeta, String)>> = match with_thread_pool(threads, move || {
        if threads <= 1 {
            let mut out = Vec::with_capacity(entries.len());
            for ent in &entries {
                let res = extract_one_entry(archive_path, output_dir, ent, decode_assets, image_only);
                extract_pb_for_work.inc(1);
                out.push(res);
            }
            Ok(out)
        } else {
            Ok(entries.par_iter().map(|ent| {
                let res = extract_one_entry(archive_path, output_dir, ent, decode_assets, image_only);
                extract_pb_for_work.inc(1);
                res
            }).collect())
        }
    }) {
        Ok(v) => { extract_pb.finish(); v }
        Err(e) => { extract_pb.finish(); return Err(e); }
    };

    let mut meta_entries = Vec::with_capacity(results.len());
    let mut skipped_non_images = 0usize;
    for item in results {
        let (rec, msg) = item?;
        if rec.image_only_skipped { skipped_non_images += 1; }
        println!("{msg}");
        meta_entries.push(rec);
    }

    if use_hash {
        let hash_pb = ProgressBarText::new("extract/hash", meta_entries.len());
        for rec in &mut meta_entries {
            let include_raw = !image_only || !decode_assets;
            rec.hash_files = watched_files_for_entry(output_dir, rec, include_raw)?;
            hash_pb.inc(1);
        }
        hash_pb.finish();
    }

    meta.entries = meta_entries;
    meta.selected_count = meta.entries.len();
    meta.decode_assets = decode_assets;
    meta.clean = decode_assets;
    meta.image_only = image_only;
    meta.use_hash = use_hash;
    meta.extract_policy = if image_only { "image_only".to_string() } else if decode_assets { "clean".to_string() } else { "raw".to_string() };
    meta.template_archive = archive_path.to_string_lossy().into_owned();
    meta.path_filter = only_path.map(|s| s.to_string());
    meta.hash_semantics = if use_hash {
        "Hashes are recorded for emitted reimportable files. Selective rebuild with --use-hash copies the template archive and appends/repoints only entries whose watched files changed.".to_string()
    } else { "disabled".to_string() };

    let meta_path = output_dir.join(META_NAME);
    fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)?;
    println!("metadata: {}", meta_path.display());
    if image_only && decode_assets { println!("image-only: emitted TPL sidecars; skipped {skipped_non_images} non-TPL entries"); }
    else if image_only { println!("image-only: emitted raw TPL files; skipped {skipped_non_images} non-TPL entries"); }
    if let Some(p) = only_path { println!("only-path: selected {} entries under {:?}", meta.selected_count, p); }
    Ok(())
}

fn get_template_archive_path(meta: &ArchiveMeta, explicit_template: Option<&Path>) -> Result<PathBuf> {
    if let Some(t) = explicit_template { return Ok(t.to_path_buf()); }
    let p = PathBuf::from(&meta.template_archive);
    if !p.exists() { bail!("template archive not found: {}; pass --template", p.display()); }
    Ok(p)
}

fn raw_path_for_entry(root: &Path, entry: &EntryMeta) -> PathBuf {
    let rel = if entry.disk_path.is_empty() { normalize_archive_path(&entry.path) } else { entry.disk_path.clone() };
    root.join(rel)
}

fn entry_original_data_from_template(template_archive: &Path, template_entries: &[ArchiveEntry], entry_meta: &EntryMeta) -> Result<Vec<u8>> {
    let idx = entry_meta.index;
    if idx >= template_entries.len() { bail!("entry index {idx} not present in template archive"); }
    let ent = &template_entries[idx];
    if ent.name != entry_meta.path { bail!("template entry name mismatch at {idx}: {} != {}", ent.name, entry_meta.path); }
    let (data, _, _) = decompress_entry_to_vec(template_archive, ent)?;
    Ok(data)
}

fn current_changed_roles(root: &Path, entry_meta: &EntryMeta) -> Result<(bool, Vec<String>)> {
    let mut changed_roles = Vec::new();
    for item in &entry_meta.hash_files {
        let p = root.join(&item.path);
        if !p.exists() { bail!("watched file missing: {}", p.display()); }
        let now = sha256_file(&p)?;
        if now != item.sha256 { changed_roles.push(item.role.clone()); }
    }
    Ok((!changed_roles.is_empty(), changed_roles))
}

fn materialize_entry_data(root: &Path, entry_meta: &EntryMeta, raw_only: bool, template_data: Option<&[u8]>, prefer_raw: bool) -> Result<Vec<u8>> {
    let raw_path = raw_path_for_entry(root, entry_meta);
    if raw_only {
        if raw_path.exists() { return fs::read(raw_path).map_err(Into::into); }
        if let Some(data) = template_data { return Ok(data.to_vec()); }
        bail!("raw file not extracted and no template data available: {}", entry_meta.path);
    }
    if prefer_raw && raw_path.exists() { return fs::read(raw_path).map_err(Into::into); }
    if entry_meta.asset_kind == "tpl" {
        if let Some(tpl_meta_rel) = &entry_meta.sidecars.tpl_meta {
            let meta_path = root.join(tpl_meta_rel);
            if meta_path.exists() {
                if raw_path.exists() {
                    return rebuild_tpl_asset_from_bytes(&fs::read(raw_path)?, &meta_path);
                }
                if let Some(data) = template_data {
                    return rebuild_tpl_asset_from_bytes(data, &meta_path);
                }
                bail!("TPL template not available for {}", entry_meta.path);
            }
        }
    }
    if entry_meta.asset_kind == "bin8192" {
        if let Some(png_rel) = &entry_meta.sidecars.png {
            let png_path = root.join(png_rel);
            if png_path.exists() { return image_to_bin8192(&png_path); }
        }
    }
    if raw_path.exists() { return fs::read(raw_path).map_err(Into::into); }
    if let Some(data) = template_data { return Ok(data.to_vec()); }
    bail!("cannot materialize entry: {}", entry_meta.path)
}

fn compress_bytes_to_writer<W: Write>(data: &[u8], writer: W, chunk_size: usize) -> Result<(u64, u64)> {
    compress_stream_to_writer(Cursor::new(data), writer, chunk_size)
}

fn compress_stream_to_writer<R: Read, W: Write>(mut reader: R, mut writer: W, chunk_size: usize) -> Result<(u64, u64)> {
    let mut curr = vec![0u8; chunk_size];
    let n = reader.read(&mut curr)?;
    if n == 0 { return Ok((0, 0)); }
    curr.truncate(n);
    let mut total_uncomp = 0u64;
    let mut total_comp = 0u64;
    loop {
        let mut next = vec![0u8; chunk_size];
        let next_n = reader.read(&mut next)?;
        next.truncate(next_n);
        let flag = if next_n == 0 { 0x00u8 } else { 0x80u8 };
        let mut comp = Vec::new();
        lzma_rs::lzma_compress(&mut Cursor::new(&curr), &mut comp)
            .map_err(|e| anyhow!("lzma compress failed: {e:?}"))?;
        if comp.len() > 0x00FF_FFFF { bail!("compressed chunk too large: {}", comp.len()); }
        let prefix = ((flag as u32) << 24) | comp.len() as u32;
        writer.write_all(&prefix.to_be_bytes())?;
        writer.write_all(&comp)?;
        let pad = (4 - ((4 + comp.len()) % 4)) % 4;
        if pad != 0 { writer.write_all(&vec![0u8; pad])?; }
        total_uncomp += curr.len() as u64;
        total_comp += (4 + comp.len() + pad) as u64;
        if next_n == 0 { break; }
        curr = next;
    }
    Ok((total_uncomp, total_comp))
}

#[derive(Clone, Debug)]
struct PreparedEntry {
    index_in_meta: usize,
    original_index: usize,
    comp_path: PathBuf,
    name_offset: u32,
    comp_size: u32,
    unknown0: u32,
    uncomp_size: u32,
    unknown1: u32,
    message: String,
}

fn compress_entry_to_temp(root: &Path, entry: &EntryMeta, temp_dir: &Path, chunk_size: usize, raw_only: bool, template_archive: Option<&Path>, template_entries: Option<&[ArchiveEntry]>, index_in_meta: usize) -> Result<PreparedEntry> {
    let comp_path = temp_dir.join(format!("entry_{index_in_meta:06}.comp"));
    let mut out = File::create(&comp_path)?;
    let raw_path = raw_path_for_entry(root, entry);

    let can_copy_template = !raw_path.exists()
        && !raw_only
        && template_archive.is_some()
        && template_entries.is_some()
        && !(entry.asset_kind == "tpl" && entry.sidecars.tpl_meta.is_some())
        && !(entry.asset_kind == "bin8192" && entry.sidecars.png.is_some());

    let (uncomp_size, comp_size) = if can_copy_template {
        let template_archive = template_archive.unwrap();
        let t_entries = template_entries.unwrap();
        let ent = t_entries.get(entry.index).ok_or_else(|| anyhow!("entry index {} not present in template", entry.index))?;
        if ent.name != entry.path { bail!("template entry mismatch at {}", entry.index); }
        let _bytes_written = copy_original_compressed_entry(template_archive, ent, &mut out)?;
        (ent.uncompressed_size as u64, ent.compressed_size as u64)
    } else if (raw_only && raw_path.exists()) || (entry.asset_kind == "raw" && raw_path.exists()) || (entry.asset_kind != "tpl" && entry.asset_kind != "bin8192" && raw_path.exists()) {
        let f = File::open(&raw_path)?;
        compress_stream_to_writer(f, &mut out, chunk_size)?
    } else {
        let template_data = if !raw_path.exists() || entry.asset_kind == "tpl" {
            if let (Some(t_archive), Some(t_entries)) = (template_archive, template_entries) {
                Some(entry_original_data_from_template(t_archive, t_entries, entry)?)
            } else { None }
        } else { None };
        let data = materialize_entry_data(root, entry, raw_only, template_data.as_deref(), false)?;
        compress_bytes_to_writer(&data, &mut out, chunk_size)?
    };

    if uncomp_size > u32::MAX as u64 || comp_size > u32::MAX as u64 { bail!("entry too large for 32-bit archive fields: {}", entry.path); }
    Ok(PreparedEntry {
        index_in_meta,
        original_index: entry.index,
        comp_path,
        name_offset: 0,
        comp_size: comp_size as u32,
        unknown0: entry.unknown0,
        uncomp_size: uncomp_size as u32,
        unknown1: entry.unknown1,
        message: format!("packed [{}/?] {} size=0x{:X} comp=0x{:X}", index_in_meta + 1, entry.path, uncomp_size, comp_size),
    })
}

fn rebuild_archive_full(extracted_dir: &Path, output_dat: &Path, raw_only: bool, chunk_size: usize, threads: usize, template_override: Option<&Path>) -> Result<()> {
    let meta_path = extracted_dir.join(META_NAME);
    if !meta_path.exists() { bail!("missing metadata: {}", meta_path.display()); }
    let meta: ArchiveMeta = serde_json::from_slice(&fs::read(&meta_path)?)?;
    if meta.entries.is_empty() { bail!("metadata has no entries"); }

    let template_path = if meta.image_only { Some(get_template_archive_path(&meta, template_override)?) } else { None };
    let template_parsed = if let Some(path) = &template_path { Some(parse_archive_table(path)?.1) } else { None };

    let parent = output_dat.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let tmp = TempDir::new_in(parent)?;
    let prep_pb = ProgressBarText::new("rebuild/prepare", meta.entries.len());
    let prep_pb_for_work = Arc::clone(&prep_pb);
    let mut prepared: Vec<PreparedEntry> = match with_thread_pool(threads, || {
        if threads <= 1 {
            let mut v = Vec::with_capacity(meta.entries.len());
            for (i, ent) in meta.entries.iter().enumerate() {
                let res = compress_entry_to_temp(extracted_dir, ent, tmp.path(), chunk_size, raw_only, template_path.as_deref(), template_parsed.as_deref(), i);
                prep_pb_for_work.inc(1);
                v.push(res?);
            }
            Ok(v)
        } else {
            meta.entries.par_iter().enumerate()
                .map(|(i, ent)| {
                    let res = compress_entry_to_temp(extracted_dir, ent, tmp.path(), chunk_size, raw_only, template_path.as_deref(), template_parsed.as_deref(), i);
                    prep_pb_for_work.inc(1);
                    res
                })
                .collect::<Result<Vec<_>>>()
        }
    }) {
        Ok(v) => { prep_pb.finish(); v }
        Err(e) => { prep_pb.finish(); return Err(e); }
    };
    prepared.sort_by_key(|p| p.index_in_meta);

    let entries_offset = meta.entries_offset as usize;
    let mut pre_entries = hex_decode(&meta.pre_entries_hex).unwrap_or_default();
    if pre_entries.len() != entries_offset {
        pre_entries = vec![0u8; entries_offset];
        write_u32_be_at(&mut pre_entries, 4, meta.header_unknown0)?;
        write_u32_be_at(&mut pre_entries, 8, meta.header_unknown1)?;
    }
    if pre_entries.len() < 16 { pre_entries.resize(16, 0); }
    write_u32_be_at(&mut pre_entries, 0, meta.entries.len() as u32)?;
    write_u32_be_at(&mut pre_entries, 12, meta.entries_offset)?;
    if pre_entries.len() < entries_offset { pre_entries.resize(entries_offset, 0); }

    let names_offset_start = entries_offset + meta.entries.len() * ARCHIVE_ENTRY_SIZE;
    let mut names = Vec::new();
    let mut name_offsets = Vec::with_capacity(meta.entries.len());
    for ent in &meta.entries {
        name_offsets.push((names_offset_start + names.len()) as u32);
        names.extend_from_slice(&string_to_latin1(&ent.path)?);
        names.push(0);
    }
    while names.len() % 4 != 0 { names.push(0); }

    let mut out = File::create(output_dat)?;
    out.write_all(&pre_entries[..entries_offset])?;
    out.write_all(&vec![0u8; meta.entries.len() * ARCHIVE_ENTRY_SIZE])?;
    out.write_all(&names)?;

    let mut records = Vec::with_capacity(prepared.len());
    let write_pb = ProgressBarText::new("rebuild/write", prepared.len());
    for (i, mut p) in prepared.into_iter().enumerate() {
        let file_offset = out.stream_position()?;
        let mut comp = File::open(&p.comp_path)?;
        io::copy(&mut comp, &mut out)?;
        p.name_offset = name_offsets[i];
        records.push((p.name_offset, p.comp_size, p.unknown0, p.uncomp_size, p.unknown1, file_offset as u32));
        write_pb.inc(1);
    }
    write_pb.finish();

    out.seek(SeekFrom::Start(entries_offset as u64))?;
    for rec in records {
        write_entry_record(&mut out, rec)?;
    }
    println!("rebuilt archive: {}", output_dat.display());
    Ok(())
}

#[derive(Clone, Debug)]
struct PreparedChange {
    entry_index: usize,
    entry_off: u64,
    comp_path: PathBuf,
    comp_size: u32,
    uncomp_size: u32,
    unknown0: u32,
    unknown1: u32,
    name_offset: u32,
    message: String,
}

fn prepare_changed_entry(root: &Path, entry: &EntryMeta, template_archive: &Path, template_entries: &[ArchiveEntry], temp_dir: &Path, raw_only: bool, chunk_size: usize, roles: Vec<String>, entries_offset: u32) -> Result<PreparedChange> {
    let template_ent = template_entries.get(entry.index).ok_or_else(|| anyhow!("entry index {} missing in template", entry.index))?;
    if template_ent.name != entry.path { bail!("template entry mismatch at {}", entry.index); }
    let prefer_raw = roles.iter().any(|r| r == "raw") && !roles.iter().any(|r| r.starts_with("tpl_") || r == "bin8192_png");
    let template_data = if entry.asset_kind == "tpl" || !raw_path_for_entry(root, entry).exists() {
        Some(entry_original_data_from_template(template_archive, template_entries, entry)?)
    } else { None };
    let data = materialize_entry_data(root, entry, raw_only, template_data.as_deref(), prefer_raw)?;
    let comp_path = temp_dir.join(format!("changed_{:06}.comp", entry.index));
    let mut out = File::create(&comp_path)?;
    let (uncomp_size, comp_size) = compress_bytes_to_writer(&data, &mut out, chunk_size)?;
    if uncomp_size > u32::MAX as u64 || comp_size > u32::MAX as u64 { bail!("entry too large for 32-bit archive fields: {}", entry.path); }
    Ok(PreparedChange {
        entry_index: entry.index,
        entry_off: entries_offset as u64 + entry.index as u64 * ARCHIVE_ENTRY_SIZE as u64,
        comp_path,
        comp_size: comp_size as u32,
        uncomp_size: uncomp_size as u32,
        unknown0: entry.unknown0,
        unknown1: entry.unknown1,
        name_offset: template_ent.name_offset,
        message: format!("reimported [{}/?] {} roles={} size=0x{:X} comp=0x{:X}", entry.index + 1, entry.path, roles.join(","), uncomp_size, comp_size),
    })
}

fn selective_reimport_archive(extracted_dir: &Path, output_dat: &Path, template_archive: Option<&Path>, raw_only: bool, chunk_size: usize, threads: usize) -> Result<()> {
    let meta_path = extracted_dir.join(META_NAME);
    if !meta_path.exists() { bail!("missing metadata: {}", meta_path.display()); }
    let meta: ArchiveMeta = serde_json::from_slice(&fs::read(&meta_path)?)?;
    if !meta.use_hash { bail!("metadata was not extracted with --use-hash; cannot selective rebuild"); }
    let template_path = get_template_archive_path(&meta, template_archive)?;
    let (_, template_entries) = parse_archive_table(&template_path)?;

    let mut changed_entries = Vec::new();
    let mut skipped = 0usize;
    for entry in &meta.entries {
        let (changed, roles) = current_changed_roles(extracted_dir, entry)?;
        if changed { changed_entries.push((entry.clone(), roles)); } else { skipped += 1; }
    }

    let parent = output_dat.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    fs::copy(&template_path, output_dat).with_context(|| format!("copy template {} -> {}", template_path.display(), output_dat.display()))?;
    let tmp = TempDir::new_in(parent)?;

    let prep_pb = ProgressBarText::new("rebuild/selective-prepare", changed_entries.len());
    let prep_pb_for_work = Arc::clone(&prep_pb);
    let mut prepared: Vec<PreparedChange> = match with_thread_pool(threads, || {
        if threads <= 1 {
            let mut v = Vec::with_capacity(changed_entries.len());
            for (entry, roles) in &changed_entries {
                let res = prepare_changed_entry(extracted_dir, entry, &template_path, &template_entries, tmp.path(), raw_only, chunk_size, roles.clone(), meta.entries_offset);
                prep_pb_for_work.inc(1);
                v.push(res?);
            }
            Ok(v)
        } else {
            changed_entries.par_iter()
                .map(|(entry, roles)| {
                    let res = prepare_changed_entry(extracted_dir, entry, &template_path, &template_entries, tmp.path(), raw_only, chunk_size, roles.clone(), meta.entries_offset);
                    prep_pb_for_work.inc(1);
                    res
                })
                .collect::<Result<Vec<_>>>()
        }
    }) {
        Ok(v) => { prep_pb.finish(); v }
        Err(e) => { prep_pb.finish(); return Err(e); }
    };
    prepared.sort_by_key(|p| p.entry_index);

    let mut out = OpenOptions::new().read(true).write(true).open(output_dat)?;
    let write_pb = ProgressBarText::new("rebuild/selective-write", prepared.len());
    for p in &prepared {
        let file_offset = out.seek(SeekFrom::End(0))?;
        let mut comp = File::open(&p.comp_path)?;
        io::copy(&mut comp, &mut out)?;
        out.seek(SeekFrom::Start(p.entry_off))?;
        write_entry_record(&mut out, (p.name_offset, p.comp_size, p.unknown0, p.uncomp_size, p.unknown1, file_offset as u32))?;
        write_pb.inc(1);
    }
    write_pb.finish();
    println!("selective rebuild: changed={}, skipped={}, output={}", prepared.len(), skipped, output_dat.display());
    Ok(())
}

fn rebuild_archive(extracted_dir: &Path, output_dat: &Path, raw_only: bool, chunk_size: usize, use_hash: bool, template_archive: Option<&Path>, threads: usize) -> Result<()> {
    let meta_path = extracted_dir.join(META_NAME);
    if !meta_path.exists() { bail!("missing metadata: {}", meta_path.display()); }
    let meta: ArchiveMeta = serde_json::from_slice(&fs::read(&meta_path)?)?;
    if use_hash {
        selective_reimport_archive(extracted_dir, output_dat, template_archive, raw_only, chunk_size, threads)
    } else {
        // Full rebuild from metadata. If extraction was image-only, template data is used for raw files that were intentionally not emitted.
        if meta.image_only && template_archive.is_some() {
            // Keep metadata path unchanged; rebuild_archive_full resolves the path from metadata.
        }
        rebuild_archive_full(extracted_dir, output_dat, raw_only, chunk_size, threads, template_archive)
    }
}

fn print_archive_info(archive_path: &Path) -> Result<()> {
    let (meta, entries) = parse_archive_table(archive_path)?;
    println!("file: {}", archive_path.display());
    println!("size: 0x{:X}", meta.archive_size);
    println!("count: {} entries_offset=0x{:X}", meta.file_count, meta.entries_offset);
    for ent in entries {
        println!(
            "[{:<04}] {} name=0x{:X} comp=0x{:X} uncomp=0x{:X} data=0x{:X} unk=({},{})",
            ent.index, ent.name, ent.name_offset, ent.compressed_size, ent.uncompressed_size, ent.file_offset, ent.unknown0, ent.unknown1
        );
    }
    Ok(())
}

fn self_test() -> Result<()> {
    println!("self-test: n/XXXX.bin 4bpp roundtrip");
    let sample: Vec<u8> = (0..BIN8192_SIZE).map(|i| ((i * 37 + 13) & 0xFF) as u8).collect();
    let td = tempfile::tempdir()?;
    let p = td.path().join("n/8020.bin");
    extract_bin8192_from_bytes(&sample, &p, true)?;
    let rebuilt = image_to_bin8192(&p.with_file_name("8020.bin.png"))?;
    if rebuilt != sample { bail!("n/XXXX.bin 4bpp PNG roundtrip failed"); }
    println!("  OK");
    println!("self-test OK");
    Ok(())
}
