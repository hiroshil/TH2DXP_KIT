use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use encoding_rs::SHIFT_JIS;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Read;
use std::sync::OnceLock;
use std::path::{Path, PathBuf};

const TOOL_NAME: &str = "scenario_eboot_structured_rust_cli";
const TOOL_VERSION: u32 = 1;
const DEFAULT_TABLE_START: usize = 0x000E_E8B4;
const DEFAULT_MAX_ENTRIES: usize = 2000;
const DEFAULT_INPLACE_OFFSET_BASE: usize = 0x0029_F000;
const DEFAULT_INPLACE_VADDR_BASE: usize = 0x002A_F000;
const DEFAULT_INPLACE_MAX_SIZE: usize = 3_734_108;
const RUST_WORKER_STACK_SIZE: usize = 64 * 1024 * 1024;
const META_FILENAME: &str = "scenario_meta.json";
const TEXT_OPCODE: [u8; 4] = [0x02, 0x0B, 0x00, 0x0A];
const DEFAULT_ENCODING: &str = "cp932";

#[derive(Debug, Clone)]
struct ProgramHeader {
    vaddr: u64,
    memsz: u64,
    offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScenarioEntry {
    entry_index: usize,
    filename: String,
    json_filename: String,
    name_ptr: u32,
    data_ptr: u32,
    file_offset: usize,
    uncomp_size: usize,
    comp_size_table: usize,
    comp_size_actual: usize,
    raw_chunk_size: usize,
    sha256_bin: String,
    #[serde(default)]
    text_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    json_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FontTableMeta {
    source: Option<String>,
    mapping_count: usize,
    duplicate_text_count: usize,
    max_byte_key_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScenarioMeta {
    #[serde(default)]
    tool: Option<String>,
    #[serde(default)]
    version: Option<u32>,
    #[serde(default)]
    source_eboot: Option<String>,
    #[serde(default)]
    source_eboot_size: Option<usize>,
    #[serde(default)]
    source_eboot_sha256: Option<String>,
    #[serde(default)]
    table_start: Option<usize>,
    #[serde(default)]
    max_entries: Option<usize>,
    #[serde(default)]
    encoding: Option<String>,
    #[serde(default)]
    extraction_mode: Option<String>,
    #[serde(default)]
    hash_algorithm: Option<String>,
    #[serde(default)]
    font_tbl: Option<FontTableMeta>,
    #[serde(default)]
    notes: Option<Vec<String>>,
    #[serde(default)]
    entries: Vec<ScenarioEntry>,
}

#[derive(Debug, Clone)]
struct TextRecordInfo {
    offset_int: usize,
    offset: String,
    length: usize,
    text: String,
    total_size: usize,
    end_int: usize,
}

#[derive(Debug, Clone)]
struct Candidate {
    entry_index: usize,
    name_ptr: u32,
    data_ptr: u32,
    uncomp_size: usize,
    comp_size_table: usize,
    file_offset: usize,
}

#[derive(Debug, Clone)]
struct ExtractedScript {
    entry: ScenarioEntry,
    units: Vec<Value>,
    records_count: usize,
    items_count: usize,
}

#[derive(Debug, Clone)]
struct RebuildTask {
    entry: ScenarioEntry,
    json_path: PathBuf,
}

#[derive(Debug, Clone)]
struct PreparedImport {
    entry: ScenarioEntry,
    uncomp_size: usize,
    lzma_alone: Vec<u8>,
}

#[derive(Parser, Debug)]
#[command(
    name = "scenario_eboot_tool",
    about = "Structured scenario extractor/rebuilder for EBOOT embedded scripts"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Extract {
        eboot: PathBuf,
        output_dir: PathBuf,
        #[arg(long)]
        use_hash: bool,
        #[arg(long = "font-tbl")]
        font_tbl: Option<PathBuf>,
        #[arg(long = "table-start", value_parser = parse_usize_arg, default_value_t = DEFAULT_TABLE_START)]
        table_start: usize,
        #[arg(long = "max-entries", default_value_t = DEFAULT_MAX_ENTRIES)]
        max_entries: usize,
        #[arg(long, default_value = DEFAULT_ENCODING)]
        encoding: String,
        #[arg(long = "debug-offsets")]
        debug_offsets: bool,
        #[arg(short = 'j', long = "thread", alias = "threads")]
        thread: Option<usize>,
    },
    Rebuild {
        eboot: PathBuf,
        input_dir: PathBuf,
        output_eboot: PathBuf,
        #[arg(long)]
        meta: Option<PathBuf>,
        #[arg(long)]
        use_hash: bool,
        #[arg(long = "allow-growth")]
        allow_growth: bool,
        #[arg(long = "no-verify-relocations")]
        no_verify_relocations: bool,
        #[arg(long = "font-tbl")]
        font_tbl: Option<PathBuf>,
        #[arg(long = "table-start", value_parser = parse_usize_arg, default_value_t = DEFAULT_TABLE_START)]
        table_start: usize,
        #[arg(long = "inplace-offset-base", value_parser = parse_usize_arg, default_value_t = DEFAULT_INPLACE_OFFSET_BASE)]
        inplace_offset_base: usize,
        #[arg(long = "inplace-vaddr-base", value_parser = parse_usize_arg, default_value_t = DEFAULT_INPLACE_VADDR_BASE)]
        inplace_vaddr_base: usize,
        #[arg(long = "inplace-max-size", value_parser = parse_usize_arg, default_value_t = DEFAULT_INPLACE_MAX_SIZE)]
        inplace_max_size: usize,
        #[arg(long, default_value = DEFAULT_ENCODING)]
        encoding: String,
        #[arg(short = 'j', long = "thread", alias = "threads")]
        thread: Option<usize>,
    },
}

fn parse_usize_arg(s: &str) -> std::result::Result<usize, String> {
    let t = s.trim().replace('_', "");
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        usize::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else {
        t.parse::<usize>().map_err(|e| e.to_string())
    }
}

fn parse_usize_value(value: &str) -> Result<usize> {
    parse_usize_arg(value).map_err(|e| anyhow::anyhow!(e))
}

#[derive(Debug, Clone, Copy)]
struct TextEncoding;

fn encoding_from_label(label: &str) -> Result<TextEncoding> {
    let normalized = label.trim().to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "cp932" | "ms932" | "windows-31j" | "windows31j" | "shift-jis" | "shiftjis" | "sjis" => Ok(TextEncoding),
        _ => bail!(
            "unsupported encoding label: {label}. Supported labels: cp932, ms932, windows-31j, shift_jis, sjis"
        ),
    }
}

fn validate_threads(thread: Option<usize>) -> Result<Option<usize>> {
    if let Some(0) = thread {
        bail!("-j/--thread must be >= 1");
    }
    Ok(thread)
}

fn run_with_optional_pool<T, F>(thread: Option<usize>, f: F) -> Result<T>
where
    T: Send,
    F: FnOnce() -> Result<T> + Send,
{
    if let Some(n) = validate_threads(thread)? {
        let pool = ThreadPoolBuilder::new()
            .num_threads(n)
            .stack_size(RUST_WORKER_STACK_SIZE)
            .build()
            .context("failed to build rayon thread pool")?;
        pool.install(f)
    } else {
        f()
    }
}

fn read_u16_le(buf: &[u8], off: usize) -> Result<u16> {
    if off + 2 > buf.len() {
        bail!("read_u16_le out of range at 0x{off:X}");
    }
    Ok(u16::from_le_bytes([buf[off], buf[off + 1]]))
}

fn read_u32_le(buf: &[u8], off: usize) -> Result<u32> {
    if off + 4 > buf.len() {
        bail!("read_u32_le out of range at 0x{off:X}");
    }
    Ok(u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]))
}

fn read_u32_be(buf: &[u8], off: usize) -> Result<u32> {
    if off + 4 > buf.len() {
        bail!("read_u32_be out of range at 0x{off:X}");
    }
    Ok(u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]))
}

fn read_u64_le(buf: &[u8], off: usize) -> Result<u64> {
    if off + 8 > buf.len() {
        bail!("read_u64_le out of range at 0x{off:X}");
    }
    Ok(u64::from_le_bytes([
        buf[off], buf[off + 1], buf[off + 2], buf[off + 3],
        buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7],
    ]))
}

fn read_u64_be(buf: &[u8], off: usize) -> Result<u64> {
    if off + 8 > buf.len() {
        bail!("read_u64_be out of range at 0x{off:X}");
    }
    Ok(u64::from_be_bytes([
        buf[off], buf[off + 1], buf[off + 2], buf[off + 3],
        buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7],
    ]))
}

fn write_u16_le(buf: &mut Vec<u8>, value: u16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_u32_le(buf: &mut [u8], off: usize, value: u32) -> Result<()> {
    if off + 4 > buf.len() {
        bail!("put_u32_le out of range at 0x{off:X}");
    }
    buf[off..off + 4].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn put_u32_be(buf: &mut [u8], off: usize, value: u32) -> Result<()> {
    if off + 4 > buf.len() {
        bail!("put_u32_be out of range at 0x{off:X}");
    }
    buf[off..off + 4].copy_from_slice(&value.to_be_bytes());
    Ok(())
}

fn sha256_bytes(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn write_json_value_pretty(path: &Path, value: &Value) -> Result<Vec<u8>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(path, &bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(bytes)
}

fn load_json_value(path: &Path) -> Result<Value> {
    let data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&data).with_context(|| format!("parse JSON {}", path.display()))
}

fn load_meta(path: &Path) -> Result<ScenarioMeta> {
    let data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&data).with_context(|| format!("parse meta {}", path.display()))
}

fn strip_tbl_comment(line: &str) -> String {
    let stripped = line.trim_start();
    if stripped.is_empty()
        || stripped.starts_with('#')
        || stripped.starts_with("//")
        || stripped.starts_with(';')
    {
        return String::new();
    }
    let mut end = line.len();
    for marker in [" //", " #", " ;"] {
        if let Some(pos) = line.find(marker) {
            end = end.min(pos);
        }
    }
    line[..end].trim_end_matches(['\r', '\n']).to_string()
}

fn parse_hex_bytes(token: &str) -> Option<Vec<u8>> {
    let mut t = token.trim().to_string();
    if t.starts_with("0x") || t.starts_with("0X") {
        t = t[2..].to_string();
    }
    t = t.replace(' ', "").replace('_', "");
    if t.is_empty() || t.len() % 2 != 0 {
        return None;
    }
    hex::decode(t).ok()
}

#[derive(Debug, Clone)]
struct FontTable {
    byte_to_text: HashMap<Vec<u8>, String>,
    text_to_bytes: HashMap<String, Vec<u8>>,
    max_key_len: usize,
    text_tokens: Vec<String>,
    source: Option<PathBuf>,
    duplicate_text_count: usize,
}

impl FontTable {
    fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("read font table {}", path.display()))?;
        let content = content.trim_start_matches('\u{feff}');
        let mut mapping: HashMap<Vec<u8>, String> = HashMap::new();

        for (idx, original_line) in content.lines().enumerate() {
            let lineno = idx + 1;
            let line = strip_tbl_comment(original_line);
            if line.trim().is_empty() {
                continue;
            }
            let parts: Option<(String, String)> = if let Some(pos) = line.find('=') {
                Some((line[..pos].to_string(), line[pos + 1..].to_string()))
            } else {
                let mut split = line.splitn(2, char::is_whitespace);
                let left = split.next().unwrap_or("");
                let right = split.next().unwrap_or("");
                if !left.is_empty() && !right.is_empty() {
                    Some((left.to_string(), right.to_string()))
                } else {
                    None
                }
            };
            let (left, right) = parts.with_context(|| {
                format!("{}:{lineno}: cannot parse mapping line: {original_line:?}", path.display())
            })?;
            let left_bytes = parse_hex_bytes(&left);
            let right_bytes = parse_hex_bytes(&right);
            let (raw, mut value) = match (left_bytes, right_bytes) {
                (Some(lb), None) => (lb, right),
                (None, Some(rb)) => (rb, left),
                (Some(lb), Some(_rb)) => (lb, right),
                (None, None) => bail!(
                    "{}:{lineno}: no hex byte sequence found: {original_line:?}",
                    path.display()
                ),
            };
            value = match value.as_str() {
                "<space>" => " ".to_string(),
                "<tab>" => "\t".to_string(),
                "<empty>" => String::new(),
                _ => value,
            };
            if value.is_empty() {
                bail!("{}:{lineno}: empty text token is not allowed", path.display());
            }
            if let Some(old) = mapping.get(&raw) {
                if old != &value {
                    bail!(
                        "{}:{lineno}: duplicate byte key {} maps to both {:?} and {:?}",
                        path.display(),
                        hex::encode_upper(&raw),
                        old,
                        value
                    );
                }
            }
            mapping.insert(raw, value);
        }

        if mapping.is_empty() {
            bail!("font table contains no mappings");
        }

        let mut text_to_bytes: HashMap<String, Vec<u8>> = HashMap::new();
        let mut duplicate_text_count = 0usize;
        for (raw, text) in &mapping {
            if let Some(old) = text_to_bytes.get(text) {
                if old != raw {
                    duplicate_text_count += 1;
                    continue;
                }
            }
            text_to_bytes.insert(text.clone(), raw.clone());
        }
        let max_key_len = mapping.keys().map(|k| k.len()).max().unwrap_or(1);
        let mut text_tokens: Vec<String> = text_to_bytes.keys().cloned().collect();
        text_tokens.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));

        Ok(Self {
            byte_to_text: mapping,
            text_to_bytes,
            max_key_len,
            text_tokens,
            source: Some(path.to_path_buf()),
            duplicate_text_count,
        })
    }

    fn decode_bytes(&self, data: &[u8], encoding: TextEncoding) -> String {
        let mut out = String::new();
        let mut i = 0usize;
        while i < data.len() {
            let mut matched = false;
            let limit = self.max_key_len.min(data.len() - i);
            for size in (1..=limit).rev() {
                let chunk = &data[i..i + size];
                if let Some(value) = self.byte_to_text.get(chunk) {
                    out.push_str(value);
                    i += size;
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }
            for size in [1usize, 2usize] {
                if i + size <= data.len() {
                    if let Ok(s) = decode_strict(&data[i..i + size], encoding) {
                        out.push_str(&s);
                        i += size;
                        matched = true;
                        break;
                    }
                }
            }
            if !matched {
                out.push('\u{fffd}');
                i += 1;
            }
        }
        out
    }

    fn encode_text(&self, text: &str, encoding: TextEncoding) -> Vec<u8> {
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < text.len() {
            let mut matched = false;
            for token in &self.text_tokens {
                if !token.is_empty() && text[i..].starts_with(token) {
                    if let Some(raw) = self.text_to_bytes.get(token) {
                        out.extend_from_slice(raw);
                        i += token.len();
                        matched = true;
                        break;
                    }
                }
            }
            if matched {
                continue;
            }
            let ch = text[i..].chars().next().unwrap();
            encode_char_ignore(ch, encoding, &mut out);
            i += ch.len_utf8();
        }
        out
    }

    fn meta(&self) -> FontTableMeta {
        FontTableMeta {
            source: self.source.as_ref().map(|p| p.to_string_lossy().to_string()),
            mapping_count: self.byte_to_text.len(),
            duplicate_text_count: self.duplicate_text_count,
            max_byte_key_len: self.max_key_len,
        }
    }
}

fn load_font_table_arg(path_arg: Option<&Path>) -> Result<Option<FontTable>> {
    match path_arg {
        Some(path) => {
            if !path.exists() {
                bail!("font.tbl not found: {}", path.display());
            }
            Ok(Some(FontTable::load(path)?))
        }
        None => Ok(None),
    }
}

fn decode_strict(data: &[u8], _encoding: TextEncoding) -> Result<String> {
    decode_cp932_strict(data)
}

fn decode_cp932_strict(data: &[u8]) -> Result<String> {
    // The game engine uses Windows CP932 / Windows-31J semantics, not strict
    // JIS Shift-JIS.  Keep ASCII 0x5C as backslash for engine controls, keep
    // JIS X 0201 half-width kana, accept CP932 vendor/EUDC double-byte ranges
    // F0..FC, and preserve CP932 single-byte PUA slots 80/A0/FD/FE/FF.
    let mut out = String::new();
    let mut i = 0usize;
    while i < data.len() {
        let b = data[i];

        if b <= 0x7F {
            out.push(char::from(b));
            i += 1;
            continue;
        }

        if let Some(ch) = cp932_single_byte_char(b) {
            out.push(ch);
            i += 1;
            continue;
        }

        if is_cp932_lead_byte(b) {
            if i + 1 >= data.len() {
                bail!("incomplete cp932 lead byte 0x{b:02X} at byte {i}");
            }
            let trail = data[i + 1];
            if !is_cp932_trail_byte(trail) {
                bail!("invalid cp932 trail byte 0x{trail:02X} after lead 0x{b:02X} at byte {i}");
            }
            let pair = [b, trail];
            let (cow, had_errors) = SHIFT_JIS.decode_without_bom_handling(&pair);
            if had_errors {
                bail!("invalid cp932 byte pair {:02X}{:02X} at byte {i}", b, trail);
            }
            out.push_str(&cow);
            i += 2;
            continue;
        }

        bail!("invalid cp932 byte 0x{b:02X} at byte {i}");
    }
    Ok(out)
}

fn cp932_single_byte_char(b: u8) -> Option<char> {
    match b {
        0x80 => Some('\u{0080}'),
        0xA0 => Some('\u{F8F0}'),
        0xA1..=0xDF => char::from_u32(0xFF61 + u32::from(b - 0xA1)),
        0xFD => Some('\u{F8F1}'),
        0xFE => Some('\u{F8F2}'),
        0xFF => Some('\u{F8F3}'),
        _ => None,
    }
}

fn cp932_byte_from_single_char(ch: char) -> Option<u8> {
    match ch {
        '\u{0080}' => Some(0x80),
        '\u{F8F0}' => Some(0xA0),
        '\u{F8F1}' => Some(0xFD),
        '\u{F8F2}' => Some(0xFE),
        '\u{F8F3}' => Some(0xFF),
        _ => {
            let cp = ch as u32;
            if (0xFF61..=0xFF9F).contains(&cp) {
                Some(0xA1 + (cp - 0xFF61) as u8)
            } else {
                None
            }
        }
    }
}

fn is_cp932_lead_byte(b: u8) -> bool {
    (0x81..=0x9F).contains(&b) || (0xE0..=0xFC).contains(&b)
}

fn is_cp932_trail_byte(b: u8) -> bool {
    (0x40..=0x7E).contains(&b) || (0x80..=0xFC).contains(&b)
}

fn decode_text_bytes(
    raw: &[u8],
    encoding: TextEncoding,
    font_table: Option<&FontTable>,
) -> Result<String> {
    if let Some(tbl) = font_table {
        Ok(tbl.decode_bytes(raw, encoding))
    } else {
        decode_strict(raw, encoding)
    }
}

fn cp932_encode_map() -> &'static HashMap<char, Vec<u8>> {
    static MAP: OnceLock<HashMap<char, Vec<u8>>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = HashMap::new();

        for b in 0u16..=0xFF {
            let raw = [b as u8];
            if let Ok(s) = decode_cp932_strict(&raw) {
                if let Some(ch) = single_decoded_char(&s) {
                    map.entry(ch).or_insert_with(|| raw.to_vec());
                }
            }
        }

        for lead in (0x81u16..=0x9F).chain(0xE0u16..=0xFC) {
            for trail in 0x40u16..=0xFC {
                if trail == 0x7F {
                    continue;
                }
                let raw = [lead as u8, trail as u8];
                if let Ok(s) = decode_cp932_strict(&raw) {
                    if let Some(ch) = single_decoded_char(&s) {
                        map.entry(ch).or_insert_with(|| raw.to_vec());
                    }
                }
            }
        }

        // Python cp932 accepts these historical Unicode aliases on encode even
        // when cp932 decode returns the Windows/compatibility form.
        map.insert('\u{301C}', vec![0x81, 0x60]); // WAVE DASH alias
        map.insert('\u{2016}', vec![0x81, 0x61]); // DOUBLE VERTICAL LINE alias
        map.insert('\u{2212}', vec![0x81, 0x7C]); // MINUS SIGN alias
        map.insert('\u{00A2}', vec![0x81, 0x91]); // CENT SIGN alias
        map.insert('\u{00A3}', vec![0x81, 0x92]); // POUND SIGN alias
        map.insert('\u{00AC}', vec![0x81, 0xCA]); // NOT SIGN alias

        map
    })
}

fn single_decoded_char(s: &str) -> Option<char> {
    let mut chars = s.chars();
    let ch = chars.next()?;
    if chars.next().is_none() { Some(ch) } else { None }
}

fn encode_char_ignore(ch: char, _encoding: TextEncoding, out: &mut Vec<u8>) {
    let cp = ch as u32;

    // Keep engine-visible ASCII byte-for-byte. This is critical for literal
    // control syntax such as \\k, \\n and \\p####.
    if cp <= 0x7F {
        out.push(cp as u8);
        return;
    }

    // CP932-specific single-byte slots, including the PUA bytes the engine can
    // use directly: A0/FD/FE/FF.  This fixes the previous skip on U+F8F1 etc.
    if let Some(b) = cp932_byte_from_single_char(ch) {
        out.push(b);
        return;
    }

    if let Some(raw) = cp932_encode_map().get(&ch) {
        out.extend_from_slice(raw);
    }
}

fn encode_lossy_ignore(text: &str, encoding: TextEncoding) -> Vec<u8> {
    let mut out = Vec::new();
    for ch in text.chars() {
        encode_char_ignore(ch, encoding, &mut out);
    }
    out
}

fn is_hex_digit_byte(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

fn split_preserving_engine_escapes(text: &str) -> Vec<(bool, String)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'\\' {
            let j = bytes[i..]
                .iter()
                .position(|&b| b == b'\\')
                .map(|p| i + p)
                .unwrap_or(bytes.len());
            out.push((false, text[i..j].to_string()));
            i = j;
            continue;
        }

        if i + 6 <= bytes.len()
            && bytes[i + 1] == b'p'
            && bytes[i + 2..i + 6].iter().all(|&b| is_hex_digit_byte(b))
        {
            out.push((true, text[i..i + 6].to_string()));
            i += 6;
            continue;
        }

        if i + 1 < bytes.len() {
            let next_start = i + 1;
            let next_ch = text[next_start..].chars().next().unwrap();
            let end = next_start + next_ch.len_utf8();
            out.push((true, text[i..end].to_string()));
            i = end;
        } else {
            out.push((true, "\\".to_string()));
            i += 1;
        }
    }
    out
}

fn encode_text_bytes(
    text: &str,
    encoding: TextEncoding,
    font_table: Option<&FontTable>,
) -> Vec<u8> {
    if let Some(tbl) = font_table {
        let mut out = Vec::new();
        for (is_escape, chunk) in split_preserving_engine_escapes(text) {
            if chunk.is_empty() {
                continue;
            }
            if is_escape {
                out.extend_from_slice(chunk.as_bytes());
            } else {
                out.extend_from_slice(&tbl.encode_text(&chunk, encoding));
            }
        }
        out
    } else {
        encode_lossy_ignore(text, encoding)
    }
}

fn encode_control_ascii(control: &str) -> Vec<u8> {
    control.as_bytes().to_vec()
}

fn read_elf_program_headers(eboot_data: &[u8]) -> Result<Vec<ProgramHeader>> {
    if eboot_data.len() < 0x40 {
        bail!("EBOOT is too small to contain an ELF64 header");
    }
    if &eboot_data[0..4] != b"\x7FELF" {
        bail!("Input is not an ELF file");
    }
    if eboot_data[4] != 2 {
        bail!("Expected ELF64");
    }
    if eboot_data[5] != 2 {
        bail!("Expected big-endian ELF");
    }
    let e_phoff = read_u64_be(eboot_data, 0x20)? as usize;
    let e_phentsize = u16::from_be_bytes([eboot_data[0x36], eboot_data[0x37]]) as usize;
    let e_phnum = u16::from_be_bytes([eboot_data[0x38], eboot_data[0x39]]) as usize;
    let mut headers = Vec::new();
    for i in 0..e_phnum {
        let ph_start = e_phoff + i * e_phentsize;
        if ph_start + e_phentsize > eboot_data.len() {
            break;
        }
        let p_type = read_u32_be(eboot_data, ph_start)?;
        if p_type == 1 {
            let p_offset = read_u64_be(eboot_data, ph_start + 8)?;
            let p_vaddr = read_u64_be(eboot_data, ph_start + 16)?;
            let p_memsz = read_u64_be(eboot_data, ph_start + 40)?;
            headers.push(ProgramHeader {
                vaddr: p_vaddr,
                memsz: p_memsz,
                offset: p_offset,
            });
        }
    }
    if headers.is_empty() {
        bail!("No PT_LOAD program headers found");
    }
    Ok(headers)
}

fn vaddr_to_fileoff(vaddr: u32, ph_headers: &[ProgramHeader]) -> usize {
    let v = vaddr as u64;
    for ph in ph_headers {
        if ph.vaddr <= v && v < ph.vaddr + ph.memsz {
            return (ph.offset + (v - ph.vaddr)) as usize;
        }
    }
    vaddr as usize
}

fn lzma_decompress_chunk(
    eboot_data: &[u8],
    file_off: usize,
    expected_uncomp_size: Option<usize>,
) -> Result<(Vec<u8>, usize, usize)> {
    if file_off + 4 > eboot_data.len() {
        bail!("Invalid LZMA chunk offset 0x{file_off:X}");
    }
    let comp_size = read_u32_be(eboot_data, file_off)? as usize;
    let start = file_off + 4;
    let end = start + comp_size;
    if end > eboot_data.len() {
        bail!("LZMA chunk 0x{file_off:X} size 0x{comp_size:X} exceeds EBOOT size");
    }
    let lzma_alone = &eboot_data[start..end];
    if lzma_alone.len() < 13 {
        bail!("LZMA chunk 0x{file_off:X} is too small");
    }
    if lzma_alone[0] != 0x5D {
        bail!(
            "LZMA chunk 0x{file_off:X} has unexpected properties byte 0x{:02X}",
            lzma_alone[0]
        );
    }
    let mut props = [0u8; 5];
    props.copy_from_slice(&lzma_alone[0..5]);
    let unpack_size_raw = read_u64_le(lzma_alone, 5)?;
    let unpack_size = if unpack_size_raw == u64::MAX {
        expected_uncomp_size.with_context(|| {
            format!("LZMA chunk 0x{file_off:X} has unknown unpack size and no table fallback")
        })?
    } else {
        usize::try_from(unpack_size_raw).context("LZMA unpack size does not fit usize")?
    };
    let payload = &lzma_alone[13..];
    let data = lzma_sdk_rs::decode_raw(payload, &props, unpack_size);
    Ok((data, comp_size, 4 + comp_size))
}

fn compress_lzma_alone(data: &[u8]) -> Vec<u8> {
    let mut props = lzma_sdk_rs::LzmaProps::for_level(5, 1 << 13);
    props.dict_size = 1 << 13;
    props.lc = 3;
    props.lp = 0;
    props.pb = 2;
    props.fb = 273;
    let raw = lzma_sdk_rs::encode(data, &props);
    let prop_byte = ((props.pb * 5 + props.lp) * 9 + props.lc) as u8;
    let mut out = Vec::with_capacity(13 + raw.len());
    out.push(prop_byte);
    out.extend_from_slice(&props.dict_size.to_le_bytes());
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&raw);
    out
}

fn read_text_records_structured(
    data: &[u8],
    encoding: TextEncoding,
    font_table: Option<&FontTable>,
) -> Vec<TextRecordInfo> {
    let mut out = Vec::new();
    let mut i = 16usize;
    while i + 6 < data.len() {
        if data[i..].starts_with(&TEXT_OPCODE) {
            let Ok(length) = read_u16_le(data, i + 4) else {
                i += 1;
                continue;
            };
            let length = length as usize;
            let start = i + 6;
            let end = start + length;
            if end <= data.len() {
                let raw = &data[start..end];
                match decode_text_bytes(raw, encoding, font_table) {
                    Ok(text) => {
                        out.push(TextRecordInfo {
                            offset_int: i,
                            offset: format!("0x{i:X}"),
                            length,
                            text,
                            total_size: 6 + length,
                            end_int: end,
                        });
                        i = end;
                        continue;
                    }
                    Err(_) => {
                        i += 1;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    out
}

fn is_p_record(text: &str) -> bool {
    let b = text.as_bytes();
    b.len() == 6 && b[0] == b'\\' && b[1] == b'p' && b[2..6].iter().all(|&c| c.is_ascii_hexdigit())
}

fn p_suffix_start(text: &str) -> Option<usize> {
    let b = text.as_bytes();
    if b.len() >= 6 {
        let s = b.len() - 6;
        if b[s] == b'\\' && b[s + 1] == b'p' && b[s + 2..s + 6].iter().all(|&c| c.is_ascii_hexdigit()) {
            return Some(s);
        }
    }
    None
}

fn split_text_control_suffix(text: &str) -> (String, String, String) {
    if let Some(v) = text.strip_suffix("\\k\\n") {
        return (v.to_string(), "\\k\\n".to_string(), "k_n_suffix".to_string());
    }
    if let Some(v) = text.strip_suffix("\\k") {
        return (v.to_string(), "\\k".to_string(), "k_suffix".to_string());
    }
    if let Some(v) = text.strip_suffix("\\n") {
        return (v.to_string(), "\\n".to_string(), "n_suffix".to_string());
    }
    if let Some(start) = p_suffix_start(text) {
        return (
            text[..start].to_string(),
            text[start..].to_string(),
            "p_suffix".to_string(),
        );
    }
    (text.to_string(), String::new(), "none".to_string())
}

fn classify_text_record(text: &str) -> (String, String, String, String) {
    if is_p_record(text) {
        return (
            "control_text".to_string(),
            String::new(),
            text.to_string(),
            "p_record".to_string(),
        );
    }
    let (visible, control, control_kind) = split_text_control_suffix(text);
    ("text".to_string(), visible, control, control_kind)
}

fn item_with_offset(mut item: Value, offset: &str, include_offsets: bool) -> Value {
    if include_offsets {
        item.as_object_mut()
            .unwrap()
            .insert("offset".to_string(), Value::String(offset.to_string()));
    }
    item
}

fn make_structured_units(
    data: &[u8],
    records: &[TextRecordInfo],
    include_offsets: bool,
) -> Vec<Value> {
    let mut units: Vec<Value> = Vec::new();
    let mut cur_items: Vec<Value> = Vec::new();
    let mut unit_index = 0usize;

    let flush = |cur_items: &mut Vec<Value>, end_offset: usize, reason: &str, units: &mut Vec<Value>, unit_index: &mut usize| {
        if cur_items.is_empty() {
            return;
        }
        let mut unit = Map::new();
        unit.insert("end_reason".to_string(), Value::String(reason.to_string()));
        unit.insert("items".to_string(), Value::Array(std::mem::take(cur_items)));
        if include_offsets {
            unit.insert("unit_index".to_string(), json!(*unit_index));
            let start = unit
                .get("items")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(|item| item.get("offset"))
                .and_then(Value::as_str)
                .unwrap_or("0x0")
                .to_string();
            unit.insert("start_offset".to_string(), Value::String(start));
            unit.insert("end_offset".to_string(), Value::String(format!("0x{end_offset:X}")));
        }
        units.push(Value::Object(unit));
        *unit_index += 1;
    };

    for (idx, rec) in records.iter().enumerate() {
        let (kind, visible, control, control_kind) = classify_text_record(&rec.text);
        let item = if kind == "control_text" {
            let mut obj = Map::new();
            obj.insert("kind".to_string(), Value::String("control_text".to_string()));
            obj.insert("length".to_string(), json!(rec.length));
            obj.insert("control".to_string(), Value::String(control.clone()));
            obj.insert("editable".to_string(), Value::Bool(false));
            if include_offsets {
                obj.insert("control_kind".to_string(), Value::String(control_kind.clone()));
            }
            item_with_offset(Value::Object(obj), &rec.offset, include_offsets)
        } else {
            let mut obj = Map::new();
            obj.insert("kind".to_string(), Value::String("text".to_string()));
            obj.insert("length".to_string(), json!(rec.length));
            obj.insert("original_text".to_string(), Value::String(visible.clone()));
            obj.insert("translated_text".to_string(), Value::String(visible));
            obj.insert("control_suffix".to_string(), Value::String(control.clone()));
            obj.insert("editable".to_string(), Value::Bool(true));
            if include_offsets {
                obj.insert("control_kind".to_string(), Value::String(control_kind.clone()));
            }
            item_with_offset(Value::Object(obj), &rec.offset, include_offsets)
        };
        cur_items.push(item);

        let next_off = records.get(idx + 1).map(|r| r.offset_int).unwrap_or(data.len());
        let gap = &data[rec.end_int..next_off];
        let has_gap = !gap.is_empty();
        if has_gap {
            let mut obj = Map::new();
            obj.insert("kind".to_string(), Value::String("opcode_gap".to_string()));
            obj.insert("length".to_string(), json!(gap.len()));
            obj.insert("bytes".to_string(), Value::String(hex::encode_upper(gap)));
            obj.insert("editable".to_string(), Value::Bool(false));
            cur_items.push(item_with_offset(Value::Object(obj), &format!("0x{:X}", rec.end_int), include_offsets));
        }

        let terminal = kind == "control_text" || control_kind != "none";
        if terminal {
            let reason = if has_gap {
                "terminal_control_plus_gap"
            } else {
                "terminal_control"
            };
            flush(&mut cur_items, next_off, reason, &mut units, &mut unit_index);
        } else if has_gap {
            flush(&mut cur_items, next_off, "opcode_gap", &mut units, &mut unit_index);
        }
    }
    flush(&mut cur_items, data.len(), "eof", &mut units, &mut unit_index);
    units
}

fn json_str<'a>(v: &'a Value, key: &str, default: &'a str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or(default)
}

fn json_bool(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn json_usize(v: &Value, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .unwrap_or(default)
}

fn item_is_insert(item: &Value) -> bool {
    json_bool(item, "insert") || json_bool(item, "new") || json_bool(item, "_insert")
}

fn unit_is_insert(unit: &Value) -> bool {
    json_bool(unit, "insert") || json_bool(unit, "new") || json_bool(unit, "_insert")
}

fn get_item_offset(item: &Value) -> Result<Option<usize>> {
    let Some(value) = item.get("offset") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    if let Some(s) = value.as_str() {
        if s.is_empty() || s == "new" || s == "NEW" {
            return Ok(None);
        }
        return Ok(Some(parse_usize_value(s)?));
    }
    if let Some(n) = value.as_u64() {
        return Ok(Some(n as usize));
    }
    bail!("invalid offset value: {value}");
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ItemSignature(Vec<String>);

fn item_signature(item: &Value) -> ItemSignature {
    let kind = json_str(item, "kind", "").to_string();
    let len = item
        .get("length")
        .and_then(Value::as_i64)
        .unwrap_or(-1)
        .to_string();
    match kind.as_str() {
        "text" => ItemSignature(vec![
            kind,
            len,
            json_str(item, "original_text", "").to_string(),
            json_str(item, "control_suffix", "").to_string(),
        ]),
        "control_text" => ItemSignature(vec![
            kind,
            len,
            json_str(item, "control", "").to_string(),
        ]),
        "opcode_gap" => ItemSignature(vec![
            kind,
            len,
            json_str(item, "bytes", "").to_ascii_uppercase(),
        ]),
        _ => ItemSignature(vec![kind]),
    }
}

fn unit_signature(unit: &Value) -> Vec<ItemSignature> {
    unit.get("items")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|item| !item_is_insert(item))
                .map(item_signature)
                .collect()
        })
        .unwrap_or_default()
}

fn unit_matches_template(unit: &Value, template_unit: &Value) -> bool {
    unit_signature(unit) == unit_signature(template_unit)
}

fn set_field(value: &mut Value, key: &str, new_value: Value) -> Result<()> {
    let obj = value
        .as_object_mut()
        .with_context(|| format!("expected object when setting {key}"))?;
    obj.insert(key.to_string(), new_value);
    Ok(())
}

fn remove_field(value: &mut Value, key: &str) {
    if let Some(obj) = value.as_object_mut() {
        obj.remove(key);
    }
}

fn as_inserted_unit(unit: &Value) -> Result<Value> {
    let mut out_unit = unit.clone();
    set_field(&mut out_unit, "insert", Value::Bool(true))?;
    remove_field(&mut out_unit, "unit_index");
    remove_field(&mut out_unit, "start_offset");
    remove_field(&mut out_unit, "end_offset");
    let items = unit
        .get("items")
        .and_then(Value::as_array)
        .context("inserted unit has no item list")?;
    let mut out_items = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        if !item.is_object() {
            bail!("inserted unit item {idx} is not an object");
        }
        let mut cloned = item.clone();
        set_field(&mut cloned, "insert", Value::Bool(true))?;
        remove_field(&mut cloned, "offset");
        out_items.push(cloned);
    }
    set_field(&mut out_unit, "items", Value::Array(out_items))?;
    Ok(out_unit)
}

fn hydrate_offsets_from_template(
    orig_data: &[u8],
    units: &[Value],
    encoding: TextEncoding,
    font_table: Option<&FontTable>,
) -> Result<Vec<Value>> {
    let records = read_text_records_structured(orig_data, encoding, font_table);
    let template_units = make_structured_units(orig_data, &records, true);
    let mut hydrated = Vec::with_capacity(units.len());
    let mut tmpl_unit_pos = 0usize;
    let mut seen_original_unit = false;

    for (json_unit_pos, unit) in units.iter().enumerate() {
        if !unit.is_object() {
            bail!("structured JSON must be a list of unit objects");
        }
        let src_items = unit
            .get("items")
            .and_then(Value::as_array)
            .with_context(|| format!("unit #{json_unit_pos} has no item list"))?;

        let explicit_insert = unit_is_insert(unit);
        let auto_insert = if explicit_insert {
            false
        } else if tmpl_unit_pos >= template_units.len() {
            true
        } else {
            !unit_matches_template(unit, &template_units[tmpl_unit_pos])
        };

        if explicit_insert || auto_insert {
            if !seen_original_unit {
                bail!(
                    "Inserted unit before the first original unit is not supported yet; insert after an existing unit"
                );
            }
            hydrated.push(as_inserted_unit(unit)?);
            continue;
        }

        let tmpl_items = template_units[tmpl_unit_pos]
            .get("items")
            .and_then(Value::as_array)
            .context("internal error: template unit has no items")?;
        tmpl_unit_pos += 1;
        seen_original_unit = true;

        let mut out_unit = unit.clone();
        let mut out_items = Vec::with_capacity(src_items.len());
        let mut tmpl_pos = 0usize;

        for (item_index, item) in src_items.iter().enumerate() {
            if !item.is_object() {
                bail!("unit #{json_unit_pos} item {item_index} is not an object");
            }
            let mut cloned = item.clone();
            if item_is_insert(&cloned) {
                remove_field(&mut cloned, "offset");
                out_items.push(cloned);
                continue;
            }
            if get_item_offset(&cloned)?.is_some() {
                if tmpl_pos < tmpl_items.len() {
                    tmpl_pos += 1;
                }
                out_items.push(cloned);
                continue;
            }
            if tmpl_pos >= tmpl_items.len() {
                bail!(
                    "unit #{json_unit_pos} has extra non-insert item #{item_index}; mark newly added items with insert=true"
                );
            }
            let tmpl = &tmpl_items[tmpl_pos];
            tmpl_pos += 1;
            if cloned.get("kind") != tmpl.get("kind") {
                bail!(
                    "unit #{json_unit_pos} item #{item_index} kind mismatch: JSON={:?}, original={:?}. Do not reorder existing items across boundaries.",
                    cloned.get("kind"),
                    tmpl.get("kind")
                );
            }
            let offset = tmpl
                .get("offset")
                .cloned()
                .context("internal error: template item has no offset")?;
            set_field(&mut cloned, "offset", offset)?;
            out_items.push(cloned);
        }
        set_field(&mut out_unit, "items", Value::Array(out_items))?;
        hydrated.push(out_unit);
    }
    Ok(hydrated)
}

fn sorted_json_items(units: &[Value]) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    for unit in units {
        if !unit.is_object() {
            bail!("structured JSON must be a list of unit objects");
        }
        let items = unit
            .get("items")
            .and_then(Value::as_array)
            .context("structured unit has no item list")?;
        for item in items {
            if !item.is_object() {
                bail!("structured unit contains a non-object item");
            }
            out.push(item.clone());
        }
    }
    Ok(out)
}

fn encode_structured_text_item(
    item: &Value,
    encoding: TextEncoding,
    font_table: Option<&FontTable>,
) -> Vec<u8> {
    let text = item
        .get("translated_text")
        .or_else(|| item.get("original_text"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let control_suffix = json_str(item, "control_suffix", "");
    let mut payload = encode_text_bytes(text, encoding, font_table);
    if !control_suffix.is_empty() {
        payload.extend_from_slice(&encode_control_ascii(control_suffix));
    }
    payload
}

fn encode_structured_control_item(item: &Value) -> Vec<u8> {
    encode_control_ascii(json_str(item, "control", ""))
}

fn make_text_record(payload: &[u8], where_label: &str) -> Result<Vec<u8>> {
    if payload.len() > 0xFFFF {
        bail!("TEXT payload at {where_label} exceeds 65535 bytes");
    }
    let mut out = Vec::with_capacity(6 + payload.len());
    out.extend_from_slice(&TEXT_OPCODE);
    write_u16_le(&mut out, payload.len() as u16);
    out.extend_from_slice(payload);
    Ok(out)
}

#[derive(Debug, Clone)]
struct RelocEvent {
    old_start: usize,
    old_end: usize,
    delta: isize,
}

fn rebuild_scenario_from_structured_json(
    orig_data: &[u8],
    units: &[Value],
    encoding: TextEncoding,
    font_table: Option<&FontTable>,
    allow_growth: bool,
    verify_relocations: bool,
) -> Result<Vec<u8>> {
    if orig_data.len() < 0x10 {
        bail!("Invalid scenario: smaller than header");
    }
    let label_count = read_u32_le(orig_data, 8)? as usize;
    let code_size = read_u32_le(orig_data, 4)? as usize;
    let code_start = 0x10 + label_count * 0x24;
    if orig_data.len() != code_start + code_size {
        bail!("Invalid scenario size formula");
    }

    let hydrated = hydrate_offsets_from_template(orig_data, units, encoding, font_table)?;
    let text_records = read_text_records_structured(orig_data, encoding, font_table);
    let text_by_off: HashMap<usize, TextRecordInfo> = text_records
        .into_iter()
        .map(|r| (r.offset_int, r))
        .collect();
    let items = sorted_json_items(&hydrated)?;
    let mut consumed_text_offsets: HashSet<usize> = HashSet::new();
    let mut new_tail = Vec::new();
    let mut last = 16usize;
    let mut reloc_events: Vec<RelocEvent> = Vec::new();

    fn copy_until(new_tail: &mut Vec<u8>, orig_data: &[u8], last: &mut usize, old_abs: usize) -> Result<()> {
        if old_abs < *last {
            bail!("structured JSON item order overlaps original bytecode: 0x{old_abs:X} < 0x{:X}", *last);
        }
        new_tail.extend_from_slice(&orig_data[*last..old_abs]);
        *last = old_abs;
        Ok(())
    }

    for item in items {
        let kind = json_str(&item, "kind", "text").to_string();
        let off = get_item_offset(&item)?;
        match kind.as_str() {
            "text" | "control_text" => {
                let payload = if kind == "text" {
                    encode_structured_text_item(&item, encoding, font_table)
                } else {
                    encode_structured_control_item(&item)
                };
                let where_label = item
                    .get("offset")
                    .and_then(Value::as_str)
                    .unwrap_or("<new>")
                    .to_string();
                let new_record = make_text_record(&payload, &where_label)?;
                if off.is_none() {
                    if !allow_growth {
                        bail!("Inserted text/control_text item requires --allow-growth");
                    }
                    let insert_at = last;
                    new_tail.extend_from_slice(&new_record);
                    reloc_events.push(RelocEvent {
                        old_start: insert_at,
                        old_end: insert_at,
                        delta: new_record.len() as isize,
                    });
                    continue;
                }
                let off = off.unwrap();
                let rec = text_by_off
                    .get(&off)
                    .with_context(|| format!("JSON references non-text offset 0x{off:X}"))?;
                if !consumed_text_offsets.insert(off) {
                    bail!("Duplicate text/control_text item for offset 0x{off:X}");
                }
                copy_until(&mut new_tail, orig_data, &mut last, off)?;
                let old_total = rec.total_size;
                let new_total = new_record.len();
                if !allow_growth && new_total > old_total {
                    bail!(
                        "Text growth is disabled at 0x{off:X}: old_total={old_total}, new_total={new_total}. Pass --allow-growth after verification."
                    );
                }
                new_tail.extend_from_slice(&new_record);
                last = rec.end_int;
                let delta = new_total as isize - old_total as isize;
                if delta != 0 {
                    reloc_events.push(RelocEvent {
                        old_start: off,
                        old_end: rec.end_int,
                        delta,
                    });
                }
            }
            "opcode_gap" => {
                let raw_hex = json_str(&item, "bytes", "");
                let raw = hex::decode(raw_hex).with_context(|| format!("invalid opcode_gap hex: {raw_hex}"))?;
                if off.is_none() {
                    if !allow_growth {
                        bail!("Inserted opcode_gap item requires --allow-growth");
                    }
                    let insert_at = last;
                    new_tail.extend_from_slice(&raw);
                    if !raw.is_empty() {
                        reloc_events.push(RelocEvent {
                            old_start: insert_at,
                            old_end: insert_at,
                            delta: raw.len() as isize,
                        });
                    }
                    continue;
                }
                let off = off.unwrap();
                let length = json_usize(&item, "length", raw.len());
                if raw.len() != length {
                    bail!("opcode_gap at 0x{off:X}: length field does not match bytes");
                }
                copy_until(&mut new_tail, orig_data, &mut last, off)?;
                if off + length > orig_data.len() {
                    bail!("opcode_gap at 0x{off:X} exceeds original data");
                }
                let old_gap = &orig_data[off..off + length];
                if old_gap != raw.as_slice() {
                    bail!(
                        "opcode_gap mismatch at 0x{off:X}: JSON={} original={}",
                        hex::encode_upper(&raw),
                        hex::encode_upper(old_gap)
                    );
                }
                new_tail.extend_from_slice(old_gap);
                last = off + length;
            }
            _ => bail!("Unsupported structured item kind: {kind:?}"),
        }
    }
    new_tail.extend_from_slice(&orig_data[last..]);

    let total_delta: isize = reloc_events.iter().map(|e| e.delta).sum();
    let new_code_size_i = code_size as isize + total_delta;
    if new_code_size_i < 0 {
        bail!("negative rebuilt code size");
    }
    let new_code_size = new_code_size_i as usize;
    if new_code_size > u32::MAX as usize {
        bail!("rebuilt code size exceeds u32");
    }

    let mut new_data = Vec::with_capacity(16 + new_tail.len());
    new_data.extend_from_slice(&orig_data[..16]);
    new_data.extend_from_slice(&new_tail);
    put_u32_le(&mut new_data, 4, new_code_size as u32)?;

    for i in 0..label_count {
        let ptr_abs = 0x10 + i * 0x24 + 0x20;
        let p = read_u32_le(&new_data, ptr_abs)? as usize;
        let old_file = code_start + p;
        let mut shift: isize = 0;
        for ev in &reloc_events {
            if ev.old_start == ev.old_end {
                if old_file >= ev.old_start {
                    shift += ev.delta;
                }
                continue;
            }
            if old_file >= ev.old_end {
                shift += ev.delta;
            } else if ev.old_start <= old_file && old_file < ev.old_end {
                if old_file != ev.old_start + 1 {
                    bail!(
                        "Label {i} points inside edited text record at 0x{old_file:X}; only record+1 anchor is currently supported"
                    );
                }
            }
        }
        if shift != 0 {
            let new_file_i = old_file as isize + shift;
            if new_file_i < code_start as isize {
                bail!("Label {i} relocated before code_start");
            }
            let new_ptr = (new_file_i as usize) - code_start;
            if new_ptr > u32::MAX as usize {
                bail!("Label {i} relocated beyond u32 range");
            }
            put_u32_le(&mut new_data, ptr_abs, new_ptr as u32)?;
        }
    }

    if verify_relocations {
        let new_code_size = read_u32_le(&new_data, 4)? as usize;
        if new_data.len() != code_start + new_code_size {
            bail!("Rebuilt scenario size formula mismatch");
        }
    }

    Ok(new_data)
}

fn scan_candidates(eboot_data: &[u8], table_start: usize, max_entries: usize) -> Result<Vec<Candidate>> {
    let ph_headers = read_elf_program_headers(eboot_data)?;
    let mut candidates = Vec::new();
    for count in 0..max_entries {
        let entry_off = table_start + count * 16;
        if entry_off + 16 > eboot_data.len() {
            break;
        }
        let name_ptr = read_u32_be(eboot_data, entry_off)?;
        let data_ptr = read_u32_be(eboot_data, entry_off + 4)?;
        let uncomp_size = read_u32_be(eboot_data, entry_off + 8)? as usize;
        let comp_size_table = read_u32_be(eboot_data, entry_off + 12)? as usize;
        if data_ptr == 0 || name_ptr == 0 {
            break;
        }
        let file_offset = vaddr_to_fileoff(data_ptr, &ph_headers);
        candidates.push(Candidate {
            entry_index: count,
            name_ptr,
            data_ptr,
            uncomp_size,
            comp_size_table,
            file_offset,
        });
    }
    Ok(candidates)
}

fn decompress_candidate(eboot_data: &[u8], cand: &Candidate) -> Option<(ScenarioEntry, Vec<u8>)> {
    let Ok((dec, comp_size_actual, raw_chunk_size)) =
        lzma_decompress_chunk(eboot_data, cand.file_offset, Some(cand.uncomp_size))
    else {
        return None;
    };
    let entry = ScenarioEntry {
        entry_index: cand.entry_index,
        filename: format!("script_{:04}.bin", cand.entry_index),
        json_filename: format!("script_{:04}.json", cand.entry_index),
        name_ptr: cand.name_ptr,
        data_ptr: cand.data_ptr,
        file_offset: cand.file_offset,
        uncomp_size: cand.uncomp_size,
        comp_size_table: cand.comp_size_table,
        comp_size_actual,
        raw_chunk_size,
        sha256_bin: sha256_bytes(&dec),
        text_count: 0,
        json_sha256: None,
    };
    Some((entry, dec))
}

fn scan_entries(
    eboot_data: &[u8],
    table_start: usize,
    max_entries: usize,
    thread: Option<usize>,
) -> Result<Vec<(ScenarioEntry, Vec<u8>)>> {
    let candidates = scan_candidates(eboot_data, table_start, max_entries)?;
    let mut entries = run_with_optional_pool(thread, || {
        let out: Vec<(ScenarioEntry, Vec<u8>)> = if thread.is_some() {
            candidates
                .par_iter()
                .filter_map(|cand| decompress_candidate(eboot_data, cand))
                .collect()
        } else {
            candidates
                .iter()
                .filter_map(|cand| decompress_candidate(eboot_data, cand))
                .collect()
        };
        Ok(out)
    })?;
    entries.sort_by_key(|(entry, _)| entry.entry_index);
    Ok(entries)
}

fn process_extracted_scripts(
    scanned: &[(ScenarioEntry, Vec<u8>)],
    encoding: TextEncoding,
    font_table: Option<&FontTable>,
    include_offsets: bool,
    thread: Option<usize>,
) -> Result<Vec<ExtractedScript>> {
    let mut processed = run_with_optional_pool(thread, || {
        let out: Vec<ExtractedScript> = if thread.is_some() {
            scanned
                .par_iter()
                .map(|(entry, dec)| {
                    let records = read_text_records_structured(dec, encoding, font_table);
                    let units = make_structured_units(dec, &records, include_offsets);
                    let mut entry = entry.clone();
                    entry.text_count = units.len();
                    let items_count = units
                        .iter()
                        .filter_map(|u| u.get("items").and_then(Value::as_array))
                        .map(|items| items.len())
                        .sum();
                    ExtractedScript {
                        entry,
                        units,
                        records_count: records.len(),
                        items_count,
                    }
                })
                .collect()
        } else {
            scanned
                .iter()
                .map(|(entry, dec)| {
                    let records = read_text_records_structured(dec, encoding, font_table);
                    let units = make_structured_units(dec, &records, include_offsets);
                    let mut entry = entry.clone();
                    entry.text_count = units.len();
                    let items_count = units
                        .iter()
                        .filter_map(|u| u.get("items").and_then(Value::as_array))
                        .map(|items| items.len())
                        .sum();
                    ExtractedScript {
                        entry,
                        units,
                        records_count: records.len(),
                        items_count,
                    }
                })
                .collect()
        };
        Ok(out)
    })?;
    processed.sort_by_key(|e| e.entry.entry_index);
    Ok(processed)
}

fn build_meta(
    eboot_path: &Path,
    eboot_data: &[u8],
    table_start: usize,
    max_entries: usize,
    entries: Vec<ScenarioEntry>,
    encoding_label: &str,
    font_table: Option<&FontTable>,
    use_hash: bool,
) -> ScenarioMeta {
    ScenarioMeta {
        tool: Some(TOOL_NAME.to_string()),
        version: Some(TOOL_VERSION),
        source_eboot: Some(eboot_path.to_string_lossy().to_string()),
        source_eboot_size: Some(eboot_data.len()),
        source_eboot_sha256: Some(sha256_bytes(eboot_data)),
        table_start: Some(table_start),
        max_entries: Some(max_entries),
        encoding: Some(encoding_label.to_string()),
        extraction_mode: Some("structured_json".to_string()),
        hash_algorithm: if use_hash { Some("sha256".to_string()) } else { None },
        font_tbl: font_table.map(FontTable::meta),
        notes: Some(vec![
            "Rust port of the structured extractor/rebuilder.".to_string(),
            "Edit only text.translated_text by default.".to_string(),
            "control_text and opcode_gap are preserved as explicit non-editable bytecode items.".to_string(),
            "Offsets are omitted by default; rebuild aligns existing items by unit/item order. Use extract --debug-offsets for diagnostics.".to_string(),
            "New inserted text/control_text/opcode_gap items must be marked insert=true and require --allow-growth.".to_string(),
        ]),
        entries,
    }
}

fn get_original_entry_data(eboot_data: &[u8], ph_headers: &[ProgramHeader], entry: &ScenarioEntry) -> Result<Vec<u8>> {
    let file_off = if entry.data_ptr != 0 {
        vaddr_to_fileoff(entry.data_ptr, ph_headers)
    } else {
        entry.file_offset
    };
    let (data, _, _) = lzma_decompress_chunk(eboot_data, file_off, Some(entry.uncomp_size))?;
    Ok(data)
}

fn prepare_rebuild_import(
    task: &RebuildTask,
    eboot_data: &[u8],
    ph_headers: &[ProgramHeader],
    encoding: TextEncoding,
    font_table: Option<&FontTable>,
    allow_growth: bool,
    verify_relocations: bool,
) -> Result<PreparedImport> {
    let units_value = load_json_value(&task.json_path)?;
    let units = units_value
        .as_array()
        .with_context(|| format!("{} must contain a JSON list", task.json_path.display()))?;
    let orig_data = get_original_entry_data(eboot_data, ph_headers, &task.entry)?;
    let rebuilt = rebuild_scenario_from_structured_json(
        &orig_data,
        units,
        encoding,
        font_table,
        allow_growth,
        verify_relocations,
    )?;
    let lzma_alone = compress_lzma_alone(&rebuilt);
    Ok(PreparedImport {
        entry: task.entry.clone(),
        uncomp_size: rebuilt.len(),
        lzma_alone,
    })
}

fn prepare_rebuild_imports(
    tasks: &[RebuildTask],
    eboot_data: &[u8],
    ph_headers: &[ProgramHeader],
    encoding: TextEncoding,
    font_table: Option<&FontTable>,
    allow_growth: bool,
    verify_relocations: bool,
    thread: Option<usize>,
) -> Result<Vec<PreparedImport>> {
    let mut imports = run_with_optional_pool(thread, || {
        let results: Result<Vec<PreparedImport>> = if thread.is_some() {
            tasks
                .par_iter()
                .map(|task| {
                    prepare_rebuild_import(
                        task,
                        eboot_data,
                        ph_headers,
                        encoding,
                        font_table,
                        allow_growth,
                        verify_relocations,
                    )
                })
                .collect()
        } else {
            tasks
                .iter()
                .map(|task| {
                    prepare_rebuild_import(
                        task,
                        eboot_data,
                        ph_headers,
                        encoding,
                        font_table,
                        allow_growth,
                        verify_relocations,
                    )
                })
                .collect()
        };
        results
    })?;
    imports.sort_by_key(|p| p.entry.entry_index);
    Ok(imports)
}

fn repack_prepared_entries_into_eboot(
    eboot_data: &mut [u8],
    table_start: usize,
    imports: &[PreparedImport],
    inplace_offset_base: usize,
    inplace_vaddr_base: usize,
    inplace_max_size: usize,
) -> Result<(usize, usize)> {
    let mut current_offset = inplace_offset_base;
    let mut success = 0usize;

    for import in imports {
        let new_comp_size = import.lzma_alone.len();
        let raw_chunk_size = 4 + new_comp_size;
        let padding = (4 - (raw_chunk_size % 4)) % 4;
        let total_chunk_size = raw_chunk_size + padding;
        let used_after = (current_offset - inplace_offset_base) + total_chunk_size;
        if inplace_max_size != 0 && used_after > inplace_max_size {
            bail!("Repack exceeds reserved area: need {used_after} bytes, limit is {inplace_max_size}");
        }
        if current_offset + total_chunk_size > eboot_data.len() {
            bail!(
                "Repack write exceeds EBOOT size at 0x{current_offset:X}; refusing to extend executable"
            );
        }
        let new_vaddr = inplace_vaddr_base + (current_offset - inplace_offset_base);
        if new_comp_size > u32::MAX as usize || new_vaddr > u32::MAX as usize || total_chunk_size > u32::MAX as usize || import.uncomp_size > u32::MAX as usize {
            bail!("repack value exceeds u32 range");
        }
        put_u32_be(eboot_data, current_offset, new_comp_size as u32)?;
        current_offset += 4;
        eboot_data[current_offset..current_offset + new_comp_size].copy_from_slice(&import.lzma_alone);
        current_offset += new_comp_size;
        if padding != 0 {
            eboot_data[current_offset..current_offset + padding].fill(0);
            current_offset += padding;
        }
        let entry_start = table_start + import.entry.entry_index * 16;
        if entry_start + 16 > eboot_data.len() {
            bail!("Index entry {} is outside EBOOT", import.entry.entry_index);
        }
        put_u32_be(eboot_data, entry_start + 4, new_vaddr as u32)?;
        put_u32_be(eboot_data, entry_start + 8, import.uncomp_size as u32)?;
        put_u32_be(eboot_data, entry_start + 12, total_chunk_size as u32)?;
        success += 1;
    }

    Ok((success, current_offset - inplace_offset_base))
}

fn command_extract(
    eboot: PathBuf,
    output_dir: PathBuf,
    use_hash: bool,
    font_tbl: Option<PathBuf>,
    table_start: usize,
    max_entries: usize,
    encoding_label: String,
    debug_offsets: bool,
    thread: Option<usize>,
) -> Result<()> {
    validate_threads(thread)?;
    let encoding = encoding_from_label(&encoding_label)?;
    let font_table = load_font_table_arg(font_tbl.as_deref())?;
    fs::create_dir_all(&output_dir)?;
    let eboot_data = fs::read(&eboot).with_context(|| format!("read {}", eboot.display()))?;
    let scanned = scan_entries(&eboot_data, table_start, max_entries, thread)?;
    let processed = process_extracted_scripts(
        &scanned,
        encoding,
        font_table.as_ref(),
        debug_offsets,
        thread,
    )?;

    let mut entries = Vec::with_capacity(processed.len());
    let mut written = 0usize;
    let mut total_units = 0usize;
    let mut total_items = 0usize;
    let mut total_records = 0usize;

    for mut script in processed {
        total_units += script.units.len();
        total_items += script.items_count;
        total_records += script.records_count;
        if !script.units.is_empty() {
            let path = output_dir.join(&script.entry.json_filename);
            let value = Value::Array(script.units);
            let bytes = write_json_value_pretty(&path, &value)?;
            written += 1;
            if use_hash {
                script.entry.json_sha256 = Some(sha256_bytes(&bytes));
            }
        }
        entries.push(script.entry);
    }

    let meta = build_meta(
        &eboot,
        &eboot_data,
        table_start,
        max_entries,
        entries,
        &encoding_label,
        font_table.as_ref(),
        use_hash,
    );
    write_json_pretty(&output_dir.join(META_FILENAME), &meta)?;
    println!(
        "Extracted {written} structured JSON files; units={total_units}, items={total_items}, underlying text_records={total_records}"
    );
    if let Some(n) = thread {
        println!("Rayon threads: {n}");
    }
    println!("Metadata saved to {}", output_dir.join(META_FILENAME).display());
    Ok(())
}

fn command_rebuild(
    eboot: PathBuf,
    input_dir: PathBuf,
    output_eboot: PathBuf,
    meta_arg: Option<PathBuf>,
    use_hash: bool,
    allow_growth: bool,
    no_verify_relocations: bool,
    font_tbl: Option<PathBuf>,
    table_start_arg: usize,
    inplace_offset_base: usize,
    inplace_vaddr_base: usize,
    inplace_max_size: usize,
    encoding_arg: String,
    thread: Option<usize>,
) -> Result<()> {
    validate_threads(thread)?;
    let meta_path = meta_arg.unwrap_or_else(|| input_dir.join(META_FILENAME));
    let meta = load_meta(&meta_path)?;
    if meta.entries.is_empty() {
        bail!("Metadata contains no entries");
    }
    let mode = meta.extraction_mode.as_deref().unwrap_or("");
    if mode != "structured_json" && mode != "clean_logical_json" {
        bail!("This Rust structured tool rebuilds only structured_json extractions");
    }
    if use_hash && meta.hash_algorithm.is_none() {
        bail!("Metadata has no JSON hashes. Re-run extract with --use-hash.");
    }

    let encoding_label = meta.encoding.clone().unwrap_or(encoding_arg);
    let encoding = encoding_from_label(&encoding_label)?;
    let font_tbl_path = font_tbl.or_else(|| {
        meta.font_tbl
            .as_ref()
            .and_then(|m| m.source.as_ref())
            .map(PathBuf::from)
    });
    let font_table = load_font_table_arg(font_tbl_path.as_deref())?;

    let mut skipped_unchanged = 0usize;
    let mut skipped_missing = 0usize;
    let mut tasks = Vec::new();
    for entry in &meta.entries {
        let path = input_dir.join(&entry.json_filename);
        if !path.exists() {
            skipped_missing += 1;
            continue;
        }
        if use_hash {
            if let Some(old_hash) = &entry.json_sha256 {
                let new_hash = sha256_file(&path)?;
                if &new_hash == old_hash {
                    skipped_unchanged += 1;
                    continue;
                }
            }
        }
        tasks.push(RebuildTask {
            entry: entry.clone(),
            json_path: path,
        });
    }

    let mut eboot_data = fs::read(&eboot).with_context(|| format!("read {}", eboot.display()))?;
    let ph_headers = read_elf_program_headers(&eboot_data)?;
    let table_start = meta.table_start.unwrap_or(table_start_arg);
    let imports = prepare_rebuild_imports(
        &tasks,
        &eboot_data,
        &ph_headers,
        encoding,
        font_table.as_ref(),
        allow_growth,
        !no_verify_relocations,
        thread,
    )?;

    if imports.is_empty() {
        println!("No changed entries selected for import; output EBOOT will be a copy.");
    }

    let (success, used) = repack_prepared_entries_into_eboot(
        &mut eboot_data,
        table_start,
        &imports,
        inplace_offset_base,
        inplace_vaddr_base,
        inplace_max_size,
    )?;

    if let Some(parent) = output_eboot.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_eboot, &eboot_data)
        .with_context(|| format!("write {}", output_eboot.display()))?;
    println!("Imported {success} structured scenario entries into {}", output_eboot.display());
    if use_hash {
        println!("Skipped unchanged JSON files: {skipped_unchanged}");
    }
    if skipped_missing != 0 {
        println!("Skipped missing input files: {skipped_missing}");
    }
    if let Some(n) = thread {
        println!("Rayon threads: {n}");
    }
    println!("Reserved area used: {used} bytes");
    Ok(())
}

fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Extract {
            eboot,
            output_dir,
            use_hash,
            font_tbl,
            table_start,
            max_entries,
            encoding,
            debug_offsets,
            thread,
        } => command_extract(
            eboot,
            output_dir,
            use_hash,
            font_tbl,
            table_start,
            max_entries,
            encoding,
            debug_offsets,
            thread,
        ),
        Commands::Rebuild {
            eboot,
            input_dir,
            output_eboot,
            meta,
            use_hash,
            allow_growth,
            no_verify_relocations,
            font_tbl,
            table_start,
            inplace_offset_base,
            inplace_vaddr_base,
            inplace_max_size,
            encoding,
            thread,
        } => command_rebuild(
            eboot,
            input_dir,
            output_eboot,
            meta,
            use_hash,
            allow_growth,
            no_verify_relocations,
            font_tbl,
            table_start,
            inplace_offset_base,
            inplace_vaddr_base,
            inplace_max_size,
            encoding,
            thread,
        ),
    }
}

fn main() -> Result<()> {
    let handle = std::thread::Builder::new()
        .name("scenario_eboot_cli".to_string())
        .stack_size(RUST_WORKER_STACK_SIZE)
        .spawn(run_cli)
        .context("failed to spawn stack-extended CLI thread")?;

    match handle.join() {
        Ok(result) => result,
        Err(payload) => {
            if let Some(message) = payload.downcast_ref::<&str>() {
                bail!("worker thread panicked: {message}");
            }
            if let Some(message) = payload.downcast_ref::<String>() {
                bail!("worker thread panicked: {message}");
            }
            bail!("worker thread panicked without a string payload");
        }
    }
}
