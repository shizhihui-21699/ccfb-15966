//! ELF loading, symbol table, and memory model for the interpreter.
//!
//! Everything the auditor knows about "what is at this address" comes from
//! here: section bytes (code + rodata, and initialised RAM), the symbol table
//! from the link (plus an optional merged `.map`/`nm` file), and the storage
//! class (code / rodata / data / bss / stack) used later to turn a permission
//! map into findings.

use anyhow::{bail, Context, Result};
use object::{Object, ObjectSection, ObjectSegment, ObjectSymbol, SectionIndex, SymbolKind};

/// Decode the ELF `sh_flags` word into our three booleans. `object` 0.40 wraps
/// the raw word in `object::elf::SectionFlags`, so unwrap it by bits.
fn elf_flags(f: object::SectionFlags) -> SectionFlags {
    match f {
        object::SectionFlags::Elf { sh_flags, .. } => {
            let raw: u64 = sh_flags.0;
            SectionFlags {
                alloc: raw & 0x2 != 0, // SHF_ALLOC
                exec: raw & 0x4 != 0,  // SHF_EXECINSTR
                write: raw & 0x1 != 0, // SHF_WRITE
            }
        }
        object::SectionFlags::None => SectionFlags::default(),
        _ => SectionFlags::default(),
    }
}
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Storage {
    Code,
    Rodata,
    Data,
    Bss,
    /// A region of RAM declared as a stack in the config and confirmed empty
    /// of other objects.
    Stack,
    /// Anything not attributable (e.g. an address between sections).
    Unknown,
}

impl Storage {
    pub fn is_writable_ram(self) -> bool {
        matches!(self, Storage::Data | Storage::Bss | Storage::Stack)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Sym {
    pub name: String,
    pub addr: u32,
    pub size: u32,
    pub storage: Storage,
    pub global: bool,
}

#[derive(Clone, Debug)]
pub struct Section {
    pub name: String,
    pub addr: u32,
    pub size: u32,
    pub flags: SectionFlags,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SectionFlags {
    pub alloc: bool,
    pub exec: bool,
    pub write: bool,
}

#[derive(Clone, Debug)]
pub struct FirmwareImage {
    pub path: String,
    pub entry: u32,
    pub sections: Vec<Section>,
    pub symbols: Vec<Sym>,
    /// Executable bytes per section (addr -> bytes)
    code: Vec<(u32, u32, Vec<u8>)>,
    /// Data bytes per section (PROGBITS only; NOBITS reads as zero). Covers
    /// rodata as well as RAM so literal-pool reads work.
    data: Vec<(u32, u32, Vec<u8>)>,
    /// The loader layout: which RAM addresses are backed at all (so the
    /// interpreter can distinguish "unknown memory" from "zero-initialised
    /// bss").
    ram_ranges: Vec<(u32, u32)>,
    /// Flattened list of section extents, used to clamp symbol-size inference.
    extents: Vec<(u32, u32)>,
}

fn section_storage(name: &str, flags: SectionFlags) -> Storage {
    if flags.exec {
        return Storage::Code;
    }
    let n = name.to_ascii_lowercase();
    if n.contains("bss") || n.contains("noinit") || n.contains("common") {
        return Storage::Bss;
    }
    if n.contains("data") || flags.write {
        return Storage::Data;
    }
    if flags.alloc {
        Storage::Rodata
    } else {
        Storage::Unknown
    }
}

impl FirmwareImage {
    pub fn load(path: &Path, extra_symbols: Option<&Path>) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let file = object::File::parse(&*bytes).with_context(|| format!("parsing {}", path.display()))?;
        if file.architecture() != object::Architecture::Arm {
            bail!("{} is not an ARM ELF (architecture {:?})", path.display(), file.architecture());
        }
        if !file.is_64() {
            // 32-bit ARM is what Cortex-M uses; 64-bit would be AArch64.
        }

        let mut sections = Vec::new();
        let mut code = Vec::new();
        let mut data = Vec::new();
        for sec in file.sections() {
            let f = sec.flags();
            let flags = elf_flags(f);
            let (alloc, exec, write) = (flags.alloc, flags.exec, flags.write);
            if !alloc || sec.address() == 0 || sec.size() == 0 {
                continue;
            }
            let name = sec.name().unwrap_or("?").to_string();
            let addr = sec.address() as u32;
            let size = sec.size() as u32;
            let is_nobits = sec.kind() == object::SectionKind::UninitializedData
                || sec.kind() == object::SectionKind::UninitializedTls;
            sections.push(Section { name: name.clone(), addr, size, flags });
            if is_nobits {
                // bss: backed, initial contents zero
                data.push((addr, size, Vec::new()));
                continue;
            }
            let content = sec.data().with_context(|| format!("reading section {name}"))?;
            let mut buf = content.to_vec();
            if buf.len() < size as usize {
                buf.resize(size as usize, 0);
            }
            if exec {
                code.push((addr, size, buf.clone()));
            }
            // Keep rodata/literal pools readable from any section (a literal
            // pool can even share the .text section), so every PROGBITS
            // section is part of the data model.
            data.push((addr, size, buf));
        }

        let mut ram_ranges: Vec<(u32, u32)> = Vec::new();
        for seg in file.segments() {
            let addr = seg.address() as u32;
            let size = seg.size() as u32;
            if size > 0 && addr >= 0x2000_0000 {
                ram_ranges.push((addr, size));
            }
        }
        // Also treat sections in the RAM range as backed.
        for s in &sections {
            if s.addr >= 0x2000_0000 {
                ram_ranges.push((s.addr, s.size));
            }
        }

        let mut symbols = Vec::new();
        for sym in file.symbols() {
            let name = match sym.name() {
                Ok(n) if !n.is_empty() => n.to_string(),
                _ => continue,
            };
            if matches!(sym.kind(), SymbolKind::Section | SymbolKind::File) {
                continue;
            }
            let addr = sym.address() as u32;
            let size = sym.size() as u32;
            let sec_idx = sym.section_index();
            let storage = match sec_idx {
                Some(idx) => {
                    let sec = file.section_by_index(idx).ok();
                    match sec {
                        Some(s) => {
                            let flags = elf_flags(s.flags());
                            section_storage(s.name().unwrap_or("?"), flags)
                        }
                        None => Storage::Unknown,
                    }
                }
                None => {
                    // ABS symbol: linker-provided (__ram_start etc.), storage
                    // depends on the range, resolved lazily by the caller.
                    Storage::Unknown
                }
            };
            symbols.push(Sym { name, addr, size, storage, global: sym.is_global() });
        }

        if let Some(extra) = extra_symbols {
            let text = std::fs::read_to_string(extra).with_context(|| format!("reading {}", extra.display()))?;
            merge_symbol_text(&text, &mut symbols, &sections);
        }


        let extents: Vec<(u32, u32)> = sections.iter().map(|s| (s.addr, s.size)).collect();
        fill_symbol_sizes(&mut symbols, &extents);

        Ok(FirmwareImage {
            path: path.display().to_string(),
            entry: file.entry() as u32,
            sections,
            symbols,
            code,
            data,
            ram_ranges,
            extents,
        })
    }

    pub fn exec_sections(&self) -> Vec<&Section> {
        self.sections.iter().filter(|s| s.flags.exec).collect()
    }

    /// True if the address is inside an executable section.
    pub fn is_code(&self, addr: u32) -> bool {
        self.code.iter().any(|(a, s, _)| addr >= *a && addr < a.wrapping_add(*s))
    }

    fn code_bytes(&self, addr: u32) -> Option<[u8; 4]> {
        for (a, s, b) in &self.code {
            let end = a.wrapping_add(*s);
            if addr >= *a && addr < end {
                let off = (addr - *a) as usize;
                let mut w = [0u8; 4];
                let avail = b.len().saturating_sub(off).min(4);
                if avail == 0 {
                    return None;
                }
                w[..avail].copy_from_slice(&b[off..off + avail]);
                return Some(w);
            }
        }
        None
    }

    fn data_word(&self, addr: u32) -> Option<u32> {
        for (a, s, b) in &self.data {
            let end = a.wrapping_add(*s);
            if addr >= *a && addr < end {
                let off = (addr - *a) as usize;
                let mut w = [0u8; 4];
                let avail = b.len().saturating_sub(off).min(4);
                for (i, byte) in w.iter_mut().enumerate().take(avail) {
                    *byte = b[off + i];
                }
                return Some(u32::from_le_bytes(w));
            }
        }
        // Backed-but-unlisted RAM inside a known range reads as zero.
        if self.ram_ranges.iter().any(|(a, s)| addr >= *a && addr < a.wrapping_add(*s)) {
            return Some(0);
        }
        None
    }

    pub fn symbol_at(&self, addr: u32) -> Option<&Sym> {
        // Smallest symbol whose [addr, addr+size) covers `addr`.
        let mut best: Option<&Sym> = None;
        for s in &self.symbols {
            if s.size == 0 {
                continue;
            }
            if addr >= s.addr && addr < s.addr.wrapping_add(s.size) {
                if best.map(|b| s.size < b.size).unwrap_or(true) {
                    best = Some(s);
                }
            }
        }
        best
    }

    pub fn symbol_by_name(&self, name: &str) -> Option<&Sym> {
        self.symbols.iter().find(|s| s.name == name)
    }

    pub fn symbols_in_range(&self, start: u32, end: u32) -> Vec<&Sym> {
        let mut v: Vec<&Sym> = self
            .symbols
            .iter()
            .filter(|s| s.size > 0 && s.addr >= start && s.addr < end)
            .collect();
        v.sort_by_key(|s| s.addr);
        v
    }

    /// Storage class of an address, preferring a named symbol, then the
    /// containing section, then the RAM/stack ranges the caller supplied.
    pub fn storage_at(&self, addr: u32) -> Storage {
        if let Some(sym) = self.symbol_at(addr) {
            if sym.storage != Storage::Unknown {
                return sym.storage;
            }
        }
        for s in &self.sections {
            if addr >= s.addr && addr < s.addr.wrapping_add(s.size) {
                let st = section_storage(&s.name, s.flags);
                if st != Storage::Unknown {
                    return st;
                }
            }
        }
        Storage::Unknown
    }

    pub fn describe_addr(&self, addr: u32) -> String {
        if let Some(s) = self.symbol_at(addr) {
            format!("{} ({:#010x})", s.name, addr)
        } else {
            format!("{addr:#010x}")
        }
    }
}

impl crate::arm::exec::Memory for FirmwareImage {
    fn read_insn(&self, addr: u32) -> Option<[u8; 4]> {
        self.code_bytes(addr)
    }
    fn read_data(&self, addr: u32) -> Option<u32> {
        self.data_word(addr)
    }
    fn symbol_at(&self, addr: u32) -> Option<String> {
        FirmwareImage::symbol_at(self, addr).map(|s| s.name.clone())
    }
}

/// Give zero-size symbols an extent so `symbol_at` works, but never let the
/// extent cross into a different section: infer only up to the next symbol
/// *within the same section*, else the section end.
fn fill_symbol_sizes(symbols: &mut [Sym], extents: &[(u32, u32)]) {
    let extents: Vec<(u32, u32)> = extents.to_vec();
    let mut addrs: Vec<(u32, usize)> = symbols
        .iter()
        .enumerate()
        .filter(|(_, s)| s.storage != Storage::Unknown)
        .map(|(i, s)| (s.addr, i))
        .collect();
    addrs.sort_unstable();
    for k in 0..addrs.len() {
        let (addr, i) = addrs[k];
        if symbols[i].size != 0 {
            continue;
        }
        let section_end = extents
            .iter()
            .find(|(a, sz)| addr >= *a && addr < a.wrapping_add(*sz))
            .map(|(a, sz)| a.wrapping_add(*sz))
            .unwrap_or(addr + 2);
        let next = if k + 1 < addrs.len() { addrs[k + 1].0 } else { section_end };
        let end = next.min(section_end);
        let size = end.saturating_sub(addr);
        symbols[i].size = size.max(2);
    }
}

/// Merge symbols from a GNU ld `.map` file or `nm` output. This lets users
/// supply symbol names that were stripped from the shipped ELF.
fn merge_symbol_text(text: &str, symbols: &mut Vec<Sym>, sections: &[Section]) {
    // `nm` style:  "08000001 T boot_mpu_init"
    // `.map` style: "                0x08000001                boot_mpu_init"
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // nm format
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() == 3 && parts[0].len() <= 8 {
            if let Ok(addr) = u32::from_str_radix(parts[0], 16) {
                let name = parts[2].to_string();
                add_merged(symbols, name, addr, sections, true);
                continue;
            }
        }
        // map format: an address token beginning with 0x
        if parts.len() >= 2 && parts[0].starts_with("0x") {
            if let Ok(addr) = u32::from_str_radix(&parts[0][2..], 16) {
                let name = parts[1].to_string();
                add_merged(symbols, name, addr, sections, false);
            }
        }
    }
}

fn add_merged(symbols: &mut Vec<Sym>, name: String, addr: u32, sections: &[Section], global: bool) {
    if symbols.iter().any(|s| s.name == name && s.addr == addr) {
        return;
    }
    let storage = sections
        .iter()
        .find(|s| addr >= s.addr && addr < s.addr.wrapping_add(s.size))
        .map(|s| section_storage(&s.name, s.flags))
        .unwrap_or(Storage::Unknown);
    symbols.push(Sym { name, addr, size: 0, storage, global });
}

/// Convenience: resolve a section index to a name, used by reports.
pub fn section_name(file: &object::File, idx: SectionIndex) -> Option<String> {
    file.section_by_index(idx).ok().and_then(|s| s.name().ok().map(|n| n.to_string()))
}

/// Group symbols into a sorted map by name for deterministic output.
pub fn sorted_symbols(image: &FirmwareImage) -> BTreeMap<String, u32> {
    image.symbols.iter().map(|s| (s.name.clone(), s.addr)).collect()
}
