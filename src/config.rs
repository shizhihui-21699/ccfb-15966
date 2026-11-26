//! TOML configuration describing the audit.
//!
//! The tool cannot guess which function is "boot", which function each task
//! switches through, or what a given RAM range is *meant* to hold. That intent
//! comes from this file. Everything else — the region register writes, the
//! resulting permission map, the overlap/shadowing, the sub-region holes — is
//! derived from the firmware itself.
//!
//! Example (also see `examples/`):
//!
//! ```toml
//! [mcu]
//! dregion = 8               # MPU_TYPE.DREGION from the manual/silicon
//!
//! [boot]
//! entry = "boot_mpu_init"   # a symbol, or a hex address
//!
//! [memory.code]             # storage classes used for finding reports
//! ranges = ["__kernel_code_start..__kernel_code_end"]
//! ...
//!
//! [[object]]                # a named memory object with an expected policy
//! name = "kernel_data"
//! range = "__kernel_data_start..__kernel_data_end"
//! kind = "data"
//! expected = { privileged = "rw", unprivileged = "none" }
//!
//! [[task]]
//! name = "pump_control"
//! switch_entry = "task_switch"        # function that programs the MPU
//! switch_arg = { symbol = "task1_cfg" } # R0 at entry
//! # or switch_arg = { value = 0x2000_1234 } / { reg = 0 }
//! stack = "__task1_stack_bottom..__task1_stack_top"
//! ```

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

use crate::symbol::FirmwareImage;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    #[serde(default)]
    pub mcu: McuConfig,
    pub boot: Option<EntrySpec>,
    /// Optional storage-class ranges; keys are lowercase storage names.
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default, rename = "object")]
    pub objects: Vec<ObjectSpec>,
    #[serde(default, rename = "task")]
    pub tasks: Vec<TaskSpec>,
    /// Policy expectations applied to any object not individually matched.
    #[serde(default)]
    pub policy: PolicyConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct McuConfig {
    /// MPU_TYPE.DREGION (number of regions).
    #[serde(default = "default_dregion")]
    pub dregion: usize,
    /// How the RBAR/RASR aliases index regions: "relative" (default) or
    /// "modulo".
    #[serde(default)]
    pub alias_mode: Option<String>,
    /// Include the default memory map (PRIVDEFENA) in maps. When unset, the
    /// firmware's own MPU_CTRL write decides.
    #[serde(default)]
    pub privdefena: Option<bool>,
}

fn default_dregion() -> usize {
    8
}

impl Default for McuConfig {
    fn default() -> Self {
        McuConfig { dregion: 8, alias_mode: None, privdefena: None }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct EntrySpec {
    /// Symbol name or hex address of the function to run.
    pub entry: Option<AddressOrSymbol>,
    /// Startup: run `boot` then the task switch, so the map is the one in
    /// effect after both (usual case).
    #[serde(default)]
    pub startup: bool,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct MemoryConfig {
    /// Address ranges of executable code.
    #[serde(default)]
    pub code: Vec<String>,
    #[serde(default)]
    pub rodata: Vec<String>,
    #[serde(default)]
    pub data: Vec<String>,
    #[serde(default)]
    pub bss: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum AddressOrSymbol {
    Symbol { symbol: String },
    Value { value: String },
}

impl AddressOrSymbol {
    pub fn resolve(&self, image: &FirmwareImage) -> Result<u32> {
        match self {
            AddressOrSymbol::Symbol { symbol } => {
                let s = image.symbol_by_name(symbol).with_context(|| format!("symbol `{symbol}` not found"))?;
                Ok(s.addr)
            }
            AddressOrSymbol::Value { value } => Ok(parse_addr(value)?),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ObjectSpec {
    pub name: String,
    pub range: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub expected: Option<ExpectedSpec>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ExpectedSpec {
    #[serde(default)]
    pub privileged: Option<String>,
    #[serde(default)]
    pub unprivileged: Option<String>,
    /// True when this object may be written by its owning task.
    #[serde(default)]
    pub writable: bool,
    /// True when this object is expected to be executable.
    #[serde(default)]
    pub executable: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TaskSpec {
    pub name: String,
    /// Function that programs the MPU for this task.
    #[serde(default)]
    pub switch_entry: Option<String>,
    #[serde(default)]
    pub switch_arg: Option<SwitchArg>,
    /// Stack range, e.g. "__task1_stack_bottom..__task1_stack_top".
    #[serde(default)]
    pub stack: Option<String>,
    #[serde(default)]
    pub expected: Option<ExpectedSpec>,
    /// Optional: source of the "argument register" if switch_arg is a symbol
    /// but the switch is done via a struct pointer placed in RAM.
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum SwitchArg {
    /// Symbol whose address is loaded into R0 at the switch entry.
    Symbol { symbol: String },
    /// Literal 32-bit value in R0 (e.g. a pointer written to RAM).
    Value { value: String },
    /// Leave R0 as it is (e.g. the switch function derives its own config).
    None,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct PolicyConfig {
    /// Treat addresses not classified as anything (in RAM) as data.
    #[serde(default)]
    pub unknown_is_data: bool,
}

pub fn parse_addr(s: &str) -> Result<u32> {
    let t = s.trim();
    let body = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    // Allow underscores as digit separators.
    let clean: String = body.chars().filter(|c| *c != '_').collect();
    u32::from_str_radix(&clean, 16).with_context(|| format!("bad address `{s}`"))
}

/// A half-open address range.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Range {
    pub start: u32,
    pub end: u32,
}

impl Range {
    pub fn contains(&self, addr: u32) -> bool {
        addr >= self.start && addr < self.end
    }
    pub fn len(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }
    pub fn overlap(&self, other: &Range) -> Range {
        Range { start: self.start.max(other.start), end: self.end.min(other.end) }
    }
    pub fn intersects(&self, other: &Range) -> bool {
        self.start.max(other.start) < self.end.min(other.end)
    }
}

/// Parse `A..B`, where A and B are each a symbol or a hex constant.
pub fn parse_range(spec: &str, image: &FirmwareImage) -> Result<Range> {
    let parts: Vec<&str> = spec.split("..").collect();
    if parts.len() != 2 {
        bail!("range `{spec}` must look like `start..end`");
    }
    let a = parse_addr_or_symbol(parts[0].trim(), image)?;
    let b = parse_addr_or_symbol(parts[1].trim(), image)?;
    if b <= a {
        bail!("range `{spec}` is empty (start {a:#x} >= end {b:#x})");
    }
    Ok(Range { start: a, end: b })
}

pub fn parse_addr_or_symbol(s: &str, image: &FirmwareImage) -> Result<u32> {
    let t = s.trim();
    if t.starts_with("0x") || t.starts_with("0X") || t.chars().all(|c| c.is_ascii_hexdigit() || c == '_') {
        if let Ok(v) = parse_addr(t) {
            // A bare hex string could also be a symbol; prefer symbol when it
            // exists, since symbols like "dead_beef" are rare but addresses
            // in decimal-looking hex are common. Hex parsing wins because the
            // grammar here is hexadecimal.
            return Ok(v);
        }
    }
    if let Some(sym) = image.symbol_by_name(t) {
        return Ok(sym.addr);
    }
    parse_addr(t).with_context(|| format!("not a symbol or address: `{t}`"))
}

pub fn load(path: &Path) -> Result<Config> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let cfg: Config = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    Ok(cfg)
}
