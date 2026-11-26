//! ARMv7-M (PMSAv7) MPU model.
//!
//! Everything here is derived from the ARMv7-M Architecture Reference Manual
//! (ARM DDI 0403C), principally:
//!   * B3.5 "Protected Memory System Architecture (PMSAv7)"
//!   * B3.5.1 "PMSAv7 compliant MPU operation" (alignment / overlap / sub-regions)
//!   * Table B3-43 (region size encoding), Table B3-47 (AP encoding),
//!     Table B3-48 (XN encoding)
//!   * B2.2.8 "Access permission checking" (CheckPermission pseudocode)
//!   * B3.5 "ValidateAddressP" MPU pseudocode
//!
//! The three rules that trip people up and that this module implements
//! literally:
//!   1. Region sizes are powers of two and the programmed base address is
//!      naturally aligned to the region size (address bits below log2(size)
//!      are ignored by the hardware, so a mis-aligned base silently expands
//!      the region downwards).
//!   2. For regions of 256 bytes or more, the eight SRD bits each disable one
//!      1/8th sub-region (LSB = lowest address). Sub-region bits are
//!      UNPREDICTABLE/SBZ below 256 bytes.
//!   3. When regions overlap, the *highest numbered* matching region wins.
//!
//! The default memory map, when PRIVDEFENA is set, behaves like region
//! number -1 for privileged accesses only.

/// One 1/8 sub-region of a ≥256-byte region.
pub const SUBREGION_COUNT: usize = 8;

pub fn rasr_size_field(rasr: u32) -> u32 {
    (rasr >> 1) & 0x1F
}
pub fn rasr_enabled(rasr: u32) -> bool {
    rasr & 1 != 0
}
pub fn rasr_ap(rasr: u32) -> u8 {
    ((rasr >> 24) & 0x7) as u8
}
pub fn rasr_xn(rasr: u32) -> bool {
    rasr & (1 << 28) != 0
}
pub fn rasr_srd(rasr: u32) -> u8 {
    ((rasr >> 8) & 0xFF) as u8
}
pub fn rbar_base(rbar: u32) -> u32 {
    rbar & 0xFFFF_FFE0
}

/// Decoded view of one region register pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub number: u8,
    pub rbar: u32,
    pub rasr: u32,
    /// latched at decode time from RASR
    pub enabled: bool,
    /// log2 of the region size, 2..=32. Values 2..=4 are "unsupported" by
    /// PMSAv7 (minimum supported is 32 bytes, i.e. field value 4).
    pub size_log2: u8,
    /// Programmed base with the low 5 bits already stripped (RBAR[31:5]).
    pub programmed_base: u32,
    /// Base actually used after natural alignment to the region size.
    pub base: u32,
    /// True when the programmed base was not naturally aligned; the hardware
    /// silently aligns it down, which usually means the author's intent and
    /// the actual protected range disagree.
    pub misaligned: bool,
    pub ap: u8,
    pub xn: bool,
    pub srd: u8,
}

impl Region {
    pub fn size_bytes(&self) -> u64 {
        if self.size_log2 >= 32 {
            1u64 << 32
        } else {
            1u64 << self.size_log2
        }
    }

    pub fn end(&self) -> u64 {
        let start = self.base as u64;
        (start + self.size_bytes()).min(1u64 << 32)
    }

    /// PMSAv7 only supports sizes 32 B .. 4 GB, i.e. size field 4..=31.
    pub fn size_supported(&self) -> bool {
        (4..=31).contains(&self.size_log2)
    }

    /// Sub-regions exist only for regions of 256 bytes (size_log2 == 8) or more.
    pub fn has_subregions(&self) -> bool {
        self.size_log2 >= 8
    }

    pub fn subregion_bytes(&self) -> u64 {
        self.size_bytes() / SUBREGION_COUNT as u64
    }

    /// Does this region cover `addr`, taking SRD into account?
    pub fn covers(&self, addr: u32) -> bool {
        if !self.enabled {
            return false;
        }
        let a = addr as u64;
        let end = self.end();
        if a < self.base as u64 || a >= end {
            // A 4GB region wraps the entire address space; the base is 0.
            if self.size_log2 == 32 {
                // covers everything
            } else {
                return false;
            }
        }
        if self.has_subregions() {
            let idx = ((a - self.base as u64) >> (self.size_log2 as u64 - 3)) as usize;
            let idx = idx.min(SUBREGION_COUNT - 1);
            if (self.srd >> idx) & 1 != 0 {
                return false;
            }
        }
        true
    }

    /// The sub-region index inside this region (only meaningful for ≥256 B).
    pub fn subregion_index(&self, addr: u32) -> Option<usize> {
        if !self.has_subregions() || !self.covers(addr) {
            return None;
        }
        let idx = ((addr as u64 - self.base as u64) >> (self.size_log2 as u64 - 3)) as usize;
        Some(idx.min(SUBREGION_COUNT - 1))
    }

    pub fn from_regs(number: u8, rbar: u32, rasr: u32) -> Region {
        let size_field = rasr_size_field(rasr);
        let size_log2 = (size_field + 1) as u8;
        let programmed_base = rbar_base(rbar);
        let size = if size_log2 >= 32 { 1u64 << 32 } else { 1u64 << size_log2 };
        let align_mask = if size >= (1u64 << 32) { 0 } else { (size - 1) as u32 };
        let misaligned = programmed_base & align_mask != 0;
        let base = programmed_base & !align_mask;
        Region {
            number,
            rbar,
            rasr,
            enabled: rasr_enabled(rasr),
            size_log2,
            programmed_base,
            base,
            misaligned,
            ap: rasr_ap(rasr),
            xn: rasr_xn(rasr),
            srd: rasr_srd(rasr),
        }
    }
}

/// Access permission pair derived from an AP field (Table B3-47).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rwx {
    pub read: bool,
    pub write: bool,
    pub exec: bool,
}

impl Rwx {
    pub const NONE: Rwx = Rwx { read: false, write: false, exec: false };
    pub const RO: Rwx = Rwx { read: true, write: false, exec: false };
    pub const RX: Rwx = Rwx { read: true, write: false, exec: true };
    pub const RW: Rwx = Rwx { read: true, write: true, exec: false };
    pub const RWX: Rwx = Rwx { read: true, write: true, exec: true };

    pub fn describe(&self) -> String {
        if !self.read && !self.write && !self.exec {
            "none".to_string()
        } else {
            let mut s = String::new();
            s.push('r');
            if self.write {
                s.push('w');
            } else {
                s.push('-');
            }
            if self.exec {
                s.push('x');
            } else {
                s.push('-');
            }
            s
        }
    }
}

/// Is `ap` a reserved/unpredictable encoding (100)?
pub fn ap_reserved(ap: u8) -> bool {
    ap == 0b100
}

/// Privileged / unprivileged RWXO for an AP value, independently of XN.
/// Returns `None` when the encoding is reserved (100).
pub fn ap_to_rwx(ap: u8) -> Option<(Rwx, Rwx)> {
    // read/write before XN; XN is applied by the caller because it only
    // affects instruction fetch.
    let (p_rw, u_rw): ((bool, bool), (bool, bool)) = match ap {
        0b000 => ((false, false), (false, false)),
        0b001 => ((true, true), (false, false)),
        0b010 => ((true, true), (true, false)),
        0b011 => ((true, true), (true, true)),
        0b101 => ((true, false), (false, false)),
        0b110 => ((true, false), (true, false)),
        0b111 => ((true, false), (true, false)),
        _ => return None,
    };
    // Instruction fetch requires read permission (CheckPermission: the AP
    // "fault" is evaluated first and XN is OR'ed in afterwards).
    let mk = |(r, w): (bool, bool)| Rwx { read: r, write: w, exec: r };
    Some((mk(p_rw), mk(u_rw)))
}

fn apply_xn(mut p: Rwx, xn: bool) -> Rwx {
    if xn {
        p.exec = false;
    }
    p
}

/// Where an effective permission came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PermSource {
    /// Region number that won the overlap resolution.
    Region(u8),
    /// Default memory map via PRIVDEFENA (privileged only).
    Background,
    /// Nothing matched and no background region is enabled.
    Fault,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Effective {
    pub privileged: Rwx,
    pub unprivileged: Rwx,
    pub source: PermSource,
    /// Lower-numbered regions that also matched this address but were
    /// shadowed by the winner (the legitimate "override" pattern).
    pub shadowed: Vec<u8>,
}

/// The full MPU register file and its behaviour.
#[derive(Clone, Debug, Default)]
pub struct Mpu {
    pub ctrl: u32,
    pub rnr: u32,
    pub regions: Vec<(u32, u32)>, // (rbar, rasr) per region
    pub alias_mode: AliasMode,
    /// Number of regions implemented by the core (MPU_TYPE.DREGION).
    pub dregion: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AliasMode {
    /// Alias n accesses the registers of region (RNR + n).
    #[default]
    Relative,
    /// Alias n accesses the registers of region (RNR + n) mod DREGION.
    Modulo,
}

pub const MPU_TYPE_ADDR: u32 = 0xE000_ED90;
pub const MPU_CTRL_ADDR: u32 = 0xE000_ED94;
pub const MPU_RNR_ADDR: u32 = 0xE000_ED98;
pub const MPU_RBAR_ADDR: u32 = 0xE000_ED9C;
pub const MPU_RASR_ADDR: u32 = 0xE000_EDA0;
// Alias pairs: (RBAR, RASR)
pub const MPU_ALIASES: [(u32, u32); 3] =
    [(0xE000_EDA4, 0xE000_EDA8), (0xE000_EDAC, 0xE000_EDB0), (0xE000_EDB4, 0xE000_EDB8)];

pub const CTRL_ENABLE: u32 = 1 << 0;
pub const CTRL_HFNMIENA: u32 = 1 << 1;
pub const CTRL_PRIVDEFENA: u32 = 1 << 2;
pub const RBAR_VALID: u32 = 1 << 4;

impl Mpu {
    pub fn new(dregion: usize) -> Self {
        Mpu { ctrl: 0, rnr: 0, regions: vec![(0, 0); dregion], alias_mode: AliasMode::Relative, dregion }
    }

    pub fn enabled(&self) -> bool {
        self.ctrl & CTRL_ENABLE != 0
    }
    pub fn privdefena(&self) -> bool {
        self.ctrl & CTRL_PRIVDEFENA != 0
    }

    /// Returns true if `addr` is one of the MPU register addresses or an alias.
    pub fn is_mpu_addr(addr: u32) -> bool {
        addr == MPU_TYPE_ADDR
            || addr == MPU_CTRL_ADDR
            || addr == MPU_RNR_ADDR
            || addr == MPU_RBAR_ADDR
            || addr == MPU_RASR_ADDR
            || MPU_ALIASES.iter().any(|(r, s)| *r == addr || *s == addr)
    }

    fn alias_target(&self, n: u32) -> Option<usize> {
        if self.dregion == 0 {
            return None;
        }
        let base = self.rnr & 0xFF;
        let idx = match self.alias_mode {
            AliasMode::Relative => base.wrapping_add(n) as usize,
            AliasMode::Modulo => ((base as usize + n as usize) % self.dregion) as usize,
        };
        if idx < self.dregion {
            Some(idx)
        } else {
            None
        }
    }

    /// Perform a 32-bit write to an MPU register. Returns a short description
    /// of the effect for tracing, or `None` if the address is not MPU-related.
    pub fn write(&mut self, addr: u32, value: u32) -> Option<MpuWrite> {
        match addr {
            MPU_TYPE_ADDR => None, // read-only
            MPU_CTRL_ADDR => {
                let old = self.ctrl;
                self.ctrl = value;
                Some(MpuWrite::Ctrl { old, new: value })
            }
            MPU_RNR_ADDR => {
                let old = self.rnr;
                self.rnr = value & 0xFF;
                Some(MpuWrite::Rnr { old, new: self.rnr })
            }
            MPU_RBAR_ADDR => Some(self.write_rbar((self.rnr & 0xFF) as u8, value)),
            MPU_RASR_ADDR => Some(self.write_rasr((self.rnr & 0xFF) as u8, value)),
            a => {
                for (i, (rbar_addr, rasr_addr)) in MPU_ALIASES.iter().enumerate() {
                    let n = (i + 1) as u32;
                    if *rbar_addr == a {
                        return self.alias_target(n).map(|idx| {
                            let mut w = self.write_rbar(idx as u8, value);
                            if let MpuWrite::Rbar { alias, .. } = &mut w {
                                *alias = Some(n);
                            }
                            w
                        });
                    }
                    if *rasr_addr == a {
                        return self.alias_target(n).map(|idx| {
                            let mut w = self.write_rasr(idx as u8, value);
                            if let MpuWrite::Rasr { alias, .. } = &mut w {
                                *alias = Some(n);
                            }
                            w
                        });
                    }
                }
                None
            }
        }
    }

    fn write_rbar(&mut self, region: u8, value: u32) -> MpuWrite {
        let mut target = region;
        if value & RBAR_VALID != 0 {
            target = (value & 0xF) as u8;
            self.rnr = target as u32;
        }
        let note = if (target as usize) >= self.dregion {
            Some(format!("region {} out of range (DREGION={})", target, self.dregion))
        } else {
            None
        };
        if (target as usize) < self.dregion {
            self.regions[target as usize].0 = value;
        }
        MpuWrite::Rbar { value, region: target, valid: value & RBAR_VALID != 0, note, alias: None, applied: (target as usize) < self.dregion }
    }

    fn write_rasr(&mut self, region: u8, value: u32) -> MpuWrite {
        let note = if (region as usize) >= self.dregion {
            Some(format!("region {} out of range (DREGION={})", region, self.dregion))
        } else {
            None
        };
        if (region as usize) < self.dregion {
            self.regions[region as usize].1 = value;
        }
        MpuWrite::Rasr { value, region, note, alias: None, applied: (region as usize) < self.dregion }
    }

    pub fn read(&self, addr: u32) -> Option<u32> {
        match addr {
            MPU_TYPE_ADDR => Some(((self.dregion as u32) << 8) | 0),
            MPU_CTRL_ADDR => Some(self.ctrl),
            MPU_RNR_ADDR => Some(self.rnr),
            MPU_RBAR_ADDR => self.regions.get((self.rnr & 0xFF) as usize).map(|p| p.0).or(Some(0)),
            MPU_RASR_ADDR => self.regions.get((self.rnr & 0xFF) as usize).map(|p| p.1).or(Some(0)),
            a => {
                for (i, (rbar_addr, rasr_addr)) in MPU_ALIASES.iter().enumerate() {
                    let n = (i + 1) as u32;
                    if *rbar_addr == a {
                        return self.alias_target(n).map(|idx| self.regions[idx].0);
                    }
                    if *rasr_addr == a {
                        return self.alias_target(n).map(|idx| self.regions[idx].1);
                    }
                }
                None
            }
        }
    }

    pub fn decoded_regions(&self) -> Vec<Region> {
        self.regions
            .iter()
            .enumerate()
            .map(|(i, (rbar, rasr))| Region::from_regs(i as u8, *rbar, *rasr))
            .collect()
    }

    /// Default memory map permission and XN for the background region
    /// (ValidateAddressP: `perms.ap = '001'`, XN from the 4-bit address
    /// region index).
    pub fn default_map(addr: u32) -> (Rwx, Rwx) {
        let xn = (addr >> 28) > 9 || (addr >> 29) == 0b010;
        let priv_rwx = apply_xn(Rwx { read: true, write: true, exec: true }, xn);
        let unpriv_rwx = apply_xn(Rwx { read: false, write: false, exec: false }, xn);
        (priv_rwx, unpriv_rwx)
    }

    /// Effective permissions at `addr` when the MPU is enabled.
    pub fn effective(&self, addr: u32) -> Effective {
        if !self.enabled() {
            let (p, _privileged_only) = Self::default_map(addr);
            // With the MPU disabled both privileged and unprivileged code see
            // the default map, i.e. AP = 011 everywhere.
            let u = Rwx { read: true, write: true, exec: !((addr >> 28) > 9 || (addr >> 29) == 0b010) };
            return Effective { privileged: p, unprivileged: u, source: PermSource::Background, shadowed: vec![] };
        }
        let regions = self.decoded_regions();
        let mut winner: Option<&Region> = None;
        let mut shadowed: Vec<u8> = Vec::new();
        for r in &regions {
            if r.enabled && r.covers(addr) {
                if let Some(w) = winner {
                    shadowed.push(w.number);
                    // keep list sorted for stability
                }
                winner = Some(r);
            }
        }
        match winner {
            Some(r) => {
                match ap_to_rwx(r.ap) {
                    Some((p, u)) => {
                        let p = apply_xn(p, r.xn);
                        let u = apply_xn(u, r.xn);
                        shadowed.sort_unstable();
                        Effective { privileged: p, unprivileged: u, source: PermSource::Region(r.number), shadowed }
                    }
                    None => {
                        // Reserved AP encoding: treat as no access and surface
                        // the invalid encoding via the source.
                        Effective { privileged: Rwx::NONE, unprivileged: Rwx::NONE, source: PermSource::Region(r.number), shadowed }
                    }
                }
            }
            None => {
                let (p, u) = Self::default_map(addr);
                if self.privdefena() {
                    Effective { privileged: p, unprivileged: u, source: PermSource::Background, shadowed: vec![] }
                } else {
                    Effective { privileged: Rwx::NONE, unprivileged: Rwx::NONE, source: PermSource::Fault, shadowed: vec![] }
                }
            }
        }
    }

    /// Snapshot of the register state, in a form the report can serialise.
    pub fn snapshot(&self) -> MpuSnapshot {
        MpuSnapshot {
            ctrl: self.ctrl,
            enabled: self.enabled(),
            privdefena: self.privdefena(),
            hfnmiena: self.ctrl & CTRL_HFNMIENA != 0,
            regions: self.decoded_regions(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MpuSnapshot {
    pub ctrl: u32,
    pub enabled: bool,
    pub privdefena: bool,
    pub hfnmiena: bool,
    pub regions: Vec<Region>,
}

#[derive(Clone, Debug)]
pub enum MpuWrite {
    Ctrl { old: u32, new: u32 },
    Rnr { old: u32, new: u32 },
    Rbar { value: u32, region: u8, valid: bool, note: Option<String>, alias: Option<u32>, applied: bool },
    Rasr { value: u32, region: u8, note: Option<String>, alias: Option<u32>, applied: bool },
}

impl MpuWrite {
    pub fn describe(&self) -> String {
        match self {
            MpuWrite::Ctrl { old, new } => format!(
                "MPU_CTRL {:#010x} -> {:#010x} (ENABLE={} PRIVDEFENA={} HFNMIENA={})",
                old,
                new,
                new & 1,
                (new >> 2) & 1,
                (new >> 1) & 1
            ),
            MpuWrite::Rnr { old, new } => format!("MPU_RNR {old} -> {new}"),
            MpuWrite::Rbar { value, region, valid, note, alias, applied } => {
                let mut s = format!(
                    "MPU_RBAR[{}] = {:#010x} base={:#010x}{}",
                    region,
                    value,
                    value & 0xFFFF_FFE0,
                    if *valid { " (VALID: selected region)" } else { "" }
                );
                if let Some(a) = alias {
                    s.push_str(&format!(" via alias A{a}"));
                }
                if let Some(n) = note {
                    s.push_str(&format!(" !! {n}"));
                }
                if !*applied {
                    s.push_str(" !! write dropped (region not implemented)");
                }
                s
            }
            MpuWrite::Rasr { value, region, note, alias, applied } => {
                let mut s = format!("MPU_RASR[{}] = {:#010x}", region, value);
                if let Some(a) = alias {
                    s.push_str(&format!(" via alias A{a}"));
                }
                if let Some(n) = note {
                    s.push_str(&format!(" !! {n}"));
                }
                if !*applied {
                    s.push_str(" !! write dropped (region not implemented)");
                }
                s
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rasr(size_field: u32, ap: u8, xn: bool, srd: u8, enable: bool) -> u32 {
        (size_field << 1)
            | ((ap as u32) << 24)
            | if xn { 1 << 28 } else { 0 }
            | ((srd as u32) << 8)
            | if enable { 1 } else { 0 }
    }

    #[test]
    fn ap_table_matches_manual() {
        // Table B3-47.
        let cases = [
            (0b000, Rwx::NONE, Rwx::NONE),
            (0b001, Rwx::RW, Rwx::NONE),
            (0b010, Rwx::RW, Rwx::RO),
            (0b011, Rwx::RW, Rwx::RW),
            (0b101, Rwx::RO, Rwx::NONE),
            (0b110, Rwx::RO, Rwx::RO),
            (0b111, Rwx::RO, Rwx::RO),
        ];
        for (ap, p, u) in cases {
            let (got_p, got_u) = ap_to_rwx(ap).unwrap();
            assert_eq!(got_p.read, p.read);
            assert_eq!(got_p.write, p.write);
            assert_eq!(got_u.read, u.read);
            assert_eq!(got_u.write, u.write);
        }
        assert!(ap_to_rwx(0b100).is_none());
    }

    #[test]
    fn xnormal_clears_only_execute() {
        let p = apply_xn(Rwx::RWX, true);
        assert_eq!(p, Rwx::RW);
        let u = apply_xn(Rwx::RW, true);
        assert_eq!(u, Rwx::RW);
    }

    #[test]
    fn size_and_alignment() {
        // 64 KB region, size field 15, misaligned base 0x2001_8000.
        let r = Region::from_regs(0, 0x2001_8000 | 0, rasr(15, 0b011, false, 0, true));
        assert_eq!(r.size_log2, 16);
        assert_eq!(r.size_bytes(), 65536);
        assert_eq!(r.programmed_base, 0x2001_8000);
        assert!(r.misaligned);
        assert_eq!(r.base, 0x2001_0000);
        assert_eq!(r.end(), 0x2002_0000);
    }

    #[test]
    fn subregion_hole() {
        // 256 KB region at 0x2000_0000, size field 17 (log2 18), size 256 KB,
        // so each sub-region is 32 KB. SRD bit 1 disables sub-region 1 =>
        // hole at 0x2000_8000..0x2001_0000.
        let r = Region::from_regs(0, 0x2000_0000, rasr(17, 0b011, false, 0b0000_0010, true));
        assert!(!r.covers(0x2000_8000));
        assert!(!r.covers(0x2000_FFFF));
        assert!(r.covers(0x2000_7000));
        assert!(r.covers(0x2001_0000));
    }

    #[test]
    fn subregion_ignored_below_256() {
        // 128 B region: SRD is SBZ/UNP, so a set bit must not punch a hole.
        let r = Region::from_regs(0, 0x2000_0000, rasr(6, 0b011, false, 0xFF, true));
        assert!(!r.has_subregions());
        assert!(r.covers(0x2000_0000));
    }

    #[test]
    fn highest_number_wins_overlap() {
        let mut m = Mpu::new(8);
        // Region 0: whole SRAM 128K, unprivileged RW.
        m.regions[0] = (0x2000_0000, rasr(16, 0b011, false, 0, true));
        // Region 3: 4K window at 0x2001_F000, unprivileged no access (carve-out).
        m.regions[3] = (0x2001_F000, rasr(11, 0b001, false, 0, true));
        m.ctrl = CTRL_ENABLE;
        let inside = m.effective(0x2001_F800);
        assert_eq!(inside.source, PermSource::Region(3));
        assert_eq!(inside.unprivileged.write, false);
        assert_eq!(inside.shadowed, vec![0]);
        let outside = m.effective(0x2001_0000);
        assert_eq!(outside.source, PermSource::Region(0));
        assert!(outside.unprivileged.write);
    }

    #[test]
    fn privdefena_background() {
        let mut m = Mpu::new(8);
        m.ctrl = CTRL_ENABLE | CTRL_PRIVDEFENA;
        let e = m.effective(0x2000_0000);
        assert_eq!(e.source, PermSource::Background);
        assert!(e.privileged.write);
        assert!(!e.unprivileged.read, "background is privileged-only");
    }

    #[test]
    fn no_region_no_background_faults() {
        let mut m = Mpu::new(8);
        m.ctrl = CTRL_ENABLE;
        let e = m.effective(0x2000_0000);
        assert_eq!(e.source, PermSource::Fault);
        assert!(!e.privileged.read);
    }

    #[test]
    fn disabled_mpu_default_map() {
        let m = Mpu::new(8);
        let e = m.effective(0x0800_0000);
        assert!(e.unprivileged.read);
        assert!(e.unprivileged.exec);
        let e = m.effective(0x4000_0000);
        assert!(!e.unprivileged.exec, "peripheral space is XN by default");
    }

    #[test]
    fn ctrl_write_and_rnr() {
        let mut m = Mpu::new(8);
        m.write(MPU_CTRL_ADDR, 5);
        assert!(m.enabled() && m.privdefena());
        m.write(MPU_RNR_ADDR, 3);
        assert_eq!(m.rnr, 3);
        let w = m.write(MPU_RBAR_ADDR, 0x2000_0000).unwrap();
        match w {
            MpuWrite::Rbar { region, valid, .. } => {
                assert_eq!(region, 3);
                assert!(!valid);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn rbar_valid_selects_region() {
        let mut m = Mpu::new(8);
        m.write(MPU_RNR_ADDR, 1);
        let w = m.write(MPU_RBAR_ADDR, 0x2000_0000 | 0x10 | 5).unwrap();
        match w {
            MpuWrite::Rbar { region, valid, .. } => {
                assert_eq!(region, 5);
                assert!(valid);
            }
            _ => panic!(),
        }
        assert_eq!(m.rnr, 5);
        assert_eq!(m.regions[5].0, 0x2000_0000 | 0x10 | 5);
    }

    #[test]
    fn alias_relative_and_modulo() {
        let mut m = Mpu::new(4);
        m.alias_mode = AliasMode::Modulo;
        m.write(MPU_RNR_ADDR, 3);
        let w = m.write(MPU_ALIASES[0].0, 0x2000_0000).unwrap();
        match w {
            MpuWrite::Rbar { region, .. } => assert_eq!(region, 0),
            _ => panic!(),
        }
        m.alias_mode = AliasMode::Relative;
        // With RNR=3 in a 4-region MPU, alias A1 targets region 4, which is
        // not implemented: the write must be dropped.
        assert!(m.write(MPU_ALIASES[0].1, rasr(11, 0b011, false, 0, true)).is_none());
        // From RNR=1 the relative alias reaches region 2.
        m.write(MPU_RNR_ADDR, 1);
        let w = m.write(MPU_ALIASES[0].1, rasr(11, 0b011, false, 0, true)).unwrap();
        match w {
            MpuWrite::Rasr { region, .. } => assert_eq!(region, 2),
            _ => panic!(),
        }
    }

    #[test]
    fn boundary_map_spans() {
        let mut m = Mpu::new(8);
        m.regions[0] = (0x2000_0000, rasr(16, 0b011, false, 0, true));
        m.regions[1] = (0x2000_0000, rasr(11, 0b001, false, 0, true)); // 4K
        m.ctrl = CTRL_ENABLE;
        assert_eq!(m.effective(0x2000_0800).source, PermSource::Region(1));
        assert_eq!(m.effective(0x2000_1000).source, PermSource::Region(0));
    }
}
