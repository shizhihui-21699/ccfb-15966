//! Minimal assembler used only by tests and by the bundled self-check
//! firmware fixtures. It emits Thumb-2 instruction words from a tiny subset
//! of mnemonics so we can build deterministic firmware images in-tree.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reg {
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
    R10,
    R11,
    R12,
    Sp,
    Lr,
    Pc,
}

impl Reg {
    pub fn num(self) -> u8 {
        match self {
            Reg::R0 => 0,
            Reg::R1 => 1,
            Reg::R2 => 2,
            Reg::R3 => 3,
            Reg::R4 => 4,
            Reg::R5 => 5,
            Reg::R6 => 6,
            Reg::R7 => 7,
            Reg::R8 => 8,
            Reg::R9 => 9,
            Reg::R10 => 10,
            Reg::R11 => 11,
            Reg::R12 => 12,
            Reg::Sp => 13,
            Reg::Lr => 14,
            Reg::Pc => 15,
        }
    }
}

/// Emits a relocatable-lite image: instructions are placed at a fixed base,
/// labels are resolved in a second pass so short/wide branches can be encoded.
pub struct Assembler {
    base: u32,
    words: Vec<u16>,
    pub labels: HashMap<String, u32>,
    pending: Vec<PendingBranch>,
}

struct PendingBranch {
    at_word: usize,
    label: String,
    kind: BranchKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BranchKind {
    /// 16-bit unconditional B (T2), +-2048
    B16,
    /// 32-bit B.W
    B32,
    /// 32-bit BL
    Bl,
}

impl Assembler {
    pub fn new(base: u32) -> Self {
        Self { base, words: Vec::new(), labels: HashMap::new(), pending: Vec::new() }
    }

    pub fn label(&mut self, name: &str) {
        let here = self.base + (self.words.len() as u32 / 2) * 2;
        self.labels.insert(name.to_string(), here);
    }

    pub fn addr_here(&self) -> u32 {
        self.base + (self.words.len() as u32 / 2) * 2
    }

    fn push(&mut self, hw: u16) {
        self.words.push(hw);
    }

    /// `(base_low, base_mid, base_high, disp_mid, disp_high)` - unused; kept
    /// for parity with the ARM binary helpers in tests.
    pub fn encode_movs_imm8(rd: Reg, imm: u8) -> u16 {
        0x2000 | ((rd.num() as u16) << 8) | (imm as u16)
    }

    pub fn movs(&mut self, rd: Reg, imm: u8) {
        self.push(Self::encode_movs_imm8(rd, imm));
    }

    pub fn movw(&mut self, rd: Reg, imm: u16) {
        let i = (imm >> 11) & 1;
        let imm4 = (imm >> 12) & 0xF;
        let imm3 = (imm >> 8) & 0x7;
        let imm8 = imm & 0xFF;
        let hw1 = 0xF240u16 | ((i as u16) << 10) | ((imm4 as u16) << 0);
        let hw2 = ((rd.num() as u16) << 8) | ((imm3 as u16) << 12) | (imm8 as u16);
        self.push(hw1);
        self.push(hw2);
    }

    pub fn movt(&mut self, rd: Reg, imm: u16) {
        let i = (imm >> 11) & 1;
        let imm4 = (imm >> 12) & 0xF;
        let imm3 = (imm >> 8) & 0x7;
        let imm8 = imm & 0xFF;
        let hw1 = 0xF2C0u16 | ((i as u16) << 10) | ((imm4 as u16) << 0);
        let hw2 = ((rd.num() as u16) << 8) | ((imm3 as u16) << 12) | (imm8 as u16);
        self.push(hw1);
        self.push(hw2);
    }

    /// `ldr rd, [rn, #imm12]` (T3 wide encoding, imm12 scaled by 4).
    pub fn ldr_imm12(&mut self, rd: Reg, rn: Reg, off_bytes: u32) {
        assert_eq!(off_bytes % 4, 0);
        let imm12 = off_bytes / 4;
        let hw1 = 0xF8D0u16 | (rn.num() as u16);
        let hw2 = ((rd.num() as u16) << 12) | (imm12 as u16);
        self.push(hw1);
        self.push(hw2);
    }

    /// `str rd, [rn, #imm12]` (T3 wide encoding, imm12 scaled by 4).
    pub fn str_imm12(&mut self, rd: Reg, rn: Reg, off_bytes: u32) {
        assert_eq!(off_bytes % 4, 0);
        let imm12 = off_bytes / 4;
        let hw1 = 0xF8C0u16 | (rn.num() as u16);
        let hw2 = ((rd.num() as u16) << 12) | (imm12 as u16);
        self.push(hw1);
        self.push(hw2);
    }

    /// `ldr rd, [rn]` T1 (16-bit).
    pub fn ldr_t1(&mut self, rd: Reg, rn: Reg) {
        let hw = 0x6800u16 | ((rn.num() as u16) << 3) | (rd.num() as u16);
        self.push(hw);
    }

    /// `str rd, [rn]` T1 (16-bit).
    pub fn str_t1(&mut self, rd: Reg, rn: Reg) {
        let hw = 0x6000u16 | ((rn.num() as u16) << 3) | (rd.num() as u16);
        self.push(hw);
    }

    pub fn adds_imm8(&mut self, rd: Reg, imm: u8) {
        self.push(0x3000 | ((rd.num() as u16) << 8) | (imm as u16));
    }

    pub fn subs_imm8(&mut self, rd: Reg, imm: u8) {
        self.push(0x3800 | ((rd.num() as u16) << 8) | (imm as u16));
    }

    pub fn b16(&mut self, label: &str) {
        self.pending.push(PendingBranch { at_word: self.words.len(), label: label.to_string(), kind: BranchKind::B16 });
        self.push(0); // placeholder
    }

    pub fn b32(&mut self, label: &str) {
        self.pending.push(PendingBranch { at_word: self.words.len(), label: label.to_string(), kind: BranchKind::B32 });
        self.push(0);
        self.push(0);
    }

    pub fn bl(&mut self, label: &str) {
        self.pending.push(PendingBranch { at_word: self.words.len(), label: label.to_string(), kind: BranchKind::Bl });
        self.push(0);
        self.push(0);
    }

    pub fn bx_lr(&mut self) {
        self.push(0x4770);
    }

    pub fn nop(&mut self) {
        self.push(0xBF00);
    }

    pub fn finish(mut self) -> (u32, Vec<u8>, HashMap<String, u32>) {
        for p in std::mem::take(&mut self.pending) {
            let target = *self.labels.get(&p.label).unwrap_or_else(|| panic!("undefined label {}", p.label));
            let pc = self.base + (p.at_word as u32 / 1) * 2;
            let disp = target as i64 - (pc as i64 + 4);
            match p.kind {
                BranchKind::B16 => {
                    assert!((-2048..=2046).contains(&disp) && disp % 2 == 0, "B16 out of range: {}", disp);
                    let imm11 = ((disp as i32 >> 1) & 0x7FF) as u16;
                    self.words[p.at_word] = 0xE000 | imm11;
                }
                BranchKind::B32 => {
                    let s = ((disp >> 24) & 1) as u16;
                    let i1 = ((disp >> 23) & 1) as u16;
                    let i2 = ((disp >> 22) & 1) as u16;
                    let imm10 = ((disp >> 12) & 0x3FF) as u16;
                    let imm11 = ((disp >> 1) & 0x7FF) as u16;
                    let j1 = (!(i1 ^ s)) & 1;
                    let j2 = (!(i2 ^ s)) & 1;
                    self.words[p.at_word] = 0xF000 | (s << 10) | imm10;
                    self.words[p.at_word + 1] = 0x9000 | (j1 << 13) | (j2 << 11) | imm11;
                }
                BranchKind::Bl => {
                    let s = ((disp >> 24) & 1) as u16;
                    let i1 = ((disp >> 23) & 1) as u16;
                    let i2 = ((disp >> 22) & 1) as u16;
                    let imm10 = ((disp >> 12) & 0x3FF) as u16;
                    let imm11 = ((disp >> 1) & 0x7FF) as u16;
                    let j1 = (!(i1 ^ s)) & 1;
                    let j2 = (!(i2 ^ s)) & 1;
                    self.words[p.at_word] = 0xF000 | (s << 10) | imm10;
                    self.words[p.at_word + 1] = 0xD000 | (j1 << 13) | (j2 << 11) | imm11;
                }
            }
        }
        let mut bytes = Vec::with_capacity(self.words.len() * 2);
        for w in &self.words {
            bytes.push((*w & 0xFF) as u8);
            bytes.push((*w >> 8) as u8);
        }
        (self.base, bytes, self.labels)
    }
}
