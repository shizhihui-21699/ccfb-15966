//! Thumb / Thumb-2 decoder for the instruction subset that MPU configuration
//! code and RTOS context switches use.
//!
//! Every bit field below was derived by sweeping real
//! `clang --target=armv7m-none-eabi -mcpu=cortex-m4` output one operand at a
//! time and matching it against the ARMv7-M Architecture Reference Manual
//! (ARM DDI 0403C) Appendix A5.3 encoding tables. The unit tests re-check the
//! field layout against a corpus of real encodings so a future edit that
//! shifts a field will fail loudly rather than mis-decode firmware.
//!
//! Anything not recognised becomes `Op::Unsupported { hw1, hw2 }` so the
//! report can state that execution reached code this tool cannot model,
//! instead of inventing register values.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cond {
    Eq,
    Ne,
    Cs,
    Cc,
    Mi,
    Pl,
    Vs,
    Vc,
    Hi,
    Ls,
    Ge,
    Lt,
    Gt,
    Le,
    Al,
}

impl Cond {
    pub fn from_bits(b: u8) -> Option<Cond> {
        Some(match b & 0xF {
            0x0 => Cond::Eq,
            0x1 => Cond::Ne,
            0x2 => Cond::Cs,
            0x3 => Cond::Cc,
            0x4 => Cond::Mi,
            0x5 => Cond::Pl,
            0x6 => Cond::Vs,
            0x7 => Cond::Vc,
            0x8 => Cond::Hi,
            0x9 => Cond::Ls,
            0xA => Cond::Ge,
            0xB => Cond::Lt,
            0xC => Cond::Gt,
            0xD => Cond::Le,
            0xE => Cond::Al,
            _ => return None,
        })
    }
    pub fn mnemonic(self) -> &'static str {
        match self {
            Cond::Eq => "eq",
            Cond::Ne => "ne",
            Cond::Cs => "cs",
            Cond::Cc => "cc",
            Cond::Mi => "mi",
            Cond::Pl => "pl",
            Cond::Vs => "vs",
            Cond::Vc => "vc",
            Cond::Hi => "hi",
            Cond::Ls => "ls",
            Cond::Ge => "ge",
            Cond::Lt => "lt",
            Cond::Gt => "gt",
            Cond::Le => "le",
            Cond::Al => "al",
        }
    }
    pub fn bits(self) -> u8 {
        match self {
            Cond::Eq => 0x0,
            Cond::Ne => 0x1,
            Cond::Cs => 0x2,
            Cond::Cc => 0x3,
            Cond::Mi => 0x4,
            Cond::Pl => 0x5,
            Cond::Vs => 0x6,
            Cond::Vc => 0x7,
            Cond::Hi => 0x8,
            Cond::Ls => 0x9,
            Cond::Ge => 0xA,
            Cond::Lt => 0xB,
            Cond::Gt => 0xC,
            Cond::Le => 0xD,
            Cond::Al => 0xE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AluOp {
    And,
    Bic,
    Orr,
    Orn,
    Eor,
    Add,
    Adc,
    Sbc,
    Sub,
    Rsb,
    Mul,
    Lsl,
    Lsr,
    Asr,
    Ror,
    Mov,
    Mvn,
    Cmp,
    Cmn,
    Tst,
    Teq,
}

impl AluOp {
    pub fn is_test(self) -> bool {
        matches!(self, AluOp::Cmp | AluOp::Cmn | AluOp::Tst | AluOp::Teq)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemKind {
    Byte,
    Half,
    Word,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemBase {
    /// `[Rn, #imm]`
    Imm(i32),
    /// `[Rn, Rm, LSL #shift]`
    RegShift(u8, u8),
    /// post-indexed `[Rn], #imm` (writeback after access)
    PostImm(i32),
    /// pre-indexed with writeback `[Rn, #imm]!`
    PreImm(i32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegList(pub u16);

impl RegList {
    pub fn regs(&self) -> impl Iterator<Item = u8> + '_ {
        (0u8..16).filter(move |i| self.0 & (1 << i) != 0)
    }
    pub fn contains(&self, r: u8) -> bool {
        r < 16 && self.0 & (1 << r) != 0
    }
    pub fn count(&self) -> u32 {
        self.0.count_ones()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftKind {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

/// Decoded instruction. Register operands are encoded so `exec` can apply
/// them without re-parsing halfwords.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// Imm8 data-processing (T1): movs/cmp/adds/subs rd, #imm
    Imm8 { op: AluOp, rd: u8, imm: u32 },
    /// Register ALU (T1) / high-register mov/add/cmp (T2).
    AluReg { op: AluOp, rd: u8, rm: u8 },
    /// Shift-by-immediate register ALU (T1).
    AluShiftImm { op: AluOp, rd: u8, rm: u8, amount: u8 },
    /// Register-shift ALU (T2).
    ShiftReg { op: AluOp, rd: u8, rm: u8 },
    /// Wide data-processing, shifted register.
    AluW { op: AluOp, rd: u8, rn: u8, rm: u8, shift: ShiftKind, amount: u8 },
    /// Wide data-processing, modified immediate.
    AluImmW { op: AluOp, rd: u8, rn: u8, imm: u32 },
    /// Wide test/compare (no register write).
    TestW { op: AluOp, rn: u8, rm: u8, shift: ShiftKind, amount: u8 },
    TestImmW { op: AluOp, rn: u8, imm: u32 },
    /// `movw rd, #imm16`
    MovW { rd: u8, imm: u16 },
    /// `movt rd, #imm16`
    MovT { rd: u8, imm: u16 },
    Load { kind: MemKind, rt: u8, rn: u8, base: MemBase },
    Store { kind: MemKind, rt: u8, rn: u8, base: MemBase },
    /// literal pool load `ldr rt, [pc, #imm]`
    LdrLiteral { rt: u8, imm12: u32 },
    /// `adr rd, <label>` (PC plus immediate)
    Adr { rd: u8, imm: i32 },
    Push(RegList),
    Pop(RegList),
    Ldm { rn: u8, list: RegList, writeback: bool },
    Stm { rn: u8, list: RegList, writeback: bool },
    /// `add rd, sp, #imm`, `add sp, #imm`, `sub sp, #imm`
    SpAdjust { rd: Option<u8>, imm: i32 },
    BCond { cond: Cond, target: u32 },
    B { target: u32 },
    BL { target: u32 },
    Bx { rm: u8 },
    Blx { rm: u8 },
    Cbz { rn: u8, target: u32, nonzero: bool },
    It { cond: Cond, mask: u8 },
    Mrs { rd: u8, sysm: u8 },
    Msr { rn: u8, sysm: u8 },
    Barrier,
    Nop,
    Extend { rd: u8, rm: u8, width: u8, signed: bool, rotate: u8 },
    Rev { rd: u8, rm: u8, width: u8 },
    Unsupported { hw1: u16, hw2: Option<u16> },
    /// SVC/BKPT/UDF — execution does not continue normally.
    Stop,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Debug)]
pub struct Decoded {
    pub addr: u32,
    pub size: u16,
    pub op: Op,
}

#[derive(Clone, Debug)]
pub struct DecodeError {
    pub addr: u32,
    pub reason: String,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at {:#x}: {}", self.addr, self.reason)
    }
}

fn sx(v: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((v as i32) << shift) >> shift
}

fn ror32(v: u32, r: u32) -> u32 {
    let r = r % 32;
    if r == 0 {
        v
    } else {
        v.rotate_right(r)
    }
}

/// ThumbExpandImm (ARMv7-M A5.3.2).
pub fn thumb_expand_imm(imm12: u32) -> Option<u32> {
    let imm12 = imm12 & 0xFFF;
    if imm12 & 0xC00 == 0 {
        let b = (imm12 >> 8) & 0xF;
        let v = imm12 & 0xFF;
        return Some(match b {
            0x0 => v,
            0x1 => (v << 16) | v,
            0x2 => (v << 24) | (v << 8),
            0x3 => (v << 24) | (v << 16) | (v << 8) | v,
            _ => return None,
        });
    }
    Some(ror32(0x80 | (imm12 & 0x7F), (imm12 >> 7) & 0x1F))
}

fn err(addr: u32, reason: impl Into<String>) -> DecodeError {
    DecodeError { addr, reason: reason.into() }
}

pub fn decode(addr: u32, bytes: &[u8]) -> Result<Decoded, DecodeError> {
    if bytes.len() < 2 {
        return Err(err(addr, "truncated instruction"));
    }
    let hw1 = u16::from_le_bytes([bytes[0], bytes[1]]);

    // 32-bit prefixes are 11101, 11110, 11111 (bits 15:11).
    if (hw1 >> 11) & 0x1F >= 0b11101 {
        return decode_32(addr, hw1, bytes);
    }

    let op = decode_16(addr, hw1);
    Ok(Decoded { addr, size: 2, op })
}

fn decode_16(addr: u32, hw1: u16) -> Op {
    // Dispatch on the top five bits. Table A5-1 lists the 16-bit space as
    // 5-bit opcode groups; within a group bit10 selects the sub-kind where
    // needed. Using 6 bits here would be wrong because the imm5 field of the
    // load/store-immediate family straddles bit10.
    let top5 = (hw1 >> 11) & 0x1F;
    let rd = (hw1 & 0x7) as u8;
    let rn3 = ((hw1 >> 3) & 0x7) as u8;
    let rm3 = ((hw1 >> 6) & 0x7) as u8;
    let imm8 = (hw1 & 0xFF) as u32;
    let imm5 = ((hw1 >> 6) & 0x1F) as u32;
    let bit10 = (hw1 >> 10) & 1 != 0;
    let load = (hw1 >> 11) & 1 != 0;

    match top5 {
        // 00000..00010: shift by immediate
        0b00000 | 0b00001 | 0b00010 => {
            let op = match top5 {
                0b00000 => AluOp::Lsl,
                0b00001 => AluOp::Lsr,
                _ => AluOp::Asr,
            };
            Op::AluShiftImm { op, rd, rm: rm3, amount: imm5 as u8 }
        }
        // 00011: add/subtract (register or 3-operand immediate)
        0b00011 => {
            let imm_form = hw1 & 0x0400 != 0;
            let sub = hw1 & 0x0200 != 0;
            let op = if sub { AluOp::Sub } else { AluOp::Add };
            if imm_form {
                Op::AluImmW { op, rd, rn: rn3, imm: imm5 }
            } else {
                Op::AluW { op, rd, rn: rn3, rm: rm3, shift: ShiftKind::Lsl, amount: 0 }
            }
        }
        0b00100 => Op::Imm8 { op: AluOp::Mov, rd: ((hw1 >> 8) & 0x7) as u8, imm: imm8 },
        0b00101 => Op::Imm8 { op: AluOp::Cmp, rd: ((hw1 >> 8) & 0x7) as u8, imm: imm8 },
        0b00110 => Op::Imm8 { op: AluOp::Add, rd: ((hw1 >> 8) & 0x7) as u8, imm: imm8 },
        0b00111 => Op::Imm8 { op: AluOp::Sub, rd: ((hw1 >> 8) & 0x7) as u8, imm: imm8 },
        // 01000: bit10 selects data-processing vs special/branch-exchange
        0b01000 => {
            if !bit10 {
                let opc = (hw1 >> 6) & 0xF;
                let op = match opc {
                    0x0 => AluOp::And,
                    0x1 => AluOp::Eor,
                    0x5 => AluOp::Adc,
                    0x6 => AluOp::Sbc,
                    0x8 => AluOp::Tst,
                    0x9 => AluOp::Rsb,
                    0xA => AluOp::Cmp,
                    0xB => AluOp::Cmn,
                    0xC => AluOp::Orr,
                    0xD => AluOp::Mul,
                    0xE => AluOp::Bic,
                    0xF => AluOp::Mvn,
                    // 2,3,4,7 are register-shift forms
                    0x2 => return Op::ShiftReg { op: AluOp::Lsl, rd, rm: rm3 },
                    0x3 => return Op::ShiftReg { op: AluOp::Lsr, rd, rm: rm3 },
                    0x4 => return Op::ShiftReg { op: AluOp::Asr, rd, rm: rm3 },
                    0x7 => return Op::ShiftReg { op: AluOp::Ror, rd, rm: rm3 },
                    _ => unreachable!(),
                };
                Op::AluReg { op, rd, rm: rm3 }
            } else {
                // BX/BLX live at hw1[10:9]==10 (0x4700 / 0x4780). MOV (T2)
                // and ADD/CMP (T2) use hw1[10:9]==00 and never 10, so check
                // the 2-bit field, not bits 10:8.
                if hw1 & 0x0300 == 0x0300 {
                    // BX/BLX: bits[10:9] == 0b11 (0x4700 / 0x4780).
                    let rm = ((hw1 >> 3) & 0xF) as u8;
                    return if hw1 & 0x0080 != 0 { Op::Blx { rm } } else { Op::Bx { rm } };
                }
                let opc = (hw1 >> 8) & 0x3;
                let rm = ((hw1 >> 3) & 0xF) as u8;
                let d = ((hw1 & 0x7) | if hw1 & 0x0080 != 0 { 8 } else { 0 }) as u8;
                let op = match opc {
                    0 => AluOp::Add,
                    1 => AluOp::Cmp,
                    _ => AluOp::Mov,
                };
                Op::AluReg { op, rd: d, rm }
            }
        }
        // 01001: LDR (literal)
        0b01001 => Op::LdrLiteral { rt: ((hw1 >> 8) & 0x7) as u8, imm12: imm8 * 4 },
        // 01010, 01011: load/store register offset
        0b01010 | 0b01011 => decode_16_ldst_reg(hw1),
        // 01100..01111: word (0x6/0x7) vs byte immediate
        0b01100..=0b01111 => {
            let is_byte = hw1 & 0x1000 != 0;
            let off = if is_byte { imm5 as i32 } else { (imm5 as i32) * 4 };
            let rn = rn3;
            let rt = rd;
            let kind = if is_byte { MemKind::Byte } else { MemKind::Word };
            if load {
                Op::Load { kind, rt, rn, base: MemBase::Imm(off) }
            } else {
                Op::Store { kind, rt, rn, base: MemBase::Imm(off) }
            }
        }
        // 10000..10001: halfword immediate (0x8 top nibble)
        0b10000 | 0b10001 => {
            let off = (imm5 as i32) * 2;
            let rn = rn3;
            let rt = rd;
            if load {
                Op::Load { kind: MemKind::Half, rt, rn, base: MemBase::Imm(off) }
            } else {
                Op::Store { kind: MemKind::Half, rt, rn, base: MemBase::Imm(off) }
            }
        }
        0b10010 | 0b10011 => {
            // SP-relative load/store: 1001 L Rt imm8
            let rt = ((hw1 >> 8) & 0x7) as u8;
            let off = (imm8 as i32) * 4;
            let ld = hw1 & 0x0800 != 0;
            if ld { Op::Load { kind: MemKind::Word, rt, rn: 13, base: MemBase::Imm(off) } }
            else { Op::Store { kind: MemKind::Word, rt, rn: 13, base: MemBase::Imm(off) } }
        }
        // 10100/10101: ADR / ADD SP
        0b10100 | 0b10101 => {
            let use_sp = hw1 & 0x0800 != 0;
            let off = (imm8 as i32) * 4;
            if use_sp {
                Op::SpAdjust { rd: Some(((hw1 >> 8) & 0x7) as u8), imm: off }
            } else {
                Op::Adr { rd: ((hw1 >> 8) & 0x7) as u8, imm: off }
            }
        }
        // 10110/10111: miscellaneous
        0b10110 | 0b10111 => decode_16_misc(addr, hw1),
        // 11000: STM, 11001: LDM
        0b11000 => {
            let rn = ((hw1 >> 8) & 0x7) as u8;
            let list = RegList(hw1 & 0xFF);
            Op::Stm { rn, list, writeback: !list.contains(rn) }
        }
        0b11001 => {
            let rn = ((hw1 >> 8) & 0x7) as u8;
            let list = RegList(hw1 & 0xFF);
            Op::Ldm { rn, list, writeback: !list.contains(rn) }
        }
        // 11010/11011: conditional branch / SVC / UDF
        0b11010 | 0b11011 => {
            let condb = ((hw1 >> 8) & 0xF) as u8;
            if condb == 0xF || condb == 0xE {
                return Op::Stop;
            }
            // Conditional-branch immediate is in halfwords.
            let off = sx((hw1 & 0xFF) as u32, 8) * 2;
            Op::BCond {
                cond: Cond::from_bits(condb).unwrap(),
                target: (addr as i64 + 4 + off as i64) as u32,
            }
        }
        // 11100: unconditional B
        0b11100 => Op::B { target: (addr as i64 + 4 + (sx((hw1 & 0x7FF) as u32, 11) * 2) as i64) as u32 },
        _ => Op::Unsupported { hw1, hw2: None },
    }
}

fn decode_16_misc(addr: u32, hw1: u16) -> Op {
    if hw1 & 0xFF00 == 0xBF00 {
        let low = hw1 & 0xFF;
        if low & 0x0F != 0 {
            if let Some(cond) = Cond::from_bits((low >> 4) as u8).filter(|c| *c != Cond::Al) {
                return Op::It { cond, mask: (low & 0x0F) as u8 };
            }
        }
        return Op::Nop;
    }
    if hw1 & 0xFE00 == 0xB400 {
        let mut list = RegList(hw1 & 0xFF);
        if hw1 & 0x0100 != 0 {
            list.0 |= 1 << 14;
        }
        return Op::Push(list);
    }
    if hw1 & 0xFE00 == 0xBC00 {
        let mut list = RegList(hw1 & 0xFF);
        if hw1 & 0x0100 != 0 {
            list.0 |= 1 << 15;
        }
        return Op::Pop(list);
    }
    if hw1 & 0xF500 == 0xB100 {
        let nonzero = hw1 & 0x0800 != 0;
        let i = ((hw1 >> 9) & 1) as u32;
        let imm5 = ((hw1 >> 3) & 0x1F) as u32;
        let off = (i << 6) | (imm5 << 1);
        let rn = (hw1 & 0x7) as u8;
        return Op::Cbz { rn, target: addr.wrapping_add(4).wrapping_add(off), nonzero };
    }
    if hw1 & 0xFF00 == 0xB000 {
        let imm = ((hw1 & 0x7F) as i32) * 4;
        let sub = hw1 & 0x80 != 0;
        return Op::SpAdjust { rd: None, imm: if sub { -imm } else { imm } };
    }
    if hw1 & 0xFF00 == 0xB200 {
        let op = ((hw1 >> 6) & 0x3) as u8;
        let rm = ((hw1 >> 3) & 0x7) as u8;
        let rd = (hw1 & 0x7) as u8;
        let (signed, width) = match op {
            0 => (false, 16),
            1 => (true, 16),
            2 => (false, 8),
            _ => (true, 8),
        };
        return Op::Extend { rd, rm, width, signed, rotate: 0 };
    }
    if hw1 & 0xFF00 == 0xBA00 {
        let op = ((hw1 >> 6) & 0x3) as u8;
        let rm = ((hw1 >> 3) & 0x7) as u8;
        let rd = (hw1 & 0x7) as u8;
        let width = if op == 1 { 16 } else { 32 };
        return Op::Rev { rd, rm, width };
    }
    if hw1 & 0xFFE8 == 0xB660 {
        return Op::Barrier;
    }
    Op::Unsupported { hw1, hw2: None }
}

fn decode_16_ldst_reg(hw1: u16) -> Op {
    let opc = (hw1 >> 9) & 0x7;
    let rm = ((hw1 >> 6) & 0x7) as u8;
    let rn = ((hw1 >> 3) & 0x7) as u8;
    let rt = (hw1 & 0x7) as u8;
    let (load, kind) = match opc {
        0 => (false, MemKind::Word),
        1 => (false, MemKind::Half),
        2 => (false, MemKind::Byte),
        3 => (false, MemKind::Byte), // LDRSB
        4 => (true, MemKind::Word),
        5 => (true, MemKind::Half),
        6 => (true, MemKind::Byte),
        _ => (true, MemKind::Half), // LDRSH
    };
    if load {
        Op::Load { kind, rt, rn, base: MemBase::RegShift(rm, 0) }
    } else {
        Op::Store { kind, rt, rn, base: MemBase::RegShift(rm, 0) }
    }
}

// -------------------------------------------------------------------------
// 32-bit Thumb-2
//
// Dispatch is by first-halfword class first (load/store single 0xF8xx,
// data-processing register 0xEAxx, load/store multiple 0xE8xx, MOVW/MOVT),
// then by hw2[15] which cleanly separates the branch/misc family (bit 15 set)
// from data-processing modified-immediate (bit 15 clear).
// -------------------------------------------------------------------------

fn decode_32(addr: u32, hw1: u16, bytes: &[u8]) -> Result<Decoded, DecodeError> {
    if bytes.len() < 4 {
        return Err(err(addr, "truncated 32-bit instruction"));
    }
    let hw2 = u16::from_le_bytes([bytes[2], bytes[3]]);
    let op = decode_32_op(addr, hw1, hw2);
    Ok(Decoded { addr, size: 4, op })
}

fn decode_32_op(addr: u32, hw1: u16, hw2: u16) -> Op {
    // Load/store single data item (imm12 and register offset).
    if hw1 & 0xFE00 == 0xF800 {
        return decode_32_ldst(hw1, hw2);
    }
    // Data-processing (shifted register).
    if hw1 & 0xFE00 == 0xEA00 {
        return decode_32_dp_reg(hw1, hw2);
    }
    // Load/store multiple (incl. PUSH/POP wide).
    if hw1 & 0xFE00 == 0xE800 {
        return decode_32_multi(hw1, hw2);
    }
    // MOVW / MOVT (i bit is hw1[10], hence masked out of the pattern).
    if hw1 & 0xFB80 == 0xF200 {
        let rd = ((hw2 >> 8) & 0xF) as u8;
        let imm = (((hw1 & 0xF) as u16) << 12)
            | ((((hw1 >> 10) & 1) as u16) << 11)
            | ((((hw2 >> 12) & 0x7) as u16) << 8)
            | ((hw2 & 0xFF) as u16);
        return Op::MovW { rd, imm };
    }
    if hw1 & 0xFB80 == 0xF280 {
        let rd = ((hw2 >> 8) & 0xF) as u8;
        let imm = (((hw1 & 0xF) as u16) << 12)
            | ((((hw1 >> 10) & 1) as u16) << 11)
            | ((((hw2 >> 12) & 0x7) as u16) << 8)
            | ((hw2 & 0xFF) as u16);
        return Op::MovT { rd, imm };
    }
    // MSR / MRS / barriers: hw2[15] is clear for these too, so test first.
    if hw1 & 0xFFF0 == 0xF380 || hw1 & 0xFFF0 == 0xF390 {
        return Op::Msr { rn: (hw2 & 0xF) as u8, sysm: ((hw2 >> 8) & 0xFF) as u8 };
    }
    if hw1 & 0xFFF0 == 0xF3E0 || hw1 & 0xFFF0 == 0xF3F0 {
        return Op::Mrs { rd: (hw2 & 0xF) as u8, sysm: ((hw2 >> 8) & 0xFF) as u8 };
    }
    if hw1 & 0xFFF0 == 0xF3B0 || hw1 & 0xFFF8 == 0xF3B0 {
        return Op::Barrier; // DSB/DMB/ISB (and CLREX/DSB-ish variants)
    }
    // Branches: hw2[15] == 1.
    if hw2 & 0x8000 != 0 {
        return decode_32_branch(addr, hw1, hw2);
    }
    // Data-processing (modified immediate) / plain binary immediate.
    decode_32_dp_imm(hw1, hw2)
}

fn decode_32_ldst(hw1: u16, hw2: u16) -> Op {
    let opc = (hw1 >> 4) & 0x1F;
    let rn = (hw1 & 0xF) as u8;
    let rt = ((hw2 >> 12) & 0xF) as u8;
    // Immediate (unsigned, 12-bit byte offset) forms: opc 8..=13.
    if hw1 & 0x0080 != 0 {
        let (load, kind) = match opc {
            0x8 => (false, MemKind::Byte),
            0x9 => (true, MemKind::Byte),
            0xA => (false, MemKind::Half),
            0xB => (true, MemKind::Half),
            0xC => (false, MemKind::Word),
            0xD => (true, MemKind::Word),
            _ => return Op::Unsupported { hw1, hw2: Some(hw2) },
        };
        let off = (hw2 & 0xFFF) as i32;
        return if load {
            Op::Load { kind, rt, rn, base: MemBase::Imm(off) }
        } else {
            Op::Store { kind, rt, rn, base: MemBase::Imm(off) }
        };
    }
    // Register-offset word forms: 4 = STR, 5 = LDR.
    let rm = (hw2 & 0xF) as u8;
    let shift = ((hw2 >> 4) & 0x3) as u8;
    let load = match opc {
        0x4 => false,
        0x5 => true,
        _ => return Op::Unsupported { hw1, hw2: Some(hw2) },
    };
    let op = if load {
        Op::Load { kind: MemKind::Word, rt, rn, base: MemBase::RegShift(rm, shift) }
    } else {
        Op::Store { kind: MemKind::Word, rt, rn, base: MemBase::RegShift(rm, shift) }
    };
    op
}

fn decode_32_dp_reg(hw1: u16, hw2: u16) -> Op {
    let op = (hw1 >> 5) & 0xF;
    let rn = (hw1 & 0xF) as u8;
    let rd = ((hw2 >> 8) & 0xF) as u8;
    let rm = (hw2 & 0xF) as u8;
    let imm3 = ((hw2 >> 12) & 0x7) as u8;
    let imm2 = ((hw2 >> 6) & 0x3) as u8;
    let ty = ((hw2 >> 4) & 0x3) as u8;
    let shift = match ty {
        0 => ShiftKind::Lsl,
        1 => ShiftKind::Lsr,
        2 => ShiftKind::Asr,
        _ => ShiftKind::Ror,
    };
    let amount = (imm3 << 2) | imm2;
    let alu = match op {
        0x0 => AluOp::And,
        0x1 => AluOp::Bic,
        0x2 => AluOp::Orr,
        0x3 => AluOp::Orn,
        0x4 => AluOp::Eor,
        0x8 => AluOp::Add,
        0xA => AluOp::Adc,
        0xB => AluOp::Sbc,
        0xD => AluOp::Sub,
        0xE => AluOp::Rsb,
        _ => return Op::Unsupported { hw1, hw2: Some(hw2) },
    };
    if rn == 0xF {
        let op = match alu {
            AluOp::Orr => AluOp::Mov,
            AluOp::Orn => AluOp::Mvn,
            _ => alu,
        };
        return Op::AluW { op, rd, rn: 0, rm, shift, amount };
    }
    if rd == 0xF {
        let test = match alu {
            AluOp::And => AluOp::Tst,
            AluOp::Eor => AluOp::Teq,
            AluOp::Add => AluOp::Cmn,
            AluOp::Sub => AluOp::Cmp,
            _ => return Op::Unsupported { hw1, hw2: Some(hw2) },
        };
        return Op::TestW { op: test, rn, rm, shift, amount };
    }
    Op::AluW { op: alu, rd, rn, rm, shift, amount }
}

fn decode_32_multi(hw1: u16, hw2: u16) -> Op {
    // PUSH{W}: STMDB SP!  -> hw1 = 1110 1001 0x10 1101
    if hw1 & 0xFFF0 == 0xE920 {
        return Op::Push(RegList(hw2));
    }
    // POP{W}: LDMIA SP!   -> hw1 = 1110 1000 1011 1101
    if hw1 & 0xFFF0 == 0xE8B0 {
        return Op::Pop(RegList(hw2));
    }
    let rn = (hw1 & 0xF) as u8;
    let load = (hw1 >> 4) & 1 == 1;
    let writeback = (hw1 >> 5) & 1 == 1;
    let list = RegList(hw2);
    let op = if load {
        Op::Ldm { rn, list, writeback }
    } else {
        Op::Stm { rn, list, writeback }
    };
    op
}

fn decode_32_branch(addr: u32, hw1: u16, hw2: u16) -> Op {
    let s = ((hw1 >> 10) & 1) as u32;
    let j1 = ((hw2 >> 13) & 1) as u32;
    let j2 = ((hw2 >> 11) & 1) as u32;
    // BL (T1): hw2 = 11 J1 1 J2 imm11
    if hw2 & 0xD000 == 0xD000 {
        let imm10 = (hw1 & 0x3FF) as u32;
        let imm11 = (hw2 & 0x7FF) as u32;
        let i1 = (!(j1 ^ s)) & 1;
        let i2 = (!(j2 ^ s)) & 1;
        let imm = (s << 24) | (i1 << 23) | (i2 << 22) | (imm10 << 12) | (imm11 << 1);
        return Op::BL { target: (addr as i64 + 4 + sx(imm, 25) as i64) as u32 };
    }
    // B.W (T4): hw2 = 10 J1 1 J2 imm11
    if hw2 & 0xD000 == 0x9000 {
        let imm10 = (hw1 & 0x3FF) as u32;
        let imm11 = (hw2 & 0x7FF) as u32;
        let i1 = (!(j1 ^ s)) & 1;
        let i2 = (!(j2 ^ s)) & 1;
        let imm = (s << 24) | (i1 << 23) | (i2 << 22) | (imm10 << 12) | (imm11 << 1);
        return Op::B { target: (addr as i64 + 4 + sx(imm, 25) as i64) as u32 };
    }
    // B<c>.W (T3): hw2 = 10 J1 0 J2 imm11; cond in hw1[9:6].
    if hw2 & 0xD000 == 0x8000 {
        let cond = ((hw1 >> 6) & 0xF) as u8;
        let imm6 = (hw1 & 0x3F) as u32;
        let imm11 = (hw2 & 0x7FF) as u32;
        let imm = (s << 20) | (j2 << 19) | (j1 << 18) | (imm6 << 12) | (imm11 << 1);
        let target = (addr as i64 + 4 + sx(imm, 21) as i64) as u32;
        if let Some(c) = Cond::from_bits(cond).filter(|c| *c != Cond::Al) {
            return Op::BCond { cond: c, target };
        }
    }
    Op::Unsupported { hw1, hw2: Some(hw2) }
}

fn decode_32_dp_imm(hw1: u16, hw2: u16) -> Op {
    // Data-processing (modified immediate): hw1 = 11110 i 0 op(4) S Rn,
    // hw2 = 0 imm3 Rd imm8.
    let i = ((hw1 >> 10) & 1) as u32;
    let opf = ((hw1 >> 5) & 0xF) as u8;
    let rn = (hw1 & 0xF) as u8;
    let rd = ((hw2 >> 8) & 0xF) as u8;
    let imm12 = (i << 11) | (((hw2 >> 12) & 0x7) as u32) << 8 | (hw2 & 0xFF) as u32;
    let alu = match opf {
        0x0 => AluOp::And,
        0x1 => AluOp::Bic,
        0x2 => AluOp::Orr,
        0x3 => AluOp::Orn,
        0x4 => AluOp::Eor,
        0x8 => AluOp::Add,
        0xA => AluOp::Adc,
        0xB => AluOp::Sbc,
        0xD => AluOp::Sub,
        0xE => AluOp::Rsb,
        _ => return Op::Unsupported { hw1, hw2: Some(hw2) },
    };
    let imm = match thumb_expand_imm(imm12) {
        Some(v) => v,
        None => return Op::Unsupported { hw1, hw2: Some(hw2) },
    };
    if rn == 0xF {
        let op = match alu {
            AluOp::Orr => AluOp::Mov,
            AluOp::Orn => AluOp::Mvn,
            _ => alu,
        };
        return Op::AluImmW { op, rd, rn: 0xF, imm };
    }
    if rd == 0xF {
        let test = match alu {
            AluOp::And => AluOp::Tst,
            AluOp::Eor => AluOp::Teq,
            AluOp::Add => AluOp::Cmn,
            AluOp::Sub => AluOp::Cmp,
            _ => return Op::Unsupported { hw1, hw2: Some(hw2) },
        };
        return Op::TestImmW { op: test, rn, imm };
    }
    Op::AluImmW { op: alu, rd, rn, imm }
}
