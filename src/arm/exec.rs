//! Register-level ARMv7-M interpreter, scoped to what MPU auditing needs.
//!
//! This is *not* a full CPU model. It executes the boot path and each task's
//! switch path with concrete register values to recover:
//!   * every constant written to MPU_RNR / MPU_RBAR / MPU_RASR (and aliases),
//!   * which way conditional branches/CBZ go for that register state, including
//!     IT blocks,
//!   * literal-pool constants, so `ldr r0, =0xE000ED9C` yields the real address.
//!
//! It stops cleanly (rather than guessing) at `bx lr`, a `bl` to an unmapped
//! helper, an unmodelled instruction, an unbacked memory read, or a budget
//! limit, and reports exactly why so the caller can print "path not fully
//! modelled". Memory *writes* are recorded but not committed: for permission
//! auditing we care about the address/size, not the stored value.
//!
//! #### IT block semantics
//! Verified against real encodings: `ITSTATE = cond[3:0] : mask[3:0]`.
//! The current condition is `ITSTATE[7:4]` (base = `cond[3:1]`, invert =
//! `ITSTATE[4]`), and each instruction advances with
//! `if ITSTATE[2:0]==0 { clear } else { ITSTATE[4:0] = ITSTATE[4:0]<<1 }`.

use super::decode::{self, AluOp, Cond, Decoded, MemBase, MemKind, Op, RegList, ShiftKind};
use std::fmt;

/// A memory access performed by the simulated code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mem {
    pub addr: u32,
    pub size: u32,
    pub write: bool,
}

/// A write to an MPU register.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MpuWriteEvent {
    pub pc: u32,
    pub addr: u32,
    pub value: u32,
    /// True when the write happened with the core in unprivileged (User)
    /// mode, which faults on real hardware.
    pub unprivileged: bool,
    pub func: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// Reached the caller-specified stop address or executed a return to the
    /// sentinel LR.
    Returned,
    /// SVC / BKPT / UDF.
    Exception,
    /// `bl`/`blx` to an address with no executable bytes (external helper).
    CalledUnknown { target: u32 },
    /// Decoder could not model the instruction.
    Unsupported { pc: u32, hw1: u16, hw2: Option<u16> },
    /// Instruction budget exhausted (likely an infinite loop).
    Budget { pc: u32 },
    /// Branched outside mapped code.
    BadBranch { target: u32 },
    /// Data read from an address the memory model does not back.
    UnknownMemory { addr: u32 },
    /// Start address has no code.
    NoCode { addr: u32 },
    /// A return via `bx lr` where LR was never set by a modelled call.
    ReturnedUnknown,
    /// Control reached an address using an interworking bit pattern the
    /// interpreter cannot follow (e.g. switching to ARM state).
    BadState { target: u32 },
}

impl StopReason {
    pub fn describe(&self) -> String {
        match self {
            StopReason::Returned => "returned normally".into(),
            StopReason::Exception => "hit an exception-generating instruction".into(),
            StopReason::CalledUnknown { target } => {
                format!("called a function outside the image at {target:#010x}")
            }
            StopReason::Unsupported { pc, hw1, hw2 } => match hw2 {
                Some(h) => format!("unmodelled instruction at {pc:#010x} ({hw1:04x} {h:04x})"),
                None => format!("unmodelled instruction at {pc:#010x} ({hw1:04x})"),
            },
            StopReason::Budget { pc } => format!("instruction budget exhausted at {pc:#010x}"),
            StopReason::BadBranch { target } => format!("branch to unmapped address {target:#010x}"),
            StopReason::UnknownMemory { addr } => format!("read of unmapped address {addr:#010x}"),
            StopReason::NoCode { addr } => format!("no code at {addr:#010x}"),
            StopReason::ReturnedUnknown => "returned to an uninitialised link register".into(),
            StopReason::BadState { target } => format!("branch needs ARM state at {target:#010x}"),
        }
    }
    /// A clean stop means execution reached a natural end and every path it
    /// took was modelled.
    pub fn is_clean(&self) -> bool {
        matches!(self, StopReason::Returned | StopReason::CalledUnknown { .. })
    }
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

/// Memory model backing the interpreter.
pub trait Memory {
    /// Instruction halfwords at `addr` (2 or 4 bytes), or `None` if `addr` is
    /// not executable.
    fn read_insn(&self, addr: u32) -> Option<[u8; 4]>;
    /// A 32-bit little-endian data read (initial RAM contents, literal pools).
    fn read_data(&self, addr: u32) -> Option<u32>;
    /// Symbol whose range contains `addr`.
    fn symbol_at(&self, addr: u32) -> Option<String>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Regs {
    pub r: [u32; 16],
}

impl Regs {
    pub fn get(&self, i: u8) -> u32 {
        self.r.get(i as usize).copied().unwrap_or(0)
    }
    pub fn set(&mut self, i: u8, v: u32) {
        if (i as usize) < 16 {
            self.r[i as usize] = v;
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    pub n: bool,
    pub z: bool,
    pub c: bool,
    pub v: bool,
}

impl Flags {
    pub fn test(&self, cond: Cond) -> bool {
        match cond {
            Cond::Eq => self.z,
            Cond::Ne => !self.z,
            Cond::Cs => self.c,
            Cond::Cc => !self.c,
            Cond::Mi => self.n,
            Cond::Pl => !self.n,
            Cond::Vs => self.v,
            Cond::Vc => !self.v,
            Cond::Hi => self.c && !self.z,
            Cond::Ls => !self.c || self.z,
            Cond::Ge => self.n == self.v,
            Cond::Lt => self.n != self.v,
            Cond::Gt => self.n == self.v && !self.z,
            Cond::Le => self.n != self.v || self.z,
            Cond::Al => true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Run {
    pub writes: Vec<MpuWriteEvent>,
    pub mem: Vec<Mem>,
    pub stop: StopReason,
    pub regs: Regs,
    pub steps: u32,
}

enum Step {
    Next,
    Jump(u32),
    Stop(StopReason),
}

pub struct Interp<'a> {
    mem: &'a dyn Memory,
    /// Bytes written by the simulated program (stack pushes, stores to RAM
    /// config structures, etc.). Real code round-trips through the stack, so
    /// the interpreter must see its own writes. Word-granularity is enough for
    /// MPU setup code, which only moves words; sub-word stores are merged.
    shadow: std::collections::HashMap<u32, u32>,
    pub regs: Regs,
    pub flags: Flags,
    pub unprivileged: bool,
    pub writes: Vec<MpuWriteEvent>,
    pub mem_events: Vec<Mem>,
    /// ITSTATE: `it_base` = cond[3:1], `it_low` = (cond[0]<<4)|mask[3:0].
    it_base: u8,
    it_low: u8,
    budget: u32,
    /// Address that means "the routine returned": normally the LR value set
    /// by the caller.
    return_sentinel: Option<u32>,
    trace: bool,
}

impl<'a> Interp<'a> {
    pub fn new(mem: &'a dyn Memory, regs: Regs, unprivileged: bool) -> Self {
        Interp {
            mem,
            shadow: std::collections::HashMap::new(),
            regs,
            flags: Flags::default(),
            unprivileged,
            writes: Vec::new(),
            mem_events: Vec::new(),
            it_base: 0,
            it_low: 0,
            budget: 500_000,
            return_sentinel: None,
            trace: false,
        }
    }

    pub fn with_budget(mut self, b: u32) -> Self {
        self.budget = b;
        self
    }
    pub fn with_return_sentinel(mut self, addr: u32) -> Self {
        self.return_sentinel = Some(addr);
        self
    }
    pub fn set_return_sentinel(&mut self, addr: u32) {
        self.return_sentinel = Some(addr);
    }
    pub fn set_trace(&mut self, t: bool) { self.trace = t; }

    fn fetch(&self, addr: u32) -> Option<Decoded> {
        let w = self.mem.read_insn(addr)?;
        let hw1 = u16::from_le_bytes([w[0], w[1]]);
        let slice: &[u8] = if (hw1 >> 11) & 0x1F >= 0b11101 { &w[..4] } else { &w[..2] };
        decode::decode(addr, slice).ok()
    }

    pub fn run(&mut self, start: u32) -> Run {
        // Function symbols carry the Thumb bit (bit 0) in their low bit; the
        // interpreter executes in Thumb state so it must start at the even
        // address.
        let mut pc = start & !1;
        let mut steps = 0u32;
        let stop = loop {
            if steps >= self.budget {
                break StopReason::Budget { pc };
            }
            steps += 1;
            if Some(pc) == self.return_sentinel {
                break StopReason::Returned;
            }
            let d = match self.fetch(pc) {
                Some(d) => d,
                None => break StopReason::NoCode { addr: pc },
            };
            if self.trace {
                eprintln!("{pc:08x}: {:?}  r0={:08x} r1={:08x} r4={:08x} r5={:08x}", d.op, self.regs.get(0), self.regs.get(1), self.regs.get(4), self.regs.get(5));
            }
            // Condition gate (IT block or the condition baked into BCond).
            let pass = self.cond_pass(&d.op);
            let next_pc_dflt = pc.wrapping_add(d.size as u32);
            let step = if pass {
                self.regs.r[15] = pc.wrapping_add(4); // PC value seen by the insn
                self.exec(&d, pc)
            } else {
                Step::Next
            };
            match step {
                Step::Next => pc = next_pc_dflt,
                Step::Jump(t) => pc = t,
                Step::Stop(r) => break r,
            }
        };
        Run {
            writes: std::mem::take(&mut self.writes),
            mem: std::mem::take(&mut self.mem_events),
            stop,
            regs: self.regs,
            steps,
        }
    }

    /// Evaluate the current condition and advance ITSTATE.
    fn cond_pass(&mut self, op: &Op) -> bool {
        // BCond carries its own condition and is never inside an IT block.
        if let Op::BCond { cond, .. } = op {
            return self.flags.test(*cond);
        }
        if self.it_low & 0xF == 0 {
            return true; // outside an IT block
        }
        let cond_bits = (self.it_base << 1) | ((self.it_low >> 4) & 1);
        let cond = Cond::from_bits(cond_bits).unwrap_or(Cond::Al);
        let pass = self.flags.test(cond);
        // ITAdvance
        if self.it_low & 0x7 == 0 {
            self.it_base = 0;
            self.it_low = 0;
        } else {
            self.it_low = (self.it_low << 1) & 0x1F;
        }
        pass
    }

    fn set_it(&mut self, cond: Cond, mask: u8) {
        let bits = cond.bits();
        self.it_base = (bits >> 1) & 0x7;
        self.it_low = ((bits & 1) << 4) | (mask & 0xF);
    }

    fn reg(&self, i: u8) -> u32 {
        // Reads of the PC inside the executed instruction return the
        // instruction address + 4 (already stashed in r[15]).
        self.regs.get(i)
    }
    fn set(&mut self, i: u8, v: u32) {
        self.regs.set(i, v);
    }

    fn set_nz(&mut self, r: u32) {
        self.flags.n = r >> 31 != 0;
        self.flags.z = r == 0;
    }
    fn add_with_carry(&mut self, a: u32, b: u32, cin: u32) -> u32 {
        let full = a as u64 + b as u64 + cin as u64;
        let r = full as u32;
        self.flags.c = full > 0xFFFF_FFFF;
        self.flags.v = ((a ^ r) & (b ^ r) & 0x8000_0000) != 0;
        self.set_nz(r);
        r
    }
    fn sub_with_borrow(&mut self, a: u32, b: u32, bin: u32) -> u32 {
        let full = a as i64 - b as i64 - bin as i64;
        let r = full as u32;
        self.flags.c = (a as u64) >= (b as u64).wrapping_add(bin as u64);
        self.flags.v = ((a ^ b) & (a ^ r) & 0x8000_0000) != 0;
        self.set_nz(r);
        r
    }

    fn apply(&mut self, op: AluOp, a: u32, b: u32) -> Option<u32> {
        let r = match op {
            AluOp::And | AluOp::Tst => a & b,
            AluOp::Bic => a & !b,
            AluOp::Orr => a | b,
            AluOp::Orn => a | !b,
            AluOp::Eor | AluOp::Teq => a ^ b,
            AluOp::Add | AluOp::Cmn => {
                let r = self.add_with_carry(a, b, 0);
                return Some(r);
            }
            AluOp::Adc => {
                let c = self.flags.c as u32;
                let r = self.add_with_carry(a, b, c);
                return Some(r);
            }
            AluOp::Sbc => {
                let c = self.flags.c as u32;
                let r = self.sub_with_borrow(a, b, 1 - c);
                return Some(r);
            }
            AluOp::Sub | AluOp::Cmp => {
                let r = self.sub_with_borrow(a, b, 0);
                return Some(r);
            }
            AluOp::Rsb => {
                let r = self.sub_with_borrow(b, a, 0);
                return Some(r);
            }
            AluOp::Mul => a.wrapping_mul(b),
            AluOp::Lsl => {
                if b >= 32 {
                    0
                } else {
                    a.wrapping_shl(b)
                }
            }
            AluOp::Lsr => {
                if b >= 32 {
                    0
                } else {
                    a >> b
                }
            }
            AluOp::Asr => {
                if b >= 32 {
                    ((a as i32) >> 31) as u32
                } else {
                    ((a as i32) >> b) as u32
                }
            }
            AluOp::Ror => {
                if b == 0 {
                    a
                } else {
                    a.rotate_right(b % 32)
                }
            }
            AluOp::Mov => a,
            AluOp::Mvn => !a,
        };
        self.set_nz(r);
        Some(r)
    }

    fn shift(&self, kind: ShiftKind, v: u32, amount: u8) -> u32 {
        match kind {
            ShiftKind::Lsl => {
                if amount == 0 { v } else if amount >= 32 { 0 } else { v.wrapping_shl(amount as u32) }
            }
            ShiftKind::Lsr => {
                if amount == 0 { 0 } else if amount >= 32 { 0 } else { v >> amount }
            }
            ShiftKind::Asr => {
                let a = if amount == 0 { 32 } else { amount as u32 };
                if a >= 32 { ((v as i32) >> 31) as u32 } else { ((v as i32) >> a) as u32 }
            }
            ShiftKind::Ror => {
                if amount == 0 { v } else { v.rotate_right((amount as u32) % 32) }
            }
        }
    }

    fn ea(&self, rn: u8, base: MemBase) -> (u32, Option<(u8, u32)>) {
        // PC-relative bases use Align(PC,4); the instruction sets r[15]=addr+4
        // before exec, so align it here when rn == 15.
        let base_val = if rn == 15 {
            self.regs.r[15] & !3
        } else {
            self.regs.get(rn)
        };
        match base {
            MemBase::Imm(o) => (base_val.wrapping_add(o as u32), None),
            MemBase::PreImm(o) => {
                let a = base_val.wrapping_add(o as u32);
                (a, Some((rn, a)))
            }
            MemBase::PostImm(o) => {
                let a = base_val;
                (a, Some((rn, base_val.wrapping_add(o as u32))))
            }
            MemBase::RegShift(rm, sh) => {
                let off = self.regs.get(rm).wrapping_shl(sh as u32);
                (base_val.wrapping_add(off), None)
            }
        }
    }

    fn read_word(&self, ea: u32) -> Option<u32> {
        let aligned = ea & !3;
        if let Some(v) = self.shadow.get(&aligned) {
            let raw = if ea == aligned { *v } else { *v >> ((ea - aligned) * 8) };
            return Some(raw);
        }
        self.mem.read_data(ea)
    }

    fn write_word(&mut self, ea: u32, size: u32, value: u32) {
        // Merge a sub-word write into the 4-byte cell, reading the current
        // contents from shadow or underlying memory.
        let aligned = ea & !3;
        let shift = (ea - aligned) * 8;
        let mask: u32 = match size {
            1 => 0xFF,
            2 => 0xFFFF,
            _ => 0xFFFF_FFFF,
        } << shift;
        let cur = self.shadow.get(&aligned).copied().or_else(|| self.mem.read_data(aligned)).unwrap_or(0);
        let merged = (cur & !mask) | ((value << shift) & mask);
        self.shadow.insert(aligned, merged);
    }

    fn do_load(&mut self, kind: MemKind, _rt: u8, ea: u32) -> Option<u32> {
        let raw = match self.read_word(ea) {
            Some(v) => v,
            None => return None,
        };
        Some(match kind {
            MemKind::Byte => raw & 0xFF,
            MemKind::Half => raw & 0xFFFF,
            MemKind::Word => raw,
        })
    }

    fn push(&mut self, list: RegList) -> Step {
        let n = list.count() as u32 * 4;
        let mut sp = self.regs.get(13).wrapping_sub(n);
        self.regs.set(13, sp);
        for r in list.regs() {
            self.mem_events.push(Mem { addr: sp, size: 4, write: true });
            let v = self.regs.get(r);
            self.write_word(sp, 4, v);
            sp = sp.wrapping_add(4);
        }
        Step::Next
    }

    fn pop(&mut self, list: RegList) -> Step {
        let n = list.count() as u32 * 4;
        let sp0 = self.regs.get(13);
        // Load all values first, then write back registers in increasing order.
        let mut loaded: Vec<(u8, u32)> = Vec::new();
        for (i, r) in list.regs().enumerate() {
            let ea = sp0.wrapping_add(i as u32 * 4);
            match self.read_word(ea) {
                Some(v) => loaded.push((r, v)),
                None => return Step::Stop(StopReason::UnknownMemory { addr: ea }),
            }
        }
        self.regs.set(13, sp0.wrapping_add(n));
        let mut branch = None;
        for (r, v) in loaded {
            if r == 15 {
                branch = Some(v);
            } else {
                self.regs.set(r, v);
            }
        }
        match branch {
            Some(t) => {
                if self.return_sentinel == Some(t) || self.return_sentinel == Some(t & !1) {
                    return Step::Stop(StopReason::Returned);
                }
                self.branch_to(t)
            }
            None => Step::Next,
        }
    }

    fn ldm(&mut self, rn: u8, list: RegList, writeback: bool) -> Step {
        let base = self.regs.get(rn);
        let mut loaded = Vec::new();
        for (i, r) in list.regs().enumerate() {
            let ea = base.wrapping_add(i as u32 * 4);
            match self.read_word(ea) {
                Some(v) => loaded.push((r, v)),
                None => return Step::Stop(StopReason::UnknownMemory { addr: ea }),
            }
        }
        let mut branch = None;
        for (r, v) in loaded {
            if r == 15 {
                branch = Some(v);
            } else {
                self.regs.set(r, v);
            }
        }
        if writeback && !list.contains(rn) {
            self.regs.set(rn, base.wrapping_add(list.count() * 4));
        }
        match branch {
            Some(t) => {
                if self.return_sentinel == Some(t) || self.return_sentinel == Some(t & !1) {
                    return Step::Stop(StopReason::Returned);
                }
                self.branch_to(t)
            }
            None => Step::Next,
        }
    }

    fn stm(&mut self, rn: u8, list: RegList, writeback: bool) -> Step {
        let base = self.regs.get(rn);
        for (i, r) in list.regs().enumerate() {
            let ea = base.wrapping_add(i as u32 * 4);
            self.mem_events.push(Mem { addr: ea, size: 4, write: true });
            let v = self.regs.get(r);
            self.write_word(ea, 4, v);
        }
        if writeback {
            self.regs.set(rn, base.wrapping_add(list.count() * 4));
        }
        Step::Next
    }

    fn branch_to(&mut self, target: u32) -> Step {
        // BLX/BX to a register: bit0 selects state.
        if target & 1 == 0 {
            // Branching to an ARM-state address is out of scope.
            return Step::Stop(StopReason::BadState { target });
        }
        let t = target & !1;
        if self.mem.read_insn(t).is_none() {
            return Step::Stop(StopReason::BadBranch { target: t });
        }
        Step::Jump(t)
    }

    fn exec(&mut self, d: &Decoded, pc: u32) -> Step {
        let op = &d.op;
        match op {
            Op::Imm8 { op, rd, imm } => {
                match op {
                    AluOp::Mov => {
                        self.set(*rd, *imm);
                        self.set_nz(*imm);
                    }
                    AluOp::Cmp => {
                        let a = self.reg(*rd);
                        self.sub_with_borrow(a, *imm, 0);
                    }
                    AluOp::Add => {
                        let a = self.reg(*rd);
                        let r = self.add_with_carry(a, *imm, 0);
                        self.set(*rd, r);
                    }
                    AluOp::Sub => {
                        let a = self.reg(*rd);
                        let r = self.sub_with_borrow(a, *imm, 0);
                        self.set(*rd, r);
                    }
                    _ => return Step::Stop(StopReason::Unsupported { pc, hw1: 0, hw2: None }),
                }
                Step::Next
            }
            Op::AluReg { op, rd, rm } => {
                let src = self.reg(*rm);
                match op {
                    // MOV/MVN take Rm as the source, not the old Rd.
                    AluOp::Mov => {
                        self.set(*rd, src);
                        self.set_nz(src);
                    }
                    AluOp::Mvn => {
                        let r = !src;
                        self.set(*rd, r);
                        self.set_nz(r);
                    }
                    // RSB (T1) is rd = #0 - Rm.
                    AluOp::Rsb => {
                        let r = self.sub_with_borrow(0, src, 0);
                        self.set(*rd, r);
                    }
                    _ if op.is_test() => {
                        let a = self.reg(*rd);
                        self.apply(*op, a, src);
                    }
                    _ => {
                        let a = self.reg(*rd);
                        if let Some(r) = self.apply(*op, a, src) {
                            self.set(*rd, r);
                        }
                    }
                }
                Step::Next
            }
            Op::AluShiftImm { op, rd, rm, amount } => {
                let kind = match op {
                    AluOp::Lsl => ShiftKind::Lsl,
                    AluOp::Lsr => ShiftKind::Lsr,
                    AluOp::Asr => ShiftKind::Asr,
                    _ => ShiftKind::Ror,
                };
                let r = self.shift(kind, self.reg(*rm), *amount);
                self.set(*rd, r);
                self.set_nz(r);
                Step::Next
            }
            Op::ShiftReg { op, rd, rm } => {
                let amt = self.reg(*rm) & 0xFF;
                let r = self.apply_shift_reg(*op, self.reg(*rd), amt);
                self.set(*rd, r);
                Step::Next
            }
            Op::AluW { op, rd, rn, rm, shift, amount } => {
                let b = self.shift(*shift, self.reg(*rm), *amount);
                match op {
                    AluOp::Mov => {
                        self.set(*rd, b);
                        self.set_nz(b);
                    }
                    AluOp::Mvn => {
                        let r = !b;
                        self.set(*rd, r);
                        self.set_nz(r);
                    }
                    AluOp::Rsb => {
                        let r = self.sub_with_borrow(0, b, 0);
                        self.set(*rd, r);
                    }
                    _ => {
                        let a = self.reg(*rn);
                        if let Some(r) = self.apply(*op, a, b) {
                            self.set(*rd, r);
                        }
                    }
                }
                Step::Next
            }
            Op::AluImmW { op, rd, rn, imm } => {
                match op {
                    AluOp::Mov => {
                        self.set(*rd, *imm);
                        self.set_nz(*imm);
                    }
                    AluOp::Mvn => {
                        let r = !*imm;
                        self.set(*rd, r);
                        self.set_nz(r);
                    }
                    AluOp::Rsb => {
                        let r = self.sub_with_borrow(0, *imm, 0);
                        self.set(*rd, r);
                    }
                    _ => {
                        let a = self.reg(*rn);
                        if let Some(r) = self.apply(*op, a, *imm) {
                            self.set(*rd, r);
                        }
                    }
                }
                Step::Next
            }
            Op::TestW { op, rn, rm, shift, amount } => {
                let b = self.shift(*shift, self.reg(*rm), *amount);
                self.apply(*op, self.reg(*rn), b);
                Step::Next
            }
            Op::TestImmW { op, rn, imm } => {
                self.apply(*op, self.reg(*rn), *imm);
                Step::Next
            }
            Op::MovW { rd, imm } => {
                self.set(*rd, *imm as u32);
                Step::Next
            }
            Op::MovT { rd, imm } => {
                let cur = self.reg(*rd);
                let v = (cur & 0xFFFF) | ((*imm as u32) << 16);
                self.set(*rd, v);
                Step::Next
            }
            Op::Load { kind, rt, rn, base } => {
                let (ea, post) = self.ea(*rn, *base);
                self.mem_events.push(Mem { addr: ea, size: kind_size(*kind), write: false });
                match self.do_load(*kind, *rt, ea) {
                    Some(v) => {
                        self.set(*rt, v);
                        if let Some((r, nv)) = post {
                            self.set(r, nv);
                        }
                        Step::Next
                    }
                    None => Step::Stop(StopReason::UnknownMemory { addr: ea }),
                }
            }
            Op::Store { kind, rt, rn, base } => {
                let (ea, post) = self.ea(*rn, *base);
                self.mem_events.push(Mem { addr: ea, size: kind_size(*kind), write: true });
                let v = self.reg(*rt);
                self.write_word(ea, kind_size(*kind), v);
                // Record a write to an MPU register as an MPU event; the
                // engine replays these into the MPU model.
                if crate::mpu::Mpu::is_mpu_addr(ea) {
                    self.writes.push(MpuWriteEvent {
                        pc,
                        addr: ea,
                        value: v,
                        unprivileged: self.unprivileged,
                        func: self.mem.symbol_at(pc),
                    });
                }
                if let Some((r, nv)) = post {
                    self.set(r, nv);
                }
                Step::Next
            }
            Op::LdrLiteral { rt, imm12 } => {
                let pcv = self.regs.r[15]; // addr + 4
                let ea = (pcv & !3).wrapping_add(*imm12);
                self.mem_events.push(Mem { addr: ea, size: 4, write: false });
                match self.read_word(ea) {
                    Some(v) => {
                        self.set(*rt, v);
                        Step::Next
                    }
                    None => Step::Stop(StopReason::UnknownMemory { addr: ea }),
                }
            }
            Op::Adr { rd, imm } => {
                let pcv = self.regs.r[15];
                self.set(*rd, (pcv & !3).wrapping_add(*imm as u32));
                Step::Next
            }
            Op::Push(list) => self.push(*list),
            Op::Pop(list) => self.pop(*list),
            Op::Ldm { rn, list, writeback } => self.ldm(*rn, *list, *writeback),
            Op::Stm { rn, list, writeback } => self.stm(*rn, *list, *writeback),
            Op::SpAdjust { rd, imm } => {
                if let Some(rd) = rd {
                    let v = self.regs.get(13).wrapping_add(*imm as u32);
                    self.set(*rd, v);
                } else {
                    let v = self.regs.get(13).wrapping_add(*imm as u32);
                    self.regs.set(13, v);
                }
                Step::Next
            }
            Op::BCond { cond, target } => {
                if self.flags.test(*cond) {
                    Step::Jump(*target)
                } else {
                    Step::Next
                }
            }
            Op::B { target } => self.branch_to(*target | 1),
            Op::BL { target } => {
                // Link register gets the return address (this insn + 4), with
                // bit0 set so a plain `bx lr` returns to Thumb state.
                let ret = pc.wrapping_add(d.size as u32) | 1;
                if self.mem.read_insn(*target & !1).is_none() {
                    return Step::Stop(StopReason::CalledUnknown { target: *target });
                }
                self.regs.set(14, ret);
                Step::Jump(*target & !1)
            }
            Op::Bx { rm } => {
                let t = self.reg(*rm);
                // Return-to-sentinel is checked first: the caller sets LR to
                // an odd sentinel that has no backing code, exactly like a
                // top-level RTOS return.
                if self.return_sentinel == Some(t) || self.return_sentinel == Some(t & !1) {
                    return Step::Stop(StopReason::Returned);
                }
                if t & 1 == 0 {
                    Step::Stop(StopReason::BadState { target: t })
                } else {
                    let target = t & !1;
                    if self.mem.read_insn(target).is_none() {
                        Step::Stop(StopReason::CalledUnknown { target })
                    } else {
                        Step::Jump(target)
                    }
                }
            }
            Op::Blx { rm } => {
                let t = self.reg(*rm);
                let target = t & !1;
                let ret = pc.wrapping_add(d.size as u32) | 1;
                self.regs.set(14, ret);
                if self.mem.read_insn(target).is_none() {
                    Step::Stop(StopReason::CalledUnknown { target })
                } else {
                    Step::Jump(target)
                }
            }
            Op::Cbz { rn, target, nonzero } => {
                let v = self.reg(*rn);
                if (v != 0) == *nonzero {
                    Step::Jump(*target)
                } else {
                    Step::Next
                }
            }
            Op::It { cond, mask } => {
                self.set_it(*cond, *mask);
                Step::Next
            }
            Op::Mrs { rd, .. } => {
                // PRIMASK/FAULTMASK/CONTROL are all zero in our model; IPSR
                // etc. are irrelevant to MPU configuration.
                self.set(*rd, 0);
                Step::Next
            }
            Op::Msr { .. } => Step::Next,
            Op::Barrier | Op::Nop => Step::Next,
            Op::Extend { rd, rm, width, signed, rotate } => {
                let mut v = self.reg(*rm);
                if *rotate != 0 {
                    v = v.rotate_right(*rotate as u32 % 32);
                }
                let v = match (*signed, *width) {
                    (false, 8) => v & 0xFF,
                    (false, 16) => v & 0xFFFF,
                    (true, 8) => ((v as i8) as u32),
                    (true, 16) => ((v as i16) as u32),
                    _ => v,
                };
                self.set(*rd, v);
                Step::Next
            }
            Op::Rev { rd, rm, width } => {
                let v = self.reg(*rm);
                let r = match *width {
                    16 => ((v as u16).swap_bytes()) as u32 | (v & 0xFFFF_0000),
                    _ => v.swap_bytes(),
                };
                self.set(*rd, r);
                Step::Next
            }
            Op::Unsupported { hw1, hw2 } => Step::Stop(StopReason::Unsupported { pc, hw1: *hw1, hw2: *hw2 }),
            Op::Stop => Step::Stop(StopReason::Exception),
        }
    }

    fn apply_shift_reg(&mut self, op: AluOp, v: u32, amt: u32) -> u32 {
        let r = match op {
            AluOp::Lsl => {
                if amt >= 32 { 0 } else { v.wrapping_shl(amt) }
            }
            AluOp::Lsr => {
                if amt >= 32 { 0 } else { v >> amt }
            }
            AluOp::Asr => {
                if amt >= 32 { (v as i32 >> 31) as u32 } else { (v as i32 >> amt) as u32 }
            }
            AluOp::Ror => {
                if amt == 0 { v } else { v.rotate_right(amt % 32) }
            }
            _ => v,
        };
        self.set_nz(r);
        r
    }
}

fn kind_size(k: MemKind) -> u32 {
    match k {
        MemKind::Byte => 1,
        MemKind::Half => 2,
        MemKind::Word => 4,
    }
}
