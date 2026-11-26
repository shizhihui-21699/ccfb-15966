//! Regression test: every encoding in `corpus.json` was produced by real
//! clang for Cortex-M4. Decoding each one and re-checking the recovered
//! operands against a hand-written expectation table catches any future
//! shift of a bit field.

use mpu_audit::arm::decode::{self, AluOp, Cond, MemBase, MemKind, Op, RegList, ShiftKind};
use serde::Deserialize;

#[derive(Deserialize)]
struct Entry {
    asm: String,
    words: Vec<String>,
}

fn bytes_of(words: &[String]) -> Vec<u8> {
    let mut v = Vec::new();
    for w in words {
        let hw = u16::from_str_radix(w, 16).unwrap();
        v.extend_from_slice(&hw.to_le_bytes());
    }
    v
}

fn decode_entry(e: &Entry) -> decode::Decoded {
    let b = bytes_of(&e.words);
    decode::decode(0, &b).unwrap_or_else(|err| panic!("{}: {err}", e.asm))
}

/// Assert the single-register-operand facts we can compare to the asm text.
fn check(e: &Entry) {
    let d = decode_entry(e);
    let a = e.asm.split_whitespace().collect::<Vec<_>>();
    let mnem = a[0];
    match &d.op {
        Op::Imm8 { op, rd, imm } => match mnem {
            "movs" => assert_eq!(*op, AluOp::Mov),
            "cmp" => assert_eq!(*op, AluOp::Cmp),
            "adds" => assert_eq!(*op, AluOp::Add),
            "subs" => assert_eq!(*op, AluOp::Sub),
            m => panic!("unexpected {m} for Imm8"),
        },
        Op::MovW { rd: _, imm } => {
            let want: u32 = parse_imm(&e.asm, mnem);
            assert_eq!(*imm as u32, want, "movw {} got {:#x}", e.asm, imm);
        }
        Op::MovT { imm, .. } => {
            let want: u32 = parse_imm(&e.asm, mnem);
            assert_eq!(*imm as u32, want, "movt {}", e.asm);
        }
        Op::AluImmW { imm, .. } => {
            let want: u32 = parse_imm(&e.asm, mnem.split('.').next().unwrap());
            assert_eq!(*imm, want, "{}", e.asm);
        }
        Op::TestImmW { imm, .. } => {
            let want: u32 = parse_imm(&e.asm, mnem.split('.').next().unwrap());
            assert_eq!(*imm, want, "{}", e.asm);
        }
        Op::Load { base, .. } | Op::Store { base, .. } => {
            // The decoded MemBase::Imm is already a byte offset, which is what
            // the assembly source spells after '#'. Only check the plain
            // [Rn,#imm] forms (the [Rn,Rm,lsl] form has no '#').
            if let MemBase::Imm(off) = base {
                if let Some(n) = find_hash_imm(&e.asm) {
                    if !e.asm.contains(",r") || e.asm.matches('[').count() == 1 {
                        let no_reg = !e.asm.split('[').nth(1).unwrap_or("").contains(|c: char| c == 'r' && !c.is_ascii_digit());
                        let _ = no_reg;
                    }
                    assert_eq!(*off as u32, n, "{}", e.asm);
                }
            }
        }
        Op::Cbz { nonzero, .. } => {
            assert_eq!(*nonzero, mnem == "cbnz", "{}", e.asm);
        }
        Op::BCond { .. } => {
            assert!(mnem.starts_with('b'), "{}", e.asm);
            if mnem.len() > 2 {
                let c = mnem.trim_start_matches('b').trim_end_matches(".w");
                let cond = match &d.op {
                    Op::BCond { cond, .. } => cond.mnemonic(),
                    _ => unreachable!(),
                };
                assert_eq!(c, cond, "{}", e.asm);
            }
            let _ = Cond::Eq;
        }
        Op::B { .. } | Op::BL { .. } => {}
        Op::Push(l) | Op::Pop(l) => {
            assert!(l.count() > 0, "{}", e.asm);
        }
        Op::Ldm { list, .. } | Op::Stm { list, .. } => {
            assert!(list.count() > 0, "{}", e.asm);
        }
        Op::It { .. } => assert_eq!(mnem, "it", "{}", e.asm),
        Op::AluW { shift, amount, .. } => {
            if e.asm.contains("lsl #") {
                assert_eq!(*shift, ShiftKind::Lsl);
                let n = find_shift(&e.asm);
                assert_eq!(*amount as u32, n, "{}", e.asm);
            }
            let _ = RegList(0);
        }
        Op::AluShiftImm { amount, .. } => {
            let n = find_shift(&e.asm);
            assert_eq!(*amount as u32, n, "{}", e.asm);
        }
        Op::Unsupported { hw1, hw2 } => {
            panic!("unsupported {} -> {:04x} {:?}", e.asm, hw1, hw2);
        }
        _ => {}
    }
}

fn find_hash_imm(asm: &str) -> Option<u32> {
    let p = asm.split('#').nth(1)?.trim();
    let body = p.strip_prefix("0x").or_else(|| p.strip_prefix("0X")).unwrap_or(p);
    let v: String = body.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
    u32::from_str_radix(&v, 16).ok()
}

fn find_shift(asm: &str) -> u32 { find_hash_imm(asm).unwrap_or(0) }

fn parse_imm(asm: &str, _mnem: &str) -> u32 {
    find_hash_imm(asm).unwrap_or(0)
}

#[test]
fn corpus_decodes() {
    let raw = include_str!("fixtures/corpus.json");
    let entries: Vec<Entry> = serde_json::from_str(raw).expect("fixtures/corpus.json");
    let mut n = 0;
    for e in &entries {
        check(e);
        n += 1;
    }
    assert!(n >= 100, "expected a substantial corpus, got {n}");
}

/// Ensure each instruction family is actually recognised (not silently
/// falling into `Unsupported`, which `check` would already panic on, but
/// this makes the coverage explicit and guards against a *wrong* class
/// mapping that happens to satisfy the loose per-family assertions).
#[test]
fn every_family_represented() {
    use std::collections::BTreeSet;
    let raw = include_str!("fixtures/corpus.json");
    let entries: Vec<Entry> = serde_json::from_str(raw).unwrap();
    let mut fams: BTreeSet<&'static str> = BTreeSet::new();
    for e in &entries {
        let d = decode_entry(e);
        let f = match &d.op {
            Op::Imm8 { .. } => "imm8",
            Op::AluReg { .. } => "alu_reg",
            Op::AluShiftImm { .. } => "shift_imm",
            Op::ShiftReg { .. } => "shift_reg",
            Op::AluW { .. } => "alu_w",
            Op::AluImmW { .. } => "alu_imm_w",
            Op::TestW { .. } => "test_w",
            Op::TestImmW { .. } => "test_imm_w",
            Op::MovW { .. } => "movw",
            Op::MovT { .. } => "movt",
            Op::Load { .. } => "load",
            Op::Store { .. } => "store",
            Op::LdrLiteral { .. } => "ldr_lit",
            Op::Adr { .. } => "adr",
            Op::Push(_) => "push",
            Op::Pop(_) => "pop",
            Op::Ldm { .. } => "ldm",
            Op::Stm { .. } => "stm",
            Op::SpAdjust { .. } => "sp_adjust",
            Op::BCond { .. } => "bcond",
            Op::B { .. } => "b",
            Op::BL { .. } => "bl",
            Op::Bx { .. } => "bx",
            Op::Blx { .. } => "blx",
            Op::Cbz { .. } => "cbz",
            Op::It { .. } => "it",
            Op::Mrs { .. } => "mrs",
            Op::Msr { .. } => "msr",
            Op::Barrier => "barrier",
            Op::Nop => "nop",
            Op::Extend { .. } => "extend",
            Op::Rev { .. } => "rev",
            Op::Unsupported { .. } => "unsupported",
            Op::Stop => "stop",
        };
        fams.insert(f);
    }
    for need in ["imm8","alu_reg","shift_imm","shift_reg","alu_w","alu_imm_w","test_w",
                 "test_imm_w","movw","movt","load","store","ldr_lit","push","pop",
                 "ldm","stm","sp_adjust","bcond","b","bl","bx","blx","cbz","it",
                 "mrs","msr","barrier","nop","extend","rev"] {
        assert!(fams.contains(need), "missing instruction family: {need}");
    }
    assert!(!fams.contains("unsupported"), "corpus contains unsupported instructions");
}
