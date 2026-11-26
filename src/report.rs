//! Human-readable and JSON rendering of audit reports.

use anyhow::Result;

use crate::engine::{Finding, PermissionView, Report, Segment, Severity, TaskReport};
use crate::symbol::FirmwareImage;

pub fn to_json(report: &Report) -> Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}

pub fn print_human(report: &Report) {
    println!("MPU audit for {}", report.firmware);
    println!("entry point: {:#010x}", report.entry);
    println!();

    if !report.warnings.is_empty() {
        println!("Warnings:");
        for w in &report.warnings {
            println!("  ! {w}");
        }
        println!();
    }

    for task in &report.tasks {
        print_task(task);
    }

    // Summary of findings across all tasks.
    let all: Vec<&Finding> = report
        .tasks
        .iter()
        .flat_map(|t| t.findings.iter())
        .collect();
    println!("{}", "=".repeat(78));
    let crit = all.iter().filter(|f| f.severity == Severity::Critical).count();
    let high = all.iter().filter(|f| f.severity == Severity::High).count();
    let med = all.iter().filter(|f| f.severity == Severity::Medium).count();
    let low = all.iter().filter(|f| f.severity == Severity::Low).count();
    println!("Findings: {} critical, {} high, {} medium, {} low", crit, high, med, low);
    for f in all.iter().filter(|f| f.severity.rank() >= Severity::High.rank()) {
        println!(
            "  [{}] {} {}{}",
            f.severity.label(),
            f.code,
            f.task.as_ref().map(|t| format!("(task {t}) ")).unwrap_or_default(),
            f.message
        );
    }
    if !report.simulated_ok {
        println!();
        println!(
            "NOTE: at least one task's boot/switch path did not finish cleanly; see the per-task \
             simulation status above. The permission map is the state reached when simulation stopped."
        );
    }
}

fn print_task(task: &TaskReport) {
    println!("Task: {}", task.name);
    if let Some(sw) = &task.switch_entry {
        println!("  switch entry: {sw}");
    }
    if let Some(s) = &task.boot_stop {
        println!("  boot path:   {s}");
    }
    if let Some(s) = &task.switch_stop {
        println!("  switch path: {s}");
    }
    let snap = &task.mpu_final;
    println!(
        "  MPU_CTRL = {} (ENABLE={}, PRIVDEFENA={}, HFNMIENA={})",
        snap.ctrl, snap.enabled as u8, snap.privdefena as u8, snap.hfnmiena as u8
    );

    println!("  Regions:");
    for r in &task.regions {
        if !r.enabled && r.size == 0 {
            continue;
        }
        let ap = ap_name(r.ap);
        let xn = if r.xn { "XN" } else { "  " };
        print!(
            "    [{}] {}..{} size={} AP={}{}{}{}",
            r.number,
            r.base,
            r.end,
            human_size(r.size),
            ap,
            if r.xn { " XN" } else { "" },
            if r.misaligned { " (base aligned down!)" } else { "" },
            if r.enabled { "" } else { " (disabled)" },
        );
        let _ = xn;
        if !r.holes.is_empty() {
            print!(" holes:");
            for h in &r.holes {
                print!(" {}..{}", h.start, h.end);
            }
        }
        println!();
    }

    println!();
    println!("  Effective permission map (unprivileged view):");
    println!("    {:>10}  {:>10}  {:<6} {:<26} {}", "start", "end", "perm", "storage", "source");
    for seg in &task.map {
        // Only print segments that are inside a "known" storage class or
        // inside a region (skip vast unmapped gaps for readability).
        if seg.storage == crate::symbol::Storage::Unknown && seg.view.source == "fault" {
            continue;
        }
        println!(
            "    {:>#10x}  {:>#10x}  {:<6} {:<26} {}",
            seg.range.start,
            seg.range.end,
            seg.view.unprivileged,
            format!("{:?}", seg.storage),
            seg.view.source,
        );
    }

    if !task.findings.is_empty() {
        println!();
        println!("  Findings:");
        for f in &task.findings {
            println!("    [{}] {}: {}", f.severity.label(), f.code, f.message);
            for e in &f.evidence {
                println!("        - {e}");
            }
        }
    }
    println!();
}

fn human_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{}MB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{}KB", bytes / 1024)
    } else {
        format!("{bytes}B")
    }
}

fn ap_name(ap: u8) -> String {
    match ap {
        0b000 => "--- (none)".into(),
        0b001 => "priv RW".into(),
        0b010 => "priv RW / unpriv RO".into(),
        0b011 => "RW (full)".into(),
        0b100 => "reserved".into(),
        0b101 => "priv RO".into(),
        0b110 => "RO".into(),
        0b111 => "RO".into(),
        _ => "?".into(),
    }
}

pub fn print_inspect(image: &FirmwareImage) {
    println!("ELF: {}", image.path);
    println!("entry: {:#010x}", image.entry);
    println!();
    println!("Sections:");
    for s in &image.sections {
        println!(
            "  {:<20} {:#010x}..{:#010x}  {}  {}",
            s.name,
            s.addr,
            s.addr.wrapping_add(s.size),
            human_size(s.size as u64),
            match (s.flags.exec, s.flags.write) {
                (true, _) => "exec",
                (false, true) => "rw",
                (false, false) => "ro",
            }
        );
    }
    println!();
    println!("Symbols (with inferred sizes):");
    let mut syms: Vec<_> = image.symbols.iter().collect();
    syms.sort_by_key(|s| s.addr);
    for s in syms {
        if s.size == 0 {
            continue;
        }
        println!(
            "  {:<28} {:#010x} size={:<8} {:?}",
            s.name,
            s.addr,
            human_size(s.size as u64),
            s.storage
        );
    }
}

/// Render a permission view for compact display.
pub fn _view(v: &PermissionView) -> String {
    format!("priv={} unpriv={} ({})", v.privileged, v.unprivileged, v.source)
}

/// Render a segment list as rows, for tests and debugging.
pub fn _segments(segs: &[Segment]) -> Vec<(u32, u32, String, String)> {
    segs.iter()
        .map(|s| (s.range.start, s.range.end, s.view.unprivileged.clone(), s.view.source.clone()))
        .collect()
}
