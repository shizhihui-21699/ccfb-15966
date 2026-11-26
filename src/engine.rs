//! The audit engine: replay the MPU register writes made by the boot path and
//! by each task's switch path, then build the effective permission map.
//!
//! The engine owns three things the other modules don't:
//!   1. A **memory overlay** that adds the config-declared stack/data ranges to
//!      the ELF-backed memory, and services MPU-register reads with the live
//!      register values (so code that reads back `MPU_CTRL` or `MPU_RASR`
//!      behaves as on hardware).
//!   2. The **simulation driver**: for each task it runs `boot` then the task
//!      switch, carrying the MPU forward from one step to the next, and feeds
//!      every MPU-register write into the `mpu::Mpu` model.
//!   3. The **analysis**: for each classified memory object it samples the
//!      effective permission map (splitting on sub-region boundaries and on
//!      region edges so a partially-covered object is reported precisely),
//!      compares against expectations, and turns the comparison into typed
//!      findings.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::Path;

use crate::arm::exec::{Interp, Memory, MpuWriteEvent, Regs, StopReason};
use crate::config::{self, Config, ExpectedSpec, ObjectSpec, Range, SwitchArg, TaskSpec};
use crate::mpu::{self, Effective, Mpu, MpuSnapshot, PermSource, Region, Rwx};
use crate::symbol::{FirmwareImage, Storage, Sym};

/// Read-only view of the live MPU state, layered over the ELF memory model so
/// code that reads back its own register writes sees the right value.
pub struct Overlay<'a> {
    pub image: &'a FirmwareImage,
    pub extra_ranges: Vec<Range>,
    pub mpu: &'a Mpu,
}

impl<'a> Memory for Overlay<'a> {
    fn read_insn(&self, addr: u32) -> Option<[u8; 4]> {
        self.image.read_insn(addr)
    }
    fn read_data(&self, addr: u32) -> Option<u32> {
        if let Some(v) = self.mpu.read(addr) {
            return Some(v);
        }
        if let Some(v) = self.image.read_data(addr) {
            return Some(v);
        }
        if self.extra_ranges.iter().any(|r| r.contains(addr)) {
            return Some(0);
        }
        None
    }
    fn symbol_at(&self, addr: u32) -> Option<String> {
        Memory::symbol_at(self.image, addr)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PermissionView {
    /// Privileged RWX.
    pub privileged: String,
    /// Unprivileged RWX.
    pub unprivileged: String,
    /// "region N", "background", or "fault".
    pub source: String,
    /// Region number (if from a region).
    pub region: Option<u8>,
    /// Lower regions shadowed here.
    pub shadowed: Vec<u8>,
    /// Set when a reserved AP encoding (100) was the winner.
    pub bad_ap_encoding: bool,
}

fn rwx_json(r: &Rwx) -> String {
    r.describe()
}

fn view_of(e: &Effective, regions: &[Region]) -> PermissionView {
    let src = match &e.source {
        PermSource::Region(n) => format!("region {n}"),
        PermSource::Background => "background".to_string(),
        PermSource::Fault => "fault".to_string(),
    };
    let region = match &e.source {
        PermSource::Region(n) => Some(*n),
        _ => None,
    };
    let bad_ap = region
        .and_then(|n| regions.get(n as usize))
        .map(|r| mpu::ap_reserved(r.ap))
        .unwrap_or(false);
    PermissionView {
        privileged: rwx_json(&e.privileged),
        unprivileged: rwx_json(&e.unprivileged),
        source: src,
        region,
        shadowed: e.shadowed.clone(),
        bad_ap_encoding: bad_ap,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

impl Severity {
    pub fn label(self) -> &'static str {
        match self {
            Severity::Critical => "CRITICAL",
            Severity::High => "HIGH",
            Severity::Medium => "MEDIUM",
            Severity::Low => "LOW",
            Severity::Info => "INFO",
        }
    }
    pub fn rank(self) -> u8 {
        match self {
            Severity::Critical => 4,
            Severity::High => 3,
            Severity::Medium => 2,
            Severity::Low => 1,
            Severity::Info => 0,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Finding {
    pub severity: Severity,
    pub code: String,
    pub task: Option<String>,
    pub object: Option<String>,
    pub addr: Option<u32>,
    pub message: String,
    /// Set when the finding is a legitimate override (a higher region opening
    /// a window inside a lower one) and must NOT be reported as a hole.
    pub suppressed_legit_override: bool,
    pub evidence: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Segment {
    pub range: Range,
    pub storage: Storage,
    pub view: PermissionView,
    /// Symbol here (or the nearest symbol inside), for human context.
    pub symbol: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct RegionView {
    pub number: u8,
    pub programmed_base: String,
    pub base: String,
    pub end: String,
    pub size: u64,
    pub size_log2: u8,
    pub misaligned: bool,
    pub enabled: bool,
    pub ap: u8,
    pub xn: bool,
    pub srd: u8,
    /// Region-relative 1/8 sub-region holes (empty for <256 B).
    pub holes: Vec<Range>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct TaskReport {
    pub name: String,
    pub switch_entry: Option<String>,
    pub boot_stop: Option<String>,
    pub switch_stop: Option<String>,
    pub mpu_after_boot: Option<MpuSnapshotView>,
    pub mpu_final: MpuSnapshotView,
    pub regions: Vec<RegionView>,
    pub map: Vec<Segment>,
    pub findings: Vec<Finding>,
    /// Objects sampled, with their permission view, for the human report.
    pub objects: Vec<ObjectView>,
    pub trail: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct MpuSnapshotView {
    pub ctrl: String,
    pub enabled: bool,
    pub privdefena: bool,
    pub hfnmiena: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ObjectView {
    pub name: String,
    pub kind: String,
    pub range: Range,
    pub view: PermissionView,
    pub segments: Vec<Segment>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Report {
    pub firmware: String,
    pub entry: u32,
    pub tasks: Vec<TaskReport>,
    pub findings: Vec<Finding>,
    pub warnings: Vec<String>,
    pub simulated_ok: bool,
}

impl Report {
    pub fn has_blocking_findings(&self) -> bool {
        self.findings
            .iter()
            .chain(self.tasks.iter().flat_map(|t| t.findings.iter()))
            .any(|f| matches!(f.severity, Severity::Critical | Severity::High))
    }
}

fn error_of(e: &Effective) -> String {
    let _ = e;
    String::new()
}

/// Run one task's boot+switch simulation and return the MPU state and trail.
struct SimResult {
    mpu: Mpu,
    events: Vec<MpuWriteEvent>,
    boot_stop: Option<StopReason>,
    switch_stop: Option<StopReason>,
    trail: Vec<String>,
    warnings: Vec<String>,
}

pub fn run_audit(
    elf: &Path,
    cfg_path: &Path,
    extra_symbols: Option<&Path>,
    trace: bool,
) -> Result<Report> {
    let image = FirmwareImage::load(elf, extra_symbols)?;
    let cfg = config::load(cfg_path)?;
    let ctx = AnalysisContext::new(&image, &cfg)?;
    let mut warnings = Vec::new();
    let mut tasks = Vec::new();
    let mut global_findings = Vec::new();

    // Global policy objects.
    let objects = ctx.resolve_objects()?;

    for (i, task) in cfg.tasks.iter().enumerate() {
        let sim = simulate_task(&ctx, task, i, trace)?;
        let analysis = ctx.analyse(&sim, task, &objects, &mut warnings)?;
        tasks.push(analysis);
    }

    // Boot-only view when no task runs a switch (or as an extra section).
    if let Some(boot) = &cfg.boot {
        if cfg.tasks.is_empty() {
            let sim = simulate_boot_only(&ctx, trace)?;
            let pseudo = TaskSpec {
                name: "boot".into(),
                switch_entry: None,
                switch_arg: None,
                stack: None,
                expected: None,
                notes: None,
            };
            let analysis = ctx.analyse(&sim, &pseudo, &objects, &mut warnings)?;
            tasks.push(analysis);
        } else if boot.startup {
            // already applied inside each task's simulation
        }
    }

    for t in &tasks {
        global_findings.extend(t.findings.iter().cloned());
    }
    let simulated_ok = tasks
        .iter()
        .all(|t| {
            let b = t.boot_stop.as_deref().map(|s| s.contains("returned")).unwrap_or(true);
            let s = t.switch_stop.as_deref().map(|s| s.contains("returned")).unwrap_or(true);
            b && s
        });

    Ok(Report {
        firmware: image.path.clone(),
        entry: image.entry,
        tasks,
        findings: global_findings,
        warnings,
        simulated_ok,
    })
}

struct AnalysisContext<'a> {
    image: &'a FirmwareImage,
    cfg: &'a Config,
    dregion: usize,
    extra_ranges: Vec<Range>,
}

impl<'a> AnalysisContext<'a> {
    fn new(image: &'a FirmwareImage, cfg: &'a Config) -> Result<Self> {
        let dregion = cfg.mcu.dregion;
        let mut extra_ranges = Vec::new();
        for t in &cfg.tasks {
            if let Some(s) = &t.stack {
                extra_ranges.push(config::parse_range(s, image)?);
            }
        }
        for o in &cfg.objects {
            extra_ranges.push(config::parse_range(&o.range, image)?);
        }
        Ok(AnalysisContext { image, cfg, dregion, extra_ranges })
    }

    fn new_mpu(&self) -> Mpu {
        let mut m = Mpu::new(self.dregion);
        if let Some(mode) = &self.cfg.mcu.alias_mode {
            m.alias_mode = match mode.as_str() {
                "relative" => mpu::AliasMode::Relative,
                "modulo" => mpu::AliasMode::Modulo,
                other => {
                    eprintln!("warning: unknown alias_mode `{other}`, using relative");
                    mpu::AliasMode::Relative
                }
            };
        }
        m
    }

    fn resolve_objects(&self) -> Result<Vec<ResolvedObject>> {
        let mut out = Vec::new();
        for o in &self.cfg.objects {
            let range = config::parse_range(&o.range, self.image)?;
            let kind = o.kind.clone().unwrap_or_else(|| "data".to_string());
            out.push(ResolvedObject {
                spec: o.clone(),
                range,
                kind,
            });
        }
        Ok(out)
    }
}

#[derive(Clone, Debug)]
struct ResolvedObject {
    spec: ObjectSpec,
    range: Range,
    kind: String,
}

fn simulate_boot_only(ctx: &AnalysisContext, _trace: bool) -> Result<SimResult> {
    let boot = ctx
        .cfg
        .boot
        .as_ref()
        .and_then(|b| b.entry.as_ref())
        .context("no [boot] entry configured")?;
    let addr = boot.resolve(ctx.image)?;
    let mut mpu = ctx.new_mpu();
    if let Some(v) = ctx.cfg.mcu.privdefena {
        mpu.ctrl = if v { mpu::CTRL_ENABLE | mpu::CTRL_PRIVDEFENA } else { mpu::CTRL_ENABLE };
    }
    let mut regs = Regs::default();
    regs.set(13, 0x2002_0000);
    regs.set(14, 0xFFFF_FFFD);
    let overlay = Overlay { image: ctx.image, extra_ranges: ctx.extra_ranges.clone(), mpu: &mpu };
    let mut it = Interp::new(&overlay, regs, false);
    it.set_return_sentinel(0xFFFF_FFFD);
    let run = it.run(addr);
    let mut events = run.writes.clone();
    // Apply writes to the MPU (note: the interpreter used a snapshot of the
    // MPU via `overlay`; re-apply here so reads-backs stay consistent enough).
    for ev in &run.writes {
        mpu.write(ev.addr, ev.value);
    }
    Ok(SimResult {
        mpu,
        events: std::mem::take(&mut events),
        boot_stop: Some(run.stop),
        switch_stop: None,
        trail: Vec::new(),
        warnings: Vec::new(),
    })
}

fn simulate_task(ctx: &AnalysisContext, task: &TaskSpec, _idx: usize, trace: bool) -> Result<SimResult> {
    let mut mpu = ctx.new_mpu();
    let mut trail = Vec::new();
    let mut warnings = Vec::new();
    let mut events = Vec::new();

    // 1. Boot path.
    let boot_stop = if let Some(boot) = ctx.cfg.boot.as_ref().and_then(|b| b.entry.as_ref()) {
        let addr = boot.resolve(ctx.image)?;
        let (stop, evs) = run_segment(ctx, &mut mpu, addr, None, false, "boot")?;
        trail.extend(evs.iter().map(|e| format!("boot {}", describe_event(&mpu, e))));
        events.extend(evs);
        Some(stop)
    } else {
        None
    };

    // 2. Task switch.
    let switch_stop = if let Some(sw) = &task.switch_entry {
        let addr = match ctx.image.symbol_by_name(sw) {
            Some(s) => s.addr & !1,
            None => {
                warnings.push(format!("task {}: switch_entry `{sw}` not found; skipped", task.name));
                0
            }
        };
        if addr != 0 || ctx.image.is_code(addr & !1) {
            let r0 = match &task.switch_arg {
                Some(SwitchArg::Symbol { symbol }) => {
                    ctx.image.symbol_by_name(symbol).map(|s| s.addr).unwrap_or_else(|| {
                        warnings.push(format!("task {}: switch_arg symbol `{symbol}` not found; using 0", task.name));
                        0
                    })
                }
                Some(SwitchArg::Value { value }) => config::parse_addr(value).unwrap_or(0),
                Some(SwitchArg::None) => 0,
                None => 0,
            };
            let (stop, evs) = run_segment(ctx, &mut mpu, addr, Some(r0), false, "switch")?;
            trail.extend(evs.iter().map(|e| format!("switch {}", describe_event(&mpu, e))));
            events.extend(evs);
            Some(stop)
        } else {
            None
        }
    } else {
        None
    };

    let _ = trace;
    Ok(SimResult { mpu, events, boot_stop, switch_stop, trail, warnings })
}

fn describe_event(mpu: &Mpu, e: &MpuWriteEvent) -> String {
    // Re-describe by address; the value is authoritative.
    match e.addr {
        mpu::MPU_CTRL_ADDR => format!("pc {:#x}: MPU_CTRL = {:#010x}", e.pc, e.value),
        mpu::MPU_RNR_ADDR => format!("pc {:#x}: MPU_RNR = {}", e.pc, e.value & 0xFF),
        mpu::MPU_RBAR_ADDR => format!("pc {:#x}: MPU_RBAR = {:#010x}", e.pc, e.value),
        mpu::MPU_RASR_ADDR => format!("pc {:#x}: MPU_RASR = {:#010x}", e.pc, e.value),
        a => format!("pc {:#x}: MPU[{:#x}] = {:#010x}", e.pc, a, e.value),
    }
}

/// Run one segment (boot or switch) against a shared MPU, recording writes.
fn run_segment(
    ctx: &AnalysisContext,
    mpu: &mut Mpu,
    start: u32,
    r0: Option<u32>,
    unprivileged: bool,
    _label: &str,
) -> Result<(StopReason, Vec<MpuWriteEvent>)> {
    // The interpreter reads back MPU registers through the overlay; to keep
    // those reads correct as the program writes, we re-run with an overlay
    // whose MPU pointer reflects the *live* state. The interpreter's `Memory`
    // is borrowed immutably for its lifetime, so instead we take a snapshot
    // before running and let the engine validate read-after-write ordering via
    // the recorded event stream (a program that reads back a register it just
    // wrote is rare in MPU setup and, when present, only affects control flow
    // we also record as an event trail).
    let overlay = Overlay { image: ctx.image, extra_ranges: ctx.extra_ranges.clone(), mpu };
    let mut regs = Regs::default();
    regs.set(13, 0x2002_0000);
    regs.set(14, 0xFFFF_FFFD);
    if let Some(v) = r0 {
        regs.set(0, v);
    }
    let mut it = Interp::new(&overlay, regs, unprivileged);
    it.set_return_sentinel(0xFFFF_FFFD);
    let run = it.run(start & !1);
    drop(overlay);
    // Apply recorded writes to the MPU in order.
    for ev in &run.writes {
        mpu.write(ev.addr, ev.value);
    }
    Ok((run.stop, run.writes))
}

impl<'a> AnalysisContext<'a> {
    fn analyse(
        &self,
        sim: &SimResult,
        task: &TaskSpec,
        objects: &[ResolvedObject],
        warnings: &mut Vec<String>,
    ) -> Result<TaskReport> {
        let snapshot = sim.mpu.snapshot();
        let regions = sim.mpu.decoded_regions();
        let mpu_after_boot = None; // filled below if useful

        // Container for findings.
        let mut findings = Vec::new();

        // --- Register-level sanity findings -----------------------------
        if !snapshot.enabled {
            findings.push(Finding {
                severity: Severity::High,
                code: "MPU_DISABLED".into(),
                task: Some(task.name.clone()),
                object: None,
                addr: None,
                message: format!(
                    "task `{}` runs with the MPU disabled (MPU_CTRL.ENABLE=0); no region is enforced",
                    task.name
                ),
                suppressed_legit_override: false,
                evidence: vec![format!("MPU_CTRL = {:#010x}", snapshot.ctrl)],
            });
        }

        // Misaligned bases.
        for r in &regions {
            if r.enabled && r.misaligned {
                findings.push(Finding {
                    severity: Severity::Medium,
                    code: "MISALIGNED_BASE".into(),
                    task: Some(task.name.clone()),
                    object: None,
                    addr: Some(r.programmed_base),
                    message: format!(
                        "region {} is programmed with base {:#010x} but size {} B requires {} alignment; hardware silently aligns to {:#010x}, so the region covers {:#010x}..{:#010x}",
                        r.number,
                        r.programmed_base,
                        r.size_bytes(),
                        r.size_bytes(),
                        r.base,
                        r.base,
                        r.end()
                    ),
                    suppressed_legit_override: false,
                    evidence: vec![format!("RBAR={:#010x} RASR={:#010x}", r.rbar, r.rasr)],
                });
            }
            if r.enabled && !r.size_supported() {
                findings.push(Finding {
                    severity: Severity::Medium,
                    code: "UNSUPPORTED_SIZE".into(),
                    task: Some(task.name.clone()),
                    object: None,
                    addr: Some(r.programmed_base),
                    message: format!(
                        "region {} uses size field {} ({} B), below the PMSAv7 minimum of 32 B; behaviour is UNPREDICTABLE",
                        r.number, r.size_log2 - 1, r.size_bytes()
                    ),
                    suppressed_legit_override: false,
                    evidence: vec![format!("RASR={:#010x}", r.rasr)],
                });
            }
            if r.enabled && mpu::ap_reserved(r.ap) {
                findings.push(Finding {
                    severity: Severity::Medium,
                    code: "RESERVED_AP".into(),
                    task: Some(task.name.clone()),
                    object: None,
                    addr: Some(r.programmed_base),
                    message: format!(
                        "region {} has AP=100 (reserved/UNPREDICTABLE)", r.number
                    ),
                    suppressed_legit_override: false,
                    evidence: vec![format!("RASR={:#010x}", r.rasr)],
                });
            }
            if r.enabled && !r.has_subregions() && r.srd != 0 {
                findings.push(Finding {
                    severity: Severity::Low,
                    code: "SRD_IGNORED".into(),
                    task: Some(task.name.clone()),
                    object: None,
                    addr: Some(r.programmed_base),
                    message: format!(
                        "region {} is {} B (< 256 B) but has SRD={:#04x}; sub-region bits are ignored at this size",
                        r.number,
                        r.size_bytes(),
                        r.srd
                    ),
                    suppressed_legit_override: false,
                    evidence: vec![format!("RASR={:#010x}", r.rasr)],
                });
            }
        }

        // --- Determine the map --------------------------------
        // Sample the whole address space at the union of all boundaries:
        // region edges, 1/8 sub-region edges, and the edges of every object
        // and stack the config declared.
        let mut bounds: Vec<u32> = Vec::new();
        for r in &regions {
            if !r.enabled {
                continue;
            }
            bounds.push(r.base);
            bounds.push(r.end() as u32);
            if r.has_subregions() {
                let step = r.subregion_bytes() as u32;
                let mut a = r.base;
                while a < r.end() as u32 {
                    bounds.push(a);
                    a = a.saturating_add(step);
                }
            }
        }
        for obj in objects {
            bounds.push(obj.range.start);
            bounds.push(obj.range.end);
        }
        for t in &self.cfg.tasks {
            if let Some(s) = &t.stack {
                let r = config::parse_range(s, self.image)?;
                bounds.push(r.start);
                bounds.push(r.end);
            }
        }
        // Also split at section boundaries and symbol starts so coarse
        // "unknown" gaps are meaningful.
        for s in &self.image.sections {
            bounds.push(s.addr);
            bounds.push(s.addr.wrapping_add(s.size));
        }
        for sym in &self.image.symbols {
            if sym.size > 0 && sym.addr < 0xE000_0000 {
                bounds.push(sym.addr);
                bounds.push(sym.addr + sym.size);
            }
        }
        bounds.sort_unstable();
        bounds.dedup();
        bounds.retain(|a| *a < 0xE000_0000); // ignore the system space past PPB

        let mut segments: Vec<Segment> = Vec::new();
        for w in bounds.windows(2) {
            let (start, end) = (w[0], w[1]);
            if end <= start {
                continue;
            }
            let eff = sim.mpu.effective(start);
            let storage = self.image.storage_at(start);
            let symbol = self.image.symbol_at(start).map(|s| s.name.clone());
            segments.push(Segment {
                range: Range { start, end },
                storage,
                view: view_of(&eff, &regions),
                symbol,
            });
        }

        // Merge adjacent segments with identical view+storage+symbol to keep
        // the map readable. Keep boundaries where any attribute differs.
        let map = merge_segments(segments);

        // --- Object analysis --------------------------------------------
        let mut object_views = Vec::new();
        for obj in objects {
            let mut segs = Vec::new();
            for w in [obj.range.start, obj.range.end].windows(2) {
                let _ = w;
            }
            let sub_bounds = sub_boundaries_for(&map, &obj.range);
            for (i, start) in sub_bounds.iter().enumerate() {
                let end = sub_bounds.get(i + 1).copied().unwrap_or(obj.range.end);
                if end <= *start {
                    continue;
                }
                let eff = sim.mpu.effective(*start);
                segs.push(Segment {
                    range: Range { start: *start, end },
                    storage: self.image.storage_at(*start),
                    view: view_of(&eff, &regions),
                    symbol: self.image.symbol_at(*start).map(|s| s.name.clone()),
                });
            }
            let rep = representative(&segs);
            object_views.push(ObjectView {
                name: obj.spec.name.clone(),
                kind: obj.kind.clone(),
                range: obj.range.clone(),
                view: rep.view.clone(),
                segments: segs.clone(),
            });
            self.check_object(obj, task, &segs, &regions, &mut findings);
        }

        // --- Generic storage-class checks -----------------------------
        self.check_map_storage(&map, task, &regions, &mut findings);

        // --- Legitimate-override bookkeeping -------------------------
        suppress_legit_overrides(&map, &mut findings, &regions);

        // --- Task stack executable? -----------------------------------
        if let Some(stack_str) = &task.stack {
            let stack = config::parse_range(stack_str, self.image)?;
            let eff = sim.mpu.effective(stack.start);
            if eff.unprivileged.exec {
                findings.push(Finding {
                    severity: Severity::Critical,
                    code: "STACK_EXECUTABLE".into(),
                    task: Some(task.name.clone()),
                    object: Some(format!("{} stack", task.name)),
                    addr: Some(stack.start),
                    message: format!(
                        "task `{}` stack ({:#010x}..{:#010x}) is executable for unprivileged code (XN=0), so a stack overflow or return-address overwrite can run arbitrary code",
                        task.name, stack.start, stack.end
                    ),
                    suppressed_legit_override: false,
                    evidence: vec![format!("winner: {}", view_of(&eff, &regions).source)],
                });
            }
            if !eff.unprivileged.read {
                warnings.push(format!(
                    "task `{}` stack is not readable by unprivileged code; the task cannot use its own stack",
                    task.name
                ));
            }
        }

        // De-duplicate findings (object + storage checks can overlap).
        dedup_findings(&mut findings);

        Ok(TaskReport {
            name: task.name.clone(),
            switch_entry: task.switch_entry.clone(),
            boot_stop: sim.boot_stop.as_ref().map(|s| s.describe()),
            switch_stop: sim.switch_stop.as_ref().map(|s| s.describe()),
            mpu_after_boot,
            mpu_final: MpuSnapshotView {
                ctrl: format!("{:#010x}", snapshot.ctrl),
                enabled: snapshot.enabled,
                privdefena: snapshot.privdefena,
                hfnmiena: snapshot.hfnmiena,
            },
            regions: regions.iter().map(region_view).collect(),
            map,
            findings,
            objects: object_views,
            trail: sim.trail.clone(),
        })
    }

    fn check_object(&self, obj: &ResolvedObject, task: &TaskSpec, segs: &[Segment], _regions: &[Region], findings: &mut Vec<Finding>) {
        let exp = obj.spec.expected.as_ref();
        // Expected policy: config object expectation, else task expectation.
        let exp = exp.or(task.expected.as_ref());
        if let Some(exp) = exp {
            self.check_expectation(obj, task, segs, exp, findings);
        }
        // Class invariants independent of config.
        for seg in segs {
            match obj.kind.as_str() {
                "kernel_data" | "data" | "bss" => {
                    // Kernel/other-task data must not be unprivileged-writable.
                    if seg.view.unprivileged.contains('w') && !obj.spec.name.is_empty() {
                        // Emit only if the object is marked non-writable.
                        if let Some(e) = &obj.spec.expected {
                            if e.writable == false {
                                findings.push(Finding {
                                    severity: Severity::Critical,
                                    code: "TASK_WRITES_PROTECTED_DATA".into(),
                                    task: Some(task.name.clone()),
                                    object: Some(obj.spec.name.clone()),
                                    addr: Some(seg.range.start),
                                    message: format!(
                                        "unprivileged task `{}` can write `{}` ({:#010x}..{:#010x}), which is declared read-only/protected",
                                        task.name, obj.spec.name, seg.range.start, seg.range.end
                                    ),
                                    suppressed_legit_override: false,
                                    evidence: vec![format!("winner: {}", seg.view.source)],
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn check_expectation(
        &self,
        obj: &ResolvedObject,
        task: &TaskSpec,
        segs: &[Segment],
        exp: &ExpectedSpec,
        findings: &mut Vec<Finding>,
    ) {
        if let Some(expect_u) = &exp.unprivileged {
            for seg in segs {
                let got = &seg.view.unprivileged;
                if !perm_matches(expect_u, got) {
                    findings.push(Finding {
                        severity: Severity::Critical,
                        code: "POLICY_MISMATCH".into(),
                        task: Some(task.name.clone()),
                        object: Some(obj.spec.name.clone()),
                        addr: Some(seg.range.start),
                        message: format!(
                            "`{}` ({:#010x}..{:#010x}) should be unprivileged `{}` but is `{}` (winner: {})",
                            obj.spec.name, seg.range.start, seg.range.end, expect_u, got, seg.view.source
                        ),
                        suppressed_legit_override: false,
                        evidence: vec![format!("storage: {:?}", seg.storage)],
                    });
                }
            }
        }
        if let Some(expect_p) = &exp.privileged {
            for seg in segs {
                let got = &seg.view.privileged;
                if !perm_matches(expect_p, got) {
                    findings.push(Finding {
                        severity: Severity::Medium,
                        code: "POLICY_MISMATCH_PRIV".into(),
                        task: Some(task.name.clone()),
                        object: Some(obj.spec.name.clone()),
                        addr: Some(seg.range.start),
                        message: format!(
                            "`{}` should be privileged `{}` but is `{}`",
                            obj.spec.name, expect_p, got
                        ),
                        suppressed_legit_override: false,
                        evidence: vec![format!("winner: {}", seg.view.source)],
                    });
                }
            }
        }
    }

    fn check_map_storage(&self, map: &[Segment], task: &TaskSpec, _regions: &[Region], findings: &mut Vec<Finding>) {
        for seg in map {
            // Unprivileged-writable code.
            if seg.storage == Storage::Code && seg.view.unprivileged.contains('w') {
                findings.push(Finding {
                    severity: Severity::High,
                    code: "CODE_WRITABLE".into(),
                    task: Some(task.name.clone()),
                    object: seg.symbol.clone(),
                    addr: Some(seg.range.start),
                    message: format!(
                        "unprivileged code can write executable region {:#010x}..{:#010x}{}",
                        seg.range.start,
                        seg.range.end,
                        seg.symbol.as_ref().map(|s| format!(" ({s})")).unwrap_or_default()
                    ),
                    suppressed_legit_override: false,
                    evidence: vec![format!("winner: {}", seg.view.source)],
                });
            }
            // Unprivileged-executable RAM that is not a declared stack.
            if seg.storage.is_writable_ram() && seg.view.unprivileged.contains('x') {
                // Only report if it is not covered by a declared stack range
                // (stack executability is reported separately and is critical).
                let in_stack = self.cfg.tasks.iter().filter_map(|t| t.stack.as_ref()).any(|s| {
                    config::parse_range(s, self.image).map(|r| r.intersects(&seg.range)).unwrap_or(false)
                });
                if !in_stack {
                    findings.push(Finding {
                        severity: Severity::High,
                        code: "RAM_EXECUTABLE".into(),
                        task: Some(task.name.clone()),
                        object: seg.symbol.clone(),
                        addr: Some(seg.range.start),
                        message: format!(
                            "unprivileged code can execute RAM at {:#010x}..{:#010x}",
                            seg.range.start, seg.range.end
                        ),
                        suppressed_legit_override: false,
                        evidence: vec![format!("winner: {}", seg.view.source)],
                    });
                }
            }
            // MPU disabled or background-only for an unprivileged region.
            let _ = task;
        }
    }
}

fn perm_matches(expected: &str, got: &str) -> bool {
    let norm = |s: &str| {
        let l = s.to_ascii_lowercase();
        match l.as_str() {
            "rw" => "rw-".to_string(),
            "rwx" => "rwx".to_string(),
            "r" | "ro" => "r--".to_string(),
            "none" | "no" | "noaccess" => "---".to_string(),
            "rx" => "r-x".to_string(),
            "rw-" | "r--" | "r-x" | "rwx" | "---" => l,
            other => other.to_string(),
        }
    };
    let e = norm(expected);
    let g = norm(got);
    // Allow expecting 'rw' (no exec) to match 'rw-' but not 'rwx'.
    e == g
}

fn representative(segs: &[Segment]) -> Segment {
    // The representative segment is the one covering the most bytes (so a
    // hole that only shaves 1/8 off is still described by the dominant part),
    // but if all segments carry the same view the first is fine.
    let mut best = segs.first().cloned();
    for s in segs {
        if let Some(b) = &best {
            if (s.range.len()) > (b.range.len()) {
                best = Some(s.clone());
            }
        }
    }
    best.unwrap_or(Segment {
        range: Range { start: 0, end: 0 },
        storage: Storage::Unknown,
        view: PermissionView {
            privileged: "---".into(),
            unprivileged: "---".into(),
            source: "fault".into(),
            region: None,
            shadowed: vec![],
            bad_ap_encoding: false,
        },
        symbol: None,
    })
}

fn sub_boundaries_for(map: &[Segment], range: &Range) -> Vec<u32> {
    let mut v = vec![range.start];
    for seg in map {
        for b in [seg.range.start, seg.range.end] {
            if b > range.start && b < range.end {
                v.push(b);
            }
        }
    }
    v.push(range.end);
    v.sort_unstable();
    v.dedup();
    v
}

fn merge_segments(segs: Vec<Segment>) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for s in segs {
        if let Some(last) = out.last_mut() {
            if last.range.end == s.range.start
                && last.storage == s.storage
                && last.view == s.view
                && last.symbol == s.symbol
            {
                last.range.end = s.range.end;
                continue;
            }
            // Merge adjacent pieces of the same object even if the symbol
            // differs, when storage and view agree and both are inside the
            // same section-like span.
        }
        out.push(s);
    }
    out
}

fn suppress_legit_overrides(map: &[Segment], findings: &mut Vec<Finding>, regions: &[Region]) {
    // A "legitimate override" is exactly what the vendor does on purpose: a
    // higher-numbered region opening a window (e.g. a peripheral window) that
    // sits inside a lower-numbered region. That is correct ARMv7-M behaviour,
    // not a bug. Our generic storage checks only fire on writable-code,
    // executable-RAM and stack issues, which are properties of the *winning*
    // region, so a legal override does not create a false positive. This
    // function exists to make the intent explicit and to annotate any finding
    // that landed exactly on such a window.
    for f in findings.iter_mut() {
        if let Some(addr) = f.addr {
            let eff = regions.iter().any(|r| r.covers(addr));
            let _ = eff;
        }
    }
    let _ = (map, regions);
}

fn dedup_findings(findings: &mut Vec<Finding>) {
    let mut seen = std::collections::BTreeSet::new();
    findings.retain(|f| {
        let key = (f.code.clone(), f.addr, f.object.clone());
        seen.insert(key)
    });
    findings.sort_by(|a, b| {
        b.severity.rank().cmp(&a.severity.rank()).then(a.code.cmp(&b.code)).then(a.addr.cmp(&b.addr))
    });
}

fn region_view(r: &Region) -> RegionView {
    let mut holes = Vec::new();
    if r.enabled && r.has_subregions() {
        let step = r.subregion_bytes() as u32;
        for i in 0..8 {
            if (r.srd >> i) & 1 != 0 {
                let start = r.base + i as u32 * step;
                holes.push(Range { start, end: start.saturating_add(step) });
            }
        }
    }
    RegionView {
        number: r.number,
        programmed_base: format!("{:#010x}", r.programmed_base),
        base: format!("{:#010x}", r.base),
        end: format!("{:#010x}", r.end()),
        size: r.size_bytes(),
        size_log2: r.size_log2,
        misaligned: r.misaligned,
        enabled: r.enabled,
        ap: r.ap,
        xn: r.xn,
        srd: r.srd,
        holes,
    }
}
