use anyhow::Result;
use clap::{Parser, Subcommand};
use mpu_audit::{engine, report, symbol::FirmwareImage};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "mpu-audit", version, about = "Offline MPU permission auditor for ARMv7-M firmware",
    long_about = "Replays the MPU region-register writes performed by a firmware's boot path and by \
each task-switch path, then reconstructs the memory-access permission view actually in effect while \
each task runs. Checks sub-region holes, overlap precedence, AP/XN combination errors, and flags \
privileged-data writes or executable stacks.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Audit the effective MPU permission map of every task in a TOML config.
    Audit {
        #[arg(long)] elf: PathBuf,
        #[arg(long)] config: PathBuf,
        /// Optional extra symbol file (GNU ld .map or `nm` output).
        #[arg(long)] symbols: Option<PathBuf>,
        #[arg(long)] json: bool,
        #[arg(long)] trace: bool,
        #[arg(long)] strict: bool,
    },
    /// Dump sections, symbols and their storage class.
    Inspect {
        #[arg(long)] elf: PathBuf,
        #[arg(long)] symbols: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Audit { elf, config, symbols, json, trace, strict } => {
            let report = engine::run_audit(&elf, &config, symbols.as_deref(), trace)?;
            if json {
                println!("{}", report::to_json(&report)?);
            } else {
                report::print_human(&report);
            }
            if strict && report.has_blocking_findings() {
                std::process::exit(2);
            }
        }
        Command::Inspect { elf, symbols } => {
            let image = FirmwareImage::load(&elf, symbols.as_deref())?;
            report::print_inspect(&image);
        }
    }
    Ok(())
}
