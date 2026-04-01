//! Integration test: runs the generator binary and verifies vk-sys compiles.
//!
//! This catches type resolution bugs that unit tests miss,any broken type
//! across 161k lines of generated Rust will surface as a compile error.
//!
//! These tests share generated output files, so they run under a process-wide
//! mutex to prevent races when cargo runs tests in parallel.

use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

/// Serialize all generator integration tests to avoid parallel writes to
/// the same output files.
static GENERATOR_LOCK: Mutex<()> = Mutex::new(());

fn cargo() -> Command {
    let mut cmd = Command::new(env!("CARGO"));
    cmd.current_dir(workspace_root());
    cmd
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR has parent")
}

#[test]
fn generator_runs_successfully() {
    let _lock = GENERATOR_LOCK.lock().expect("GENERATOR_LOCK poisoned");
    let output = cargo()
        .args(["run", "-p", "generator"])
        .output()
        .expect("failed to launch generator");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "generator failed:\n{stderr}");
}

#[test]
fn generated_output_compiles() {
    let _lock = GENERATOR_LOCK.lock().expect("GENERATOR_LOCK poisoned");
    // Ensure generator has run first.
    let run = cargo()
        .args(["run", "-p", "generator"])
        .output()
        .expect("failed to launch generator");
    assert!(run.status.success(), "generator failed before build check");

    let output = cargo()
        .args(["build", "-p", "vk-sys"])
        .output()
        .expect("failed to launch cargo build");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "vk-sys failed to compile:\n{stderr}"
    );
}

/// Running the generator twice must produce byte-identical output.
///
/// Catches non-determinism (HashMap iteration order leaking into output).
/// Runs two sequential generator invocations and compares their output,
/// making the test self-contained regardless of parallel test execution.
#[test]
fn generator_output_is_deterministic() {
    let _lock = GENERATOR_LOCK.lock().expect("GENERATOR_LOCK poisoned");
    let root = workspace_root();

    let generated_files: Vec<std::path::PathBuf> = vec![
        // vk-sys
        root.join("vk-sys/src/handles.rs"),
        root.join("vk-sys/src/enums.rs"),
        root.join("vk-sys/src/bitmasks.rs"),
        root.join("vk-sys/src/constants.rs"),
        root.join("vk-sys/src/structs.rs"),
        root.join("vk-sys/src/builders.rs"),
        root.join("vk-sys/src/commands.rs"),
        root.join("vk-sys/src/lib.rs"),
        // vk-engine wrappers
        root.join("vk-engine/src/generated/entry_wrappers.rs"),
        root.join("vk-engine/src/generated/instance_wrappers.rs"),
        root.join("vk-engine/src/generated/device_wrappers.rs"),
        root.join("vk-engine/src/generated/mod.rs"),
    ];

    // Run 1.
    let run1 = cargo()
        .args(["run", "-p", "generator"])
        .output()
        .expect("failed to launch generator (run 1)");
    assert!(run1.status.success(), "generator run 1 failed");

    // Snapshot after run 1.
    let snapshot: Vec<(&std::path::Path, Vec<u8>)> = generated_files
        .iter()
        .map(|p| {
            let content =
                std::fs::read(p).unwrap_or_else(|e| panic!("failed to read {}: {e}", p.display()));
            (p.as_path(), content)
        })
        .collect();

    // Run 2.
    let run2 = cargo()
        .args(["run", "-p", "generator"])
        .output()
        .expect("failed to launch generator (run 2)");
    assert!(run2.status.success(), "generator run 2 failed");

    // Compare run 1 vs run 2.
    let mut mismatches = Vec::new();
    for (path, original) in &snapshot {
        let regenerated = std::fs::read(path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        if *original != regenerated {
            mismatches.push(path.display().to_string());
        }
    }

    assert!(
        mismatches.is_empty(),
        "Generator output is non-deterministic,these files changed between run 1 and run 2:\n  {}",
        mismatches.join("\n  ")
    );
}
