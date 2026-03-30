//! Integration test: runs the generator binary and verifies vk-sys compiles.
//!
//! This catches type resolution bugs that unit tests miss — any broken type
//! across 161k lines of generated Rust will surface as a compile error.

use std::path::Path;
use std::process::Command;

fn cargo() -> Command {
    let mut cmd = Command::new(env!("CARGO"));
    cmd.current_dir(workspace_root());
    cmd
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

#[test]
fn generator_runs_successfully() {
    let output = cargo()
        .args(["run", "-p", "generator"])
        .output()
        .expect("failed to launch generator");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "generator failed:\n{stderr}"
    );
}

#[test]
fn generated_output_compiles() {
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
