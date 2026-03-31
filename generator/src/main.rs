use std::fs;
use std::path::Path;

use generator::{
    emit_bitmasks, emit_builders, emit_commands, emit_constants, emit_enums, emit_handles,
    emit_layout_check, emit_structs, emit_wrappers, parse, validate,
};

fn main() {
    let vk_xml = Path::new(env!("CARGO_MANIFEST_DIR")).join("vk.xml");
    let registry = parse::parse_registry(&vk_xml);

    print_summary(&registry);
    validate::check_type_completeness(&registry);

    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vk-sys/src");

    write_module(
        &out_dir,
        "handles.rs",
        emit_handles::emit_handles(&registry),
    );
    write_module(&out_dir, "enums.rs", emit_enums::emit_enums(&registry));
    write_module(
        &out_dir,
        "bitmasks.rs",
        emit_bitmasks::emit_bitmasks(&registry),
    );
    write_module(
        &out_dir,
        "constants.rs",
        emit_constants::emit_constants(&registry),
    );
    write_module(
        &out_dir,
        "structs.rs",
        emit_structs::emit_structs(&registry),
    );
    write_module(
        &out_dir,
        "builders.rs",
        emit_builders::emit_builders(&registry),
    );
    write_module(
        &out_dir,
        "commands.rs",
        emit_commands::emit_commands(&registry),
    );

    update_lib_rs(&out_dir);

    // Generate C ↔ Rust cross-validation programs.
    let test_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vk-sys/tests");
    let c_check = emit_layout_check::emit_c_layout_check(&registry);
    let c_path = test_dir.join("c_layout_check.c");
    fs::write(&c_path, &c_check).unwrap_or_else(|e| {
        panic!("failed to write {}: {e}", c_path.display());
    });
    println!("  wrote c_layout_check.c ({} lines)", c_check.lines().count());

    let rs_check = emit_layout_check::emit_rust_layout_check(&registry);
    let bin_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vk-sys/src/bin");
    fs::create_dir_all(&bin_dir).unwrap_or_else(|e| {
        panic!("failed to create {}: {e}", bin_dir.display());
    });
    let rs_path = bin_dir.join("rust_layout_check.rs");
    fs::write(&rs_path, &rs_check).unwrap_or_else(|e| {
        panic!("failed to write {}: {e}", rs_path.display());
    });
    println!("  wrote rust_layout_check.rs ({} lines)", rs_check.lines().count());

    // Generate ergonomic wrapper methods for vk-engine.
    let engine_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vk-engine/src/generated");
    fs::create_dir_all(&engine_dir).unwrap_or_else(|e| {
        panic!("failed to create {}: {e}", engine_dir.display());
    });

    let (entry_wrappers, instance_wrappers, device_wrappers) =
        emit_wrappers::emit_wrappers(&registry);

    write_module(&engine_dir, "entry_wrappers.rs", entry_wrappers);
    write_module(&engine_dir, "instance_wrappers.rs", instance_wrappers);
    write_module(&engine_dir, "device_wrappers.rs", device_wrappers);
    write_engine_mod_rs(&engine_dir);

    // Run rustfmt on vk-engine generated files so the output matches
    // `cargo fmt` exactly. prettyplease and rustfmt disagree on import
    // ordering, line wrapping, and argument formatting.
    // vk-sys is skipped — it has `disable_all_formatting = true`.
    rustfmt_engine();

    println!("\n=== generation complete ===");
    println!("  vk-sys output:   {}", out_dir.display());
    println!("  vk-engine output: {}", engine_dir.display());
}

fn write_module(out_dir: &Path, filename: &str, tokens: proc_macro2::TokenStream) {
    let file = syn::parse2(tokens).unwrap_or_else(|e| {
        panic!("failed to parse generated {filename}: {e}");
    });
    let formatted = prettyplease::unparse(&file);

    let path = out_dir.join(filename);
    fs::write(&path, &formatted).unwrap_or_else(|e| {
        panic!("failed to write {}: {e}", path.display());
    });

    let lines = formatted.lines().count();
    println!("  wrote {filename} ({lines} lines)");
}

fn rustfmt_engine() {
    let status = std::process::Command::new(env!("CARGO"))
        .args(["fmt", "-p", "vk-engine"])
        .current_dir(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("CARGO_MANIFEST_DIR has parent"),
        )
        .status();
    match status {
        Ok(s) if s.success() => println!("  rustfmt vk-engine: ok"),
        Ok(s) => eprintln!("  warning: cargo fmt exited with {s}"),
        Err(e) => eprintln!("  warning: cargo fmt not available ({e}), skipping"),
    }
}

fn update_lib_rs(out_dir: &Path) {
    let content = "\
//! Raw Vulkan FFI types generated from `vk.xml`.
//!
//! Do not edit by hand — regenerate with the `generator` crate.
//!
//! Every type carries a spec link, and structs include metadata from
//! vk.xml: extension provenance, pNext chain relationships, member
//! annotations (optional, length-of, thread safety).

#![no_std]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

pub mod handles;
pub mod enums;
pub mod bitmasks;
pub mod constants;
pub mod structs;
pub mod builders;
pub mod commands;
";
    let path = out_dir.join("lib.rs");
    fs::write(&path, content).unwrap_or_else(|e| {
        panic!("failed to write {}: {e}", path.display());
    });
}

fn write_engine_mod_rs(out_dir: &Path) {
    let content = "\
//! Generated wrapper methods for Entry, Instance, and Device.
//!
//! These methods are auto-generated from `vk.xml` by the `generator` crate.
//! Do not edit by hand — run `cargo run -p generator` to regenerate.
//!
//! Each method wraps a single Vulkan command, adding:
//! - Output-parameter returns (instead of out-pointer + `VkResult`)
//! - Two-call enumeration for array-returning commands
//! - Spec links, error codes, and safety documentation

mod entry_wrappers;
mod instance_wrappers;
mod device_wrappers;
";
    let path = out_dir.join("mod.rs");
    fs::write(&path, content).unwrap_or_else(|e| {
        panic!("failed to write {}: {e}", path.display());
    });
}

fn print_summary(registry: &parse::VkRegistry) {
    use parse::DispatchLevel;

    let unions = registry.structs.iter().filter(|s| s.is_union).count();
    let structs = registry.structs.len() - unions;
    let bitmasks_64 = registry
        .bitmasks
        .iter()
        .filter(|b| b.bitwidth == 64)
        .count();
    let entry_cmds = registry
        .commands
        .iter()
        .filter(|c| c.dispatch_level == DispatchLevel::Entry)
        .count();
    let instance_cmds = registry
        .commands
        .iter()
        .filter(|c| c.dispatch_level == DispatchLevel::Instance)
        .count();
    let device_cmds = registry
        .commands
        .iter()
        .filter(|c| c.dispatch_level == DispatchLevel::Device)
        .count();

    println!("=== vk.xml parse summary ===");
    println!("  handles:       {}", registry.handles.len());
    println!("  structs:       {structs}");
    println!("  unions:        {unions}");
    println!("  enums:         {}", registry.enums.len());
    println!(
        "  bitmasks:      {} ({bitmasks_64} are 64-bit)",
        registry.bitmasks.len()
    );
    println!(
        "  commands:      {} (entry={entry_cmds}, instance={instance_cmds}, device={device_cmds})",
        registry.commands.len()
    );
    println!("  constants:     {}", registry.constants.len());
    println!("  funcpointers:  {}", registry.func_pointers.len());
    println!("  extensions:    {}", registry.extensions.len());
    println!("  platforms:     {}", registry.platforms.len());
    println!(
        "  aliases:       {} (type={}, command={}, bitmask={})",
        registry.aliases.len(),
        registry
            .aliases
            .iter()
            .filter(|a| a.kind == parse::AliasKind::Type)
            .count(),
        registry
            .aliases
            .iter()
            .filter(|a| a.kind == parse::AliasKind::Command)
            .count(),
        registry
            .aliases
            .iter()
            .filter(|a| a.kind == parse::AliasKind::Bitmask)
            .count(),
    );
}
