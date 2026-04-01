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

    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vulkan-rs-sys/src");

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
    let test_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vulkan-rs-sys/tests");
    let c_check = emit_layout_check::emit_c_layout_check(&registry);
    let c_path = test_dir.join("c_layout_check.c");
    fs::write(&c_path, &c_check).unwrap_or_else(|e| {
        panic!("failed to write {}: {e}", c_path.display());
    });
    println!(
        "  wrote c_layout_check.c ({} lines)",
        c_check.lines().count()
    );

    let rs_check = emit_layout_check::emit_rust_layout_check(&registry);
    let bin_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vulkan-rs-sys/src/bin");
    fs::create_dir_all(&bin_dir).unwrap_or_else(|e| {
        panic!("failed to create {}: {e}", bin_dir.display());
    });
    let rs_path = bin_dir.join("rust_layout_check.rs");
    fs::write(&rs_path, &rs_check).unwrap_or_else(|e| {
        panic!("failed to write {}: {e}", rs_path.display());
    });
    println!(
        "  wrote rust_layout_check.rs ({} lines)",
        rs_check.lines().count()
    );

    // Generate ergonomic wrapper methods for vulkan-rs.
    let engine_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vulkan-rs/src/generated");
    fs::create_dir_all(&engine_dir).unwrap_or_else(|e| {
        panic!("failed to create {}: {e}", engine_dir.display());
    });

    let (entry_wrappers, instance_wrappers, device_wrappers) =
        emit_wrappers::emit_wrappers(&registry);

    write_module(&engine_dir, "entry_wrappers.rs", entry_wrappers);
    write_module(&engine_dir, "instance_wrappers.rs", instance_wrappers);
    write_module(&engine_dir, "device_wrappers.rs", device_wrappers);
    write_engine_mod_rs(&engine_dir);

    // Run rustfmt on vulkan-rs generated files so the output matches
    // `cargo fmt` exactly. prettyplease and rustfmt disagree on import
    // ordering, line wrapping, and argument formatting.
    // vulkan-rs-sys is skipped,it has `disable_all_formatting = true`.
    rustfmt_engine();

    println!("\n=== generation complete ===");
    println!("  vulkan-rs-sys output:   {}", out_dir.display());
    println!("  vulkan-rs output: {}", engine_dir.display());
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
        .args(["fmt", "-p", "vulkan-rs"])
        .current_dir(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("CARGO_MANIFEST_DIR has parent"),
        )
        .status();
    match status {
        Ok(s) if s.success() => println!("  rustfmt vulkan-rs: ok"),
        Ok(s) => eprintln!("  warning: cargo fmt exited with {s}"),
        Err(e) => eprintln!("  warning: cargo fmt not available ({e}), skipping"),
    }
}

fn update_lib_rs(out_dir: &Path) {
    let content = "\
//! Raw Vulkan FFI types generated from `vk.xml`.
//!
//! Do not edit by hand,regenerate with the `generator` crate.
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
//! Do not edit by hand,run `cargo run -p generator` to regenerate.
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

#[cfg(test)]
fn empty_registry() -> parse::VkRegistry {
    parse::VkRegistry {
        handles: vec![],
        structs: vec![],
        enums: vec![],
        bitmasks: vec![],
        commands: vec![],
        constants: vec![],
        func_pointers: vec![],
        extensions: vec![],
        platforms: vec![],
        aliases: vec![],
        base_types: Default::default(),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    // -- write_module ---------------------------------------------------------

    #[test]
    fn write_module_creates_formatted_file() {
        let dir = std::env::temp_dir().join("gen_test_write_module");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let tokens = quote::quote! {
            pub const FOO: u32 = 42u32;
        };
        write_module(&dir, "test_out.rs", tokens);

        let content = fs::read_to_string(dir.join("test_out.rs")).unwrap();
        assert!(content.contains("pub const FOO: u32 = 42u32;"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[should_panic(expected = "failed to parse")]
    fn write_module_panics_on_invalid_tokens() {
        let dir = std::env::temp_dir().join("gen_test_write_invalid");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        // Valid tokens but not a valid syn::File (bare expression).
        let tokens = quote::quote! { 1 + 2 };
        write_module(&dir, "bad.rs", tokens);
    }

    // -- update_lib_rs --------------------------------------------------------

    #[test]
    fn update_lib_rs_writes_expected_content() {
        let dir = std::env::temp_dir().join("gen_test_lib_rs");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        update_lib_rs(&dir);

        let content = fs::read_to_string(dir.join("lib.rs")).unwrap();
        assert!(content.contains("#![no_std]"));
        assert!(content.contains("pub mod handles;"));
        assert!(content.contains("pub mod enums;"));
        assert!(content.contains("pub mod bitmasks;"));
        assert!(content.contains("pub mod constants;"));
        assert!(content.contains("pub mod structs;"));
        assert!(content.contains("pub mod builders;"));
        assert!(content.contains("pub mod commands;"));

        let _ = fs::remove_dir_all(&dir);
    }

    // -- write_engine_mod_rs --------------------------------------------------

    #[test]
    fn write_engine_mod_rs_writes_expected_content() {
        let dir = std::env::temp_dir().join("gen_test_engine_mod");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        write_engine_mod_rs(&dir);

        let content = fs::read_to_string(dir.join("mod.rs")).unwrap();
        assert!(content.contains("mod entry_wrappers;"));
        assert!(content.contains("mod instance_wrappers;"));
        assert!(content.contains("mod device_wrappers;"));

        let _ = fs::remove_dir_all(&dir);
    }

    // -- print_summary --------------------------------------------------------

    #[test]
    fn print_summary_empty_registry() {
        // Should not panic with an empty registry.
        print_summary(&empty_registry());
    }

    #[test]
    fn print_summary_with_data() {
        let mut registry = empty_registry();
        registry.handles.push(parse::HandleDef {
            name: "VkInstance".to_string(),
            dispatchable: true,
            parent: None,
            object_type: Some("VK_OBJECT_TYPE_INSTANCE".to_string()),
            provided_by: None,
        });
        registry.structs.push(parse::StructDef {
            name: "VkExtent2D".to_string(),
            members: vec![],
            extends: vec![],
            returned_only: false,
            is_union: false,
            provided_by: None,
        });
        registry.structs.push(parse::StructDef {
            name: "VkClearValue".to_string(),
            members: vec![],
            extends: vec![],
            returned_only: false,
            is_union: true,
            provided_by: None,
        });
        registry.enums.push(parse::EnumDef {
            name: "VkResult".to_string(),
            variants: vec![],
        });
        registry.bitmasks.push(parse::BitmaskDef {
            name: "VkAccessFlags2".to_string(),
            flags_name: "VkAccessFlagBits2".to_string(),
            bitwidth: 64,
            bits: vec![],
        });
        registry.commands.push(parse::CommandDef {
            name: "vkCreateInstance".to_string(),
            return_type: "VkResult".to_string(),
            params: vec![],
            success_codes: vec![],
            error_codes: vec![],
            dispatch_level: parse::DispatchLevel::Entry,
            provided_by: None,
        });
        registry.aliases.push(parse::AliasDef {
            name: "VkFoo".to_string(),
            target: "VkBar".to_string(),
            kind: parse::AliasKind::Type,
        });
        registry.aliases.push(parse::AliasDef {
            name: "vkFoo".to_string(),
            target: "vkBar".to_string(),
            kind: parse::AliasKind::Command,
        });

        // Should not panic and should count correctly.
        print_summary(&registry);
    }

    // -- write_module round-trip formatting ------------------------------------

    #[test]
    fn write_module_formats_with_prettyplease() {
        let dir = std::env::temp_dir().join("gen_test_pretty");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        // Intentionally ugly token stream.
        let tokens = quote::quote! {
            pub fn foo(a:u32,b:u32,c:u32)->u32{a+b+c}
        };
        write_module(&dir, "pretty.rs", tokens);

        let content = fs::read_to_string(dir.join("pretty.rs")).unwrap();
        // prettyplease should add spaces and formatting.
        assert!(content.contains("pub fn foo"));
        assert!(content.contains("-> u32"));

        let _ = fs::remove_dir_all(&dir);
    }
}
