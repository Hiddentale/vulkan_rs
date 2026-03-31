//! Regression tests that run against the real vk.xml registry and generated output.
//!
//! These tests catch classes of bugs that would otherwise only surface when
//! vk.xml is updated to a newer Vulkan version. Each test guards a specific
//! invariant that the generator must uphold across all types and commands.

use std::path::Path;

/// Load the real vk.xml registry for testing.
fn load_registry() -> generator::parse::VkRegistry {
    let vk_xml = Path::new(env!("CARGO_MANIFEST_DIR")).join("vk.xml");
    generator::parse::parse_registry(&vk_xml)
}

/// Read generated file content as a string.
fn read_generated(relative: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    std::fs::read_to_string(root.join(relative))
        .unwrap_or_else(|e| panic!("failed to read {relative}: {e}"))
}

// ═══════════════════════════════════════════════════════════════════
// Test 1: No raw u64/u32 fallback for aliased Flags types
// ═══════════════════════════════════════════════════════════════════

/// If vk.xml defines a bitmask alias (e.g. VkPipelineCreateFlags2KHR →
/// VkPipelineCreateFlags2), the generated output must emit a proper type
/// alias, not a raw `u64` or `u32` fallback.
#[test]
fn no_raw_fallback_for_aliased_flags() {
    let registry = load_registry();
    let structs_rs = read_generated("vk-sys/src/structs.rs");

    // Collect all bitmask alias names from the registry.
    let bitmask_alias_names: Vec<&str> = registry
        .aliases
        .iter()
        .filter(|a| a.name.contains("Flags") && a.target.contains("Flags"))
        .map(|a| a.name.strip_prefix("Vk").unwrap_or(&a.name))
        .collect();

    let mut bad = Vec::new();
    for name in &bitmask_alias_names {
        // Check if this alias appears as `pub type XxxFlags = u32;` or `= u64;`
        let pattern_u32 = format!("pub type {name} = u32;");
        let pattern_u64 = format!("pub type {name} = u64;");
        if structs_rs.contains(&pattern_u32) || structs_rs.contains(&pattern_u64) {
            bad.push(*name);
        }
    }

    assert!(
        bad.is_empty(),
        "These aliased Flags types fell back to raw u32/u64 instead of a proper type alias:\n  {}",
        bad.join("\n  ")
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 4: Two-call method count regression
// ═══════════════════════════════════════════════════════════════════

/// The number of generated two-call methods (enumerate_two_call + fill_two_call)
/// must not decrease. A decrease means a command was misclassified or dropped.
#[test]
fn two_call_method_count_does_not_regress() {
    let instance_wrappers = read_generated("vk-engine/src/generated/instance_wrappers.rs");
    let device_wrappers = read_generated("vk-engine/src/generated/device_wrappers.rs");
    let all = format!("{instance_wrappers}\n{device_wrappers}");

    let enumerate_count = all.matches("enumerate_two_call").count();
    let fill_count = all.matches("fill_two_call").count();
    let total = enumerate_count + fill_count;

    // Current baseline: 51 two-call methods. This must only grow.
    assert!(
        total >= 51,
        "Two-call method count regressed: found {total} (enumerate={enumerate_count}, fill={fill_count}), expected >= 51"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 5: No two-call for single-output commands
// ═══════════════════════════════════════════════════════════════════

/// Every command classified as Enumerate or Fill must actually have the
/// output count/array pair in its parameters (validated via vk.xml metadata).
#[test]
fn two_call_commands_have_output_pairs() {
    let registry = load_registry();
    let pnext = generator::wrapper_utils::build_pnext_struct_set(&registry);

    let mut false_positives = Vec::new();

    for cmd in &registry.commands {
        let roles = generator::wrapper_utils::classify_params(cmd, &pnext);
        let pattern = generator::wrapper_utils::classify_command(cmd, &roles);

        if matches!(
            pattern,
            generator::wrapper_utils::CommandPattern::Enumerate
                | generator::wrapper_utils::CommandPattern::Fill
        ) {
            // Must have at least one OutputArray role.
            let has_output_array = roles
                .iter()
                .any(|r| matches!(r, generator::wrapper_utils::ParamRole::OutputArray { .. }));
            if !has_output_array {
                false_positives.push(cmd.name.as_str());
            }
        }
    }

    assert!(
        false_positives.is_empty(),
        "Commands classified as Enumerate/Fill but missing OutputArray:\n  {}",
        false_positives.join("\n  ")
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 6: Return type consistency for two-call commands
// ═══════════════════════════════════════════════════════════════════

/// Enumerate commands must return VkResult. Fill commands must return void.
#[test]
fn two_call_return_type_consistency() {
    let registry = load_registry();
    let pnext = generator::wrapper_utils::build_pnext_struct_set(&registry);

    let mut mismatches = Vec::new();

    for cmd in &registry.commands {
        let roles = generator::wrapper_utils::classify_params(cmd, &pnext);
        let pattern = generator::wrapper_utils::classify_command(cmd, &roles);

        match pattern {
            generator::wrapper_utils::CommandPattern::Enumerate => {
                if cmd.return_type != "VkResult" {
                    mismatches.push(format!(
                        "{}: Enumerate but returns '{}'",
                        cmd.name, cmd.return_type
                    ));
                }
            }
            generator::wrapper_utils::CommandPattern::Fill => {
                if cmd.return_type != "void" {
                    mismatches.push(format!(
                        "{}: Fill but returns '{}'",
                        cmd.name, cmd.return_type
                    ));
                }
            }
            _ => {}
        }
    }

    assert!(
        mismatches.is_empty(),
        "Two-call return type mismatches:\n  {}",
        mismatches.join("\n  ")
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 7: No duplicate type definitions
// ═══════════════════════════════════════════════════════════════════

/// Scan generated output for duplicate `pub type`, `pub struct`, or
/// `pub union` definitions. Promoted extensions can cause duplication.
#[test]
fn no_duplicate_type_definitions() {
    let files = [
        "vk-sys/src/handles.rs",
        "vk-sys/src/enums.rs",
        "vk-sys/src/bitmasks.rs",
        "vk-sys/src/structs.rs",
        "vk-sys/src/constants.rs",
        "vk-sys/src/commands.rs",
    ];

    let mut all_duplicates = Vec::new();

    for file in &files {
        let content = read_generated(file);
        let mut type_names: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for line in content.lines() {
            let trimmed = line.trim();
            let name = if let Some(rest) = trimmed.strip_prefix("pub struct ") {
                rest.split(|c: char| c == '(' || c == '{' || c == '<' || c.is_whitespace())
                    .next()
            } else if let Some(rest) = trimmed.strip_prefix("pub union ") {
                rest.split(|c: char| c == '{' || c == '<' || c.is_whitespace())
                    .next()
            } else if let Some(rest) = trimmed.strip_prefix("pub type ") {
                rest.split(|c: char| c == '=' || c.is_whitespace()).next()
            } else {
                None
            };

            if let Some(name) = name
                && !name.is_empty()
            {
                *type_names.entry(name.to_string()).or_default() += 1;
            }
        }

        for (name, count) in &type_names {
            if *count > 1 {
                all_duplicates.push(format!("{file}: '{name}' defined {count} times"));
            }
        }
    }

    assert!(
        all_duplicates.is_empty(),
        "Duplicate type definitions found:\n  {}",
        all_duplicates.join("\n  ")
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 8: All struct members resolve to known types
// ═══════════════════════════════════════════════════════════════════

/// Every struct member type in the registry must resolve to either a
/// primitive, a generated type, or a valid alias. This catches broken
/// type references when new types are added to vk.xml.
#[test]
fn all_struct_members_resolve() {
    let registry = load_registry();

    // Collect all known type names (handles, structs, enums, bitmasks, aliases).
    let mut known: std::collections::HashSet<String> = std::collections::HashSet::new();

    for h in &registry.handles {
        known.insert(h.name.clone());
    }
    for s in &registry.structs {
        known.insert(s.name.clone());
    }
    for e in &registry.enums {
        known.insert(e.name.clone());
    }
    for b in &registry.bitmasks {
        known.insert(b.name.clone());
        known.insert(b.flags_name.clone());
    }
    for a in &registry.aliases {
        known.insert(a.name.strip_prefix("Vk").unwrap_or(&a.name).to_string());
        known.insert(a.target.strip_prefix("Vk").unwrap_or(&a.target).to_string());
    }
    for fp in &registry.func_pointers {
        known.insert(fp.name.clone());
    }
    for name in registry.base_types.keys() {
        known.insert(name.clone());
    }

    // Primitives and known C types.
    let primitives = [
        "void", "char", "float", "double", "int", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "int32_t", "int64_t", "size_t",
    ];
    for p in &primitives {
        known.insert(p.to_string());
    }

    // Reserved Flags types (empty bitmasks with no FlagBits) are resolved as
    // u32/u64 aliases in the generated output. They're valid types even though
    // they don't appear in the bitmask list. Recognize any VkXxxFlags pattern.
    let is_known_flags = |name: &str| -> bool {
        let stripped = name.strip_prefix("Vk").unwrap_or(name);
        stripped.contains("Flags") && !stripped.contains("FlagBits")
    };

    let mut unresolved = Vec::new();

    for s in &registry.structs {
        for m in &s.members {
            let stripped = m.type_name.strip_prefix("Vk").unwrap_or(&m.type_name);
            if !known.contains(stripped)
                && !known.contains(&m.type_name)
                && !m.type_name.starts_with("StdVideo")
                && !is_known_flags(&m.type_name)
                && generator::type_map::c_type_to_rust(&m.type_name).is_none()
            {
                unresolved.push(format!("{}.{}: type '{}'", s.name, m.name, m.type_name));
            }
        }
    }

    assert!(
        unresolved.is_empty(),
        "Unresolved struct member types:\n  {}",
        unresolved.join("\n  ")
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 9: pNext builder lifetime safety
// ═══════════════════════════════════════════════════════════════════

/// Every `push_next` method must take `&'a mut T` so the borrow checker
/// ties the pNext target's lifetime to the builder. If this pattern
/// breaks for any struct, pNext chain safety is silently lost.
#[test]
fn push_next_methods_have_lifetime_tie() {
    let builders_rs = read_generated("vk-sys/src/builders.rs");

    let push_next_count = builders_rs.matches("fn push_next").count();
    let lifetime_count = builders_rs.matches("&'a mut T").count();

    assert!(
        push_next_count > 0,
        "No push_next methods found in builders.rs"
    );
    assert_eq!(
        push_next_count, lifetime_count,
        "Some push_next methods are missing the &'a mut T lifetime tie: \
         found {push_next_count} push_next but only {lifetime_count} with &'a mut T"
    );
}

/// Builder count must not decrease. A decrease means extensible structs
/// stopped getting builders, breaking the ergonomic API.
#[test]
fn builder_count_does_not_regress() {
    let builders_rs = read_generated("vk-sys/src/builders.rs");

    let builder_count = builders_rs.matches("pub struct ").count();
    let push_next_count = builders_rs.matches("fn push_next").count();

    // Current baseline: 1227 builders, 978 with push_next.
    assert!(
        builder_count >= 1200,
        "Builder count regressed: found {builder_count}, expected >= 1200"
    );
    assert!(
        push_next_count >= 950,
        "push_next count regressed: found {push_next_count}, expected >= 950"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 10: Type alias targets resolve
// ═══════════════════════════════════════════════════════════════════

/// Every type alias chain must eventually resolve to a concrete type
/// in the registry. A dangling chain means the generator would emit
/// `pub type Foo = Bar;` where `Bar` is never defined.
#[test]
fn type_alias_chains_resolve() {
    let registry = load_registry();

    let strip_vk = |s: &str| -> String { s.strip_prefix("Vk").unwrap_or(s).to_string() };

    // Concrete types (Vk-stripped): handles, structs, enums, bitmasks, etc.
    let mut concrete: std::collections::HashSet<String> = std::collections::HashSet::new();
    for h in &registry.handles {
        concrete.insert(h.name.clone());
    }
    for s in &registry.structs {
        concrete.insert(s.name.clone());
    }
    for e in &registry.enums {
        concrete.insert(e.name.clone());
    }
    for b in &registry.bitmasks {
        concrete.insert(b.name.clone());
        concrete.insert(b.flags_name.clone());
    }
    for fp in &registry.func_pointers {
        concrete.insert(fp.name.clone());
    }
    for name in registry.base_types.keys() {
        concrete.insert(name.clone());
    }

    // Build alias map for chain following. Keys/values are Vk-stripped.
    let alias_map: std::collections::HashMap<String, String> = registry
        .aliases
        .iter()
        .filter(|a| a.kind == generator::parse::AliasKind::Type)
        .map(|a| (strip_vk(&a.name), strip_vk(&a.target)))
        .collect();

    // Flags types (e.g. AccessFlags2) are resolved to their FlagBits
    // form by the generator, so they're valid even if not in the concrete set.
    let is_flags_type =
        |name: &str| -> bool { name.contains("Flags") && !name.contains("FlagBits") };

    let mut dangling = Vec::new();
    for a in &registry.aliases {
        if a.kind != generator::parse::AliasKind::Type {
            continue;
        }

        // Follow the chain up to 10 hops (all Vk-stripped).
        let mut current = strip_vk(&a.target);
        let mut resolved = concrete.contains(&current) || is_flags_type(&current);
        for _ in 0..10 {
            if resolved {
                break;
            }
            if let Some(next) = alias_map.get(&current) {
                current = next.clone();
                resolved = concrete.contains(&current) || is_flags_type(&current);
            } else {
                break;
            }
        }

        if !resolved {
            dangling.push(format!("{} → … → {current}", a.name));
        }
    }

    assert!(
        dangling.is_empty(),
        "Type alias chains with no concrete target:\n  {}",
        dangling.join("\n  ")
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 11: Command alias targets resolve
// ═══════════════════════════════════════════════════════════════════

/// Every command alias must point to a command that exists in the
/// registry. A dangling command alias means the fallback loader will
/// reference a function pointer field that doesn't exist.
#[test]
fn command_alias_targets_resolve() {
    let registry = load_registry();

    let known_commands: std::collections::HashSet<&str> =
        registry.commands.iter().map(|c| c.name.as_str()).collect();

    let mut dangling = Vec::new();
    for a in &registry.aliases {
        if a.kind == generator::parse::AliasKind::Command
            && !known_commands.contains(a.target.as_str())
        {
            dangling.push(format!("{} → {}", a.name, a.target));
        }
    }

    assert!(
        dangling.is_empty(),
        "Command aliases with missing targets:\n  {}",
        dangling.join("\n  ")
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 12: Command dispatch level classification
// ═══════════════════════════════════════════════════════════════════

/// For every command in InstanceCommands, its first parameter must be
/// VkInstance or VkPhysicalDevice. For DeviceCommands, it must be
/// VkDevice, VkQueue, or VkCommandBuffer. Misclassification means
/// loading via the wrong proc addr.
#[test]
fn command_dispatch_level_is_correct() {
    let registry = load_registry();

    let instance_handles = ["VkInstance", "VkPhysicalDevice"];
    let device_handles = ["VkDevice", "VkQueue", "VkCommandBuffer"];

    let mut misclassified = Vec::new();

    for cmd in &registry.commands {
        let first_param_type = cmd
            .params
            .first()
            .map(|p| p.type_name.as_str())
            .unwrap_or("");

        match cmd.dispatch_level {
            generator::parse::DispatchLevel::Instance => {
                if !instance_handles.contains(&first_param_type) {
                    misclassified.push(format!(
                        "{}: Instance-level but first param is '{first_param_type}'",
                        cmd.name
                    ));
                }
            }
            generator::parse::DispatchLevel::Device => {
                if !device_handles.contains(&first_param_type) {
                    misclassified.push(format!(
                        "{}: Device-level but first param is '{first_param_type}'",
                        cmd.name
                    ));
                }
            }
            generator::parse::DispatchLevel::Entry => {
                // Entry commands can have any first param (or none).
            }
        }
    }

    assert!(
        misclassified.is_empty(),
        "Commands with incorrect dispatch level:\n  {}",
        misclassified.join("\n  ")
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 13: Command alias fallback loading
// ═══════════════════════════════════════════════════════════════════

/// Every command alias must produce a fallback load in the generated
/// commands.rs. Without it, calling the promoted name on an older
/// driver silently returns None instead of trying the extension name.
#[test]
fn command_aliases_have_fallback_loading() {
    let registry = load_registry();
    let commands_rs = read_generated("vk-sys/src/commands.rs");

    let mut missing = Vec::new();
    for a in &registry.aliases {
        if a.kind != generator::parse::AliasKind::Command {
            continue;
        }
        // The alias name (e.g. "vkCreateRenderPass2KHR") should appear
        // in a cstr literal as a fallback load target.
        let needle = format!("{}\\0", a.name);
        if !commands_rs.contains(&needle) {
            missing.push(a.name.as_str());
        }
    }

    assert!(
        missing.is_empty(),
        "Command aliases missing fallback loading in commands.rs:\n  {}",
        missing.join("\n  ")
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 14: All structs have spec links
// ═══════════════════════════════════════════════════════════════════

/// Every struct in the generated output must have a Khronos spec link
/// in a preceding doc comment. Missing links mean the generator's doc
/// pipeline broke for that type.
#[test]
fn all_structs_have_spec_links() {
    let structs_rs = read_generated("vk-sys/src/structs.rs");

    let mut missing = Vec::new();
    let lines: Vec<&str> = structs_rs.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("pub struct ") {
            let name = rest
                .split(|c: char| c == '{' || c == '(' || c == '<' || c.is_whitespace())
                .next()
                .unwrap_or("");
            if name.is_empty()
                || name == "BaseOutStructure"
                || name == "BaseInStructure"
                || name.starts_with("StdVideo")
            {
                continue;
            }
            // Search preceding lines for a spec link.
            // Some structs have very long Extended By lists (300+ lines).
            let start = i.saturating_sub(500);
            let has_link = lines[start..i]
                .iter()
                .any(|l| l.contains("registry.khronos.org"));
            if !has_link {
                missing.push(name.to_string());
            }
        }
    }

    assert!(
        missing.is_empty(),
        "Structs missing spec links:\n  {}",
        missing.join("\n  ")
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 15: All wrapper methods have spec links
// ═══════════════════════════════════════════════════════════════════

/// Every generated wrapper method must have a Khronos spec link.
#[test]
fn all_wrapper_methods_have_spec_links() {
    let instance_wrappers = read_generated("vk-engine/src/generated/instance_wrappers.rs");
    let device_wrappers = read_generated("vk-engine/src/generated/device_wrappers.rs");
    let all = format!("{instance_wrappers}\n{device_wrappers}");

    let mut missing = Vec::new();
    let lines: Vec<&str> = all.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("pub unsafe fn ") {
            let name = trimmed
                .strip_prefix("pub unsafe fn ")
                .unwrap_or("")
                .split('(')
                .next()
                .unwrap_or("");
            // Doc overrides can push the spec link further back.
            let start = i.saturating_sub(50);
            let has_link = lines[start..i]
                .iter()
                .any(|l| l.contains("registry.khronos.org"));
            if !has_link {
                missing.push(name.to_string());
            }
        }
    }

    assert!(
        missing.is_empty(),
        "Wrapper methods missing spec links:\n  {}",
        missing.join("\n  ")
    );
}

// ═══════════════════════════════════════════════════════════════════
// Test 16: All wrapper methods returning Result have # Errors
// ═══════════════════════════════════════════════════════════════════

/// Every generated wrapper whose return type contains `Result` AND
/// whose command has error codes in vk.xml must have an `# Errors`
/// section. Commands without error codes in vk.xml are exempt.
#[test]
fn result_returning_wrappers_have_error_docs() {
    let registry = load_registry();
    let instance_wrappers = read_generated("vk-engine/src/generated/instance_wrappers.rs");
    let device_wrappers = read_generated("vk-engine/src/generated/device_wrappers.rs");
    let all = format!("{instance_wrappers}\n{device_wrappers}");

    // Build set of commands that have error codes.
    let cmds_with_errors: std::collections::HashSet<String> = registry
        .commands
        .iter()
        .filter(|c| !c.error_codes.is_empty())
        .map(|c| {
            // Convert vkCmdFoo to snake_case field name.
            let stripped = c.name.strip_prefix("vk").unwrap_or(&c.name);
            <str as heck::ToSnakeCase>::to_snake_case(stripped)
        })
        .collect();

    let mut missing = Vec::new();
    let lines: Vec<&str> = all.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if !trimmed.starts_with("pub unsafe fn ") {
            continue;
        }
        let name = trimmed
            .strip_prefix("pub unsafe fn ")
            .unwrap_or("")
            .split('(')
            .next()
            .unwrap_or("");

        // Only check commands that have error codes in the registry.
        if !cmds_with_errors.contains(name) {
            continue;
        }

        let start = i.saturating_sub(25);
        let has_errors = lines[start..i].iter().any(|l| l.contains("# Errors"));
        if !has_errors {
            missing.push(name.to_string());
        }
    }

    assert!(
        missing.is_empty(),
        "Result-returning wrappers missing # Errors section:\n  {}",
        missing.join("\n  ")
    );
}
